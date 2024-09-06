# -*- coding: utf-8 -*-
import random
from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb_client import WritePrecision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-1])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from res.config_cloud import Config
from res.power import *
import threading
import time
import pika
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import json
from influxdb_client.client.flux_table import FluxStructureEncoder
from res.utils import MQ_decode


# from config import Config

class Monitor:
    def __init__(self, args, monitor_path, agent_num, eliminate):
        self.host = args.cloudhost
        self.mqport = args.mqport
        self.influxdb_port = args.influxdb_port
        self.bucker_name = args.bucker_name  # 数据库名称
        self.org = args.org  # 组织
        self.INFLUX_TOKEN = args.INFLUX_TOKEN
        self.client = influxdb_client.InfluxDBClient(url='http://' + self.host + ':' + str(self.influxdb_port),
                                                     token=self.INFLUX_TOKEN, org=self.org)
        self.client_aux = influxdb_client.InfluxDBClient(url='http://' + self.host + ':' + str(self.influxdb_port),
                                                     token=self.INFLUX_TOKEN, org=self.org)
        self.inference_result_channel = self._declare_mq(self.host, self.mqport, args.mquser, args.mqpw)
        # self.power_get_channel = self._declare_mq(self.host, self.mqport, args.mquser, args.mqpw)
        self.ppo_update_channel = self._declare_mq(self.host, self.mqport, args.mquser, args.mqpw)
        self.queue_channel = self._declare_mq(self.host, self.mqport, args.mquser, args.mqpw)
        self.bandwidth_channel = self._declare_mq(self.host, self.mqport, args.mquser, args.mqpw)
        self.monitor_path = monitor_path
        self.agent_num = agent_num
        self.update_times = 1
        self.latency_edge = []
        self.latency_cloud = []
        self.latency_all = []
        self.PPOUpdate_flag = False  # 各个设备是否需要更新PPO模型的标志

        self.start_flag = False
        self.eliminate = eliminate
        self.start_flag_ensure = [0] * agent_num

        upper_bound_acc_path = ROOT + "/res/max_acc.txt"
        self.df = pd.read_csv(upper_bound_acc_path, header=None, names=['videoID', 'chunkID', 'acc'])

        # delete measurement
        # start = "1970-01-01T00:00:00Z"
        # stop = "2024-07-27T21:00:00Z"
        # delete_api = self.client.delete_api()
        # # delete_api.delete(start, stop, '_measurement="all_power"', bucket=self.bucker_name, org=self.org)
        # delete_api.delete(start, stop, '_measurement="latency"', bucket=self.bucker_name, org=self.org)
        # delete_api.delete(start, stop, '_measurement="acc"', bucket=self.bucker_name, org=self.org)
        # delete_api.delete(start, stop, '_measurement="throughput"', bucket=self.bucker_name, org=self.org)
        # delete_api.delete(start, stop, '_measurement="queue_size"', bucket=self.bucker_name, org=self.org)
        # delete_api.delete(start, stop, '_measurement="queue_wait"', bucket=self.bucker_name, org=self.org)
        # delete_api.delete(start, stop, '_measurement="bandwidth"', bucket=self.bucker_name, org=self.org)
        # print("delete")
        # latency acc throughput
        latency_acc_throughput_monitor = threading.Thread(target=self._latency_acc_throughput_monitor)
        latency_acc_throughput_monitor.start()

        # queue and bandwidth
        queue_monitor = threading.Thread(target=self._queue_monitor)
        queue_monitor.start()

        bandwidth_monitor = threading.Thread(target=self._bandwidth_monitor)
        bandwidth_monitor.start()

        ppo_update = threading.Thread(target=self.ppo_update)
        ppo_update.start()

    def _declare_mq(self, host='10.12.11.144', port=5672, username='guest', password='guest'):
        user_info = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host, port, credentials=user_info, blocked_connection_timeout=999999,
                                      heartbeat=0))  # 连接服务器上的RabbitMQ服务

        # 创建channel
        channel = connection.channel()
        print('connect to rabbitmq')
        ### TODO declare more queues for the system
        return channel

    def ppo_update(self):

        self.ppo_update_channel.queue_declare(queue='ppo_update_flag')

        def callback(ch, method, properties, body):
            data = MQ_decode(body)
            if data['flag'] == 2 and not self.start_flag:
                self.start_flag_ensure[data['device_ID'] - 1] = 1
                if np.sum(self.start_flag_ensure) == self.agent_num:
                    self.start_flag = True
                    self.start_flag_ensure = [0] * self.agent_num

        self.ppo_update_channel.basic_consume(queue='ppo_update_flag', on_message_callback=callback, auto_ack=True)

        print('ppo_update_flag start consuming!!')
        self.ppo_update_channel.start_consuming()

    def _latency_acc_throughput_monitor(self):

        self.inference_result_channel.queue_declare(queue='inference_result')

        def callback(ch, method, properties, body):
            data = MQ_decode(body)
            device_ID = str(data['device_ID'])
            produce_device = str(data['produce_device'])
            latency = time.time() - data['timestamp']

            # if device_ID == '0':
            #     self.latency_cloud.append(latency)
            # else:
            #     self.latency_edge.append(latency)
            write_api = self.client.write_api(write_options=SYNCHRONOUS)

            # with open(self.monitor_path + produce_device + ".txt", "a") as file:
            #     # with open("./experiment/11.21_w2_L1_train_four_device_episode100_big_bitrate_acc_agent" + produce_device + ".txt", "a") as file:
            #     file.write(str(data['video_ID']) + ', ' + str(data['chunk_ID']) + ', ' + str(data['acc_chunk']) + '\n')

            latency = influxdb_client.Point("latency").tag("deviceID", device_ID).tag("produce_device",produce_device).tag("step_index", data['step_index']).field("latency", latency).time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=self.bucker_name, record=latency)

            acc = influxdb_client.Point("acc").tag("deviceID", device_ID).tag("produce_device", produce_device).tag("step_index", data['step_index']).field("videoID", data['video_ID']).field("chunkID",data['chunk_ID']).field("acc", data['acc_chunk']).time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=self.bucker_name, record=acc)

            throughput = influxdb_client.Point("throughput").tag("deviceID", device_ID).tag("produce_device",produce_device).tag("step_index", data['step_index']).field("throughput", data['frame_num']).time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=self.bucker_name, record=throughput)

            queue_wait = influxdb_client.Point("queue_wait").tag("deviceID", device_ID).tag("produce_device",produce_device).tag("step_index", data['step_index']).field("queue_wait", data['queue_wait']).time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=self.bucker_name, record=queue_wait)

        self.inference_result_channel.basic_consume(queue='inference_result', on_message_callback=callback,
                                                    auto_ack=True)

        print('Result start consuming!!')
        self.inference_result_channel.start_consuming()

    def _queue_monitor(self):
        self.queue_channel.queue_declare(queue='queue_size')

        def callback(ch, method, properties, body):
            data = MQ_decode(body)
            device_ID = str(data['device_ID'])
            write_api = self.client.write_api(write_options=SYNCHRONOUS)

            queue_size = influxdb_client.Point("queue_size").tag("deviceID", device_ID).field("transmission_queue",data['transmission_queue']). \
                field("inference_queue", data['inference_queue']).time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=self.bucker_name, record=queue_size)

        self.queue_channel.basic_consume(queue='queue_size', on_message_callback=callback, auto_ack=True)

        print('Queue start consuming!!')
        self.queue_channel.start_consuming()

    def _bandwidth_monitor(self):
        self.bandwidth_channel.queue_declare(queue='bandwidth')

        def callback(ch, method, properties, body):
            data = MQ_decode(body)
            device_ID = str(data['device_ID'])
            write_api = self.client.write_api(write_options=SYNCHRONOUS)
            bandwidth = influxdb_client.Point("bandwidth").tag("deviceID", device_ID).field("bandwidth",data['bandwidth']).time(datetime.utcnow(), WritePrecision.NS)
            write_api.write(bucket=self.bucker_name, record=bandwidth)

        self.bandwidth_channel.basic_consume(queue='bandwidth', on_message_callback=callback, auto_ack=True)

        print('Bandwidth start consuming!!')
        self.bandwidth_channel.start_consuming()

    def _get_throughput(self, atime, agent_num, step):
        all_throughput = []
        # 获取每秒系统能够处理的帧数
        """
                from(bucket: "Monitor") |> range(start: start: -24h)
                |> filter(fn: (r) => r["_measurement"] == "throughput")
                |> filter(fn: (r) => r["_field"] == "throughput")
                |> filter(fn: (r) => r["deviceID"] == "2")
                |> aggregateWindow(every: 5s, fn: sum, createEmpty: false)
                |> last()
        """
        for agent_id in range(agent_num):
            query = 'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "throughput")' \
                    '|> filter(fn: (r) => r["_field"] == "throughput")' \
                    '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '")' \
                    + ('|> filter(fn: (r) => r["step_index"] == "' + str(step) + '") ' if self.eliminate == True else "")
                    # '|> filter(fn: (r) => r["step_index"] == "' + str(step) + '") '
            # '|> aggregateWindow(every: ' + str(atime) + 's, fn: sum, createEmpty: false)'
            data = self.client.query_api().query(query=query, org=self.org)
            output_throughput = data.to_values(columns=['_value'])
            if not output_throughput:
                print("Agent" + str(agent_id + 1) + "throughput no data found, set to default 0")
                all_throughput.append(0)
            else:
                sum_throughput = sum(list(np.array(output_throughput).ravel()))  # 降维
                all_throughput.append(sum_throughput / atime)
        return all_throughput

    def _get_edge_queue_size(self, agent_num):
        all_inference_queue_size = []
        all_offloading_queue_size = []
        for agent_id in range(agent_num):
            query = 'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "queue_size") ' \
                    '|> filter(fn: (r) => r["deviceID"] == "' + str(agent_id + 1) + '") ' \
                                                                                    '|> mean()'
            data = self.client.query_api().query(query=query, org=self.org)
            output_queue_size = data.to_values(columns=['_value'])
            queue_size = list(np.array(output_queue_size).ravel())  # 降维
            # print("queue_size====", queue_size)
            if not queue_size:
                print("Agent" + str(agent_id + 1) + " edge queue size no data found set to default 0")
                all_inference_queue_size.append(0)
                all_offloading_queue_size.append(0)
            else:
                all_inference_queue_size.append(queue_size[0])
                all_offloading_queue_size.append(queue_size[1])
        return all_inference_queue_size, all_offloading_queue_size

    def _get_cloud_queue_size(self, agent_num):
        all_cloud_transmission_queue_size = []
        query = 'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "queue_size") ' \
                '|> filter(fn: (r) => r["deviceID"] == "0") ' \
                '|> mean()'
        data = self.client.query_api().query(query=query, org=self.org)
        output_queue_size = data.to_values(columns=['_value'])
        if not output_queue_size:
            print("cloud queue size no data found set to default 0")
            all_cloud_transmission_queue_size.append(0)
        else:
            queue_size = list(np.array(output_queue_size).ravel())  # 降维
            # cloud_inference_queue_size = queue_size[0]
            all_cloud_transmission_queue_size.append(queue_size[1])
        all_cloud_transmission_queue_size *= agent_num
        return all_cloud_transmission_queue_size

    def _get_system_average_latency(self, agent_num, step):
        all_AVG_latncy = []
        all_average_latency_cloud_edge = []
        all_latency_cloud_edge = []
        latency_flag = [True] * agent_num
        for agent_id in range(agent_num):
            query = 'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "latency") ' \
                    '|> filter(fn: (r) => r["_field"] == "latency") ' \
                    '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
                    + ('|> filter(fn: (r) => r["step_index"] == "' + str(step) + '") ' if self.eliminate == True else "")
                    # '|> filter(fn: (r) => r["step_index"] == "' + str(step) + '") '
            # '|> aggregateWindow(every: ' + str(atime) + 's, fn: sum, createEmpty: false)'\
            # '|> last()'
            data = self.client.query_api().query(query=query, org=self.org)
            output_latency = data.to_values(columns=['_value', 'deviceID'])
            latency_cloud_edge = [[], []]  # [[cloud_latency], [edge_latency]]
            for latency in output_latency:
                if latency[1] == '0':  # 云的延迟
                    latency_cloud_edge[0].append(latency[0])
                else:  # 边的延迟
                    latency_cloud_edge[1].append(latency[0])
            # print("Agent" + str(agent_id+1) + " latency=", latency_cloud_edge)

            latency_cloud_edge_copy = latency_cloud_edge[:]
            one_cloud_edge_latency = []
            one_cloud_edge_latency.append(
                0 if latency_cloud_edge_copy[0] == [] else np.mean(latency_cloud_edge_copy[0]))  # 0表示 没收到对应的延迟
            one_cloud_edge_latency.append(
                0 if latency_cloud_edge_copy[1] == [] else np.mean(latency_cloud_edge_copy[1]))  # 0表示 没收到对应的延迟

            all_average_latency_cloud_edge.append(one_cloud_edge_latency)
            # all_latency_cloud_edge.append(latency_cloud_edge)
            if not output_latency:
                print("Agent" + str(agent_id + 1) + " system latency no data found, set to default 5")
                all_AVG_latncy.append(5.0)
                latency_flag[agent_id] = False
            else:
                latency = list(np.concatenate(latency_cloud_edge))  # 降维
                num = len(latency)
                latency_AV = np.float32(np.sum(latency) / num)
                all_AVG_latncy.append(latency_AV)
        return all_AVG_latncy, all_average_latency_cloud_edge, latency_flag

    def _get_system_average_acc(self, agent_num, step):
        all_acc = []
        all_gap_acc = []
        all_videoID_list = []
        all_chunkID_list = []
        acc_flag = [True] * agent_num
        for agent_id in range(agent_num):
            # query = 'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "acc")' \
            #         '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
            #         '|> filter(fn: (r) => r["_field"] == "videoID") |> yield(name: "videoID")' \
            #         'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "acc")' \
            #         '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
            #         '|> filter(fn: (r) => r["_field"] == "chunkID") |> yield(name: "chunkID")' \
            #         'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "acc")' \
            #         '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
            #         '|> filter(fn: (r) => r["_field"] == "acc") |> yield(name: "acc")'

            query = 'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "acc")' \
                    '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
                    + ('|> filter(fn: (r) => r["step_index"] == "' + str(step) + '") ' if self.eliminate == True else "")
                    # '|> filter(fn: (r) => r["step_index"] == "' + str(step) + '") '

            data = self.client.query_api().query(query=query, org=self.org)
            output = data.to_values(columns=['_field', '_value'])
            videoID_list = [x[1] for x in output if x[0] == 'videoID']
            chunkID_list = [x[1] for x in output if x[0] == 'chunkID']
            acc_list = [x[1] for x in output if x[0] == 'acc']
            all_videoID_list.append(videoID_list)
            all_chunkID_list.append(chunkID_list)
            acc_gap = []
            for i in range(len(acc_list)):
                videoID = videoID_list[i]
                chunkID = chunkID_list[i]
                acc = acc_list[i]
                upper_acc_df = self.df[(self.df['videoID'] == videoID) & (self.df['chunkID'] == chunkID)]
                upper_acc = upper_acc_df['acc']  # 获取某videoID和chunkID下的最高精度
                acc_gap.append(upper_acc - acc)
            # print(acc_gap)
            all_gap_acc.append(1 if not output else np.mean(acc_gap))
            if not output:
                print("Agent" + str(agent_id + 1) + " system acc no data found, set to default 0.5")
                all_acc.append(0.5)
                acc_flag[agent_id] = False
            else:
                acc_AV = np.float32(np.mean(acc_list))
                all_acc.append(acc_AV)
        # print(all_gap_acc)
        return all_acc, all_gap_acc, all_videoID_list, all_chunkID_list, acc_flag

    def _get_system_average_queue_wait(self, agent_num, step):
        all_queue_wait = []
        all_average_qwait_cloud_edge = []
        for agent_id in range(agent_num):
            query = 'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "queue_wait")' \
                    '|> filter(fn: (r) => r["_field"] == "queue_wait")' \
                    '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
                    + ('|> filter(fn: (r) => r["step_index"] == "' + str(step) + '") ' if self.eliminate == True else "")
                    # '|> filter(fn: (r) => r["step_index"] == "' + str(step) + '") '
            data = self.client.query_api().query(query=query, org=self.org)
            output_queue_wait = data.to_values(columns=['_value', 'deviceID'])

            qwait_cloud_edge = [[], []]  # [[cloud_qwait], [edge_qwait]]
            for latency in output_queue_wait:
                if latency[1] == '0':  # 云的延迟
                    qwait_cloud_edge[0].append(latency[0])
                else:  # 边的延迟
                    qwait_cloud_edge[1].append(latency[0])

            qwait_cloud_edge_copy = qwait_cloud_edge[:]
            one_cloud_edge_qwait = []
            one_cloud_edge_qwait.append(
                0 if qwait_cloud_edge_copy[0] == [] else np.mean(qwait_cloud_edge_copy[0]))  # 0表示 没收到对应的延迟
            one_cloud_edge_qwait.append(
                0 if qwait_cloud_edge_copy[1] == [] else np.mean(qwait_cloud_edge_copy[1]))  # 0表示 没收到对应的延迟

            all_average_qwait_cloud_edge.append(one_cloud_edge_qwait)

            if not output_queue_wait:
                print("Agent" + str(agent_id + 1) + " system queue wait no data found, set to default 5")
                all_queue_wait.append(5.0)
            else:
                queue_wait = list(np.concatenate(qwait_cloud_edge))  # 降维
                queue_wait_AV = np.float32(np.mean(queue_wait))
                all_queue_wait.append(queue_wait_AV)
        return all_queue_wait, all_average_qwait_cloud_edge

    def _get_bandwidth(self, agent_num):
        all_bandwidth = []
        for agent_id in range(agent_num):
            query = 'from(bucket:"Monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "bandwidth") ' \
                    '|> filter(fn: (r) => r["deviceID"] == "' + str(agent_id + 1) + '") ' \
                                                                                    '|> aggregateWindow(every: 10s, fn: mean, createEmpty: false)' \
                                                                                    '|> yield(name:"mean")'
            data = self.client.query_api().query(query=query, org=self.org)
            output_bandwidth = data.to_values(columns=['_value'])
            # print("queue wait====", output_queue_wait)
            if not output_bandwidth:
                random_bandwidth = round(random.uniform(2, 25), 1)  # 生成一个小于等于 25 的随机小数并保留一位小数
                # random_bandwidth = 10.0
                print("Agent" + str(agent_id + 1) + " bandwidth no data found, set to default " + str(
                    random_bandwidth) + "MB")
                all_bandwidth.append(random_bandwidth)
            else:
                bandwidth = list(np.array(output_bandwidth).ravel())  # 降维
                all_bandwidth.append(bandwidth[0])
        return all_bandwidth

    def _get_step_lt_acc(self, agent_num):
        all_acc = [[] for _ in range(agent_num)]
        all_lt = [[] for _ in range(agent_num)]
        all_gap_acc = [[] for _ in range(agent_num)]

        all_acc_temp = [[] for _ in range(agent_num)]
        all_videoID = [[] for _ in range(agent_num)]
        all_chunkID = [[] for _ in range(agent_num)]

        for agent_id in range(agent_num):
            for step_index in range(12):
                stream_time = (12 - step_index + 6) * 10
                query_acc = 'from(bucket:"Monitor") |> range(start: -' + str(stream_time) + 's) |> filter(fn: (r) => r["_measurement"] == "acc")' \
                                   '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
                                   '|> filter(fn: (r) => r["step_index"] == "' + str(step_index + 1) + '")'

                # query = 'from(bucket:"monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "acc")' \
                #         '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
                #         '|> filter(fn: (r) => r["_field"] == "videoID") |> yield(name: "videoID")' \
                #         'from(bucket:"monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "acc")' \
                #         '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
                #         '|> filter(fn: (r) => r["_field"] == "chunkID") |> yield(name: "chunkID")' \
                #         'from(bucket:"monitor") |> range(start: -10s) |> filter(fn: (r) => r["_measurement"] == "acc")' \
                #         '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
                #          '|> filter(fn: (r) => r["_field"] == "acc") |> yield(name: "acc")'

                data_acc = self.client.query_api().query(query=query_acc, org=self.org)

                query_lt = 'from(bucket:"Monitor") |> range(start: -' + str(
                    stream_time) + 's) |> filter(fn: (r) => r["_measurement"] == "latency")' \
                                   '|> filter(fn: (r) => r["_field"] == "latency")' \
                                   '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") ' \
                                   '|> filter(fn: (r) => r["step_index"] == "' + str(step_index + 1) + '") '
                data_lt = self.client.query_api().query(query=query_lt, org=self.org)

                output_acc = data_acc.to_values(columns=['_field', '_value'])
                videoID_list = [x[1] for x in output_acc if x[0] == 'videoID']
                chunkID_list = [x[1] for x in output_acc if x[0] == 'chunkID']
                acc_list = [x[1] for x in output_acc if x[0] == 'acc']
                if not output_acc:
                    all_acc[agent_id].append(0.5)
                else:
                    all_acc[agent_id].append(np.mean(acc_list))
                all_acc_temp[agent_id].append(acc_list)
                all_chunkID[agent_id].append(chunkID_list)  # [[[step1], [step2], ... , [stepN]],[[step1], [step2], ... , [stepN]]]
                all_videoID[agent_id].append(videoID_list)  # agent1                          agent2

                output_lt = data_lt.to_values(columns=['_value'])
                if not output_lt:
                    print("error")
                    all_lt[agent_id].append(5.0)
                else:
                    lt_list = list(np.array(output_lt).ravel())  # 降维
                    all_lt[agent_id].append(np.mean(lt_list))

        for agent_id in range(agent_num):
            for step_index in range(12):
                gap_list = []
                for i in range(len(all_acc_temp[agent_id][step_index])):
                    acc = all_acc_temp[agent_id][step_index][i]
                    videoID = all_videoID[agent_id][step_index][i]
                    chunkID = all_chunkID[agent_id][step_index][i]
                    upper_acc_df = self.df[(self.df['videoID'] == int(videoID)) & (self.df['chunkID'] == int(chunkID))]
                    upper_acc = upper_acc_df['acc']  # 获取某videoID和chunkID下的最高精度
                    gap_list.append(upper_acc - acc)
                if not gap_list:
                    all_gap_acc[agent_id].append(0.5)
                else:
                    all_gap_acc[agent_id].append(np.mean(gap_list))
        return all_acc, all_lt, all_gap_acc, all_videoID, all_chunkID

    def _get_steps_lt_acc(self, agent_num, start_step, end_step, aux_flag):  # 左右包含
        all_acc = [[] for _ in range(agent_num)]
        all_lt = [[] for _ in range(agent_num)]
        all_gap_acc = [[] for _ in range(agent_num)]

        all_videoID = [[] for _ in range(agent_num)]
        all_chunkID = [[] for _ in range(agent_num)]

        num_step = end_step - start_step + 1
        for agent_id in range(agent_num):
            step_acc_temp = [[] for _ in range(num_step)]
            step_videoID_temp = [[] for _ in range(num_step)]
            step_chunkID_temp = [[] for _ in range(num_step)]
            step_lt_temp = [[] for _ in range(num_step)]

            stream_time = num_step * 10 + 300
            query_acc = 'from(bucket:"Monitor") |> range(start: -' + str(
                stream_time) + 's) |> filter(fn: (r) => r["_measurement"] == "acc")' \
                               '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") '

            query_lt = 'from(bucket:"Monitor") |> range(start: -' + str(
                stream_time) + 's) |> filter(fn: (r) => r["_measurement"] == "latency")' \
                               '|> filter(fn: (r) => r["_field"] == "latency")' \
                               '|> filter(fn: (r) => r["produce_device"] == "' + str(agent_id + 1) + '") '

            if aux_flag:
                data_acc = self.client_aux.query_api().query(query=query_acc, org=self.org)
                data_lt = self.client_aux.query_api().query(query=query_lt, org=self.org)
            else:
                data_acc = self.client.query_api().query(query=query_acc, org=self.org)
                data_lt = self.client.query_api().query(query=query_lt, org=self.org)

            output_acc = data_acc.to_values(columns=['_field', 'step_index', '_value'])
            output_lt = data_lt.to_values(columns=['step_index', '_value'])

            if not output_acc:
                all_acc[agent_id].append(0.5)
                all_gap_acc[agent_id].append(0.5)
            else:
                for data in output_acc:
                    if int(data[1]) >= start_step and int(data[1]) <= end_step:
                        step_id = int(data[1]) - start_step
                        if data[0] == 'videoID':
                            step_videoID_temp[step_id].append(data[2])
                        elif data[0] == 'chunkID':
                            step_chunkID_temp[step_id].append(data[2])
                        else:
                            step_acc_temp[step_id].append(data[2])
                for step_id in range(num_step):
                    videoID_list_step = step_videoID_temp[step_id]
                    chunkID_list_step = step_chunkID_temp[step_id]
                    acc_list_step = step_acc_temp[step_id]
                    gap_list = []
                    for i in range(len(acc_list_step)):
                        acc = acc_list_step[i]
                        videoID = videoID_list_step[i]
                        chunkID = chunkID_list_step[i]
                        upper_acc_df = self.df[(self.df['videoID'] == int(videoID)) & (self.df['chunkID'] == int(chunkID))]
                        upper_acc = upper_acc_df['acc']  # 获取某videoID和chunkID下的最高精度
                        gap_list.append(upper_acc - acc)
                    if not gap_list:
                        all_gap_acc[agent_id].append(0.5)
                    else:
                        all_gap_acc[agent_id].append(np.mean(gap_list))
                    if acc_list_step:
                        all_acc[agent_id].append(np.mean(acc_list_step))
                    else:
                        all_acc[agent_id].append(0.5)
                    all_chunkID[agent_id].append(chunkID_list_step)
                    all_videoID[agent_id].append(videoID_list_step)

            for data in output_lt:
                if int(data[0]) >= start_step and int(data[0]) <= end_step:
                    step_id = int(data[0]) - start_step
                    step_lt_temp[step_id].append(data[1])
            for step_id in range(num_step):
                if not step_lt_temp[step_id]:
                    all_lt[agent_id].append(5.0)
                else:
                    all_lt[agent_id].append(np.mean(step_lt_temp[step_id]))

        return all_acc, all_lt, all_gap_acc, all_videoID, all_chunkID


if __name__ == '__main__':
    args = Config(8)
    client = Monitor(args,"",8)
