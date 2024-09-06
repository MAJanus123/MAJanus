# -*- coding: utf-8 -*-
"""
Created on Fri Jul 1 09:52 2022

@author: Huifeng-Hu
"""
import argparse
import math
import os
import queue
import subprocess
import sys
import socket
import threading
import time
from multiprocessing import Pool, Manager, Process, shared_memory
from pathlib import Path
import ast
from struct import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-1])
sys.path.insert(0, ROOT+'/res')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from res.power import EngUbuntuGPU
import torch
import ffmpeg
from res.RtpPacket_chunk import RtpPacket
import base64
import json
import numpy as np
import pika
import logging
from res.utils import get_labels_chunk, ap_per_class
from res.ultralytics.yolo.utils.metrics import box_iou
from res.video_stream import bytes2numpy
from res.config_cloud import Config
from res.ultralytics.yolo.engine.model import YOLO
from res.ultralytics.yolo.utils.ops import xywh2xyxy

listen_queue = Manager().Queue()
inference_queue = Manager().Queue()
DATA_PACKET_SIZE = 8688

transmission_time = []
decode_time = []
inference_time = []
acc_time = []
All_part = []
acc = []

seed = 42
# 固定NumPy的随机数生成器
np.random.seed(seed)

def _listen(cloudhost, cloudport, listen_queue):
    # dataSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
    dataSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP
    dataSocket.settimeout(999999)
    dataSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 就是它，在bind前加
    dataSocket.bind((cloudhost, cloudport))
    print('init socket success ... ' + str(cloudport) + 'start to listen ...')
    dataSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 99999999)
    dataSocket.listen(1)
    # dataSocket.setblocking(False)  # 非阻塞
    data_array = bytearray()
    while True:
        clientSocket, clientAddr = dataSocket.accept()
        print("一个新客户端已连接：%s" % str(clientAddr))
        while True:
            data = clientSocket.recv(DATA_PACKET_SIZE)
            if data[-4:] == b'\\EOF':  # marker=2的包
                data_array.extend(data)
                listen_queue.put((data_array, time.time()))
                data_array = bytearray()
            else:
                data_array.extend(data)


def _cloud_power_monitor(sudopw, device_ID):
    print("start to monitor power")
    user_info = pika.PlainCredentials("guest", "guest")
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('10.12.11.144', 5672, credentials=user_info,
                                  blocked_connection_timeout=999999, heartbeat=0))  # 连接服务器上的RabbitMQ服务
    power_publish_channel = connection.channel()
    power_publish_channel.queue_declare('power')

    power_monitor = EngUbuntuGPU(sudopw, fq=0.1)

    while True:
        time.sleep(1)
        power = power_monitor.get()
        power_dic = {'device_ID': device_ID,
                     'power': power,
                     }
        message = str(power_dic).encode('ascii')
        message_byte = base64.b64encode(message)
        power_publish_channel.basic_publish(exchange='',  # default exchange
                                            routing_key='power',  # 指定消息要发送到哪个queue
                                            body=message_byte  # 指定要发送的消息
                                            )
        power_monitor.reset()


def _queue_monitor(device_ID, cloudhost, mqport):
    print("start to monitor queue")
    user_info = pika.PlainCredentials("guest", "guest")
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(cloudhost, mqport, credentials=user_info,
                                  blocked_connection_timeout=999999, heartbeat=0))  # 连接服务器上的RabbitMQ服务
    queue_channel = connection.channel()
    queue_channel.queue_declare('queue_size')
    while True:
        time.sleep(1)
        queue_dic = {
            "device_ID": device_ID,
            "inference_queue": inference_queue.qsize(),
            "transmission_queue": listen_queue.qsize()
        }
        message = str(queue_dic).encode('ascii')
        message_byte = base64.b64encode(message)
        queue_channel.basic_publish(exchange='',
                                    routing_key='queue_size',  # 指定消息要发送到哪个queue
                                    body=message_byte  # 指定要发送的消息
                                    )


class CloudServer:
    def __init__(self, args):
        logging.getLogger("pika").setLevel(logging.WARNING)  # 屏蔽mq日志
        self.cloudhost = args.cloudhost
        self.cloudport = [1081 + i for i in range(args.agent_num)]
        self.sudopw = args.sudopw
        self.mqport = args.mqport
        self.mquser = args.mquser
        self.mqpw = args.mqpw
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_ID = int(args.clouddeviceid)

        # 从本地读取视频名称和对应的videoid
        with open(os.path.join(ROOT, 'res', 'all_video_names_easy.txt'), 'r') as f:
            self.all_video_names = eval(f.read())
        self.imgsz = (540, 960)  # 图片尺寸
        self.modelID = 'm'
        model_dic = {'01': '/res/ultralytics/yolo/weights/yolov8n_trained.pt',
                     '02': '/res/ultralytics/yolo/weights/yolov8s_trained.pt',
                     '03': '/res/ultralytics/yolo/weights/yolov8m_trained.pt'}
        self.yolo = YOLO(ROOT + model_dic['03'], task='detect')

        # warm up
        random_array = np.random.rand(1, self.imgsz[0], self.imgsz[1], 3)
        pred_bantch = self.yolo.predict(random_array, half=True, imgsz=self.imgsz[0])

        self.inference_result_channel = self._declare_mq(self.cloudhost, self.mqport, args.mquser, args.mqpw)
        self.all_Sequence = {}  # 存储所有被接收包的序号
        self.all_Payload = {}  # 存储所有device传的视频在process过程中的字节流
        self.all_num = {}  # 存储所有chunk收到包的个数

        user_info = pika.PlainCredentials("guest", "guest")
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.cloudhost, self.mqport, credentials=user_info,
                                      blocked_connection_timeout=999999, heartbeat=0))  # 连接服务器上的RabbitMQ服务
        self.bandwidth_channel = connection.channel()
        self.bandwidth_channel.queue_declare('bandwidth')

        self.listen_process_pool = Pool(len(self.cloudport))
        self.queue_monitor = Pool(1)

        for port in self.cloudport:
            self.listen_process_pool.apply_async(func=_listen, args=(self.cloudhost, port, listen_queue))
        self.listen_process_pool.daemon = True

        self.queue_monitor.apply_async(func=_queue_monitor, args=(self.device_ID, self.cloudhost, self.mqport))
        self.queue_monitor.daemon = True

        self.control_thread = threading.Thread(target=self._control_client, daemon=True)
        self.control_thread.start()

        # self.count = threading.Thread(target=self._count, daemon=True)
        # self.count.start()

        self._process(listen_queue)

    def _count(self):
        time.sleep(300)
        print("transmission====", np.mean(transmission_time[5:]))
        print("transmission====", transmission_time)
        print("decode====", np.mean(decode_time))
        print("inference====", np.mean(inference_time) * 2)
        print("acc time====", np.mean(acc_time))
        print("All_part====", np.mean(All_part))

    def _process(self, listen_queue):  # 处理listen得到的数据
        print('start to process')
        while True:
            data = listen_queue.get()
            start_process = time.time()
            queue_wait = time.time() - data[1]
            data_array = data[0]
            payload = data_array[:-40]  # data     20(header)+16(pack)+4
            head = data_array[-40:-4]  # 最后一个包的信息
            imagePacket = RtpPacket()
            imagePacket.decode(head)
            timestamp = imagePacket.getTimestamp()
            device_ID = imagePacket.getDeviceID()
            video_ID = imagePacket.getVideoID()
            chunk_ID = imagePacket.getChunkID()
            step_index = imagePacket.getStepIndex()
            episode = imagePacket.getEpisode()
            hw = imagePacket.getPayload()
            (height, width) = unpack('ll', hw)
            transmission_time.append(time.time() - timestamp)
            # print("transmission_time", time.time() - timestamp)

            send_timestamp = imagePacket.getSendTimestamp()
            send_time = data[1] - send_timestamp
            size = len(data_array)
            if send_time < 0.01:
                bandwidth = 25.0
            else:
                bandwidth = size / send_time * 8 / 1024 / 1024
                if bandwidth > 25.0:
                    bandwidth = 25.0
            print(size, send_time, bandwidth)
            queue_dic = {
                "device_ID": int(device_ID),
                "bandwidth": bandwidth
            }
            message = str(queue_dic).encode('ascii')
            message_byte = base64.b64encode(message)
            self.bandwidth_channel.basic_publish(exchange='',
                                        routing_key='bandwidth',  # 指定消息要发送到哪个queue
                                        body=message_byte  # 指定要发送的消息
                                        )
            start = time.time()
            # 转码
            decode_process = (ffmpeg
                              .input('pipe:', format='h264')
                              .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
                              .run_async(pipe_stdin=True, pipe_stdout=True)
                              )
            decode_process.communicate(payload)
            # decode_process.stdin.write(payload)
            # decode_process.stdin.close()
            # in_bytes = decode_process.stdout.read(width * height * 3 * 30)
            in_bytes, err = decode_process.communicate()
            try:
                frame = bytes2numpy(30, in_bytes, height, width)
            except ValueError:
                print("error1")
                continue
            decode_time.append(time.time() - start)
            self._cloud_inference((timestamp, frame, device_ID, video_ID, chunk_ID, queue_wait, height, width, step_index, episode))
            print("process time==", time.time() - start_process)
            listen_queue.task_done()

    def _cloud_inference(self, frame_chunk):
        labels_chunk = get_labels_chunk(ROOT, self.all_video_names, frame_chunk[3], frame_chunk[4])  # get chunk labels
        if len(frame_chunk[1]) == len(labels_chunk):
            # assert len(frame_chunk[1]) == len(
            #     labels_chunk), "inference date length {len1} != labels length {len2}, {video}  {chunk}".format(
            #     len1=len(frame_chunk[1]), len2=len(labels_chunk), video=frame_chunk[3], chunk=frame_chunk[4])
            # iouv = torch.linspace(0.5, 0.95, 10)
            (height, width) = (frame_chunk[6], frame_chunk[7])
            iouv = torch.full((1,), 0.5)
            preds = []
            inference_batchsize = 15
            rounds = int(np.ceil(len(labels_chunk) / inference_batchsize))
            start = time.time()
            for num in range(rounds):
                if num < rounds - 1:
                    inference_data = frame_chunk[1][num * inference_batchsize:(num + 1) * inference_batchsize]
                else:
                    inference_data = frame_chunk[1][num * inference_batchsize:]
                # inference_data = torch.tensor(inference_data).to(self.device)
                # torch.tensor(inference_data).to(self.device)
                # pred_bantch = self.yolo.predict(inference_data, half=True, imgsz=height)
                pred_bantch = self.yolo.predict(inference_data, half=True, imgsz=544)
                # inference_time.append(time.time() - start)
                preds.extend([x.boxes.data.cpu() for x in pred_bantch])  # 一个chunk的推理结果
            print("inference time===", time.time() - start)
            start_acc = time.time()
            # 精度评估
            stats = []
            for i, pred in enumerate(preds):
                if (len(np.shape(labels_chunk[i]))) == 1:
                    labels_chunk[i] = labels_chunk[i][None, :]
                cls = torch.tensor(labels_chunk[i])[:, 0]
                correct_bboxes = torch.zeros(pred.shape[0], len(iouv), dtype=torch.bool)  # init
                if len(pred) == 0:
                    if len(labels_chunk[i]):  # label文件不为空
                        stats.append((correct_bboxes, *torch.zeros((2, 0)), cls))
                    continue

                if len(labels_chunk[i]):
                    labels_chunk[i][:, 1:5] = xywh2xyxy(labels_chunk[i][:, 1:5]) * ([width, height, width, height])  # target boxes
                    # for p in pred:
                    #     p[5] = torch.full_like(p[5], self.convert.get(int(p[5]), 3))  # 3表示其他类型
                    correct_bboxes = self.process_batch_acc(pred, torch.tensor(labels_chunk[i]), iouv)
                stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls))  # (tp, conf, pcls, tcls)
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
            try:
                ap = ap_per_class(*stats)
            except TypeError:
                print("error  acc")
                return
                # print(ap)
            mAP = [np.mean(x) for x in ap]
            acc_time.append(time.time() -start_acc)
            one_acc_chunk = np.mean(mAP)
            MQ_dic = {
                'device_ID': self.device_ID,  # 从哪个设备产生推理结果      int
                'produce_device': frame_chunk[2],  # 视频流从哪个设备产生    int
                'video_ID': frame_chunk[3],  # 视频名称(代号)       int
                'chunk_ID': frame_chunk[4],  # 第几块视频           int
                'acc_chunk': one_acc_chunk,  # 该chunk的精度评估结果
                'timestamp': frame_chunk[0],  # read前的时间戳，用于计算latency
                'frame_num': len(labels_chunk),  # 当前chunk的帧数      int
                'queue_wait': frame_chunk[5],
                'step_index': frame_chunk[8],
                'episode': frame_chunk[9]
            }

            print("MQ_dic:  ", MQ_dic)
            self._sendMQ(self.inference_result_channel, MQ_dic, 'inference_result')  # 推理结果发送到MQ
            All_part.append(time.time() - frame_chunk[0])
        else:
            print("error2")
    def process_batch_acc(self, detections, labels, iouv):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:  # 如果找到了满足条件的位置
                #  将位置信息、IoU值以及其他相关信息连接起来形成一个匹配矩阵。使用torch.stack()函数将位置信息堆叠在一起，
                # 然后使用torch.cat()函数将堆叠后的张量与IoU值进行连接。最后使用.cpu().numpy()将结果转换为NumPy数组。
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),1).cpu().numpy()   # [labels, detect, iou]
                if x[0].shape[0] > 1:  # 如果匹配矩阵中有多个匹配项。
                    matches = matches[matches[:, 2].argsort()[::-1]] # 根据IoU值对匹配矩阵进行降序排序。
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 去除匹配矩阵中重复的检测结果
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 去除匹配矩阵中重复的标签
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    def _declare_mq(self, host='10.12.11.144', port=5672, username='guest', password='guest'):
        user_info = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host, port, credentials=user_info,
                                      blocked_connection_timeout=999999, heartbeat=0))  # 连接服务器上的RabbitMQ服务

        # 创建channel
        channel = connection.channel()
        print('connect to rabbitmq')
        ### TODO declare more queues for the system
        return channel

    def _sendMQ(self, channel, message, queue):
        message = str(message).encode('ascii')
        message_byte = base64.b64encode(message)
        channel.queue_declare(queue=queue)

        channel.basic_publish(exchange='',
                              routing_key=queue,  # 指定消息要发送到哪个queue
                              body=message_byte  # 指定要发送的消息
                              )
        # print('mesage have send to ', queue)

    def _control_client(self):
        user_info = pika.PlainCredentials(self.mquser, self.mqpw)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.cloudhost, self.mqport, credentials=user_info,
                                      blocked_connection_timeout=999999, heartbeat=0))  # 连接服务器上的RabbitMQ服务
        control_channel = connection.channel()

        control_channel.exchange_declare(exchange='command', exchange_type='fanout')

        result = control_channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue

        control_channel.queue_bind(exchange='command', queue=queue_name)

        def callback(ch, method, properties, body):
            command = base64.b64decode(body)
            command = ast.literal_eval(command.decode('ascii'))
            if command['type'] == 'reset':
                print('server reset')
            elif command['type'] == 'action':
                # self.modelID = command['value']['cloud_model']
                print('server set action, index = ', command['step'])
            else:
                time.sleep(25)
                while not inference_queue.empty():
                    try:
                        inference_queue.get_nowait()
                    except queue.Empty:
                        pass
                while not listen_queue.empty():
                    try:
                        listen_queue.get_nowait()
                    except queue.Empty:
                        pass

        control_channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True)

        control_channel.start_consuming()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_num", required=False, default=4, type=int,
                        help="the agent num")
    # 解析命令行参数
    all_args = parser.parse_args()

    args = Config(all_args.agent_num)
    with torch.no_grad():
        client = CloudServer(args)
