import multiprocessing
import queue
import signal
import time
import os
import sys
import ast

from pathlib import Path

# sys.path.append(str('/home/hhf/xyb/Video-Analytics-Task-Offloading/'))  # add ROOT to PATH
# print(sys.path)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-1])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from res.video_stream import VideoStream, bytes2numpy
import threading
import torch
import ffmpeg
import pika
import numpy as np
from res.RtpPacket import RtpPacket
from edge.RtpClient import RtpClient

from res.yolov5.YOLO import YOLO
from res.yolov5.utils.general import xywh2xyxy

import math
import base64
import logging
from res.power import *
import argparse
from multiprocessing import Process, Queue, Manager, Pool
from res.yolov5.utils.metrics import box_iou
from camera import Camera
from res.utils import getchunk_labels
from config import Config

logging.getLogger("pika").setLevel(logging.WARNING)

transmission_queue = Manager().Queue()
inference_queue = Manager().Queue()


def _process_inference_process(queue, stream, labels, i, j, device_ID):
    """
    i is (video_ID - 1)
    j is (chunk_ID - 1)
    """
    inference_process = stream.inference_process(j)  # 格式转化
    while True:
        timestamp = time.time()
        edge_frame = inference_process.stdout.read(stream.width * stream.height * 3 * 30)
        if not edge_frame:
            # queue.put(None)
            break
        else:
            frame = bytes2numpy(30, edge_frame, stream.height, stream.width)  # 720 * 1280
            labels_chunk = getchunk_labels(labels, j)
            # (frame, 当前chunkID对应的标签文件名称, 时间戳, deviceIDv, ideoID, chunkID)
            frame_chunk = (frame, labels_chunk, timestamp, device_ID, i + 1, j + 1)
            queue.put(frame_chunk)
            # print("inference_queue: ", inference_queue.qsize())
            # print("end")
    # print("pid=======", os.getpid())
    os.kill(os.getpid(), signal.SIGKILL)


def _process_transmission_process(transmission_queue, stream, labels, i, j, device_ID):
    transmission_process = stream.transmission_process(j)
    # 从这里开始计算cloud延迟
    chunk = transmission_process.stdout.read()
    timestamp = time.time()
    chunk = (chunk, timestamp, device_ID, i + 1, j + 1)  # (chunk, timestamp, deviceID, videoID, chunkID)
    transmission_queue.put(chunk)
    # print("transmission_queue: ", transmission_queue.qsize())
    # print("end2")
    # print("pid=======", os.getpid())
    os.kill(os.getpid(), signal.SIGKILL)


class EdgeClient:
    def __init__(self, args):
        self.cloudhost = args.cloudhost
        self.cloudport = args.cloudport
        self.device_ID = int(args.deviceid)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.path_ = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))  # path= /../Video-Analytics-Task-Offloading
        # print('root path : ', self.path_)
        sys.path.insert(0, self.path_ + '/res/yolov5')

        self.device_type = args.devicetype
        self.sudo_passward = args.sudopw
        self.mquser = args.mquser
        self.mqpw = args.mqpw
        self.mqport = args.mqport
        self.model_dic = {'01':  '/res/yolov5/weights/5n_b32_e20_p.pt',
                          '02':  '/res/yolov5/weights/5s_b32_e20_p.pt',
                          '03':  '/res/yolov5/weights/5m_b32_e50_p.pt'}

        self.camera_dataset = Camera()
        self.chunk_duration = 1

        self.imgsz = (720, 1280)
        self.data_path = self.path_ + '/res/yolov5/data/bdd100k.yaml'
        self.yolo = YOLO(self.path_ + self.model_dic['02'])
        self.batchsize = 30

        self.offload_rate = 0.6  # 卸载比率
        self.po = Pool(2)

        self.rpt_client = RtpClient(self.cloudhost, self.cloudport, socket_type=0)

        self.stream_channel = self._declare_mq(args.mqhost, args.mqport, args.mquser, args.mqpw)
        self.inference_result = self._declare_mq(args.mqhost, args.mqport, args.mquser, args.mqpw)
        self.queue_channel = self._declare_mq(args.mqhost, args.mqport, args.mquser, args.mqpw)

        self.power_monitor_thread = threading.Thread(target=self._power_monitor)
        self.power_monitor_thread.start()

        self.queue_monitor_thread = threading.Thread(target=self._queue_monitor)
        self.queue_monitor_thread.start()

        self.edge_inference = threading.Thread(target=self._edge_inference)
        self.edge_inference.start()

        self.sender_thread = threading.Thread(target=self._sender)
        self.sender_thread.start()

        self.control_thread = threading.Thread(target=self._control_client)
        self.control_thread.start()

        self.done_flag = False


    def _declare_mq(self, host='10.12.11.61', port=5670, username='guest', password='guest'):
        user_info = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host, port, credentials=user_info))  # 连接服务器上的RabbitMQ服务

        # 创建channel
        channel = connection.channel()
        # print('connect to rabbitmq')
        ### TODO declare more queues for the system
        return channel

    def _power_monitor(self):
        self.stream_channel.queue_declare('power')

        if self.device_type == 'NX':
            power_monitor = EngNX(self.sudo_passward, fq=0.1)
        elif self.device_type == 'CPU':
            power_monitor = EngUbuntuCPU(self.sudo_passward, fq=0.1)
        else:
            ## TODO change later for other device
            power_monitor = EngUbuntuGPU(self.sudo_passward, fq=0.1)
            pass

        while True:
            time.sleep(1)
            power = power_monitor.get()
            # print("power==", power)
            power_dic = {'device_ID': self.device_ID,
                         'power': power
                         }
            message = str(power_dic).encode('ascii')
            message_byte = base64.b64encode(message)
            self.stream_channel.basic_publish(exchange='',
                                              routing_key='power',  # 指定消息要发送到哪个queue
                                              body=message_byte  # 指定要发送的消息
                                              )
            power_monitor.reset()

    def _load_camera_data(self, done_flag):
        print('start to load camera data: done_flag %s, %s' % (str(done_flag()), str(threading.get_ident())))
        # for i in range(len(self.camera_dataset)):
        i = 0
        while i < 30:
            sample = self.camera_dataset[i]
            stream = sample['stream']  # 一个视频
            labels = sample['labels']
            # label_names = sample['label_names']
            num_chunks = math.ceil(stream.video_duration / self.chunk_duration)
            num_local_chunk = math.ceil(
                stream.video_duration / self.chunk_duration * (1 - self.offload_rate))  # 在边缘端的块数
            num_cloud_chunk = int(num_chunks - num_local_chunk)

            print('all chunks=', str(num_chunks), '  cloud chunks= ', str(num_cloud_chunk), '  local chunks=',
                  str(num_local_chunk))

            j = 0
            while j < num_chunks:
                if j < num_local_chunk:
                    # print('put to local')
                    self.po.apply_async(func=_process_inference_process,
                                        args=(inference_queue, stream, labels, i, j, self.device_ID))
                    # p = Process(target=self._process_inference_process, args=(self.inference_queue, stream, labels, i, j))
                    # p.start()
                else:
                    # print('put to offload')
                    self.po.apply_async(func=_process_transmission_process,
                                        args=(transmission_queue, stream, labels, i, j, self.device_ID))
                    # p = Process(target=self._process_transmission_process, args=(self.transmission_queue, stream, labels, i, j))
                    # p.start()
                j += 1
                time.sleep(1)

            if done_flag():
                print("  Exiting loading camera data loop. kill thread %s" % str(threading.get_ident()))
                break
            i = i + 1

    def _sender(self):
        # 加header 发送包
        print('video sender thread start to work')
        while True:
            if transmission_queue.qsize() > 0:
                transmission_chunk = transmission_queue.get()
                payload = transmission_chunk[0]  # image_bytes
                timestamp = transmission_chunk[1]
                deviceID = transmission_chunk[2]
                videoID = transmission_chunk[3]
                chunkID = transmission_chunk[4]
                # print("send device %d video %d chunk %d" % (deviceID, videoID, chunkID))
                self.rpt_client.send_image(timestamp, payload, deviceID, videoID, chunkID)
            else:
                continue

    def _edge_inference(self):

        print("start to inference!")
        while True:
            if inference_queue.qsize() > 0:
                start = time.time()
                # print("inference_queue_size: ", inference_queue.qsize())
                frame_chunk = inference_queue.get()
                print("queue get", time.time() - start)
                acc_chunk = []
                for num in range(3):
                    if (num + 1) * 10 < len(frame_chunk[1]):
                        inference_data = frame_chunk[0][num * 10:(num + 1) * 10]
                    else:
                        inference_data = frame_chunk[0][num * 10:]
                    torch.tensor(inference_data).to(self.device)
                    start_time = time.time()
                    pred = self.yolo.inference(inference_data)
                    print("time====", time.time() - start_time)
                    # print(pred)
                    # 精度评估
                    iouv = torch.linspace(0.5, 0.95, 10)
                    for i in range(len(inference_data)):
                        if len(frame_chunk[1][num * 10 + i]) == 0:  # label文件为空
                            if len(pred[i]) == 0:  # 推理结果为空
                                acc_chunk.append(1)
                            else:
                                acc_chunk.append(0)
                        else:
                            if len(pred[i]) == 0:
                                acc_chunk.append(0)
                            else:
                                if (len(np.shape(frame_chunk[1][num * 10 + i]))) == 1:
                                    frame_chunk[1][num * 10 + i] = frame_chunk[1][num * 10 + i][None, :]
                                frame_chunk[1][num * 10 + i][:, 1:5] = xywh2xyxy(
                                    frame_chunk[1][num * 10 + i][:, 1:5]) * ([1280, 720, 1280, 720])  # target boxes
                                stats = self.process_batch_acc(pred[i].cpu(),
                                                               torch.tensor(frame_chunk[1][num * 10 + i]), iouv)
                                acc_chunk.append(stats.cpu().numpy()[:, 0].mean())
                self.MQ_dic = {
                    'device_ID': frame_chunk[3],  # 边缘设备ID          int
                    'video_ID': frame_chunk[4],  # 视频名称(代号)       int
                    'chunk_ID': frame_chunk[5],  # chunkID            int
                    'acc_chunk': np.mean(acc_chunk),  # 该chunk的精度评估结果
                    'timestamp': frame_chunk[2],  # read前的时间戳，用于计算latency
                    'frame_num': len(frame_chunk[1])  # 当前chunk的帧数      int
                }

                print("MQ_dic:  ", self.MQ_dic)
                message = str(self.MQ_dic).encode('ascii')
                message_byte = base64.b64encode(message)
                self._sendMQ(self.inference_result, message_byte, 'inference_result')  # 推理结果发送到MQ
                print("all", time.time() - start)
            else:
                continue

    def _queue_monitor(self):
        self.queue_channel.queue_declare('queue_size')
        while True:
            time.sleep(1)
            queue_dic = {
                "device_ID": self.device_ID,
                "inference_queue": inference_queue.qsize(),
                "transmission_queue": transmission_queue.qsize()
            }
            # print(queue_dic)
            message = str(queue_dic).encode('ascii')
            message_byte = base64.b64encode(message)
            self.queue_channel.basic_publish(exchange='',
                                             routing_key='queue_size',  # 指定消息要发送到哪个queue
                                             body=message_byte  # 指定要发送的消息
                                             )

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
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [labels, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    def _sendMQ(self, channel, message, queue):
        channel.queue_declare(queue=queue)

        channel.basic_publish(exchange='',
                              routing_key=queue,  # 指定消息要发送到哪个queue
                              body=message  # 指定要发送的消息
                              )
        # print('mesage have send to ', queue)

    def _control_client(self):
        user_info = pika.PlainCredentials(self.mquser, self.mqpw)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.cloudhost, self.mqport, credentials=user_info))

        control_channel = connection.channel()

        control_channel.exchange_declare(exchange='command', exchange_type='fanout')

        result = control_channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue

        control_channel.queue_bind(exchange='command', queue=queue_name)

        def callback(ch, method, properties, body):
            command = base64.b64decode(body)
            command = ast.literal_eval(command.decode('ascii'))
            if command['type'] == 'reset':
                self.done_flag = command['value']['done']
                self.first_flag = command['value']['first']

                # self.done_flag == False when run the first episode,
                # self.done_flag == True when finish one episode, need to reset the whole system.
                if self.first_flag == True:
                    reset_thread = threading.Thread(target=self._load_camera_data, args=([lambda: self.done_flag]))
                    reset_thread.start()
                else:
                    self.done_flag = True
                    time.sleep(10)
                    while not transmission_queue.empty():
                        transmission_queue.get_nowait()
                    while not inference_queue.empty():
                        inference_queue.get_nowait()
                    self.done_flag = False
                    reset_thread = threading.Thread(target=self._load_camera_data, args=([lambda: self.done_flag]))
                    reset_thread.start()

            elif command['type'] == 'action':
                self.offload_rate = command['value']['offloading_rate'][0]
                # print(self.offload_rate)
                print('set offloading rate as %s' % str(command['value']['offloading_rate']))


        control_channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True)

        control_channel.start_consuming()

    def _reset(self, id, done):
        """
        After receiving the reset command from the RL environment, we do two things:
        1. clear all the cache in the system,
        2. reset the camera and start from the begin
        
        """
        # reset the commands
        if done == False:
            self._load_camera_data()


    def _set_action(self, offload_rate):
        self.offload_rate = offload_rate



if __name__ == '__main__':
    args = Config()
    args.devicetype = 'NX'
    args.deviceid = '02'
    with torch.no_grad():
        client = EdgeClient(args)
