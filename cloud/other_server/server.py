# -*- coding: utf-8 -*-
"""
Created on Fri Jul 1 09:52 2022

@author: Huifeng-Hu
"""
import os
import sys
import queue
import socket
import threading
import time
from multiprocessing import Pool, Manager, Process
from pathlib import Path
import ast



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-1])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from res.power import EngUbuntuGPU
import torch
import ffmpeg
from res.RtpPacket import RtpPacket
from res.yolov5.YOLO import YOLO
import base64
import numpy as np
import pika
import logging
from res.utils import get_labels_chunk
from res.yolov5.utils.general import xywh2xyxy
from res.yolov5.utils.metrics import box_iou
from res.video_stream import bytes2numpy
from config import Config

cloud_trans_queue = Manager().Queue()
inference_queue = Manager().Queue()



def _listen(dataSocket, cloud_trans_queue):
    print('start to listen')
    while True:
        # conn, addr = dataSocket.accept()  # 接受TCP连接，并返回新的套接字与IP地址
        # while True:
        #     data = conn.recv(4096)  # 把接收的数据实例化
        #     # data, address = dataSocket.recvfrom(4096)  # 字节  UDP
        #     cloud_trans_queue.put(data)
        data, address = dataSocket.recvfrom(32768)  # 字节  UDP
        cloud_trans_queue.put(data)

# def _listen(dataSocket, cloud_trans_queue):
#     print('start to listen')
#     connection, client_address = dataSocket.accept()
#     while True:
#         try:
#             while True:
#                 data = connection.recv(4096)
#                 if data:
#                     cloud_trans_queue.put(data)
#                 else:
#                     break
#         finally:
#             connection.close()




def _process(all_Payload, all_Sequence, all_num):  # 处理listen得到的数据
    print('start to process')
    while True:
        if cloud_trans_queue.qsize() > 0:
            print('len')
            data = cloud_trans_queue.get()
            print(len(data))
            # if len(data) !=1028:
            #     print(len(data))
            if data:
                # 解包
                imagePacket = RtpPacket()
                imagePacket.decode(data)

                currentSequenceNum = imagePacket.seqNum()
                currentMarker = imagePacket.Marker()
                payload = imagePacket.getPayload()  # 视频数据
                timestamp = imagePacket.getTimestamp()
                device_ID = imagePacket.getDeviceID()
                video_ID = imagePacket.getVideoID()
                chunk_ID = imagePacket.getChunkID()
                print('num')
                print(currentSequenceNum)
                print()
                if currentMarker == 0:  # 这个包为数据的第一个包
                    print("Cloud get the first packet form video %d chunk %d of device %d" % (video_ID, chunk_ID, device_ID))
                    if device_ID not in all_Payload:
                        all_Payload[device_ID] = {}
                        all_Sequence[device_ID] = {}
                        all_num[device_ID] = {}
                    if video_ID not in all_Payload[device_ID]:
                        all_Payload[device_ID][video_ID] = {}
                        all_Sequence[device_ID][video_ID] = {}
                        all_num[device_ID][video_ID] = {}
                    if chunk_ID not in all_Payload[device_ID][video_ID]:
                        all_Payload[device_ID][video_ID][chunk_ID] = b"" + payload
                        all_Sequence[device_ID][video_ID][chunk_ID] = 1
                        all_num[device_ID][video_ID][chunk_ID] = 1
                elif currentMarker == 2:
                    print("Cloud get the last packet form video %d chunk %d of device %d" % (video_ID, chunk_ID, device_ID))
                    # 转码
                    _process = (ffmpeg
                                .input('pipe:', format='h264')
                                .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
                                .run_async(pipe_stdin=True, pipe_stdout=True)
                                )
                    _process.stdin.write(all_Payload[device_ID][video_ID][chunk_ID])
                    _process.stdin.close()
                    in_bytes = _process.stdout.read(1280 * 720 * 3 * 30)
                    frame = bytes2numpy(30, in_bytes, 720, 1280)
                    inference_queue.put((timestamp, frame, device_ID, video_ID, chunk_ID))
                    all_Payload[device_ID][video_ID][chunk_ID] = b""
                    print("inference_queue size====", inference_queue.qsize())
                else:
                    if currentSequenceNum > all_Sequence[device_ID][video_ID][chunk_ID]:  # Discard the late packet
                        all_Sequence[device_ID][video_ID][chunk_ID] = currentSequenceNum
                        all_Payload[device_ID][video_ID][chunk_ID] += payload
                        all_num[device_ID][video_ID][chunk_ID] += 1
            cloud_trans_queue.task_done()
    # cloud_trans_queue.join()

class CloudServer:
    def __init__(self, args):
        logging.getLogger("pika").setLevel(logging.WARNING)  # 屏蔽mq日志
        self.cloudhost = args.cloudhost
        self.cloudport = args.cloudport
        self.sudopw = args.sudopw
        self.mqport = args.mqport
        self.mquser = args.mquser
        self.mqpw = args.mqpw
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device_ID = int(args.deviceid)
        self.path_ = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))  # path= /../Video-Analytics-Task-Offloading
        print('root path : ', self.path_)
        sys.path.insert(0, self.path_ + '/res/yolov5')

        self.SOCKET_TYPE = 'SOCK_DGRAM'

        # 从本地读取视频名称和对应的videoid
        with open(os.path.join(ROOT, 'res', 'all_video_names.txt'), 'r') as f:
            self.all_video_names = eval(f.read())
        self.imgsz = (720, 1280)  # 图片尺寸
        self.data_path = self.path_ + '/res/yolov5/data/try_mot.yaml'
        self.yolo = YOLO()

        self.inference_result_channel = self._declare_mq(self.cloudhost, self.mqport, args.mquser, args.mqpw)
        self.power_publish_channel = self._declare_mq(self.cloudhost, self.mqport, args.mquser, args.mqpw)
        self.queue_channel = self._declare_mq(self.cloudhost, self.mqport, args.mquser, args.mqpw)
        self.all_Sequence = {}  # 存储所有被接收包的序号
        self.all_Payload = {}   # 存储所有device传的视频在process过程中的字节流
        self.all_num = {}       # 存储所有chunk收到包的个数

        self._initSocket()
        self.listen = Pool(1)
        self.process = Pool(10)
        # self.inference = Pool(4)

        self.listen.apply_async(func=_listen, args=(self.dataSocket, cloud_trans_queue))
        # self.listen_thread = threading.Thread(target=self._listen)
        # self.listen_thread.start()

        self.process.apply_async(func=_process, args=(self.all_Payload, self.all_Sequence, self.all_num))

        # self.power_monitor_thread = threading.Thread(target=self.cloud_power_monitor)
        # self.power_monitor_thread.start()

        self.queue_monitor_thread = threading.Thread(target=self._queue_monitor)
        self.queue_monitor_thread.start()

        self.control_thread = threading.Thread(target=self._control_client)
        self.control_thread.start()

        self._cloud_inference()

    def _listen(self):
        print('start to listen')
        connection, client_address = self.dataSocket.accept()
        while True:
            try:
                while True:
                    data = connection.recv(32768)
                    if data:
                        cloud_trans_queue.put(data)
                    else:
                        break
            finally:
                connection.close()

    def _initSocket(self):
        # self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # # self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.dataSocket.bind((self.cloudhost, self.cloudport))
        # # self.dataSocket.listen(5)  # 5表示表示的是服务器拒绝(超过限制数量的)连接之前，操作系统可以挂起的最大连接数量，可以看作是"排队的数量"
        # print('init socket success...')

        if self.SOCKET_TYPE == 'SOCK_DGRAM':
            self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.dataSocket.bind((self.cloudhost, self.cloudport))
        elif self.SOCKET_TYPE == 'SOCK_STREAM':
            self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dataSocket.bind((self.cloudhost, self.cloudport))
            self.dataSocket.listen(5)
        print('init socket success...')


    # def _listen(self):
    #     print('start to listen')
    #     while True:
    #         data, address = self.dataSocket.recvfrom(4096)  # 字节
    #         cloud_trans_queue.put(data)

    def cloud_power_monitor(self):
        self.power_publish_channel.queue_declare('power')

        power_monitor = EngUbuntuGPU(self.sudopw, fq=0.1)

        while True:
            time.sleep(1)
            power = power_monitor.get()
            power_dic = {'device_ID': self.device_ID,
                         'power': power,
                         }
            message = str(power_dic).encode('ascii')
            message_byte = base64.b64encode(message)
            self.power_publish_channel.basic_publish(exchange='',   # default exchange
                                              routing_key='power',  # 指定消息要发送到哪个queue
                                              body=message_byte     # 指定要发送的消息
                                              )
            power_monitor.reset()

    def _queue_monitor(self):
        self.queue_channel.queue_declare('queue_size')
        while True:
            time.sleep(1)
            queue_dic = {
                "device_ID": self.device_ID,
                "inference_queue": inference_queue.qsize(),
                "transmission_queue": cloud_trans_queue.qsize()
            }
            message = str(queue_dic).encode('ascii')
            message_byte = base64.b64encode(message)
            self.queue_channel.basic_publish(exchange='',
                                             routing_key='queue_size',  # 指定消息要发送到哪个queue
                                             body=message_byte  # 指定要发送的消息
                                             )

    def _cloud_inference(self):

        print("start to inference!")
        while True:
            if inference_queue.qsize() > 0:
                start = time.time()
                # print("inference_queue_size: ", inference_queue.qsize())
                frame_chunk = inference_queue.get()
                print("queue get", time.time() - start)
                labels_chunk = get_labels_chunk(ROOT, self.all_video_names, frame_chunk[3], frame_chunk[4])  # get chunk labels
                assert len(frame_chunk[1]) == len(labels_chunk), "inference date length != labels length"
                iouv = torch.linspace(0.5, 0.95, 10)
                acc_chunk = []
                inference_batchsize = 15
                rounds = int(np.ceil(len(labels_chunk) / inference_batchsize))
                for num in range(rounds):
                    if num < rounds - 1:
                        inference_data = frame_chunk[1][num * inference_batchsize:(num + 1) * inference_batchsize]
                    else:
                        inference_data = frame_chunk[1][num * inference_batchsize:]
                    torch.tensor(inference_data).to(self.device)
                    start_time = time.time()
                    pred = self.yolo.inference(inference_data)
                    print("time====", time.time() - start_time)
                    # 精度评估
                    for i in range(num * inference_batchsize, num * inference_batchsize + len(pred)):
                        if len(labels_chunk[i]) == 0:  # label文件为空
                            if len(pred[i - num * inference_batchsize]) == 0:  # 推理结果为空
                                acc_chunk.append(1)
                            else:
                                acc_chunk.append(0)
                        else:
                            # acc_chunk.append(0)
                            if len(pred[i - num * inference_batchsize]) == 0:
                                acc_chunk.append(0)
                            else:
                                if (len(np.shape(labels_chunk[i]))) == 1:
                                    labels_chunk[i] = labels_chunk[i][None, :]
                                labels_chunk[i][:, 1:5] = xywh2xyxy(labels_chunk[i][:, 1:5]) * (
                                    [1280, 720, 1280, 720])  # target boxes
                                stats = self.process_batch_acc(pred[i - num * inference_batchsize].cpu(),
                                                               torch.tensor(labels_chunk[i]), iouv)
                                acc_chunk.append(stats.cpu().numpy()[:, 0].mean())

                MQ_dic = {
                    'device_ID': self.device_ID,            # 边缘设备ID          int
                    'video_ID': frame_chunk[3],             # 视频名称(代号)       int
                    'chunk_ID': frame_chunk[4],             # 第几块视频           int
                    'acc_chunk': np.mean(acc_chunk),        # 该chunk的精度评估结果
                    'timestamp': frame_chunk[0],            # read前的时间戳，用于计算latency
                    'frame_num': len(labels_chunk),         # 当前chunk的帧数      int

                }

                print("MQ_dic:  ", MQ_dic)
                message = str(MQ_dic).encode('ascii')
                message_byte = base64.b64encode(message)
                self._sendMQ(self.inference_result_channel, message_byte, 'inference_result')  # 推理结果发送到MQ
                print("all", time.time() - start)

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

    def _declare_mq(self, host='10.12.11.61', port=5670, username='guest', password='guest'):
        user_info = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host, port, credentials=user_info))  # 连接服务器上的RabbitMQ服务

        # 创建channel
        channel = connection.channel()
        print('connect to rabbitmq')
        ### TODO declare more queues for the system
        return channel

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
                print('server reset')
            elif command['type'] == 'action':
                print('server set action')

        control_channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True)

        control_channel.start_consuming()


if __name__ == '__main__':
    args = Config()
    with torch.no_grad():
        client = CloudServer(args)
