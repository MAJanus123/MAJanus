import ast
import math
import os
import signal
import sys
import time

from pathlib import Path
from struct import *

import ffmpeg

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-1])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import socket
import torch
import pika
from res.RtpPacket_chunk import RtpPacket
from res.yolov5.YOLO import YOLO
from res.yolov5.utils.general import xywh2xyxy
import base64
import logging
from res.power import *
from multiprocessing import Process, Queue, Manager, Pool, Pipe
from res.yolov5.utils.metrics import box_iou
from res.video_stream import bytes2numpy
from res.utils import get_labels_chunk
from res.config_cloud import Config
from edge.RtpClient_chunk import RtpClient
from edge.camera import Camera

logging.getLogger("pika").setLevel(logging.WARNING)

inference_queue = Manager().Queue()
transmission_queue = Manager().Queue()
DATA_PACKET_SIZE = 8688
PACKET_NUM = DATA_PACKET_SIZE / 1448

transmission_time = Manager().list()
decode_time = Manager().list()
inference_time = []
acc_time = []
All_part = []
All_acc = []


def _process_transmission_process(transmission_queue, stream, i, j, device_ID, step_index):
    start = time.time()
    transmission_process = stream.transmission_process(j)
    # 从这里开始计算cloud延迟
    chunk = transmission_process.stdout.read()
    timestamp = time.time()
    chunk = (chunk, timestamp, device_ID, i + 1, j + 1, stream.height,
             stream.width, step_index, start)  # (frame, timestamp, deviceID, videoID, chunkID, height, width, step_index)
    transmission_queue.put(chunk)
    os.kill(os.getpid(), signal.SIGKILL)

def _process_inference_process(inference_queue, stream, i, j, device_ID, step_index):
    (width, height) = (854, 480)
    inference_process = stream.inference_process(j, resolution='854x480')  # 格式转化
    # 从这里开始计算cloud延迟
    start = time.time()
    edge_frame = inference_process.stdout.read(width * height * 3 * 30)
    print("read time", time.time() - start)
    timestamp = time.time()
    frame = bytes2numpy(30, edge_frame, height, width)
    chunk = (frame, timestamp, device_ID, i + 1, j + 1, stream.height,
             stream.width, step_index, time.time())  # (frame, timestamp, deviceID, videoID, chunkID, height, width, step_index, 放入队列的时间)
    start2 = time.time()
    inference_queue.put(chunk)
    print("put in time==", time.time() - start2)
    os.kill(os.getpid(), signal.SIGKILL)

def _sender(rtp_client_cloud, transmission_queue):
    # 加header 发送包
    print('video sender thread start to work')
    while True:
        transmission_chunk = transmission_queue.get()
        payload = transmission_chunk[0]  # image_bytes
        timestamp = transmission_chunk[1]
        deviceID = transmission_chunk[2]
        videoID = transmission_chunk[3]
        chunkID = transmission_chunk[4]
        height = transmission_chunk[5]
        width = transmission_chunk[6]
        step_index = transmission_chunk[7]
        # print("send device %d video %d chunk %d" % (deviceID, videoID, chunkID))
        rtp_client_cloud.send_image_cloud(timestamp, payload, deviceID, videoID, chunkID, height, width, step_index, transmission_chunk[8])


def _power_monitor(device_type, sudo_passward, device_ID, cloudhost, mqport):
    print("start to monitor power")
    user_info = pika.PlainCredentials("guest", "guest")
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(cloudhost, mqport, credentials=user_info))

    stream_channel = connection.channel()
    stream_channel.queue_declare('power')

    if device_type == 'NX':
        power_monitor = EngNX(sudo_passward, fq=0.1)
    elif device_type == 'CPU':
        power_monitor = EngUbuntuCPU(sudo_passward, fq=0.1)
    else:
        ## TODO change later for other device
        power_monitor = EngUbuntuGPU(sudo_passward, fq=0.1)
        pass

    while True:
        time.sleep(1)
        power = power_monitor.get()
        # print("power==", power)
        power_dic = {'device_ID': device_ID,
                     'power': power
                     }
        message = str(power_dic).encode('ascii')
        message_byte = base64.b64encode(message)
        stream_channel.basic_publish(exchange='',
                                     routing_key='power',  # 指定消息要发送到哪个queue
                                     body=message_byte  # 指定要发送的消息
                                     )
        power_monitor.reset()


def _queue_monitor(device_ID, cloudhost, mqport):
    print("start to monitor queue")
    user_info = pika.PlainCredentials("guest", "guest")
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(cloudhost, mqport, credentials=user_info))

    queue_channel = connection.channel()
    queue_channel.queue_declare('queue_size')
    while True:
        time.sleep(1)
        queue_dic = {
            "device_ID": device_ID,
            "inference_queue": inference_queue.qsize(),
            "transmission_queue": transmission_queue.qsize()
        }
        # print(queue_dic)
        message = str(queue_dic).encode('ascii')
        message_byte = base64.b64encode(message)
        queue_channel.basic_publish(exchange='',
                                    routing_key='queue_size',  # 指定消息要发送到哪个queue
                                    body=message_byte  # 指定要发送的消息
                                    )


class EdgeClient:
    def __init__(self, args):
        self.cloudhost = args.cloudhost
        self.cloudport = args.cloudport
        self.edgehost = args.edgehost
        self.edgeport = args.edgeport
        self.clouddeviceid = int(args.clouddeviceid)
        self.edgedeviceid = int(args.edgedeviceid)
        self.device_type = args.devicetype
        self.sudo_passward = args.sudopw
        self.mquser = args.mquser
        self.mqpw = args.mqpw
        self.mqport = args.mqport
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.path_ = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))  # path= /../Video-Analytics-Task-Offloading
        # print('root path : ', self.path_)
        sys.path.insert(0, self.path_ + '/res/yolov5')

        self.chunk_duration = 1
        self.imgsz = (480, 854)
        with open(os.path.join(ROOT, 'res', 'all_video_names.txt'), 'r') as f:
            self.all_video_names = eval(f.read())
        model_dic = {'01': '/res/yolov5/weights/5n_b32_e20_p.pt',
                     '02': '/res/yolov5/weights/5s_b32_e20_p.pt',
                     '03': '/res/yolov5/weights/5m_b32_e50_p.pt'}
        self.yolo = YOLO(ROOT + model_dic['02'], imgsz=self.imgsz)
        self.camera_dataset = Camera()
        self.offload_rate = 0.8  # 卸载比率
        self.step_index = 0
        self.done_flag = False
        self.set_flag = False
        self.prefetch = ()  # 预先从inference_queue中取出一个
        self.rtp_client_cloud = RtpClient(self.cloudhost, self.cloudport, socket_type=1)
        self.inference_result_channel = self._declare_mq(args.mqhost, args.mqport, args.mquser, args.mqpw)
        # self.power_monitor = Pool(1)
        self.queue_monitor = Pool(1)
        self.encode = Pool(3)
        self.sender = Pool(1)

        self.queue_monitor.apply_async(func=_queue_monitor, args=(self.edgedeviceid, self.cloudhost, self.mqport))
        self.queue_monitor.daemon = True

        self.control_thread = threading.Thread(target=self._control_client, daemon=True)
        self.control_thread.start()

        # self.count = threading.Thread(target=self._count, args=(transmission_time, decode_time), daemon=True)
        # self.count.start()

        self.data_prefetch = threading.Thread(target=self._Prefetch, args=(inference_queue,), daemon=True)
        self.data_prefetch.start()

        self.sender.apply_async(func=_sender, args=(self.rtp_client_cloud, transmission_queue))
        self.sender.daemon = True

        self._process()

    def _count(self, transmission_time, decode_time):
        time.sleep(500)
        print("All_part====", np.mean(All_part))
        print("transmission====", np.mean(transmission_time))
        print("decode====", np.mean(decode_time))
        print("inference====", np.mean(inference_time) * 2)
        print("acc====", np.mean(acc_time) * 2)
        print("acc chunk", np.mean(All_acc))

    def _declare_mq(self, host='10.12.11.61', port=5670, username='guest', password='guest'):
        user_info = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host, port, credentials=user_info,
                                      blocked_connection_timeout=999999, heartbeat=0))  # 连接服务器上的RabbitMQ服务

        # 创建channel
        channel = connection.channel()
        # print('connect to rabbitmq')
        ### TODO declare more queues for the system
        return channel

    def _Prefetch(self, inference_queue):
        while True:
            if self.prefetch == ():
                data = inference_queue.get()
                self.prefetch = data
                inference_queue.task_done()
            else:
                time.sleep(0.01)

    def _load_camera_data(self, done_flag):
        print('start to load camera data: done_flag %s, %s' % (str(done_flag()), str(threading.get_ident())))
        i = 0
        # while i < len(self.camera_dataset):
        while True:
            videoNum = i % 50
            sample = self.camera_dataset[videoNum]
            stream = sample['stream']  # 一个视频
            num_chunks = math.ceil(stream.video_duration / self.chunk_duration)
            num_local_chunk = math.ceil(
                num_chunks * (1 - self.offload_rate))  # 在边缘端的块数
            num_cloud_chunk = int(num_chunks - num_local_chunk)
            print('all chunks=', str(num_chunks), '  cloud chunks= ', str(num_cloud_chunk), '  local chunks=',
                  str(num_local_chunk), '    videoID=', i + 1)
            index = self.step_index
            j = 0
            while j < num_chunks:
                if self.set_flag == True:
                    self.set_flag = False
                    break
                if j < num_local_chunk:
                    # inference_queue.put((stream, videoNum, j, time.time(), index))
                    self.encode.apply_async(func=_process_inference_process,
                                            args=(inference_queue, stream, videoNum, j, self.edgedeviceid, index))
                else:
                    self.encode.apply_async(func=_process_transmission_process,
                                            args=(transmission_queue, stream, videoNum, j, self.clouddeviceid, index))
                j += 1
                time.sleep(1)

            if done_flag():
                print("  =============Exiting loading camera data loop. kill thread %s" % str(threading.get_ident()))
                break
            i = i + 1

    # def _process(self, inference_queue):  # 处理listen得到的数据
    #     print('start to process')
    #     while True:
    #         data = inference_queue.get()  # (stream, videoNum, chunkNum, queue_timestamp, step_index)
    #         queue_wait = time.time() - data[3]
    #         stream = data[0]
    #         # print("======", queue_wait)
    #         (height, width) = self.imgsz
    #         inference_process = stream.inference_process(data[2], resolution='854x480')  # 格式转化
    #
    #         edge_frame = inference_process.stdout.read(width * height * 3 * 30)
    #
    #         # To numpy
    #         timestamp = time.time() - queue_wait  # add the time of queue_wait
    #         frame = bytes2numpy(30, edge_frame, height, width)
    #         decode_time.append(time.time() - timestamp)
    #         self._edge_inference((timestamp, frame, self.edgedeviceid, data[1] + 1, data[2] + 1, data[4]))
    #         inference_queue.task_done()

    def _process(self):  # 处理listen得到的数据
        print('start to process')
        while True:
            if self.prefetch != ():
                data = self.prefetch  # (frame, timestamp, deviceID, videoID, chunkID, height, width, step_index)
                self.prefetch = ()
                queue_wait = time.time() - data[8]  # 从放入队列到取出的时间
                timestamp = data[1]
                frame = data[0]
                videoID = data[3]
                chunkID = data[4]
                indexID = data[7]
                print("queue_wait", queue_wait)
                self._edge_inference((timestamp, frame, self.edgedeviceid, videoID, chunkID, indexID))
            else:
                time.sleep(0.01)

    def _edge_inference(self, frame_chunk):

        def _sendMQ(channel, message, queue):
            channel.queue_declare(queue=queue)

            channel.basic_publish(exchange='',
                                  routing_key=queue,  # 指定消息要发送到哪个queue
                                  body=message  # 指定要发送的消息
                                  )
            # print('mesage have send to ', queue)

        labels_chunk = get_labels_chunk(ROOT, self.all_video_names, frame_chunk[3], frame_chunk[4])  # get chunk labels
        assert len(frame_chunk[1]) == len(
            labels_chunk), "inference date length {len1} != labels length {len2}".format(
            len1=len(frame_chunk[1]), len2=len(labels_chunk), video=frame_chunk[3], chunk=frame_chunk[4])
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
            inference_time.append(time.time() - start_time)
            # 精度评估
            acc_start = time.time()
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
                        # labels_chunk[i][:, 1:5] = xywh2xyxy(labels_chunk[i][:, 1:5]) * (
                        #     [1280, 720, 1280, 720])  # target boxes
                        labels_chunk[i][:, 1:5] = xywh2xyxy(labels_chunk[i][:, 1:5]) * (
                            [854, 480, 854, 480])  # target boxes
                        stats = self.process_batch_acc(pred[i - num * inference_batchsize].cpu(),
                                                       torch.tensor(labels_chunk[i]), iouv)
                        acc_chunk.append(stats.cpu().numpy()[:, 0].mean())
            acc_time.append(time.time() - acc_start)
        one_acc_chunk = np.mean(acc_chunk)
        MQ_dic = {
            'device_ID': self.edgedeviceid,  # 边缘设备ID          int
            'video_ID': frame_chunk[3],  # 视频名称(代号)       int
            'chunk_ID': frame_chunk[4],  # 第几块视频           int
            'acc_chunk': one_acc_chunk,  # 该chunk的精度评估结果
            'timestamp': frame_chunk[0],  # read前的时间戳，用于计算latency
            'frame_num': len(labels_chunk),  # 当前chunk的帧数      int
            'step_index': frame_chunk[5],  # step index
        }

        print("MQ_dic:  ", MQ_dic)
        message = str(MQ_dic).encode('ascii')
        message_byte = base64.b64encode(message)
        _sendMQ(self.inference_result_channel, message_byte, 'inference_result')  # 推理结果发送到MQ
        All_part.append(time.time() - MQ_dic['timestamp'])
        All_acc.append(one_acc_chunk)

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
                self.offload_rate = 0.5
                self.step_index = 0
                # self.done_flag == False when run the first episode,
                # self.done_flag == True when finish one episode, need to reset the whole system.
                if self.first_flag == True:
                    reset_thread = threading.Thread(target=self._load_camera_data, args=([lambda: self.done_flag]))
                    reset_thread.start()
                else:
                    self.done_flag = True
                    while not inference_queue.empty():
                        inference_queue.get_nowait()
                    while not transmission_queue.empty():
                        transmission_queue.get_nowait()
                    time.sleep(7)
                    self.done_flag = False
                    reset_thread = threading.Thread(target=self._load_camera_data, args=([lambda: self.done_flag]))
                    reset_thread.start()
            elif command['type'] == 'action':
                # while not inference_queue.empty():
                #     inference_queue.get_nowait()
                self.offload_rate = command['value']['offloading_rate'][0]
                self.set_flag = True
                self.step_index = command['index']
                # print(self.offload_rate)
                print('set offloading rate as %s' % str(command['value']['offloading_rate']), 'index = ', self.step_index)

        control_channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True)

        control_channel.start_consuming()


if __name__ == '__main__':
    args = Config()
    args.devicetype = 'NX'
    with torch.no_grad():
        client = EdgeClient(args)
