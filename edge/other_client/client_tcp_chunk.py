import ast
import math
import os
import sys

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

logging.getLogger("pika").setLevel(logging.WARNING)

inference_queue = Manager().Queue()
listen_queue = Manager().Queue()
DATA_PACKET_SIZE = 8688
PACKET_NUM = DATA_PACKET_SIZE / 1448

transmission_time = Manager().list()
decode_time = Manager().list()
inference_time = []
acc_time = []
All_part = []
All_acc = []

def _listen(edgehost, edgeport, listen_queue):
    # dataSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dataSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dataSocket.settimeout(999999)
    dataSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 就是它，在bind前加
    dataSocket.bind((edgehost, edgeport))
    print('init socket success ... start to listen ...')
    dataSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 9999999)
    dataSocket.listen(5)  # 5表示表示的是服务器拒绝(超过限制数量的)连接之前，操作系统可以挂起的最大连接数量，可以看作是"排队的数量"
    data_array = bytearray()
    connection, client_address = dataSocket.accept()
    while True:
        data = connection.recv(DATA_PACKET_SIZE)
        if len(data) > 0:
            if data[-4:] == b'\\EOF':  # marker=2的包
                data_array.extend(data)
                listen_queue.put(data_array)
                data_array = bytearray()
            else:
                data_array.extend(data)

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
            "transmission_queue": listen_queue.qsize()
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

        self.device_ID = int(args.edgedeviceid)

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
        self.model_dic = {'01': '/res/yolov5/weights/5n_b32_e20_p.engine',
                          '02': '/res/yolov5/weights/5s_b32_e20_p.pt',
                          '03': '/res/yolov5/weights/5m_b32_e50_p.pt'}

        self.chunk_duration = 1
        with open(os.path.join(ROOT, 'res', 'all_video_names.txt'), 'r') as f:
            self.all_video_names = eval(f.read())

        self.imgsz = (480, 854)
        self.data_path = self.path_ + '/res/yolov5/data/bdd100k.yaml'
        self.yolo = YOLO(self.path_ + self.model_dic['02'], imgsz=self.imgsz)
        self.batchsize = 30

        self.all_Sequence = {}  # 存储所有被接收包的序号
        self.all_Payload = {}  # 存储所有device传的视频在process过程中的字节流
        self.all_num = {}  # 存储所有chunk收到包的个数

        # self.rtp_client = RtpClient(self.cloudhost, self.cloudport, socket_type=0)
        self.inference_result_channel = self._declare_mq(args.mqhost, args.mqport, args.mquser, args.mqpw)
        self.listen_process_send_pool = Pool(1)
        self.power_monitor = Pool(1)
        self.queue_monitor = Pool(1)

        self.listen_process_send_pool.apply_async(func=_listen, args=(self.edgehost, self.edgeport, listen_queue))
        # self.listen_process_send_pool.apply_async(func=_process, args=(
        #     self.all_Payload, self.all_Sequence, self.all_num, listen_queue))
        # self.listen_process_send_pool.apply_async(func=_sender, args=(self.rtp_client, cloud_trans_queue))
        self.listen_process_send_pool.daemon = True

        # self.power_monitor.apply_async(func=_power_monitor, args=(
        # self.device_type, self.sudo_passward, self.device_ID, self.cloudhost, self.mqport))
        # self.power_monitor.daemon = True

        self.queue_monitor.apply_async(func=_queue_monitor, args=(self.device_ID, self.cloudhost, self.mqport))
        self.queue_monitor.daemon = True

        self.control_thread = threading.Thread(target=self._control_client, daemon=True)
        self.control_thread.start()

        # self.count = threading.Thread(target=self._count, args=(transmission_time, decode_time), daemon=True)
        # self.count.start()

        self._process(listen_queue)


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
            pika.ConnectionParameters(host, port, credentials=user_info, blocked_connection_timeout=999999))  # 连接服务器上的RabbitMQ服务

        # 创建channel
        channel = connection.channel()
        # print('connect to rabbitmq')
        ### TODO declare more queues for the system
        return channel

    def _process(self, listen_queue):  # 处理listen得到的数据
        print('start to process')
        while True:
            data_array = listen_queue.get()  # (data, counter)
            payload = data_array[:-16]  # data
            head = data_array[-16:]  # 包头信息

            imagePacket = RtpPacket()
            imagePacket.decode(head)
            timestamp = imagePacket.getTimestamp()
            device_ID = imagePacket.getDeviceID()
            video_ID = imagePacket.getVideoID()
            chunk_ID = imagePacket.getChunkID()
            # (height, width) = unpack('ll', payload)
            (height, width) = (480, 854)
            transmission_time.append(time.time() - timestamp)
            start = time.time()
            # 转码
            decode_process = (ffmpeg
                              .input('pipe:', format='h264')
                              .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
                              .run_async(pipe_stdin=True, pipe_stdout=True)
                              )
            decode_process.stdin.write(payload)
            decode_process.stdin.close()
            in_bytes = decode_process.stdout.read(width * height * 3 * 30)
            frame = bytes2numpy(30, in_bytes, height, width)
            decode_time.append(time.time() - start)
            self._edge_inference((timestamp, frame, device_ID, video_ID, chunk_ID))
            listen_queue.task_done()


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
            'device_ID': self.device_ID,  # 边缘设备ID          int
            'video_ID': frame_chunk[3],  # 视频名称(代号)       int
            'chunk_ID': frame_chunk[4],  # 第几块视频           int
            'acc_chunk': one_acc_chunk,  # 该chunk的精度评估结果
            'timestamp': frame_chunk[0],  # read前的时间戳，用于计算latency
            'frame_num': len(labels_chunk),  # 当前chunk的帧数      int

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
                print('server reset')
                while not inference_queue.empty():
                    inference_queue.get()
            # elif command['type'] == 'action':
            #     print('server set action')

        control_channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True)

        control_channel.start_consuming()

if __name__ == '__main__':
    args = Config()
    args.devicetype = 'NX'
    with torch.no_grad():
        client = EdgeClient(args)
