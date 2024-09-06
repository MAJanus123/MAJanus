import argparse
import ast
import math
import multiprocessing
import os
import signal
import sys
import time

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-1])
sys.path.insert(0, ROOT + '/res')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
import torch
import pika
import base64
import logging
from res.power import *
from multiprocessing import Process, Queue, Manager, Pool, Pipe
import queue
from res.video_stream import bytes2numpy
from res.utils import get_labels_chunk, ap_per_class, get_seq, get_seq_random
from res.config_edge import Config
from edge.RtpClient_chunk import RtpClient
from edge.camera import Camera
from res.ultralytics.yolo.engine.model import YOLO
from res.ultralytics.yolo.utils.ops import xywh2xyxy
from res.ultralytics.yolo.utils.metrics import box_iou

logging.getLogger("pika").setLevel(logging.WARNING)

inference_queue = queue.Queue()
transmission_queue = Manager().Queue()
DATA_PACKET_SIZE = 8688
PACKET_NUM = DATA_PACKET_SIZE / 1448

send_time = multiprocessing.Value('f', 0.0)  # 发送一个包的时间
size = multiprocessing.Value('f', 0.0)  # 发送包的大小
inference_queue_size = multiprocessing.Value('i', 0)
time_flag = multiprocessing.Value('i', 0)  # 是否sleep一秒


offload_encode = Manager().list()
local_decode = Manager().list()
inference_time = []
queue_waittime = []
acc_time = []
All_part = []
All_thread = []

def _process_transmission_process(transmission_queue, stream, i, j, device_ID, step_index, resolution, bitrate, width, height):
    # 从这里开始计算cloud延迟
    timestamp = time.time()
    transmission_process = stream.transmission_process(j, resolution=resolution, bitrate=bitrate)
    chunk = transmission_process.stdout.read()
    offload_encode.append(time.time() - timestamp)
    chunk = (chunk, timestamp, device_ID, i + 1, j + 1, height,
             width, step_index)  # (frame, timestamp, deviceID, videoID, chunkID, height, width, step_index, 是否需要更新MAPPO)
    transmission_queue.put(chunk)
    # os.kill(os.getpid(), signal.SIGKILL)
    print("cloud encode time=", time.time() - timestamp)

def _sender(rtp_client_cloud, transmission_queue):
    # 加header 发送包
    print('video sender thread start to work')
    global send_time, size
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
        rtp_client_cloud.send_image_cloud(timestamp, payload, deviceID, videoID, chunkID,
                                                                        height, width, step_index,
                                                                        )


def _queue_bandwidth_monitor(device_ID, cloudhost, mqport, edgehost):
    print("start to monitor queue and bandwidth")
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
            "inference_queue":  inference_queue_size.value,
            "transmission_queue": transmission_queue.qsize(),
            # "bandwidth": bandwidth
        }
        message = str(queue_dic).encode('ascii')
        message_byte = base64.b64encode(message)
        queue_channel.basic_publish(exchange='',
                                    routing_key='queue_size',  # 指定消息要发送到哪个queue
                                    body=message_byte  # 指定要发送的消息
                                    )

def timer(loadtime):
    time.sleep(loadtime)
    time_flag.value = 1


class EdgeClient:
    def __init__(self, args, extra_args):
        self.cloudhost = args.cloudhost
        self.cloudport = args.cloudport
        self.edgehost = args.edgehost
        self.clouddeviceid = int(args.clouddeviceid)
        self.edgedeviceid = int(args.edgedeviceid)
        self.device_type = args.devicetype
        self.sudo_passward = args.sudopw
        self.mquser = args.mquser
        self.mqpw = args.mqpw
        self.mqport = args.mqport
        self.chunk_duration = args.stream_time
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.imgsz = (360, 640)
        self.resolution = '640x360'
        self.modelID = 's'
        self.offload_rate = 0.5  # 卸载比率
        self.bitrate = 1000000

        self.loadtime = 1
        self.camera_change_flag = 1  # loadtime小于1时，得load两遍视频
        self.loadtime_list = extra_args.loadtime_list
        if self.edgedeviceid in self.loadtime_list:
            self.loadtime = extra_args.loadtime
            print("change loadtime to 0.5")
            self.camera_change_flag = 0
        print(self.loadtime, self.camera_change_flag)

        with open(os.path.join(ROOT, 'res', 'all_video_names_easy.txt'), 'r') as f:
            self.all_video_names = eval(f.read())
        model_dic = {'01': '/res/ultralytics/yolo/weights/yolov8n_trained.pt',
                     '02': '/res/ultralytics/yolo/weights/yolov8s_trained.pt',
                     '03': '/res/ultralytics/yolo/weights/yolov8m_trained.pt'}
        self.yolo_m = YOLO(ROOT + model_dic['03'], task='detect')
        self.yolo_s = YOLO(ROOT + model_dic['02'], task='detect')
        self.yolo_n = YOLO(ROOT + model_dic['01'], task='detect')

        # warm up
        random_array = np.random.rand(1, self.imgsz[0], self.imgsz[1], 3)
        pred_bantch = self.yolo_m.predict(random_array, half=True, imgsz=self.imgsz[0])
        pred_bantch = self.yolo_s.predict(random_array, half=True, imgsz=self.imgsz[0])
        pred_bantch = self.yolo_n.predict(random_array, half=True, imgsz=self.imgsz[0])

        self.camera_dataset = Camera()
        self.step_index = 0
        self.reset_times = 0
        self.done_flag = False
        self.done_flag_double = False
        self.use_agent_ID = [1,2,3,4]
        self.rtp_client_cloud = RtpClient(self.cloudhost, self.cloudport, socket_type=1)
        self.inference_result_channel = self._declare_mq(args.mqhost, args.mqport, args.mquser, args.mqpw)
        self.ppo_update_channel = self._declare_mq(args.mqhost, args.mqport, args.mquser, args.mqpw)
        self.queue_monitor = Pool(1)
        self.timer = Pool(1)
        self.sender = Pool(1)
        self.encoder = Pool(2)

        self.queue_monitor.apply_async(func=_queue_bandwidth_monitor,
                                       args=(self.edgedeviceid, self.cloudhost, self.mqport, self.edgehost))
        self.queue_monitor.daemon = True

        self.control_thread = threading.Thread(target=self._control_client, daemon=True)
        self.control_thread.start()

        self.sender.apply_async(func=_sender, args=(self.rtp_client_cloud, transmission_queue))
        self.sender.daemon = True

        self._process()

    def _process_inference_process(self, inference_queue, stream, i, j, device_ID, step_index, resolution, width,
                                   height):
        # 从这里开始计算edge延迟
        timestamp = time.time()
        inference_process = stream.inference_process(j, resolution=resolution)  # 格式转化
        edge_frame = inference_process.stdout.read(width * height * 3 * 30)
        frame = bytes2numpy(30, edge_frame, height, width)
        local_decode.append(time.time() - timestamp)
        put_time = time.time()
        chunk = (frame, timestamp, device_ID, i + 1, j + 1, height,
                 width, step_index,
                 put_time)  # (frame, timestamp, deviceID, videoID, chunkID, height, width, step_index, 放入队列的时间)
        inference_queue.put(chunk)
        # print("encode time==", time.time() - timestamp)

    def _declare_mq(self, host='10.12.11.144', port=5672, username='guest', password='guest'):
        user_info = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host, port, credentials=user_info,
                                      blocked_connection_timeout=999999, heartbeat=0))  # 连接服务器上的RabbitMQ服务

        # 创建channel
        channel = connection.channel()
        ### TODO declare more queues for the system
        return channel

    def _load_camera_data(self, reset_time):
        print('start to load camera data, reset_time==', reset_time, True)
        i = 0
        while True:
            videoNum = i % 8
            sample = self.camera_dataset[videoNum]
            stream = sample['stream']  # 一个视频
            self.num_chunks = math.ceil(stream.video_duration / self.chunk_duration)
            self.num_local_chunk = math.ceil(self.num_chunks * (1 - self.offload_rate))  # 在边缘端的块数
            self.num_cloud_chunk = int(self.num_chunks - self.num_local_chunk)
            self.load_seq = get_seq_random(self.num_local_chunk, self.num_cloud_chunk)
            # load_seq = get_seq(num_local_chunk, num_cloud_chunk)
            # print(load_seq)
            print('all chunks=', str(self.num_chunks), '  cloud chunks= ', str(self.num_cloud_chunk), '  local chunks=',
                  str(self.num_local_chunk), '    videoID=', i + 1, self.load_seq)
            j = 0
            while j < self.num_chunks:
                # 启动timer
                time_flag.value = 0
                self.timer.apply_async(func=timer, args=(self.loadtime, ))
                self.timer.daemon = True
                inference_queue_size.value = inference_queue.qsize()
                if self.load_seq[j] == 0:  # 表示处理边端数据
                    # inference_queue.put((stream, videoNum, j, time.time(), index))
                    self.local_encode = threading.Thread(target=self._process_inference_process,
                                                         args=(inference_queue, stream, videoNum, j, self.edgedeviceid,
                                                               self.step_index, self.resolution, self.imgsz[1],
                                                               self.imgsz[0]),
                                                         daemon=True)
                    self.local_encode.start()
                else:
                    self.encoder.apply_async(func=_process_transmission_process,
                                            args=(
                                            transmission_queue, stream, videoNum, j, self.edgedeviceid, self.step_index,
                                            self.resolution, self.bitrate, self.imgsz[1], self.imgsz[0]))
                    self.encoder.daemon = True

                if self.done_flag:
                    self.done_flag = False
                    self.done_flag_double = True
                    break

                time.sleep(self.loadtime - 0.2)
                while True:
                    if time_flag.value == 1:
                        break
                    else:
                        time.sleep(0.01)

                if videoNum == 7 and j == self.num_chunks - 1:   # 一个dataset的最后，发送ppo标志位  正常reset
                    self.camera_change_flag += 1
                    if self.camera_change_flag == 2:
                        message = {
                            'device_ID': self.edgedeviceid,
                            'flag': 1,
                            'reset_times': reset_time
                        }
                        self._sendMQ(self.ppo_update_channel, message, 'ppo_update_flag')
                j += 1

            if self.done_flag_double:  # 说明此时camera需要重置，退出此camera线程
                self.done_flag_double = False
                print("  =============Exiting loading camera data loop.")
                break
            i = i + 1

    def _process(self):  # 处理listen得到的数据
        print('start to process')
        while True:
            data = inference_queue.get()  # (frame, timestamp, deviceID, videoID, chunkID, height, width, step_index, 放入队列时间, 是否更新ppo)
            inference_queue.task_done()
            queue_wait = time.time() - data[8]  # 从放入队列到取出的时间
            queue_waittime.append(queue_wait)
            timestamp = data[1]
            frame = data[0]
            videoID = data[3]
            chunkID = data[4]
            (height, width) = (data[5], data[6])
            step_index = data[7]
            # print("queue_wait", queue_wait)
            self._edge_inference((timestamp, frame, self.edgedeviceid, videoID, chunkID, queue_wait, height, width, step_index))

    def _sendMQ(self, channel, message, queue):
        message = str(message).encode('ascii')
        message_byte = base64.b64encode(message)
        channel.queue_declare(queue=queue)

        channel.basic_publish(exchange='',
                              routing_key=queue,  # 指定消息要发送到哪个queue
                              body=message_byte  # 指定要发送的消息
                              )

    def _edge_inference(self, frame_chunk):
        (height, width) = (frame_chunk[6], frame_chunk[7])
        labels_chunk = get_labels_chunk(ROOT, self.all_video_names, frame_chunk[3], frame_chunk[4])  # get chunk labels
        # assert len(frame_chunk[1]) == len(
        #     labels_chunk), "inference date length {len1} != labels length {len2}".format(
        #     len1=len(frame_chunk[1]), len2=len(labels_chunk), video=frame_chunk[3], chunk=frame_chunk[4])
        if len(frame_chunk[1]) == len(labels_chunk):
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
                length = math.ceil(inference_data.shape[0] / 2)
                # torch.tensor(inference_data).to(self.device)
                if self.modelID == 'm':
                    pred_bantch = self.yolo_m.predict(inference_data, half=True, imgsz=height)
                    pred_bantch_extra = self.yolo_m.predict(inference_data[:length], half=True, imgsz=height)
                elif self.modelID == 's':
                    pred_bantch = self.yolo_s.predict(inference_data, half=True, imgsz=height)
                    pred_bantch_extra = self.yolo_s.predict(inference_data[:length], half=True, imgsz=height)
                    pred_bantch_extra = self.yolo_s.predict(inference_data[:math.ceil(length / 4)], half=True, imgsz=height)
                else:
                    pred_bantch = self.yolo_n.predict(inference_data, half=True, imgsz=height)

                preds.extend([x.boxes.data.cpu() for x in pred_bantch])  # 一个chunk的推理结果
            inferenceTime = time.time() - start
            # inference_time.append(inferenceTime)
            print("inference time====", inferenceTime)
            # 精度评估
            start_acc = time.time()
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
                    labels_chunk[i][:, 1:5] = xywh2xyxy(labels_chunk[i][:, 1:5]) * (
                    [width, height, width, height])  # target boxes
                    # for p in pred:
                    #     p[5] = torch.full_like(p[5], self.convert.get(int(p[5]), 3))  # 3表示其他类型
                    correct_bboxes = self.process_batch_acc(pred, torch.tensor(labels_chunk[i]), iouv)
                stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls))  # (tp, conf, pcls, tcls)
            stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

            ap = ap_per_class(*stats)
            # print(ap)
            mAP = [np.mean(x) for x in ap]
            acc_time.append(time.time() - start_acc)
            one_acc_chunk = np.mean(mAP)
            MQ_dic = {
                'device_ID': self.edgedeviceid,  # 从哪个设备产生推理结果      int
                'produce_device': frame_chunk[2],  # 视频流从哪个设备产生    int
                'video_ID': frame_chunk[3],  # 视频名称(代号)       int
                'chunk_ID': frame_chunk[4],  # 第几块视频           int
                'acc_chunk': one_acc_chunk,  # 该chunk的精度评估结果
                'timestamp': frame_chunk[0],  # read前的时间戳，用于计算latency
                'frame_num': len(labels_chunk),  # 当前chunk的帧数      int
                'queue_wait': frame_chunk[5],  # queue wait time
                'step_index': frame_chunk[8]
            }

            print("MQ_dic:  ", MQ_dic, "process_time: ", time.time() - MQ_dic['timestamp'], "MQ_dic size: ", sys.getsizeof(MQ_dic))
            if self.edgedeviceid in self.use_agent_ID:
                self._sendMQ(self.inference_result_channel, MQ_dic, 'inference_result')  # 推理结果发送到MQ
        else:
            print("error")

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
                self.use_agent_ID = command['use_agent_ID']  # 确定该设备是否是真实在RL训练中跑的设备
                self.reset_times += 1
                self.done_flag = command['value']['done']
                self.first_flag = command['value']['first']
                # reset
                if self.edgedeviceid in self.use_agent_ID:
                    self.offload_rate = 0.5
                    self.step_index = 0
                    self.resolution = '640x360'
                    image = self.resolution.split('x')
                    image = list(map(int, image))
                    self.imgsz = (image[1], image[0])
                    self.bitrate = 1000000
                    self.modelID = 's'
                else:
                    self.offload_rate = 0
                    self.step_index = 0
                    self.resolution = '488x240'
                    image = self.resolution.split('x')
                    image = list(map(int, image))
                    self.imgsz = (image[1], image[0])
                    self.bitrate = 500000
                    self.modelID = 'n'
                if self.edgedeviceid in self.loadtime_list:
                    self.camera_change_flag = 0
                else:
                    self.camera_change_flag = 1
                if self.first_flag == True:  # 第一次启动
                    self.reset_times = 1  # 能够实现多个RL_train, 连续执行
                    print(self.use_agent_ID)
                self.reset_thread = threading.Thread(target=self._load_camera_data, args=(self.reset_times,))
                self.reset_thread.start()
            elif command['type'] == 'action':
                if self.edgedeviceid in command['use_agent_ID']:
                    self.offload_rate = command['value']['offloading_rate'][self.edgedeviceid - 1]
                    self.modelID = command['value']['edge_model'][self.edgedeviceid - 1]
                    self.resolution = command['value']['resolution'][self.edgedeviceid - 1]
                    self.bitrate = command['value']['bitrate'][self.edgedeviceid - 1]
                else:
                    self.offload_rate = 0
                    self.modelID = 'n'
                    self.resolution = '488x240'
                    self.bitrate = 500000

                self.step_index = command['index']
                image = self.resolution.split('x')
                image = list(map(int, image))
                self.imgsz = (image[1], image[0])

                self.num_local_chunk = math.ceil(self.num_chunks * (1 - self.offload_rate))  # 在边缘端的块数
                self.num_cloud_chunk = int(self.num_chunks - self.num_local_chunk)
                self.load_seq = get_seq_random(self.num_local_chunk, self.num_cloud_chunk)

                print('set offloading rate as %s' % str(command['value']['offloading_rate'][0]), 'index = ',
                      self.step_index)
            else:
                self.done_flag = True
                if self.reset_thread.is_alive():
                    self.reset_thread.join()
                while not inference_queue.empty():
                    try:
                        inference_queue.get_nowait()
                    except queue.Empty:
                        pass
                while not transmission_queue.empty():
                    try:
                        transmission_queue.get_nowait()
                    except queue.Empty:
                        pass
                if self.edgedeviceid in self.use_agent_ID:
                    message = {
                        'device_ID': self.edgedeviceid,
                        'flag': 2
                    }
                    self._sendMQ(self.ppo_update_channel, message, 'ppo_update_flag')
                    print("kill over")
                self.done_flag = False

        control_channel.basic_consume(
            queue=queue_name, on_message_callback=callback, auto_ack=True)

        control_channel.start_consuming()


if __name__ == '__main__':
    args = Config()
    args.devicetype = 'NX'
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadtime", required=False, default=1, type=float,
                        help="loadtime")
    parser.add_argument("--loadtime_list", nargs="+", type=int, default=[1,2,3,4,5,6,7,8],
                        help="List of loadtime")
    extra_args = parser.parse_args()
    with torch.no_grad():
        client = EdgeClient(args, extra_args)
