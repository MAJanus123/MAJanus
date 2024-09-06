import multiprocessing
import queue
import signal
import time
import os
import sys

from pathlib import Path

# sys.path.append(str('/home/hhf/xyb/Video-Analytics-Task-Offloading/'))  # add ROOT to PATH
# print(sys.path)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-3])
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
from edge.camera import Camera
from res.utils import getchunk_labels

logging.getLogger("pika").setLevel(logging.WARNING)

transmission_queue = Manager().Queue()
inference_queue = Manager().Queue()
encode_queue = Manager().Queue()

def _process_inference_process(queue, stream, labels, i, j, device_ID):
    """
    i is (video_ID - 1)
    j is (chunk_ID - 1)
    """
    timestamp = time.time()
    (height, width) = (480, 854)
    inference_process = stream.inference_process(j)  # 格式转化
    while True:

        edge_frame = inference_process.stdout.read(width * height * 3 * 30)
        if not edge_frame:
            # queue.put(None)
            break
        else:
            frame = bytes2numpy(30, edge_frame, height, width)  # 720 * 1280
    encode_queue.put(time.time() - timestamp)
    print("encode_queue: ", encode_queue.qsize())


def _process_transmission_process(transmission_queue, stream, labels, i, j, device_ID):
    timestamp = time.time()
    transmission_process = stream.transmission_process(j)
    # 从这里开始计算cloud延迟
    chunk = transmission_process.stdout.read()
    chunk = (chunk, timestamp, device_ID, i + 1, j + 1)  # (chunk, deviceID, videoID, chunkID)
    transmission_queue.put(chunk)
    encode_queue.put(time.time() - timestamp)
    print("transmission_queue: ", transmission_queue.qsize())
    print("end2")



class EdgeClient:
    def __init__(self, args):
        self.cloud_host = args.cloudhost
        self.cloud_port = args.cloudport
        self.device_ID = int(args.deviceid)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.path_ = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))  # path= /../Video-Analytics-Task-Offloading
        print('root path : ', self.path_)
        sys.path.insert(0, self.path_ + '/res/yolov5')

        self.device_type = args.devicetype
        self.sudo_passward = args.sudopw

        self.model_dic = {'01': self.path_ + '/res/yolov5/weights/5n_b32_e20_p.pt',
                          '02': self.path_ + '/res/yolov5/weights/5s_b32_e20_p.pt',
                          '03': self.path_ + '/res/yolov5/weights/5m_b32_e50_p.pt'}

        self.camera_dataset = Camera()
        self.chunk_duration = 1

        self.imgsz = (720, 1280)
        self.data_path = self.path_ + '/res/yolov5/data/bdd100k.yaml'
        self.batchsize = 30

        self.offload_rate = 1  # 卸载比率
        self.po = Pool(2)

        self._load_camera_data()
        time.sleep(10)
        sum = 0
        length = encode_queue.qsize()
        for i in range(length):
            sum += encode_queue.get()
        print("length", length)
        print("平均latency============", sum/length)

    def _load_camera_data(self):
        # for i in range(len(self.camera_dataset)):
        for i in range(10):
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
                    print('put to local')
                    self.po.apply_async(func=_process_inference_process,
                                        args=(inference_queue, stream, labels, i, j, self.device_ID))
                else:
                    print('put to offload')
                    self.po.apply_async(func=_process_transmission_process,
                                        args=(transmission_queue, stream, labels, i, j, self.device_ID))
                j += 1
                time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloudhost", required=False, default='10.12.11.61', type=str,
                        help="the host ip of the cloud")
    parser.add_argument("--cloudport", required=False, default=1080, type=int,
                        help="the host port")
    parser.add_argument("--deviceid", required=False, default='02', type=str,
                        help="local device id")
    parser.add_argument("--mqhost", required=False, default='10.12.11.61', type=str,
                        help='the rabbitmq server ip')
    parser.add_argument("--mqport", required=False, default=5670, type=int,
                        help='the rabbitmq server port')
    parser.add_argument("--mquser", required=False, default='guest', type=str,
                        help='the rabbitmq server username')
    parser.add_argument("--mqpw", required=False, default='guest', type=str,
                        help='the rabbitmq server password')
    parser.add_argument("--devicetype", required=False, default='NX', type=str,
                        help='device specific ')
    parser.add_argument("--sudopw", required=False, default='1223', type=str,
                        help='root access, do not share')
    args = parser.parse_args()
    print(args)

    with torch.no_grad():
        client = EdgeClient(args)
