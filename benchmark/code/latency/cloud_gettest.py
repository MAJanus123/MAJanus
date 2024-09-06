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

inference_queue = Queue()


def _process_inference_process(queue, stream, labels, i, j, device_ID, encode_bitrate, encode_resolution):
    """
    i is (video_ID - 1)
    j is (chunk_ID - 1)
    """
    timestamp = time.time()
    inference_process = stream.inference_process(j)  # 格式转化
    while True:
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
            print("inference_queue: ", inference_queue.qsize())
            print("end")


class EdgeClient:
    def __init__(self, args):
        self.cloud_host = args.cloudhost
        self.cloud_port = args.cloudport
        self.device_ID = int(args.deviceid)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.path_ = ROOT
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
        self.modelID = args.modelID
        self.yolo = YOLO(weights=self.model_dic[self.modelID])
        self.batchsize = args.batchsize
        self.encode_bitrate = args.bitrate
        self.encode_resolution = args.resolution
        self.offload_rate = 0  # 卸载比率
        self.inference_latency = []

        self._load_camera_data()

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
                    _process_inference_process(inference_queue, stream, labels, i, j, self.device_ID, self.encode_bitrate, self.encode_resolution)
                    time.sleep(3)
                    self._edge_inference()
                j += 1

    def _edge_inference(self):

        if inference_queue.qsize() > 0:
            print("inference_queue_size: ", inference_queue.qsize())
            start = time.time()
            frame_chunk = inference_queue.get()
            print("time------", time.time() - start)
            rounds = int(30/self.batchsize)
            for num in range(rounds):
                if (num + 1) * self.batchsize <= len(frame_chunk[1]):
                    inference_data = frame_chunk[0][num * self.batchsize:(num + 1) * self.batchsize]
                elif num * self.batchsize < len(frame_chunk[1]):
                    inference_data = frame_chunk[0][num * self.batchsize:]
                else:
                    self.inference_latency.append(0)
                    continue
                torch.tensor(inference_data).to(self.device)
                start_time = time.time()
                pred = self.yolo.inference(inference_data)
                latency = time.time() - start_time
                print("time====", latency)
                # print(pred)
                self.inference_latency.append(latency)
            # inference_data = frame_chunk[0]
            # torch.tensor(inference_data).to(self.device)
            # start_time = time.time()
            # pred = self.yolo.inference(inference_data)
            # latency = time.time() - start_time
            # print("time====", latency)
            # # print(pred)
            # self.inference_latency.append(latency)





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

    parser.add_argument("--modelID", required=False, default='03', type=str,
                        help='the model for inference')
    parser.add_argument("--batchsize", required=False, default=15, type=int,
                        help='the batchsize of a chunk')
    parser.add_argument("--resolution", required=False, default='1280x720', type=str,
                        help='the encode resolution')
    parser.add_argument("--bitrate", required=False, default='1000', type=str,
                        help='the encode bitrate')
    args = parser.parse_args()
    print(args)

    with torch.no_grad():
        client = EdgeClient(args)
