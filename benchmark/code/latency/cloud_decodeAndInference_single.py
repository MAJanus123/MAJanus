import multiprocessing
import pickle
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



def _process_transmission_process(transmission_queue, stream, labels, i, j, device_ID, encode_bitrate, encode_resolution):
    transmission_process = stream.transmission_process(j)
    labels_chunk = getchunk_labels(labels, j)
    chunk = transmission_process.stdout.read()
    chunk = (chunk, labels_chunk)
    transmission_queue.put(chunk)
    print("transmission_queue: ", transmission_queue.qsize())
    print("end2")

def _process(decode_data):  # 处理listen得到的数据
    print('start to process')
    for key in decode_data:
        data = decode_data[key]
        # 转码
        _process = (ffmpeg
                    .input('pipe:', format='h264')
                    .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
                    .run_async(pipe_stdin=True, pipe_stdout=True)
                    )
        _process.stdin.write(data[0])
        _process.stdin.close()
        in_bytes = _process.stdout.read(1280 * 720 * 3 * 30)
        frame = bytes2numpy(30, in_bytes, 720, 1280)
        inference_queue.put((frame, data[1]))


class CloudClient:
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

        # self._load_camera_data()

        try:
            with open('decode.pickle', 'rb+') as f:
                decode_data = pickle.load(f)
        except EOFError:  # 捕获异常EOFError 后返回None
            pass
        print(len(decode_data))
        self.process = threading.Thread(target=_process, args=(decode_data,))
        self.process.start()

        self.cloud_inference = threading.Thread(target=self._cloud_inference, daemon=True)
        self.cloud_inference.start()
        time.sleep(45)
        with open("cloud_decodeAndInference_single.txt", "a") as file:
            # file.write(
            #      "modelID:" + self.modelID + "   bitrate =" + str(self.encode_bitrate) + "   resolution = " + self.encode_resolution + "  batchsize = " + str(self.batchsize) +
            #      "  inference length = " + str(len(self.inference_latency)) + "  edge_inference latency = " + str(np.mean(self.inference_latency) * int(30/self.batchsize)) + "\n")
            file.write(
                "modelID:" + self.modelID + "  batchsize = " + str(
                    self.batchsize) +
                "  inference length = " + str(len(self.inference_latency)) + "  cloud_inference latency = " + str(
                    np.mean(self.inference_latency) * int(30 / self.batchsize)) + "\n")

    def _load_camera_data(self):
        # for i in range(len(self.camera_dataset)):
        num = 0
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
                    _process_transmission_process(transmission_queue, stream, labels, i, j, self.device_ID, self.encode_bitrate, self.encode_resolution)
                j += 1

        # 将queue存入pickle
        print("start write decode======")
        decode_content_dict = {}
        with open('decode.pickle', 'wb') as f:
            while not transmission_queue.empty():
                decode_content = transmission_queue.get()
                num += 1
                decode_content_dict.update({num: decode_content})
                transmission_queue.task_done()
            pickle.dump(decode_content_dict, f)
        decode_content_dict = {}
        print("num=====", num)


    def _cloud_inference(self):
        print("Start to inference")
        while True:
            if not transmission_queue.empty():
                print("inference_queue_size: ", inference_queue.qsize())
                frame_chunk = inference_queue.get()
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
        client = CloudClient(args)
