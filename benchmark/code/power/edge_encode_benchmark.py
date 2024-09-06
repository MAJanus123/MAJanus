import argparse
import pickle
import queue
import time
import os
import io

from matplotlib import pyplot as plt

from video_stream import VideoStream
import sys
import threading
import torch
import ffmpeg
import cv2
import numpy as np
from RtpPacket import RtpPacket
from RtpClient import RtpClient
# from yolov5_master.myinference_beifen import Inference
from myinference import Inference
from yolov5_master.mydataset import preprocess
import math
import json
import base64
import logging
from eval import _eval
from power import EngNX
import pandas as pd
from evaluation import _evaluation

divide1_queue = queue.Queue()
divide2_queue = queue.Queue()
_offloading_queue = queue.Queue()
edge_infer_queue = queue.Queue()


class EdgeClient:
    def __init__(self, args):
        self.all_power_list = []  # 存放实时功率
        self.start_time = 0
        self.use_time = 0
        logging.getLogger("pika").setLevel(logging.WARNING)
        path_ = os.path.dirname(os.path.abspath(__file__))
        # self.file_path = path_ + '/video/chunk10s.mp4'
        # self.file_path = file_path
        self.edge_device_id = str(args.deviceid)

        sys.path.insert(0, path_ + '/yolov5_master')
        self.model_dic = {'01': path_ + '/yolov5_master/mymodels/new-bdd100k/5n_b32_e20_p.pt',
                          '02': path_ + '/yolov5_master/mymodels/new-bdd100k/5s_b32_e20_p.pt',
                          '03': path_ + '/yolov5_master/mymodels/new-bdd100k/5m_b32_e50_p.pt'}
        self.video_dic = {1: os.path.dirname(os.path.abspath(__file__)) + '/video/03a2c043-647da9c7.mp4',
                          2: os.path.dirname(os.path.abspath(__file__)) + '/video/03a3f4e9-262be322.mp4',
                          3: os.path.dirname(os.path.abspath(__file__)) + '/video/03a30c44-d9c029ff.mp4',
                          4: os.path.dirname(os.path.abspath(__file__)) + '/video/03a041fd-2061eec6.mp4',
                          5: os.path.dirname(os.path.abspath(__file__)) + '/video/03a1130c-86adf6ea.mp4'}

        self.label_dic = {1: os.path.dirname(os.path.abspath(__file__)) + '/labels/03a2c043-647da9c7',
                          2: os.path.dirname(os.path.abspath(__file__)) + '/labels/03a3f4e9-262be322',
                          3: os.path.dirname(os.path.abspath(__file__)) + '/labels/03a30c44-d9c029ff',
                          4: os.path.dirname(os.path.abspath(__file__)) + '/labels/03a041fd-2061eec6',
                          5: os.path.dirname(os.path.abspath(__file__)) + '/labels/03a1130c-86adf6ea'}

        self.weights = self.model_dic[args.modelid]
        self.video_ID = 5
        self.video_name = self.video_dic[int(self.video_ID)]  # 视频的path

        self.data_path = path_ + '/yolov5_master/data/try_mot.yaml'
        self.imgsz = (640, 640)
        self.torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.infer = Inference(self.imgsz, self.torch_device, weights=self.weights, data=self.data_path)  # 初始化
        self.img_start_point = 0

        self.cloud_host = args.cloudhost    # 云端设备ip
        self.cloud_port = args.cloudport       # 云端设备端口

        self.chunkDuration = 1  # 每个视频块的时间长度，1为1秒
        self.rate = 0.5  # 卸载比率
        self.batchsize = 30
        self._video_chunks = {}
        self.all_chunks_string = {}  # 所有视频块转换成流后保存在这个字典中

        self.stream = VideoStream(self.video_name, self.chunkDuration)  # 获取视频数据
        self.width = self.stream.width  # width=1280
        self.height = self.stream.height  # height=720
        self.video_duration = float(self.stream.video_duration)  # video_duration=10.0s  视频持续时间
        # self.updata_video()          # 更新视频(更换视频)

        self.if_val = True
        self.change_model = True


        self.out, self.result_count, self.val_out = None, None, None

        self._divide(args)

        time.sleep(20)

        self._video_chunks = {}

        print('start to plt')
        y = self.all_power_list
        x = range(0, len(y))
        self.all_power_list = []
        print("power_list=", y)
        plt.plot(x, y, 'ro-')

        plt.xlabel("time(s)")  # X轴标签
        plt.ylabel("power(w)")  # Y轴标签
        plt.title("edge_encode power")  # 标题
        # plt.show()
        plt.savefig("edge" + str(args.deviceid) + "_resolution" + args.resolution + "_bitrate" + args.bitrate + "_preset" + args.preset + "_encode.png")
        with open("edge_encode.txt", "a") as file:
            file.write("edge encode deviceid:" + args.deviceid + "  resolution:" + args.resolution + "  bitrate:" + args.bitrate + " preset:" + args.preset + "   encode used time:" + str(self.use_time) + "\n")
            file.write(" ".join([str(x) for x in y]) + "\n\n")

    def _get_power(self):
        sudoPassword = '1223'
        print('start to get power')
        fq = 0.1
        get_power = EngNX(sudoPassword, fq)
        while True:
            time.sleep(1)
            self.all_power_list.append(get_power._get())
            get_power._reset()


    def _divide(self, args):
        # 视频分块
        """
        ---- divide edge and offloading chunks ----
        file_path: the path of video
        rate: the rate of offloading chunks
        chunkDuration: duration of each chunk
        """


        # 卸载比率
        self.all_num_chunks = math.ceil(self.video_duration / self.chunkDuration)  # 总块数
        self.local_chunks = math.ceil(self.video_duration / self.chunkDuration * (1 - self.rate))  # 在边缘端的块数
        self.offloading_chunks = int(self.all_num_chunks - self.local_chunks)

        print('all chunks=', self.all_num_chunks, '  offloading_chunks= ', self.offloading_chunks, '  local_chunks=',
              self.local_chunks)

        resolution_list = args.resolution.split('x')
        width = int(resolution_list[0])
        height = int(resolution_list[1])

        # 监控实时功率
        listen_thread = threading.Thread(target=self._get_power, daemon=True)
        listen_thread.start()
        self.start_time = time.time()
        # 重复多次
        for num in range(5):
            i = 0
            while i < self.offloading_chunks:   # 分块要卸载到云端的
                print('i 1 = ', i, '  self.offloading_chunks=', self.offloading_chunks)
                process = (ffmpeg
                           .input(filename=self.video_name, ss=i, t=self.chunkDuration)
                           .output('pipe:', format='h264', preset=args.preset, loglevel='quiet', s=args.resolution, b=args.bitrate)
                           # preset='ultrafast', preset='superfast', preset='veryfast', preset='faster', preset='fast', preset='medium', preset='slow', preset='slower', preset='veryslow', preset='placebo'
                           .run_async(pipe_stdout=True)
                           )

                self._video_chunks[i] = process.stdout.read()
                i += int(self.chunkDuration)

            while self.offloading_chunks <= i < self.video_duration:  # 分块本地的
                print('i 1 = ', i, '  self.edge_inference_chunks=', self.all_num_chunks - self.offloading_chunks)
                process = (ffmpeg
                           .input(self.video_name, ss=i, t=self.chunkDuration)
                           .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet', s=args.resolution, b=args.bitrate, preset=args.preset)
                           .run_async(pipe_stdout=True)
                           )
                i += int(self.chunkDuration)
                while True:
                    edge_frame = process.stdout.read(width * height * 3)
                    if not edge_frame:
                        # print('edge_infer_queue.put(None)')
                        edge_infer_queue.put(None)
                        break
                    else:
                        edge_infer_queue.put(edge_frame)
        _imgs = []
        im0s = []
        seted_imgsz = self.imgsz
        while True:
            in_bytes = edge_infer_queue.get()
            if in_bytes is None:
                break
            else:
                in_frame = (
                    np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                )
                img, seted_imgsz = preprocess(in_frame, self.imgsz)
                _imgs.append(img)
                im0s.append(in_frame)
        self.use_time = time.time() - self.start_time
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloudhost", required=False, default='10.12.11.61', type=str,
                    help="the host ip of the cloud")
    parser.add_argument("--cloudport", required=False, default=1080, type=int,
                        help="the host port")
    parser.add_argument("--deviceid", required=False, default='02', type=str,
                        help="local device id")
    parser.add_argument("--modelid", required=False, default='02', type=str,
                        help="cloud model id")
    parser.add_argument("--mqhost", required=False, default='10.12.11.61', type=str,
                        help='the rabbitmq server ip')
    parser.add_argument("--mqport", required=False, default=5672, type=int,
                        help='the rabbitmq server port')
    parser.add_argument("--mquser", required=False, default='hhf', type=str,
                        help='the rabbitmq server username')
    parser.add_argument("--mqpw", required=False, default='1223', type=str,
                        help='the rabbitmq server password')

    parser.add_argument("--resolution", required=False, default='1280x720', type=str,
                        help='the encode resolution')
    parser.add_argument("--bitrate", required=False, default='1000', type=str,
                        help='the encode bitrate')
    parser.add_argument("--preset", required=False, default='ultrafast', type=str,
                        help='the encode preset')
    args = parser.parse_args()
    print(args)

    with torch.no_grad():
        a = EdgeClient(args)
        # a = EdgeClient(file_path=file_path, edge_device_id='01', video_name=video_name[:-4])
