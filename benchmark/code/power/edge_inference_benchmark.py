import argparse
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
        self.rate = 0  # 卸载比率
        self.batchsize = 30
        self._video_chunks = {}
        self.all_chunks_string = {}  # 所有视频块转换成流后保存在这个字典中

        self.stream = VideoStream(self.video_name, self.chunkDuration)  # 获取视频数据
        self.width = self.stream.width  # width=1920
        self.height = self.stream.height  # height=1080
        self.video_duration = float(self.stream.video_duration)  # video_duration=10.0s  视频持续时间
        # self.updata_video()          # 更新视频(更换视频)

        self.if_val = True
        self.change_model = True

        self.out, self.result_count, self.val_out = None, None, None

        self._divideAndInference()

        time.sleep(20)
        print('start to plt')
        y = self.all_power_list
        x = range(0, len(y))
        self.all_power_list = []
        print("power_list=", y)
        plt.plot(x, y, 'ro-')

        plt.xlabel("time(s)")  # X轴标签
        plt.ylabel("power(w)")  # Y轴标签
        plt.title("edge_inference power")  # 标题
        # plt.show()
        plt.savefig("edge" + str(args.deviceid) + "_modelid" + args.modelid + "_inference.png")

        with open("edge_inference.txt", "a") as file:
            file.write("edge inference deviceid:" + args.deviceid + "  modelid:" + args.modelid + "   inference used time:" + str(self.use_time) + "\n")
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


    def _divideAndInference(self):
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


        i = 0
        while i < self.video_duration:  # 分块本地的
            print('i 1 = ', i)
            process = (ffmpeg
                        .input(self.video_name, ss=i, t=self.chunkDuration)
                        .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
                        .run_async(pipe_stdout=True)
                        )
            i += int(self.chunkDuration)
            while True:
                edge_frame = process.stdout.read(self.width * self.height * 3)
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
                    np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                )
                img, seted_imgsz = preprocess(in_frame, self.imgsz)
                _imgs.append(img)
                im0s.append(in_frame)
        time.sleep(2)
        # 监控实时功率
        listen_thread = threading.Thread(target=self._get_power, daemon=True)
        listen_thread.start()
        self.start_time = time.time()
        for i in range(20):
            self.result_count = self.infer.inference(_imgs, im0s, self.batchsize, seted_imgsz, val=self.if_val)
        self.use_time = time.time() - self.start_time



def img2video(dir_path='/home/hhf/yolov5/datasets/try_mot/images/val/011ad6b2-1dfff443',
              save_path='/home/hhf/tryout'):
    """"
    args example:
    dir_path = '/home/hhf/yolov5/datasets/try_mot/images/val/011ad6b2-1dfff443'
    save_path = '/home/hhf/tryout'
    """
    fps, w, h = 30, 1280, 720
    print('dir_path_end=', dir_path[-17:])
    save_path = save_path + '/' + dir_path[-17:] + '.mp4'
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print('len(os.listdir(dir_path))=', len(os.listdir(dir_path)))

    for i in range(len(os.listdir(dir_path))):
        a = '%03d' % (i + 1)
        img_name = dir_path + '/' + dir_path[-17:] + '-0000' + a + '.jpg'
        im0 = cv2.imread(img_name, cv2.IMREAD_COLOR)
        vid_writer.write(im0)


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
    args = parser.parse_args()
    print(args)
    with torch.no_grad():
        a = EdgeClient(args)
        # a = EdgeClient(file_path=file_path, edge_device_id='01', video_name=video_name[:-4])
