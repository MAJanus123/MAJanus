# -*- coding: utf-8 -*-
"""
Created on Fri Jul 1 09:52 2022

@author: Huifeng-Hu
"""
import argparse
import os
import pickle
from multiprocessing import Manager
import sys
import queue
import socket
import threading
import time
import torch
import ffmpeg
from RtpPacket import RtpPacket
from yolov5_master.myinference import Inference
from yolov5_master.mydataset import preprocess
from eval import _eval
# from yolov5_master.evaluation import _evaluation
import base64
import json
import numpy as np
import pika
import logging
import re
from power import EngUbuntuGPU
import matplotlib.pyplot as plt



data_queue = Manager().Queue()
cloud_trans_queue = Manager().Queue()
inference_queue = Manager().Queue()


class CloudServer:
    def __init__(self, args):
        self.all_power_list = []  # 存放空载实时功率
        self.start_time = 0
        self.use_time = 0

        logging.getLogger("pika").setLevel(logging.WARNING)  # 屏蔽mq日志
        path_ = os.path.dirname(os.path.abspath(__file__))  # 获取当前py所在的目录

        self.model_dic = {'01': path_ + '/yolov5_master/mymodels/new-bdd100k/5n_b32_e20_p.pt',
                          '02': path_ + '/yolov5_master/mymodels/new-bdd100k/5s_b32_e20_p.pt',
                          '03': path_ + '/yolov5_master/mymodels/new-bdd100k/5m_b32_e50_p.pt'}
        self.label_dic = {'1': os.path.dirname(os.path.abspath(__file__)) + '/labels/03a2c043-647da9c7',
                          '2': os.path.dirname(os.path.abspath(__file__)) + '/labels/03a3f4e9-262be322',
                          '3': os.path.dirname(os.path.abspath(__file__)) + '/labels/03a30c44-d9c029ff',
                          '4': os.path.dirname(os.path.abspath(__file__)) + '/labels/03a041fd-2061eec6',
                          '5': os.path.dirname(os.path.abspath(__file__)) + '/labels/03a1130c-86adf6ea'}

        weights = self.model_dic[args.modelid]
        data_path = path_ + '/yolov5_master/data/try_mot.yaml'
        sys.path.insert(0, path_ + '/yolov5_master')
        self.imgsz = (640, 640)  # 图片尺寸

        self.torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 选择推理设备
        self.infer = Inference(self.imgsz, self.torch_device, weights=weights, data=data_path)  # 初始化
        time.sleep(2)

        self.HOST = args.cloudhost  # 云端设备ip
        self.DATA_PORT = args.cloudport  # 云端设备端口


        # 存储结果数据
        self.cloud_data_dic = {1: {1: {'results': {0: (), 1: ()},   # 设备ID videoID 块ID
                                           'counts': 0,
                                           'all_num': 0,
                                           'acc': None
                                           }
                                     }
                               }
        # 延迟字典（卸载到云端推理的延迟）
        self.latency_dic = {'1': [], '2': [], '3': [], '4': [], '5': []}   # 设备ID 每块延迟
        # 推理结果数量
        self.result_dic = {'1': [], '2': [], '3': [], '4': [], '5': []}

        self.start_flag = False

        # init socket 传输协议
        self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dataSocket.bind((self.HOST, self.DATA_PORT))
        self.dataSocket.listen(5)   # 5表示表示的是服务器拒绝(超过限制数量的)连接之前，操作系统可以挂起的最大连接数量，可以看作是"排队的数量"

        # # 推理
        # process_thread = threading.Thread(target=self._inference)
        # process_thread.start()
        self._inference()
        time.sleep(30)
        print('start to plt')
        y = self.all_power_list
        x = range(0, len(y))
        self.all_power_list = []
        print("power_list=", y)
        plt.plot(x, y, 'ro-')

        plt.xlabel("time(s)")  # X轴标签
        plt.ylabel("power(w)")  # Y轴标签
        plt.title("cloud_inference power")  # 标题
        # plt.show()
        plt.savefig("cloud_inference" + "_modelID" + args.modelid + ".png")

        with open("cloud_inference.txt", "a") as file:
            file.write("cloud inference modelID:" + args.modelid + "   inference used time:" + str(self.use_time) + "\n")
            file.write(" ".join([str(x) for x in y]) + "\n\n")

    def _get_power(self):
        sudoPassword = '1223'
        print('start to get power')
        fq = 0.1
        get_power = EngUbuntuGPU(sudoPassword, fq)
        while True:
            time.sleep(1)
            self.all_power_list.append(get_power._get())
            get_power._reset()


    def _inference(self):  # 推理
        '''
        init new thread and deal with client information
        '''
        try:
            with open('_imgs.pickle', 'rb+') as f:
                _imgs = pickle.load(f)
        except EOFError:  # 捕获异常EOFError 后返回None
            return None
        try:
            with open('im0s.pickle', 'rb+') as f:
                im0s = pickle.load(f)
        except EOFError:  # 捕获异常EOFError 后返回None
            return None

        time.sleep(2)

        print('start to inference')
        seted_imgsz = self.imgsz
        batchsize = 30  # 一秒钟三十张照片
        # savepath = '/home/hw2/cloud/runout/000'+str(inference_tuple[4])+'.mp4'
        # 监控实时功率
        listen_thread = threading.Thread(target=self._get_power, daemon=True)
        listen_thread.start()
        self.start_time = time.time()
        for i in range(5):
            out, result_count = self.infer.inference(_imgs, im0s, batchsize, seted_imgsz, val=True)  # 推理
        self.use_time = time.time() - self.start_time
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloudhost", required=False, default='10.12.11.61', type=str,
                    help="the host ip of the cloud")
    parser.add_argument("--cloudport", required=False, default=1080, type=int,
                        help="the host port")
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
        CloudServer(args)
