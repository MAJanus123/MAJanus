# -*- coding: utf-8 -*-
"""
Created on Fri Jul 1 09:52 2022

@author: Huifeng-Hu
"""
import argparse
from multiprocessing import Manager
import os
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
import pickle


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

        self.num = 0

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

        # # 边缘设备控制
        # control_thread = threading.Thread(target=self.edge_wake)
        # control_thread.start()
        #
        # # 接收edge发送的视频数据
        # listen_thread = threading.Thread(target=self._listen)
        # listen_thread.start()
        #
        # # 侦听到数据 处理listen得到的数据 解包
        # process_thread = threading.Thread(target=self._process)
        # process_thread.start()


        # # 将queue存入pickle
        # time.sleep(90)
        # print("start write decode======")
        # decode_content_dict = {}
        # with open('decode.pickle', 'wb') as f:
        #     while not inference_queue.empty():
        #         decode_content = inference_queue.get()
        #         decode_content_dict.update({self.num: decode_content})
        #         self.num += 1
        #         inference_queue.task_done()
        #     pickle.dump(decode_content_dict, f)
        # decode_content_dict = {}
        # print("num=====", self.num)


        # # decode
        # process_thread = threading.Thread(target=self._decode)
        # process_thread.start()

        self._decode()

        time.sleep(20)
        # 画图
        print('start to plt')
        y = self.all_power_list
        x = range(0, len(y))
        self.all_power_list = []
        print("power_list=", y)
        plt.plot(x, y, 'ro-')

        plt.xlabel("time(s)")  # X轴标签
        plt.ylabel("power(w)")  # Y轴标签
        plt.title("cloud_decode power")  # 标题
        plt.show()
        plt.savefig("cloud_decode.png")


        with open("cloud_decode.txt", "a") as file:
            file.write("cloud decode " + "   decode used time:" + str(self.use_time) + "\n")
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

    def _listen(self):  # 接收edge发送的 视频数据   对应_sender()
        print('start to listen')
        img_data = b""
        while True:   # 不停监听是否有device发送数据
            clint_socket, client_addr = self.dataSocket.accept()
            while True:   # 不停接收该机器发送的数据
                recv_data = clint_socket.recv(2560)  # 接收数据
                if recv_data:
                    img_data += recv_data
                else:
                    cloud_trans_queue.put(img_data)
                    img_data = b""
                    break
            clint_socket.close()

    def _process(self):  # 处理listen得到的数据
        """n
        Listen to RTP port, receive connection from the client
        """
        print('start to process')

        while True:
            data = cloud_trans_queue.get()
            self.start_flag = True
            if data:
                # print('if data')
                # 解包
                imagePacket = RtpPacket()
                imagePacket.decode(data)
                # print('packet_memmory_size=', imagePacket.__sizeof__())
                currentSequenceNum = imagePacket.seqNum()
                # print('SequenceNum=', currentSequenceNum, type(currentSequenceNum))
                currentMarker = imagePacket.Marker()  # edgeID和视频名称(代号)
                # print('Marker=', currentMarker, type(currentMarker))
                payload = imagePacket.getPayload()  # 视频数据
                timestamp_start = imagePacket.getTimestamp()

                edgeID = int(currentMarker / 10)   # 机器ID
                videoID = currentMarker - int(currentMarker / 10) * 10   # 视频ID
                chunkID = currentSequenceNum   # 块

                if edgeID not in self.cloud_data_dic:
                    self.cloud_data_dic[edgeID] = {}
                if videoID not in self.cloud_data_dic[edgeID]:
                    self.cloud_data_dic[edgeID][videoID] = {}
                    self.cloud_data_dic[edgeID][videoID]['results'] = {}
                    self.cloud_data_dic[edgeID][videoID]['counts'] = 0  # 推理检测出的数量
                    self.cloud_data_dic[edgeID][videoID]['all_num'] = 7

                inference_queue.put((timestamp_start, payload, edgeID, videoID, chunkID))


            else:
                break
            cloud_trans_queue.task_done()  # 表示由生产者产生的任务结束
        cloud_trans_queue.join()   # 如果还有任务，会一直挂起阻塞

    def _decode(self):
        # 从pickle读出数据
        print("start read decode======")
        try:
            with open('decode.pickle', 'rb+') as f:
                decode_data = pickle.load(f)
        except EOFError:  # 捕获异常EOFError 后返回None
            return None

        i = 0
        _imgs = []
        im0s = []
        # 监控实时功率
        listen_thread = threading.Thread(target=self._get_power, daemon=True)
        listen_thread.start()
        self.start_time = time.time()
        for key in decode_data:
            i += 1
            data = decode_data[key]
            inference_tuple = data
            # 转码
            _process = (ffmpeg
                        .input('pipe:', format='h264')
                        .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
                        .run_async(pipe_stdin=True, pipe_stdout=True)
                        )

            _process.stdin.write(inference_tuple[1])
            _process.stdin.close()
            _size = len(inference_tuple[1])

            while True:  # 每张图片一循环
                in_bytes = _process.stdout.read(1280 * 720 * 3)
                if not in_bytes:
                    break
                else:
                    in_frame = (np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]))  # 图片格式转化
                    img, seted_imgsz = preprocess(in_frame, self.imgsz)  # 预处理
                    _imgs.append(img)
                    im0s.append(in_frame)
        self.use_time = time.time() - self.start_time
        print("处理的视频块数===", i)

        # # 将decode完的数据固化
        # with open('_imgs.pickle', 'wb') as f:
        #     pickle.dump(_imgs, f)
        # with open('im0s.pickle', 'wb') as f:
        #     pickle.dump(im0s, f)
        _imgs = []
        im0s = []




    def send_instruction(self, message="01_05_01"):  # 发送函数

        user_info = pika.PlainCredentials('hhf', '1223')  # 用户名和密码
        connection = pika.BlockingConnection(pika.ConnectionParameters('10.12.11.61', 5672, '/', user_info))

        instruction_channel = connection.channel()
        instruction_channel.exchange_declare(exchange='inst', exchange_type='fanout')

        # 1:start ;    0:stop
        #  message="01_05_01"     # device=1, rate=0.5, val=True
        # instruction_channel.queue_declare(queue='instruction')
        instruction_channel.basic_publish(exchange='inst',  # 当前是一个简单模式，所以这里设置为空字符串就可以了
                                          routing_key='',  # 指定消息要发送到哪个queue
                                          body=message  # 指定要发送的消息
                                          )
        print('send data ', message)
        connection.close()

    def edge_wake(self):

        control_way = input('\n是否手动控制输入?  --Yes(y)/No(n)\n')
        if control_way == 'YES' or control_way == 'yes' or control_way == 'Yes' or control_way == 'y' or control_way == 'Y':
            while True:
                _show = input('\n是否现在就下达控制指令?  --Yes(y):现在下达  --No(n):稍后下达\n')
                if _show == 'YES' or _show == 'yes' or _show == 'Yes' or _show == 'y' or _show == 'Y':
                    message = input(
                        '\n请输入控制指令：  --例如"01_05_01"用于设备1以50%卸载比率进行卸载且需要精度评估， 或"01_01"用于设备1更换模型为1. 退出本程序请输入"exit".\n')
                    if message == 'exit':
                        break
                    else:
                        if len(message) == 5 or len(message) == 8:
                            self.send_instruction(message)
                            time.sleep(1)
                        else:
                            print('输入有误，请重新输入!')
                elif _show == 'NO' or _show == 'no' or _show == 'n' or _show == 'N' or _show == 'No':
                    print('系统将等待30秒...')
                    time.sleep(30)
                    self.edge_wake()
                else:
                    print('输入有误，请重新输入!')
        elif control_way == 'NO' or control_way == 'no' or control_way == 'n' or control_way == 'N' or control_way == 'No':
            print('进入自动控制模式...')
            while True:
                _show = input('\n输入0-10开始控制\n')
                if _show == '0':
                    message = '02_00_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '1':
                    message = '02_01_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '2':
                    message = '02_02_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '3':
                    message = '02_03_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '4':
                    message = '02_04_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '5':
                    message = '02_05_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '6':
                    message = '02_06_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '7':
                    message = '02_07_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '8':
                    message = '02_08_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '9':
                    message = '02_09_01'
                    self.send_instruction(message)
                    time.sleep(1)
                elif _show == '10':
                    message = '02_10_01'
                    self.send_instruction(message)
                    time.sleep(1)
        else:
            print('输入有误，请重新输入!')
            self.edge_wake()


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
