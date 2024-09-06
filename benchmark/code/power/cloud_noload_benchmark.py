# -*- coding: utf-8 -*-
"""
Created on Fri Jul 1 09:52 2022

@author: Huifeng-Hu
"""
import argparse
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



data_queue = queue.Queue()
cloud_trans_queue = queue.Queue()
inference_queue = queue.Queue()


class CloudServer:
    def __init__(self, args):
        self.all_power_list = []  # 存放空载实时功率

        # 监控实时功率
        listen_thread = threading.Thread(target=self._get_power, daemon=True)
        listen_thread.start()

        logging.getLogger("pika").setLevel(logging.WARNING)  # 屏蔽mq日志
        path_ = os.path.dirname(os.path.abspath(__file__))   # 获取当前py所在的目录

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

        self.HOST = args.cloudhost    # 云端设备ip
        self.DATA_PORT = args.cloudport       # 云端设备端口



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



        # 接收edge发送的视频数据
        listen_thread = threading.Thread(target=self._listen, daemon=True)
        listen_thread.start()

        # 侦听到数据 处理listen得到的数据 解包
        process_thread = threading.Thread(target=self._process, daemon=True)
        process_thread.start()

        # 推理
        inference_thread = threading.Thread(target=self._inference, daemon=True)
        inference_thread.start()

        # 定期把字典里保存的结果存储到文件里
        counter_thread = threading.Thread(target=self._counter, daemon=True)
        counter_thread.start()

        # MQ

        # 边缘设备控制
        control_thread = threading.Thread(target=self.edge_wake, daemon=True)
        control_thread.start()
        # 接收边缘设备的监控
        getMQ_thread = threading.Thread(target=self.getMQ, daemon=True)
        getMQ_thread.start()
        # 能耗
        get_eng_thread = threading.Thread(target=self.get_eng, daemon=True)
        get_eng_thread.start()


        time.sleep(40)
        print('start to plt')
        y = self.all_power_list
        x = range(0, len(y))
        self.all_power_list = []
        print("power_list=", y)
        plt.plot(x, y, 'ro-')

        plt.xlabel("time(s)")  # X轴标签
        plt.ylabel("power(w)")  # Y轴标签
        plt.title("cloud_noload power")  # 标题
        plt.show()
        plt.savefig("cloud_noload.png")

        with open("cloud_noload.txt", "a") as file:
            file.write("cloud noload:\n")
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



    def _counter(self):
        print('start to count')
        while True:
            if self.start_flag:
                print('start to count for 60s')
                time.sleep(12000)
                # self.stop_flag = True
                ## _3_10, 30% to the cloud, 10 fps
                for device in range(1, 3):
                    with open('./logs/device' + str(device) + 'latency.data', 'w+') as f:
                        for latency in self.latency_dic[str(device)]:
                            f.write(str(latency) + '\n')
                    with open('./logs/device' + str(device) + 'count.data', 'w+') as cf:
                        for count in self.result_dic[str(device)]:
                            cf.write((str(count) + '\n'))
                    with open('./logs/device' + str(device) + 'mAP.data', 'w+') as mf:
                        for i in range(len(self.cloud_data_dic[device])):
                             mf.write((str(self.cloud_data_dic[device][i+1]['acc']) + '\n'))
                    if self.latency_dic[str(device)]:
                        print('device:', device, ', average latency=', np.average(self.latency_dic[str(device)]))
                        print('code test---------------------------------------------')
                        # print('device:', device, ', average latency=', np.average(self.latency_dic[str(device)]) / 30)
                break

    def _listen(self):  # 接收edge发送的 视频数据   对应_sender()
        print('start to listen')
        img_data = b""
        get_data = b""
        while True:   # 不停监听是否有device发送数据
            clint_socket, client_addr = self.dataSocket.accept()
            while True:   # 不停接收该机器发送的数据
                recv_data = clint_socket.recv(2560)  # 接收数据
                # if recv_data:
                #     get_data += recv_data
                # else:
                #     data_queue.put(get_data)
                #     get_data = b""
                #     break
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
            cloud_trans_queue.task_done()
        cloud_trans_queue.join()

    def _inference(self):  # 推理并处理process好的数据
        '''
        init new thread and deal with client information
        '''
        print('start to inference')
        while True:
            inference_tuple = inference_queue.get()
            # 转码
            _process = (ffmpeg
                        .input('pipe:', format='h264')
                        .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
                        .run_async(pipe_stdin=True, pipe_stdout=True)
                        )

            _process.stdin.write(inference_tuple[1])
            _process.stdin.close()
            _size = len(inference_tuple[1])
            # print('_size=============================', _size/1024)
            count = 0

            _imgs = []
            im0s = []
            seted_imgsz = self.imgsz
            batchsize = 30  # 一秒钟三十张照片
            while True:  # 每张图片一循环
                in_bytes = _process.stdout.read(1280 * 720 * 3)
                if not in_bytes:
                    _time_infer_start = time.time()
                    break
                else:
                    in_frame = (np.frombuffer(in_bytes, np.uint8).reshape([720, 1280, 3]))  # 图片格式转化
                    img, seted_imgsz = preprocess(in_frame, self.imgsz)  # 预处理
                    _imgs.append(img)
                    im0s.append(in_frame)
                    count += 1

            # savepath = '/home/hw2/cloud/runout/000'+str(inference_tuple[4])+'.mp4'
            out, result_count = self.infer.inference(_imgs, im0s, batchsize, seted_imgsz, val=True)  # 推理
            _time_infer_end = time.time()
            edge_device = inference_tuple[2]
            """
            inference_tuple[0] = timestamp   --float
            inference_tuple[1] = video data  --byte
            inference_tuple[2] = dege ID     --int
            inference_tuple[3] = video ID    --int
            inference_tuple[4] = chunk ID    --int
            """
            label_path = self.label_dic[str(inference_tuple[3])]


            # 精度评估
            result, eval_t = _eval(out[0], label_path, batchsize, chunk_id=inference_tuple[4], device=self.torch_device)
            self.cloud_data_dic[inference_tuple[2]][inference_tuple[3]]['results'][inference_tuple[4]] = result  # 将map50存进dic
            self.cloud_data_dic[inference_tuple[2]][inference_tuple[3]]['counts'] += result_count                # 将目标计数加进dic
            # self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['all_num']) =                # 需要知道所有的chunk数量，不然没法做平均

            print(len(self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['results']), '==???==', self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['all_num'])
            # print(' self.cloud_data_dic=',  self.cloud_data_dic)
            # 如果当前视频块为最后一个  全部卸载到云端推理的情况
            if len(self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['results']) == self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['all_num']:
                # print('video is ok!!!!!!!!!!!!!1')
                print('acc=', self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['results'])
                all_acc = 0
                for i in self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['results']:
                    all_acc += self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['results'][i]
                avg_acc = all_acc/self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['all_num']
                print('avg_acc=', avg_acc)
                # print(' self.cloud_data_dic=',  self.cloud_data_dic)
                # 清零
                self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['results'] = {}
                self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['counts'] = 0
                self.cloud_data_dic[int(inference_tuple[2])][int(inference_tuple[3])]['all_num'] = None
                print("此设备处理完毕，内存清空")

            # print('len(results)', len(self.cloud_data_dic[inference_tuple[2]][inference_tuple[3]]['results']))
            latency = round(time.time() - inference_tuple[0], 2)  # 每一块 从打包发送到推理结束的延迟
            if str(inference_tuple[2]) not in self.latency_dic:
                self.latency_dic[str(inference_tuple[2])] = []
                self.result_dic[str(inference_tuple[2])] = []
            self.latency_dic[str(inference_tuple[2])].append(latency)
            self.result_dic[str(inference_tuple[2])].append(result_count)
            print(
                'cloud inference, data from edge device: %s, %s objective, total %s frames, takes %s s,   inference '
                'time= %s s, chunk size= %s KB' % (
                    str(edge_device), str(result_count), count, str(latency), round(_time_infer_end - _time_infer_start, 3),
                    float(_size / 1024)))
            inference_queue.task_done()


    def get_eng(self):
        # 边缘端一开始就会传一次   接收边缘设备id、videoID、块数、能耗等等
        user_info = pika.PlainCredentials('hhf', '1223')
        connection = pika.BlockingConnection(pika.ConnectionParameters('10.12.11.61', 5672, '/', user_info))
        channel = connection.channel()
        channel.queue_declare(queue='energy')
        # body 存数据
        def callback(ch, method, properties, body):
            print('body=',body,type(body))
            body = body.decode(encoding="utf-8")
            edge_data = re.findall(r'\d+', body, re.S)
            print('edge[0123]', int(edge_data[0]), int(edge_data[1]), int(edge_data[2]), int(edge_data[3]))

            if int(edge_data[0]) not in self.cloud_data_dic:
                self.cloud_data_dic[int(edge_data[0])] = {}
            if int(edge_data[1]) not in self.cloud_data_dic[int(edge_data[0])]:
                self.cloud_data_dic[int(edge_data[0])][int(edge_data[1])] = {}
                self.cloud_data_dic[int(edge_data[0])][int(edge_data[1])]['results'] = {}
                self.cloud_data_dic[int(edge_data[0])][int(edge_data[1])]['counts'] = 0
                self.cloud_data_dic[int(edge_data[0])][int(edge_data[1])]['all_num'] = int(edge_data[2])



            self.cloud_data_dic[int(edge_data[0])][int(edge_data[1])]['all_num'] = int(edge_data[2])
            # print('云端收到能耗报告：', body, 'eng=', int(edge_data[3])+float(edge_data[4])/100)

        channel.basic_consume(queue='energy',  # 接收指定queue的消息
                              auto_ack=True,  # 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息
                              on_message_callback=callback  # 设置收到消息的回调函数
                              )

        channel.start_consuming()

    def getMQ(self):  # 接收边缘端处理好的数据并整合
        # MQ的声明
        user_info = pika.PlainCredentials('hhf', '1223')
        connection = pika.BlockingConnection(pika.ConnectionParameters('10.12.11.61', 5672, '/', user_info))
        channel = connection.channel()
        channel.queue_declare(queue='edge2cloud')  # 通道

        def callback(ch, method, properties, body):
            print('cloud_MQ 收到', type(body), ' size=', len(body) / 1024, 'kb')
            msg_bytes = base64.b64decode(body)
            ascii_msg = msg_bytes.decode('ascii')
            ascii_msg = ascii_msg.replace("'", "\"")
            MQ_dic = json.loads(ascii_msg)

            print('MQ_dic["id"]=', MQ_dic['id'])

            if int(MQ_dic['id']) not in self.cloud_data_dic:
                print('self.cloud_data_dic[int(MQ_dic["id"])] = {}')
                self.cloud_data_dic[int(MQ_dic['id'])] = {}
            if int(MQ_dic['video_name']) not in self.cloud_data_dic[int(MQ_dic['id'])]:
                print(222222222)
                self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])] = {}
                self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['results'] = {}
                self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['counts'] = 0
                self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['all_num'] = None

            print('self.cloud_data_dic=', self.cloud_data_dic)
            # 将传来的二进制数据处理好后存到字典里
            self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['counts'] += MQ_dic['result_count']
            self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['all_num'] = MQ_dic['all_num']
            self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['results'][MQ_dic['num']] = MQ_dic['map50']
            if len(self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['results']) == self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['all_num']:
                print('video is ok!!!!!!!!!!!!!1')
                print('acc=', self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['results'])
                all_acc = 0
                for i in self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['results']:
                    all_acc += self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['results'][i]
                avg_acc = all_acc / self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['all_num']
                print('avg_acc=', avg_acc)
                self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['results'] = {}
                self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['counts'] = 0
                self.cloud_data_dic[int(MQ_dic['id'])][int(MQ_dic['video_name'])]['all_num'] = None
                print(111111111)


        channel.basic_consume(queue='edge2cloud',  # 接收指定queue的消息
                              auto_ack=True,  # 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息
                              on_message_callback=callback  # 设置收到消息的回调函数
                              )

        channel.start_consuming()

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
