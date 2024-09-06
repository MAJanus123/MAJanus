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
import pika
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
        self.all_power_list = []  # 存放空载实时功率
        # 监控实时功率
        listen_thread = threading.Thread(target=self._get_power, daemon=True)
        listen_thread.start()

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
        self.rate = 0.1  # 卸载比率
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

        # self.video2string()     # 调用函数，将视频转换成流，ffmpeg

        # print('self.all_chunks_string=', self.all_chunks_string)

        self.out, self.result_count, self.val_out = None, None, None

        # print('power=', self.power._get())

        self.control_thread = threading.Thread(target=self._control_client, daemon=True)
        self.control_thread.start()

        print('start to plt')
        time.sleep(40)
        y = self.all_power_list
        x = range(0, len(y))
        self.all_power_list = []
        print("power_list=", y)
        plt.plot(x, y, 'ro-')

        plt.xlabel("time(s)")  # X轴标签
        plt.ylabel("power(w)")  # Y轴标签
        plt.title("edge_noload power")  # 标题
        # plt.show()
        plt.savefig("edge" + str(args.deviceid) + "_noload.png")

        with open("edge_noload.txt", "a") as file:
            file.write("edge noload deviceid:" + args.deviceid + "\n")
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


    def _divide(self):
        # 视频分块
        """
        ---- divide edge and offloading chunks ----
        file_path: the path of video
        rate: the rate of offloading chunks
        chunkDuration: duration of each chunk
        """
        # self.offloading_chunks = int(self.video_duration / self.chunkDuration * self.rate)
        # self.local_chunks = math.ceil(self.video_duration / self.chunkDuration * (1 - self.rate))
        # self.all_num_chunks = math.ceil(self.video_duration / self.chunkDuration)

        # 卸载比率
        self.all_num_chunks = math.ceil(self.video_duration / self.chunkDuration)  # 总块数
        self.local_chunks = math.ceil(self.video_duration / self.chunkDuration * (1 - self.rate))  # 在边缘端的块数
        self.offloading_chunks = int(self.all_num_chunks - self.local_chunks)

        print('all chunks=', self.all_num_chunks, '  offloading_chunks= ', self.offloading_chunks, '  local_chunks=',
              self.local_chunks)
        i = 0
        while i < self.offloading_chunks:   # 分块要卸载到云端的
            print('i 1 = ', i, '  self.offloading_chunks=', self.offloading_chunks)
            ttt1 = time.time()
            process = (ffmpeg
                       .input(filename=self.video_name, ss=i, t=self.chunkDuration)
                       .output('pipe:', format='h264', preset='ultrafast', loglevel='quiet')
                       # preset='ultrafast', preset='superfast', preset='veryfast', preset='faster', preset='fast', preset='medium', preset='slow', preset='slower', preset='veryslow', preset='placebo'
                       .run_async(pipe_stdout=True)
                       )

            self._video_chunks[i] = process.stdout.read()
            ttt2 = time.time()
            print('time=============================', ttt2 - ttt1)
            i += int(self.chunkDuration)

        while self.offloading_chunks <= i < self.video_duration:  # 分块本地的
            print('i 2 = ', i)
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

    def _producer(self):
        # 解码的块 打包
        """
        simulate the input framerate, run the experiment for 60s.
        """
        print('producer sender thread start to work')
        j = 0
        while j < self.offloading_chunks:
            chunk_bytes = self._video_chunks[j]
            packet = RtpPacket()
            _send_start_time = time.time()
            print('j===', j, '  self.offloading_chunks=', self.offloading_chunks, 'self.all_num_chunks', self.all_num_chunks)
            # print('mark======', int(self.edge_device_id)*10+int(self.video_ID))
            packet.encode(j, int(self.edge_device_id) * 10 + int(self.video_ID), chunk_bytes, _send_start_time)
            _offloading_queue.put(packet)
            # _offloading_queue.join()

            j += int(self.chunkDuration)
            # if j == self.all_num_chunks:
            #     print('device'+str(self.edge_device_id)+'eng'+str('%.2f'%(self.eng._get())))
            #     self._sendMQ('eng'+str(self.edge_device_id)+str('%.2f'%(self.eng._get())))
        _offloading_queue.put(None)

        # print('offloading_chunks=', self.offloading_chunks)
        # for j in range(self.offloading_chunks):
        #     print('j=', j)
        #     chunk_bytes = self.all_chunks_string[j]
        #     packet = RtpPacket()
        #     _send_start_time = time.time()
        #     packet.encode(0, 0, chunk_bytes, _send_start_time)
        #
        #     _offloading_queue.put(packet)
        #     # print('_offloading_queue.put(packet)')
        #     _offloading_queue.join()
        #
        #     j += int(self.chunkDuration)
        # _offloading_queue.put(None)


    def _sender(self):
        # 发送包
        print('video sender thread start to work')
        while True:
            packet = _offloading_queue.get()
            if packet is None:
                break
            client = RtpClient(host=self.cloud_host, port=self.cloud_port, socket_type=1)
            timestamp = packet.getTimestamp()
            image_bytes = packet.getPayload()
            deviceID_videoName = packet.Marker()
            chunkID = packet.seqNum()
            print('deviceID_videoName=', deviceID_videoName, '   chunkID=', chunkID)
            client.send_image(timestamp, image_bytes, deviceID_videoName, chunkID)
            _offloading_queue.task_done()  # 结束get函数产生的一个任务
            if chunkID+1 == self.all_num_chunks:
                self.change_model = True
                print('send1!!!', 'device' + str(self.edge_device_id) + 'videoID' + str(self.video_ID) + 'all' + str(self.all_num_chunks) + 'eng' + str('%.2f' % (self.power._get())))
                # 为最后一个包时（有可能全部卸载），向云端发送信息 通过get_eng()接收
                self._sendMQ('device' + str(self.edge_device_id) + 'videoID' + str(self.video_ID) + 'all' + str(self.all_num_chunks) + 'eng' + str('%.2f' % (self.power._get())), queue='energy')


    def _edge_inference(self):

        k = self.offloading_chunks
        while self.offloading_chunks <= k < math.ceil(self.video_duration):
            print('k=', k)
            _imgs = []
            im0s = []
            seted_imgsz = self.imgsz
            frame_num = 0
            s_t = time.time()

            while True:
                in_bytes = edge_infer_queue.get()
                # edge_infer_queue.task_done()
                if in_bytes is None:
                    # print("edge_infer_queue.get(None)")
                    break
                else:
                    in_frame = (
                        np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                    )
                    img, seted_imgsz = preprocess(in_frame, self.imgsz)
                    _imgs.append(img)
                    im0s.append(in_frame)
                    frame_num += 1

            if not self.if_val:
                self.result_count = self.infer.inference(_imgs, im0s, self.batchsize, seted_imgsz, val=self.if_val)
                print('edge inference, %s objective, total %s frames' % (str(self.result_count), frame_num))
            else:
                self.out, self.result_count = self.infer.inference(_imgs, im0s, self.batchsize, seted_imgsz, val=True)
                e_t = time.time()
                # for i in range(len(self.out)):
                #     self.out[i] = self.out[i].cpu().numpy().tolist()
                # self.out = np.array(self.out).tolist()

                print('len(self.out)=', len(self.out))
                label_path = self.label_dic[int(self.video_ID)]
                map50, eval_time = _eval(self.out[0], label_path, self.batchsize, chunk_id=k, device=self.torch_device)
                print('map50=', map50, 'eval_time=', eval_time)

                print('edge inference, %s objective, total %s frames' % (str(self.result_count), frame_num))
                # energy = self.eng._get()
                self.MQ_dic = {
                    'id': self.edge_device_id,  # 边缘设备ID       str
                    'video_name': self.video_ID,  # 视频名称(代号)   str
                    'num': k,  # 第几块视频       int
                    'all_num': self.all_num_chunks,  # 总共视频块数     int
                    'result_count': self.result_count,  # 边缘推理结果(count)  int
                    'map50': map50,  # 模型输出   list:1
                    'infer_t': e_t-s_t,  # inference time  每一块从开始推理到发送前的时间
                    'eval_t': eval_time  # evaluation time
                    # 'eng': energy          # energy consumption
                }
                message = str(self.MQ_dic).encode('ascii')
                message_byte = base64.b64encode(message)

                # msg_bytes = base64.b64decode(message_byte)
                # ascii_msg = msg_bytes.decode('ascii')
                # ascii_msg = ascii_msg.replace("'", "\"")
                # output_dic = json.loads(ascii_msg)
                # send_data = json.dumps(self.MQ_dic)
                self._sendMQ(message_byte)
                # print('k=', k, '   all_num=', self.all_num_chunks)
                if k + 1 == self.all_num_chunks:
                    self.change_model = True
                    print('send2!!!', 'device' + str(self.edge_device_id) + 'videoID' + str(self.video_ID) + 'all' + str(self.all_num_chunks) + 'eng' + str('%.2f' % (self.power._get())))
                    # 当前为视频最后一块
                    self._sendMQ('device' + str(self.edge_device_id) + 'videoID' + str(self.video_ID) + 'all' + str(self.all_num_chunks) + 'eng' + str('%.2f' % (self.power._get())), queue='energy')
            k += 1
            # filepath = '/home/hhf/yolov5/datasets/try_mot/images/val/011ad6b2-1dfff443'
            # self.val_out = _evaluation(self.out, filepath, self.batchsize, imgsz=seted_imgsz[0], device=self.torch_device)
            # print('edge inference, %s objective, total %s frames, mAP50= %s' % (str(self.result_count), frame_num, self.val_out))

    def _sendMQ(self, message, queue='edge2cloud'):  # 和cloud的 _getMQ()对应
        user_info = pika.PlainCredentials('hhf', '1223')  # 用户名和密码
        connection = pika.BlockingConnection(
            pika.ConnectionParameters('10.12.11.61', 5672, '/', user_info))  # 连接服务器上的RabbitMQ服务

        # 创建一个channel
        channel = connection.channel()

        # 如果指定的queue不存在，则会创建一个queue，如果已经存在 则不会做其他动作，官方推荐，每次使用时都可以加上这句
        channel.queue_declare(queue=queue)

        channel.basic_publish(exchange='',  # 当前是一个简单模式，所以这里设置为空字符串就可以了
                              routing_key=queue,  # 指定消息要发送到哪个queue
                              body=message  # 指定要发送的消息
                              )
        print('mesage have send to ', queue)

        # 关闭连接
        connection.close()

    def _control_client(self):
        # 接收来自云端的控制指令
        # server ip
        user_info = pika.PlainCredentials('hhf', '1223')
        connection = pika.BlockingConnection(pika.ConnectionParameters('10.12.11.61', 5672, '/', user_info))

        # MQ的声明
        # channel
        instruction_channel = connection.channel()

        # exchange (choose exchange)
        instruction_channel.exchange_declare(exchange='inst', exchange_type='fanout')  # 交换机形式 扇形or一对一

        # queue
        result = instruction_channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue

        instruction_channel.queue_bind(exchange='inst', queue=queue_name)

        # 回调函数
        def callback(ch, method, properties, body):
            # print('edge收到:{}'.format(body))
            _instruct = body.decode()
            # print('edge收到:{}'.format(_instruct))
            print('_instruct=', _instruct, "type=", type(_instruct), 'len=', len(_instruct))
            if len(_instruct) == 8:
                target_device = _instruct[:2]
                if target_device == self.edge_device_id:  # offloading
                    self.change_model = False
                    self.rate = int(_instruct[3:5]) / 10
                    self.if_val = True if _instruct[6:] == '01' else False

                    self.power._reset()  # 先清零
                    self._divide()   # 视频分块
                    # self.eng._reset()

                    print('send0!!!', 'device' + str(self.edge_device_id) + 'videoID' + str(self.video_ID) + 'all' + str(self.all_num_chunks) + 'eng' + str('%.2f' % (0)))
                    # 第一次给云端发送数据
                    self._sendMQ('device' + str(self.edge_device_id) + 'videoID' + str(self.video_ID) + 'all' + str(self.all_num_chunks) + 'eng' + str('%.2f' % (0)), queue='energy')

                    # self.divide()

                    # self.producer_thread = threading.Thread(target=self._producer)
                    # self.producer_thread.start()
                    self.sender_thread = threading.Thread(target=self._sender, daemon=True)
                    self.sender_thread.start()

                    self._producer()   # 打包完数据后才开始推理

                    self._edge_inference()

            if len(_instruct) == 5:  # 变换模型
                if _instruct[:2] == self.edge_device_id:
                    print('000')
                    while True:
                        if self.model_dic[str(_instruct[3:5])] == self.weights:
                            print(111)
                            break
                        if self.change_model:
                            print(222)
                            self.weights = self.model_dic[str(_instruct[3:5])]
                            self.infer = Inference(self.imgsz, self.torch_device, weights=self.weights,
                                                   data=self.data_path)
                            break

            else:
                print(' ')

        # channel: 包含channel的一切属性和方法
        # method: 包含 consumer_tag, delivery_tag, exchange, redelivered, routing_key
        # properties: basic_publish 通过 properties 传入的参数
        # body: basic_publish发送的消息

        instruction_channel.basic_consume(queue=queue_name,  # 接收指定queue的消息
                                          auto_ack=True,  # 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息
                                          on_message_callback=callback  # 设置收到消息的回调函数
                                          )

        print('Waiting for messages. To exit press CTRL+C')

        # 一直处于等待接收消息的状态，如果没收到消息就一直处于阻塞状态，收到消息就调用上面的回调函数
        instruction_channel.start_consuming()


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
