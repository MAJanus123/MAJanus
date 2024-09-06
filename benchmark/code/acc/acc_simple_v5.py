import argparse
import ast
import math
import os
import signal
import sys
import time

from pathlib import Path
from struct import *

import ffmpeg
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-3])
sys.path.append(ROOT+'/res')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
print(sys.path)
import socket
import torch
from res.ultralytics.yolo.engine.model import YOLO
from res.ultralytics.yolo.utils.ops import xywh2xyxy
import base64
import logging
from res.power import *
from multiprocessing import Process, Queue, Manager, Pool, Pipe
from res.ultralytics.yolo.utils.metrics import box_iou
from res.video_stream import bytes2numpy
from res.utils import get_labels_chunk
from res.config_cloud import Config
from edge.RtpClient_chunk import RtpClient
from edge.camera import Camera

logging.getLogger("pika").setLevel(logging.WARNING)

inference_queue = Manager().Queue()
transmission_queue = Manager().Queue()
DATA_PACKET_SIZE = 8688
PACKET_NUM = DATA_PACKET_SIZE / 1448

All_acc = []
latency = []
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class EdgeClient:
    def __init__(self, args):
        self.cloudhost = args.cloudhost
        self.cloudport = args.cloudport
        self.deviceID = args.deviceid
        self.mquser = args.mquser
        self.mqpw = args.mqpw
        self.mqport = args.mqport
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.path_ = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))  # path= /../Video-Analytics-Task-Offloading
        # print('root path : ', self.path_)
        self.modelID = args.modelID
        self.resolution = args.resolution
        self.bitrate = args.bitrate
        self.modelimgsz = args.modelimgsz
        image = self.resolution.split('x')
        image = list(map(int, image))
        self.imgsz = (image[1], image[0])

        modelimgsz = self.modelimgsz.split('x')
        modelimgsz = list(map(int, modelimgsz))
        self.model_inputsize = (modelimgsz[1], modelimgsz[0])
        self.flag = False
        self.chunk_duration = 1

        model_dic = {'01': '/res/ultralytics/yolo/weights/yolov8n.pt',
                     '02': '/res/ultralytics/yolo/weights/yolov8s.pt',
                     '03': '/res/ultralytics/yolo/weights/yolov8m.pt'}
        # self.yolo = YOLO(ROOT + model_dic[self.modelID], imgsz=self.imgsz)
        self.yolo = YOLO(ROOT + model_dic['03'], task='detect')
        self.camera_dataset = Camera()
        with open(os.path.join(ROOT, 'res', 'all_video_names_v5.txt'), 'r') as f:
            self.all_video_names = eval(f.read())

        conditions = [2, 5, 7]
        values = [2, 0, 1]
        self.convert = dict(zip(conditions, values))

        self._process_h264Tobgr24()


    def _process_h264Tobgr24(self):  # 处理listen得到的数据
        i = 0
        while i < len(self.camera_dataset):
            # while True:
            videoNum = i
            sample = self.camera_dataset[videoNum]
            stream = sample['stream']  # 一个视频
            num_chunks = math.ceil(stream.video_duration / self.chunk_duration)
            j = 0
            while j < num_chunks:
                data = (stream, videoNum, j, time.time())  # (stream, videoNum, chunkNum, queue_timestamp)
                stream = data[0]
                queue_wait = time.time() - data[3]
                # print("======", queue_wait)
                (height, width) = self.imgsz
                transmission_process = stream.transmission_process(data[2], resolution=self.resolution,
                                                                   bitrate=self.bitrate)  # 格式转化
                edge_frame = transmission_process.stdout.read()
                decode_process = (ffmpeg
                                  .input('pipe:', format='h264')
                                  .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet')
                                  .run_async(pipe_stdin=True, pipe_stdout=True)
                                  )
                decode_process.communicate(edge_frame)
                out, err = decode_process.communicate()
                # To numpy
                timestamp = time.time() - queue_wait  # add the time of queue_wait
                try:
                    frame = bytes2numpy(30, out, height, width)
                except ValueError:
                    continue
                self._edge_inference((timestamp, frame, self.deviceID, data[1] + 1, data[2] + 1))
                j += 1
            i = i + 1
        time.sleep(1)
        print(np.mean(All_acc))
        print(np.mean(latency[1:])*2)


    def _edge_inference(self, frame_chunk):

        labels_chunk = get_labels_chunk("/data0/yubo.xuan/Video-Analytics-Task-Offloading/", self.all_video_names, frame_chunk[3], frame_chunk[4])  # get chunk labels
        assert len(frame_chunk[1]) == len(
            labels_chunk), "inference date length {len1} != labels length {len2}, {video}  {chunk}".format(
            len1=len(frame_chunk[1]), len2=len(labels_chunk), video=frame_chunk[3], chunk=frame_chunk[4])
        iouv = torch.linspace(0.5, 0.95, 10)
        acc_chunk = []
        inference_batchsize = 15
        rounds = int(np.ceil(len(labels_chunk) / inference_batchsize))
        for num in range(rounds):
            if num < rounds - 1:
                inference_data = frame_chunk[1][num * inference_batchsize:(num + 1) * inference_batchsize]
            else:
                inference_data = frame_chunk[1][num * inference_batchsize:]
            # inference_data = torch.tensor(inference_data).to(self.device)
            # torch.tensor(inference_data).to(self.device)
            start = time.time()
            pred = self.yolo(inference_data)
            inference_latency = time.time() - start
            latency.append(inference_latency)
            print("inference time====", inference_latency)
            # 精度评估
            acc_start = time.time()
            for i in range(num * inference_batchsize, num * inference_batchsize + len(pred)):
                # print(pred[i - num * inference_batchsize].boxes.data.cpu(),"-----")
                if len(labels_chunk[i]) == 0:  # label文件为空
                    if len(pred[i - num * inference_batchsize]) == 0:  # 推理结果为空
                        acc_chunk.append(1)
                    else:
                        acc_chunk.append(0)
                else:
                    # acc_chunk.append(0)
                    if len(pred[i - num * inference_batchsize].boxes.data) == 0:
                        acc_chunk.append(0)
                    else:
                        if (len(np.shape(labels_chunk[i]))) == 1:
                            labels_chunk[i] = labels_chunk[i][None, :]
                        labels_chunk[i][:, 1:5] = xywh2xyxy(labels_chunk[i][:, 1:5]) * (
                        [self.imgsz[1], self.imgsz[0], self.imgsz[1], self.imgsz[0]])  # target boxes
                        prediction = pred[i - num * inference_batchsize].boxes.data.cpu()
                        for p in prediction:
                            p[5] = torch.full_like(p[5], self.convert.get(int(p[5]), p[5]))
                        stats = self.process_batch_acc(prediction,
                                                       torch.tensor(labels_chunk[i]), iouv)
                        acc_chunk.append(stats.cpu().numpy()[:, 0].mean())
        one_acc_chunk = np.mean(acc_chunk)
        MQ_dic = {
            'device_ID': self.deviceID,  # 边缘设备ID          int
            'video_ID': frame_chunk[3],  # 视频名称(代号)       int
            'chunk_ID': frame_chunk[4],  # 第几块视频           int
            'acc_chunk': one_acc_chunk,  # 该chunk的精度评估结果
            'timestamp': frame_chunk[0],  # read前的时间戳，用于计算latency
            'frame_num': len(labels_chunk),  # 当前chunk的帧数      int
        }
        All_acc.append(one_acc_chunk)
        print("MQ_dic:  ", MQ_dic)
        # with open("upper_bound_r360P_b2500_m480P_cloud.txt", "a") as file:
        #     file.write(str(frame_chunk[3]) + ', ' + str(frame_chunk[4]) + ', ' + str(one_acc_chunk) + '\n')

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
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

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
    parser.add_argument("--resolution", required=False, default='320x224', type=str,
                        help='the encode resolution')
    parser.add_argument("--bitrate", required=False, default='2500000', type=str,
                        help='the encode bitrate')
    parser.add_argument("--modelimgsz", required=False, default='640x640', type=str,
                        help='the encode bitrate')
    args = parser.parse_args()

    with torch.no_grad():
        client = EdgeClient(args)
