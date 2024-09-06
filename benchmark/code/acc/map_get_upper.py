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
sys.path.insert(0, ROOT+'/res')
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
        self.video_acc = []
        model_dic = {'01': '/res/ultralytics/yolo/weights/yolov8n_trained.pt',
                     '02': '/res/ultralytics/yolo/weights/yolov8s_trained.pt',
                     '03': '/res/ultralytics/yolo/weights/yolov8m_trained.pt'}
        self.yolo = YOLO(ROOT + model_dic[self.modelID], task='detect')
        self.camera_dataset = Camera()
        with open(os.path.join(ROOT, 'res', 'all_video_names_easy.txt'), 'r') as f:
            self.all_video_names = eval(f.read())

        conditions = [2, 5, 7]
        values = [0, 2, 1]
        self.convert = dict(zip(conditions, values))

        self._process_h264Tobgr24()


    def _process_h264Tobgr24(self):  # 处理listen得到的数据
        i = 0
        acc_dict = {}
        while i < len(self.camera_dataset):
            self.video_acc = []
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
            acc_dict[self.all_video_names[i]] = np.mean(self.video_acc)
            i = i + 1
        time.sleep(1)
        print(acc_dict)

    def _edge_inference(self, frame_chunk):

        labels_chunk = get_labels_chunk(ROOT, self.all_video_names, frame_chunk[3], frame_chunk[4])  # get chunk labels
        assert len(frame_chunk[1]) == len(
            labels_chunk), "inference date length {len1} != labels length {len2}, {video}  {chunk}".format(
            len1=len(frame_chunk[1]), len2=len(labels_chunk), video=frame_chunk[3], chunk=frame_chunk[4])
        # iouv = torch.linspace(0.5, 0.95, 10)
        iouv = torch.full((1,), 0.5)
        acc_chunk = []
        preds = []

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
            pred_bantch = self.yolo.predict(inference_data, half=True, imgsz=self.model_inputsize[0])
            inference_latency = time.time() - start
            latency.append(inference_latency)
            print("inference time====", inference_latency)
            preds.extend([x.boxes.data.cpu() for x in pred_bantch])  # 一个chunk的推理结果
        # 精度评估
        stats = []
        for i, pred in enumerate(preds):
            if (len(np.shape(labels_chunk[i]))) == 1:
                labels_chunk[i] = labels_chunk[i][None, :]
            cls = torch.tensor(labels_chunk[i])[:,0]
            correct_bboxes = torch.zeros(pred.shape[0], len(iouv), dtype=torch.bool)  # init
            if len(pred) == 0:
                if len(labels_chunk[i]):  # label文件不为空
                    stats.append((correct_bboxes, *torch.zeros((2, 0)), cls))
                continue

            if len(labels_chunk[i]):
                labels_chunk[i][:, 1:5] = xywh2xyxy(labels_chunk[i][:, 1:5]) * ([self.imgsz[1], self.imgsz[0], self.imgsz[1], self.imgsz[0]])  # target boxes
                # for p in pred:
                #     p[5] = torch.full_like(p[5], self.convert.get(int(p[5]), 3))  # 3表示其他类型
                correct_bboxes = self.process_batch_acc(pred, torch.tensor(labels_chunk[i]), iouv)
            stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls))  # (tp, conf, pcls, tcls)
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        ap = self.ap_per_class(*stats)
        print(ap)
        mAP = [np.mean(x) for x in ap]
        one_acc_chunk = np.mean(mAP)
        MQ_dic = {
            'device_ID': self.deviceID,  # 边缘设备ID          int
            'video_ID': frame_chunk[3],  # 视频名称(代号)       int
            'chunk_ID': frame_chunk[4],  # 第几块视频           int
            'acc_chunk': one_acc_chunk,  # 该chunk的精度评估结果
            'timestamp': frame_chunk[0],  # read前的时间戳，用于计算latency
            'frame_num': len(labels_chunk),  # 当前chunk的帧数      int
        }
        All_acc.append(one_acc_chunk)
        self.video_acc.append(one_acc_chunk)
        print("MQ_dic:  ", MQ_dic)
        with open("upper_bound.txt", "a") as file:
            file.write(str(frame_chunk[3]) + ', ' + str(frame_chunk[4]) + ', ' + str(one_acc_chunk) + '\n')

    def process_batch_acc(self, detections, labels, iouv):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [labels, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    def ap_per_class(self, tp, conf, pred_cls, target_cls, plot=False, save_dir=Path(), names=(), eps=1e-16, prefix=''):
        """
        Computes the average precision per class for object detection evaluation.

        Args:
            tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
            conf (np.ndarray): Array of confidence scores of the detections.
            pred_cls (np.ndarray): Array of predicted classes of the detections.
            target_cls (np.ndarray): Array of true classes of the detections.
            plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
            save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
            names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
            prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

        Returns:
            (tuple): A tuple of six arrays and one array of unique classes, where:
                tp (np.ndarray): True positive counts for each class.
                fp (np.ndarray): False positive counts for each class.
                p (np.ndarray): Precision values at each confidence threshold.
                r (np.ndarray): Recall values at each confidence threshold.
                f1 (np.ndarray): F1-score values at each confidence threshold.
                ap (np.ndarray): Average precision for each class at different IoU thresholds.
                unique_classes (np.ndarray): An array of unique classes that have data.

        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions
            if n_p == 0 or n_l == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = self.compute_ap(recall[:, j], precision[:, j])

        return ap

    def compute_ap(self, recall, precision):
        """
        Compute the average precision (AP) given the recall and precision curves.

        Arguments:
            recall (list): The recall curve.
            precision (list): The precision curve.

        Returns:
            (float): Average precision.
            (np.ndarray): Precision envelope curve.
            (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

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
    parser.add_argument("--resolution", required=False, default='320x240', type=str,
                        help='the encode resolution')
    parser.add_argument("--bitrate", required=False, default='2500000', type=str,
                        help='the encode bitrate')
    parser.add_argument("--modelimgsz", required=False, default='320x320', type=str,
                        help='the encode bitrate')

    args = parser.parse_args()

    with torch.no_grad():
        client = EdgeClient(args)
