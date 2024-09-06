import base64
import json
import math
import random
import warnings
from pathlib import Path
import torch
import cv2
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

# load队列
def get_seq(num_local_chunk, num_cloud_chunk):
    num = num_local_chunk + num_cloud_chunk
    if num_local_chunk == 0:
        return [1] * num
    if num_cloud_chunk == 0:
        return [0] * num
    if num_local_chunk > num_cloud_chunk:
        per_interval = float(num_local_chunk/num_cloud_chunk)
        big = num_local_chunk
        small = num_cloud_chunk
        a = 0   # 原本序列里的数
        b = 1   # 要插入的数
    else:
        per_interval = float(num_cloud_chunk / num_local_chunk)
        big = num_cloud_chunk
        small = num_local_chunk
        a = 1
        b = 0
    result = [a] * big
    decimal = per_interval % 1
    integer = int(per_interval)
    index = integer
    if decimal < 0.5:
        for i in range(small):
            result.insert(index, b)
            index += integer+1
    else:
        for i in range(small):
            result.insert(index, b)
            index += integer + 2 if i % 2 == 0 else integer + 1
    return result

def get_seq_random(num_local_chunk, num_cloud_chunk):
    list_edge = [0] * num_local_chunk
    list_cloud = [1] * num_cloud_chunk
    seq = list_edge + list_cloud
    # 随机打乱这个列表
    random.shuffle(seq)
    return seq
# 从labes中获取当前chunk的label数据
def getchunk_labels(labels, j):
    if ((j+1)*30) <= len(labels):
        return labels[j*30:(j+1)*30]
    else:
        return labels[j*30:len(labels)]

# 通过label_names从本地获取当前videoid chunkid对应的label数据
def get_labels_chunk(ROOT, all_video_names, video_ID, chunk_ID):
    video_name = all_video_names[video_ID - 1]
    video_label_names = os.listdir(os.path.join(ROOT, 'res', 'labels_noblack_easy_mini', video_name.split('.')[0]))
    video_label_names.sort(key=lambda x: int(x[-9:-4]))
    chunk_label_names = getchunk_labels(video_label_names, chunk_ID-1)
    chunk_label_list = []
    for name in chunk_label_names:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            chunk_label_list.append(np.loadtxt(os.path.join(ROOT, 'res', 'labels_noblack_easy_mini', video_name.split('.')[0], name)))
    return chunk_label_list

# 将MQ收到的字节流转换为json
def MQ_decode(body):
    msg_bytes = base64.b64decode(body)
    ascii_msg = msg_bytes.decode('ascii')
    ascii_msg = ascii_msg.replace("'", "\"")
    # print("ascii_msg after", ascii_msg)
    return json.loads(ascii_msg)

def resolution_select(x):
    if x <= 1/3:
        return '448x240'
    elif x <= 2/3:
        return '640x360'
    else:
        return '960x540'

# def resolution_select(x):
#     return '512x288'

def find_closest_index(a, value_list):
    closest_index = None
    min_difference = float('inf')

    for i, value in enumerate(value_list):
        difference = abs(a - value)
        if difference < min_difference or (difference == min_difference and value > value_list[closest_index]):
            closest_index = i
            min_difference = difference

    return closest_index

def bitrate_select(x):
    if x <= 1/3:
        return 300000
    elif x <= 2 / 3:
        return 800000
    else:
        return 3000000

# def bitrate_select(x):
#     return 500000


def edge_model_select(x):
    if x <= 1/3:
        return 'n'
    elif x <= 2 / 3:
        return 's'
    else:
        return 'm'

def cloud_model_select(x):
    if x <= 1 / 3:
        return 'n'
    elif x <= 2 / 3:
        return 's'
    else:
        return 'm'

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def compute_delay_reward(agent_num, delayed_feedback, gamma):
    returns_true = []
    for i in range(agent_num):
        return_temp = []
        discounted_reward = 0
        for reward in reversed(delayed_feedback[i]):
            discounted_reward = np.float32(reward + (gamma * discounted_reward))
            return_temp.insert(0, discounted_reward)
        returns_true.extend(return_temp)
    returns_true = torch.stack([torch.tensor(arr).squeeze() for arr in returns_true])
    return returns_true

def generate_RL_reward(ROOT, agent_num, log_path, log_acc_path):
    path_root = ROOT + "/RL_training/experiment/" + log_path
    path_acc_root = ROOT + "/RL_training/experiment/" + log_acc_path
    for agent_number in range(agent_num):
        path = path_root + str(agent_number+1) + '.txt'
        path_acc = path_acc_root + str(agent_number+1) + '.txt'
        df = pd.read_csv(path, header=None, names=['agent', 'action_mean', 'std', 'latency', 'throughput', 'reward', 'acc', 'queue_wait', 'offload_queue', 'edge_inference_queue', 'bandwidth', 'offload_rate', 'resolution', 'bitrate', 'edge_model', 'cloud_queue', 'cloud_latency', 'edge_latency'])
        # df = pd.read_csv(path, header=None, names=['agent', 'latency', 'throughput', 'reward', 'acc', 'queue_wait', 'offload_queue', 'edge_inference_queue', 'bandwidth', 'offload_rate', 'resolution', 'bitrate', 'edge_model', 'cloud_queue', 'cloud_latency', 'edge_latency'])

        # df = df[df['latency'] <= 5]
        reward_list = df['reward'].tolist()
        offload_rate = df['offload_rate'].tolist()
        resolution = df['resolution'].tolist()
        bitrate = df['bitrate'].tolist()
        edge_model = df['edge_model'].tolist()
        latency = df['latency'].tolist()
        acc = df['acc'].tolist()
        queue_wait = df['queue_wait'].tolist()
        bandwidth = df['bandwidth'].tolist()
        throughput = df['throughput'].tolist()
        edge_latency = df['edge_latency'].tolist()
        cloud_latency = df['cloud_latency'].tolist()

        df_acc = pd.read_csv(path_acc, header=None, names=['videoID', 'chunkID', 'acc'])
        dataset_acc = []
        temp_acc = []
        print(len(df_acc))
        for i in range(len(df_acc) - 1):
            temp_acc.append(df_acc.loc[i,'acc'])
            if (df_acc.loc[i,'videoID'] == 8) and (df_acc.loc[i+1,'videoID'] == 1):
                dataset_acc.append(np.mean(temp_acc))
                temp_acc = []

        # 处理最后一行
        temp_acc.append(df_acc.loc[len(df_acc)-1, 'acc'])
        dataset_acc.append(np.mean(temp_acc))
        print(len(dataset_acc), dataset_acc)

        # dataset_acc = dataset_acc[0:15]
        n = len(reward_list)
        x = range(0, n)

        # 处理reward
        reward = reward_list
        num = int(len(reward)/12)
        new_reward = []
        for i in range(num):
            new_reward.append(np.mean(reward[i*12:(i+1)*12]))
        if num * 12 < len(reward):
            new_reward.append(np.mean(reward[num*12:]))

        # # 处理reward
        num = int(len(acc) / 12)
        new_acc = []
        for i in range(num):
            new_acc.append(np.mean(acc[i * 12:(i + 1) * 12]))
        if num * 12 < len(acc):
            new_acc.append(np.mean(acc[num * 12:]))

        # print(len(new_reward))
        # 创建画布和子图
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 12), gridspec_kw={'hspace': 0.3})
        axs[0,0].plot(range(0,len(new_reward)), new_reward)
        axs[0,0].set_title('reward')

        axs[0,1].plot(x, offload_rate)
        axs[0,1].set_title('offload_rate')

        axs[0,2].plot(x, resolution)
        axs[0,2].set_title('resolution')

        axs[1,0].plot(x, bitrate)
        axs[1,0].set_title('bitrate')

        axs[1,1].plot(x, edge_model)
        axs[1,1].set_title('edge_model')

        axs[1, 2].plot(x, latency)
        axs[1, 2].set_title('latency')

        # axs[2, 0].plot(range(0,len(dataset_acc)), dataset_acc)
        axs[2, 0].plot(range(0,len(new_acc)), new_acc)
        axs[2, 0].set_title('dataset_acc')

        axs[2, 1].plot(x, queue_wait)
        axs[2, 1].set_title('queue_wait')

        axs[2, 2].plot(x, bandwidth)
        axs[2, 2].set_title('bandwidth')

        fig.suptitle(log_path+str(agent_number+1))
        plt.savefig("../RL_training/experiment/" + log_path + str(agent_number+1) + ".png")


def get_acc_actions(ROOT):
    path = ROOT + "/RL_training/RL_result/MAPPO7_nolatency_separate_w1_agent2.txt"
    df = pd.read_csv(path, header=None,
                     names=['agent', 'action_mean', 'std', 'latency', 'reward', 'acc', 'queue_wait', 'offload_rate',
                            'resolution', 'bitrate', 'edge_model'])
    # new_df = df[(df['acc'] > 0.7) & (df['acc'] < 0.8)]
    new_df = df[df['reward'] > 0.5]
    reward_list = new_df['reward'].tolist()
    offload_rate = new_df['offload_rate'].tolist()
    resolution = new_df['resolution'].tolist()
    bitrate = new_df['bitrate'].tolist()
    edge_model = new_df['edge_model'].tolist()
    latency = new_df['latency'].tolist()
    acc = new_df['acc'].tolist()
    n = len(reward_list)
    x = range(0, n)
    # 创建画布和子图
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 12), gridspec_kw={'hspace': 0.3})
    axs[0, 0].plot(x, reward_list)
    axs[0, 0].set_title('reward')

    axs[0, 1].plot(x, offload_rate)
    axs[0, 1].set_title('offload_rate')

    axs[0, 2].plot(x, resolution)
    axs[0, 2].set_title('resolution')

    axs[1, 0].plot(x, bitrate)
    axs[1, 0].set_title('bitrate')

    axs[1, 1].plot(x, edge_model)
    axs[1, 1].set_title('edge_model')

    axs[1, 2].plot(x, latency)
    axs[1, 2].set_title('latency')

    axs[2, 0].plot(x, acc)
    axs[2, 0].set_title('acc')
    fig.suptitle('MAPPO7_nolatency_separate_w1_agent1')
    plt.savefig("../RL_training/RL_result/MAPPO7_nolatency_separate_w1_agent1.png")


def get_better_configurations(ROOT):
    path = ROOT + "/RL_training/RL_result/MAPPO12_w0.9_dataset_agent3.txt"
    df = pd.read_csv(path, header=None,
                     names=['agent', 'action_mean', 'std', 'latency', 'reward', 'acc', 'bandwidth', 'queue_wait',
                            'offload_rate',
                            'resolution', 'bitrate', 'edge_model', 'acc_gap'])
    # df = pd.read_csv(path, header=None,
    #                  names=['agent', 'action_mean', 'std', 'latency', 'reward', 'acc', 'queue_wait', 'offload_rate',
    #                         'resolution', 'bitrate', 'edge_model'])
    new_df = df[:1000]
    new_df = new_df[new_df['acc'] < 0.70]
    # new_df = new_df[(new_df['acc'] > 0.70) & (new_df['acc'] < 0.80)]
    new_df = new_df.reset_index(drop=True)
    new_df = new_df.loc[:, ['resolution', 'bitrate', 'edge_model','offload_rate']]

    for i in range(len(new_df)):
        new_df.loc[i,'bitrate'] = 1.0 if math.ceil(new_df.loc[i,'bitrate']*3) == 0 else math.ceil(new_df.loc[i,'bitrate']*3)
        new_df.loc[i,'resolution'] = 1.0 if math.ceil(new_df.loc[i,'resolution']*3) == 0 else math.ceil(new_df.loc[i,'resolution']*3)
        new_df.loc[i,'edge_model'] = 1.0 if math.ceil(new_df.loc[i,'edge_model']*3) == 0 else math.ceil(new_df.loc[i,'edge_model']*3)
        new_df.loc[i,'offload_rate'] = round(new_df.loc[i,'offload_rate'],1)
    print(new_df)
    # 计算组合数量并按数量排序
    counts = new_df.groupby(['resolution', 'bitrate', 'edge_model','offload_rate']).size().reset_index(name='count')
    sorted_counts = counts.sort_values(by='count', ascending=False)

    # 显示结果
    # for i in range(len(sorted_counts)):
    #     print(sorted_counts.loc[i])
    print(sorted_counts[:20])

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=Path(), names=(), eps=1e-16, prefix=''):
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
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    return ap

def compute_ap(recall, precision):
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

if __name__ == "__main__":
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    ROOT = str(ROOT).split("/")
    ROOT = '/'.join(ROOT[0:-1])
    # generate_RL_reward(ROOT, 6, "1.3_lr0.0001_seperated_w0.5_agent", "1.3_lr0.0001_seperated_w0.5_acc_agent")
    # generate_RL_reward(ROOT, 6, "1.3_lr0.0001_seperated_w1_agent", "1.3_lr0.0001_seperated_w1_acc_agent")
    generate_RL_reward(ROOT, 4, "3.10_w8_T1.5_lr0.0001_df0.97_exploration10_agent", "3.10_w8_T1.5_lr0.0001_df0.97_exploration10_acc_agent")