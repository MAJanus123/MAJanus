import base64
import json
import warnings
from pathlib import Path

import cv2
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def get_acc(ROOT):
    path1 = ROOT + "/acc/upper_bound_r360P_b5000_m720P_cloud.txt"
    path2 = ROOT + "/acc/upper_bound_r360P_b10000_m720P_cloud.txt"
    path3 = ROOT + "/acc/upper_bound_r360P_b5000_m480P_cloud.txt"
    path4 = ROOT + "/acc/upper_bound_r360P_b5000_m480P_edge.txt"
    path5 = ROOT + "/acc/upper_bound_r360P_b5000_m360P_cloud.txt"
    path6 = ROOT + "/acc/upper_bound_r360P_b5000_m360P_edge.txt"
    path7 = ROOT + "/acc/upper_bound_r360P_b5000_m600P_cloud.txt"
    path8 = ROOT + "/acc/upper_bound_r360P_b5000_m600P_edge.txt"
    path9 = ROOT + "/acc/upper_bound_r720P_b10000_m600P_cloud.txt"
    path10 = ROOT + "/acc/upper_bound_r360P_b2500_m480P_cloud.txt"
    df1 = pd.read_csv(path1, header=None,
                     names=['videoID', 'chunkID', 'acc'])
    df2 = pd.read_csv(path2, header=None,
                      names=['videoID', 'chunkID', 'acc'])
    df3 = pd.read_csv(path3, header=None,
                      names=['videoID', 'chunkID', 'acc'])
    df4 = pd.read_csv(path4, header=None,
                      names=['videoID', 'chunkID', 'acc'])
    df5 = pd.read_csv(path5, header=None,
                      names=['videoID', 'chunkID', 'acc'])
    df6 = pd.read_csv(path6, header=None,
                      names=['videoID', 'chunkID', 'acc'])
    df7 = pd.read_csv(path7, header=None,
                      names=['videoID', 'chunkID', 'acc'])
    df8 = pd.read_csv(path8, header=None,
                      names=['videoID', 'chunkID', 'acc'])
    df9 = pd.read_csv(path9, header=None,
                      names=['videoID', 'chunkID', 'acc'])
    df10 = pd.read_csv(path10, header=None,
                      names=['videoID', 'chunkID', 'acc'])

    acc1 = df1['acc'].tolist()
    acc2 = df2['acc'].tolist()
    acc3 = df3['acc'].tolist()
    acc4 = df4['acc'].tolist()
    acc5 = df5['acc'].tolist()
    acc6 = df6['acc'].tolist()
    acc7 = df7['acc'].tolist()
    acc8 = df8['acc'].tolist()
    acc9 = df9['acc'].tolist()
    acc10 = df10['acc'].tolist()
    # sum_acc1 = 0
    # sum_acc2 = 0
    # for i in range(len(acc1)):
    #     if acc4[i] - acc8[i] > 0:
    #         sum_acc1 += 1
    #     if acc4[i] - acc8[i] < 0:
    #         sum_acc2 += 1
    # print(len(acc1))
    # print("sum_acc1", sum_acc1)
    # print("sum_acc2", sum_acc2)

    print(np.mean(acc1))
    print(np.mean(acc2))
    print(np.mean(acc3))
    print(np.mean(acc4))
    print(np.mean(acc5))
    print(np.mean(acc6))
    print(np.mean(acc7))
    print(np.mean(acc8))
    print(np.mean(acc9))
    print(np.mean(acc10))
if __name__ == "__main__":
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    ROOT = str(ROOT).split("/")
    ROOT = '/'.join(ROOT[0:-1])
    get_acc(ROOT)