#!/bin/bash

# 启动 Python 文件
#  1次/h
python3 ./client_tcp_chunk_camera_dataset_update.py &
sleep 23520
ps -ef |grep client_tcp_chunk_camera_dataset_update.py |awk '{print $2}'|xargs kill -9

python3 ./client_camera_frequency.py --loadtime 0.5 --loadtime_list 1 2 3 4 5 6 7 8 &
sleep 23520
ps -ef |grep client_camera_frequency.py|awk '{print $2}'|xargs kill -9


exit
