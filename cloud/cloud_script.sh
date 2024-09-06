#!/bin/bash

# 启动 Python 文件
python3 ./server_tcp_chunk.py --agent_num 6 &
sleep 90
python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_ablation3/main_synchronous_updates.py --experiment_seq 1 &
sleep 22700
#sleep 200
ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9


python3 ./server_tcp_chunk.py --agent_num 6 &
sleep 90
python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_ablation3/main_synchronous_updates.py --experiment_seq 2 &
sleep 22700
#sleep 200
ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9

#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 90
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_ablation2/main_synchronous_updates.py --experiment_seq 2 &
#sleep 22700
##sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 90
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_ablation2/main_synchronous_updates.py --experiment_seq 4 &
#sleep 45400
##sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9

#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 90
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/main_synchronous_updates.py --experiment_seq 4 &
#sleep 22700
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#
#python3 ./server_tcp_chunk.py --agent_num 8 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 5 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 8 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 6 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 8 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 7 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 8 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 8 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9


#python3 ./server_tcp_chunk.py --agent_num 8 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 9 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 8 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 10 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 8 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 11 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 8 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 12 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 13 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 14 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 15 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 16 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 17 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 18 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 19 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 20 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 21 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 22 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 23 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 6 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 24 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 25 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 26 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 27 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 28 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 29 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 30 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 31 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 32 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 33 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 34 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 35 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 4 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 36 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 37 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 38 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 39 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 40 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 41 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 42 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 43 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 44 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 45 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 46 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 47 &
##sleep 22700
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
#
#python3 ./server_tcp_chunk.py --agent_num 2 &
#sleep 70
#python3 /home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_baseline/main_synchronous_updates.py --experiment_seq 48 &
#sleep 200
#ps -ef |grep server_tcp_chunk.py |awk '{print $2}'|xargs kill -9
#ps -ef |grep main_synchronous_updates.py |awk '{print $2}'|xargs kill -9
exit
