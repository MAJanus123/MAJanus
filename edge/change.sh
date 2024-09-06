#!/bin/bash
ps -ef |grep ./change_bandwidth.sh |awk '{print $2}'|xargs kill -9
# 定义要设置的带宽值，单位是 mbit
bandwidth=(6 8 8 6 6 8)
latency=(60 40 40 60 60 40)
# 循环设置带宽
i=0

while true; do
    # 使用 tc 命令来改变带宽设置（请确保替换为适合您网络设备的接口名称）
    # 这里假设您要改变 eth0 接口的出口带宽限制
    bandwidth_now=$((i % 6))
    echo "nvidia" | sudo -S tc qdisc delete dev eth0 root
    echo "nvidia" | sudo -S tc qdisc add dev eth0 root tbf rate "${bandwidth[bandwidth_now]}mbit" latency "${latency[bandwidth_now]}ms" burst 1600
    # 输出当前时间和带宽设置
    echo "带宽设置为 ${bandwidth[bandwidth_now]} mbps 延迟为 ${latency[bandwidth_now]}ms - $(date)"

    # 等待20s再继续循环执行
    sleep 20
    ((i++))
done
