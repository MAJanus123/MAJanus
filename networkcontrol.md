安装iproute  sudo apt-get install iproute2
sudo bash ./dynamicEdge.sh clear 0.1mbit 0.01mbit 500ms

5G NO WORKLOAD 20mbit 2mbit 20ms
sudo bash ./dynamicEdge.sh set 20mbit 2mbit 20ms

4G NO WORKLOAD 8mbit 1mbit 40ms
sudo bash ./dynamicEdge.sh set 8mbit 1mbit 40ms

sudo bash ./dynamicEdge.sh set 6mbit 0.5mbit 50ms
sudo bash ./dynamicEdge.sh set 4mbit 0.5mbit 60ms

3G NO WORKLOAD 1mbit 0.1mbit 80ms
sudo bash ./dynamicEdge.sh set 1mbit 0.1mbit 80ms

2.5G NO WORKLOAD 0.1mbit 0.01mbit 500ms
sudo bash ./dynamicEdge.sh set 0.1mbit 0.01mbit 500ms

iperf3 -s -B 10.12.0.186 -p 8888   服务端
iperf3 -c 10.12.0.186 -p 8888     客户端


sudo tc qdisc add dev eth0 root tbf burst 0.1mbit rate 4mbit latency 40ms
tc -s qdisc ls dev eth0  # 查看添加的 tc 规则
 

sudo tc qdisc add dev eth0 root handle 1: htb default 12
sudo tc class add dev eth0 parent 1:1 classid 1:12 htb rate 2mbit
sudo tc qdisc add dev eth0 parent 1:12 netem delay 80ms
tc qdisc add dev eth0 parent 1:1 handle 2:0 netem delay 20ms

sudo tc qdisc delete dev eth0 root  # 删除添加的 tc 规则

sudo tc qdisc add dev eth0 root handle 1:0 htb default 1
sudo tc class add dev eth0 parent 1:0 classid 1:1 htb rate 1mbit


迪哥的
sudo tc qdisc add dev eth0 root tbf rate 25mbit latency 20ms burst 1600


