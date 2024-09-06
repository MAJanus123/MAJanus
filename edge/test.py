import threading
import subprocess
import time



if __name__ == "__main__":
    command_change1 = "ps -ef |grep ./change_bandwidth.sh |awk '{print $2}'|xargs kill -9"  # 替换为你想执行的Linux命令
    process = subprocess.Popen(command_change1, shell=True)
    time.sleep(0.1)
    command_change2 = "./change_bandwidth.sh &"  # 替换为你想执行的Linux命令
    process1 = subprocess.Popen(command_change2, shell=True)
    time.sleep(20)
    command_change3 = "ps -ef |grep ./change_bandwidth.sh |awk '{print $2}'|xargs kill -9"  # 替换为你想执行的Linux命令
    process2 = subprocess.Popen(command_change3, shell=True)
    time.sleep(0.1)
    command_change4 = "./change_bandwidth.sh &"  # 替换为你想执行的Linux命令
    process3 = subprocess.Popen(command_change4, shell=True)
    print("over!")