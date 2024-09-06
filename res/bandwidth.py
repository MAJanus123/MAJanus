#
# import json
# import subprocess
# import time
#
# # 设置iperf3客户端和服务器的IP地址
# server_ip = "10.12.11.61"
# client_ip = "10.12.11.218"
#
# # 循环运行iperf3命令并输出带宽
# while True:
#     # 运行iperf3命令并捕获输出
#     cmd = f"iperf3 -c {server_ip} -B {client_ip} -J -t 1 -p 5201"
#     result = subprocess.run(cmd.split(), capture_output=True, text=True)
#
#     # 解析iperf3输出并获取带宽
#     if result.returncode == 0:
#         output = result.stdout.strip()
#         json_output = json.loads(output)
#         bandwidth = json_output["end"]["sum_sent"]["bits_per_second"] / 1024 / 1024
#         print(f"带宽：{bandwidth:.2f} Mbps")
#
#     # 等待1秒后再次运行iperf3命令
#     time.sleep(1)

    # output = subprocess.check_output(["iperf3", "-c", server_ip, "-B", client_ip, "-t", "1"])
    # # 解码输出并按行拆分
    # lines = output.decode().split('\n')
    # # 获取统计信息行并按空格拆分
    # stats = lines[3].split()
    # print(stats[6])


import iperf3
import threading

# Define the IP address and ports to listen on
# Change these values to match your network configuration
SERVER_IP = "0.0.0.0"
PORTS = [5001, 5002, 5003]

# Function to handle a single iperf3 client connection
def handle_client(client, port):
    print(f"New client connected on port {port}")
    result = client.run()
    print(f"Test results for port {port}:")
    print(f"  Duration: {result.duration:.2f} seconds")
    print(f"  Bandwidth: {result.sent_Mbps:.2f} Mbps")

# Function to start the iperf3 server on a specific port
def start_server(port):
    print(f"Starting server on port {port}")
    server = iperf3.Server()
    server.bind_address = SERVER_IP
    server.port = port
    server.verbose = False
    server.run(handle_client, port)

# Start the server threads for each port
for port in PORTS:
    t = threading.Thread(target=start_server, args=(port,))
    t.start()

# Wait for all server threads to complete
for t in threading.enumerate():
    if t != threading.current_thread():
        t.join()


