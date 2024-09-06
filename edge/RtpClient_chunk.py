"""
Implement the rtp protocol 
send the filtered video frames to the server for aggregating information
"""
# from PIL import Image
import io
import socket
import time
from struct import *

from res.RtpPacket_chunk import RtpPacket

DATA_PACKET_SIZE = 8688
# DATA_PACKET_SIZE = 1024
DATA_HEADER_SIZE = 20


class RtpClient:  # Rtp客户端

    def __init__(self, host='10.12.11.144', port=1080, socket_type=0):
        if socket_type == 1:
            self.SOCKET_TYPE = 'SOCK_STREAM'
        elif socket_type == 0:
            self.SOCKET_TYPE = 'SOCK_DGRAM'
        self.PORT = port
        self.HOST = host
        self._connect()
        # TRICK: data sequence number is used to identify image / inference result:
        # self.DATA_SEQUENCE == 0 ? inference result : image data
        self.DATA_SEQUENCE = 0

    def _connect(self):
        # print("connect to server------")
        if self.SOCKET_TYPE == 'SOCK_DGRAM':
            socket.setdefaulttimeout(99999)
            self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # self.dataSocket.settimeout(999999)
            print(self.HOST, self.PORT)
            self.dataSocket.connect((self.HOST, self.PORT))
        elif self.SOCKET_TYPE == 'SOCK_STREAM':  # tcp
            socket.setdefaulttimeout(99999)
            self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.dataSocket.settimeout(999999)
            print(self.HOST, self.PORT)
            self.dataSocket.connect((self.HOST, self.PORT))
        # print('Connect to %s %d success...' % (self.HOST, self.PORT))

    def _sendRTPPacket(self, timestamp, marker, device_ID, video_ID, chunk_ID, step_index, data, episode, start_send):

        packet = RtpPacket()
        self.DATA_SEQUENCE = self.DATA_SEQUENCE + 1
        if marker == 2:
            packet.encode(self.DATA_SEQUENCE, marker, timestamp, device_ID, video_ID, chunk_ID, step_index, data, episode, start_send)
            sendData = packet.getPacket()
            sendData.extend(b'\\EOF')
            # print(len(sendData))
        else:
            sendData = data
        try:
            if self.SOCKET_TYPE == 'SOCK_DGRAM':
                self.dataSocket.sendto(sendData, (self.HOST, self.PORT))
            elif self.SOCKET_TYPE == 'SOCK_STREAM':
                # print("send buff", self.dataSocket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
                # self.dataSocket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 9999999)
                # print("update send buff", self.dataSocket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
                self.dataSocket.sendall(sendData)

        except:
            self.dataSocket.close()
            print("sending packet fail !!")

    def send_image_cloud(self, timestamp, image_bytes, device_ID, video_ID, chunk_ID, height, width, step_index, episode):
        marker = 1  # 1 for normal image
        self.DATA_SEQUENCE = 0
        start_send = time.time()
        self._sendRTPPacket(timestamp, marker, device_ID, video_ID, chunk_ID, step_index, image_bytes, episode, start_send)
        marker = 2  # 2 for end image
        self._sendRTPPacket(timestamp, marker, device_ID, video_ID, chunk_ID, step_index, pack('ll', height, width), episode, start_send)  # pack包为16B
        print("The last DATA_SEQUENCE is ", self.DATA_SEQUENCE, "step_index=", step_index, "time=", time.time() - timestamp)
        # send_time = time.time() - start_send
        # print("send_time", send_time, "size=", len(image_bytes))
        # return send_time, len(image_bytes)