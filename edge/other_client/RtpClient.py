"""
Implement the rtp protocol 
send the filtered video frames to the server for aggregating information
"""
# from PIL import Image
import io
import socket
import time
from struct import *

from res.RtpPacket import RtpPacket

DATA_PACKET_SIZE = 8688
DATA_HEADER_SIZE = 12


class RtpClient:  # Rtp客户端

    def __init__(self, host='10.12.11.61', port=1080, socket_type=0):
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
            self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(self.HOST, self.PORT)
            self.dataSocket.connect((self.HOST, self.PORT))
        elif self.SOCKET_TYPE == 'SOCK_STREAM':
            self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(self.HOST, self.PORT)
            self.dataSocket.connect((self.HOST, self.PORT))
        # print('Connect to %s %d success...' % (self.HOST, self.PORT))

    def _sendRTPPacket(self, timestamp, data, marker, device_ID, video_ID, chunk_ID):

        packet = RtpPacket()
        self.DATA_SEQUENCE = self.DATA_SEQUENCE + 1
        packet.encode(self.DATA_SEQUENCE, marker, data, timestamp, device_ID, video_ID, chunk_ID)
        sendData = packet.getPacket()
        if marker == 2:
            sendData.extend(b'\\EOF')
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

    def send_image_edge(self, timestamp, image_bytes, device_ID, video_ID, chunk_ID, height, width):
        marker = 0
        self.DATA_SEQUENCE = 0
        image_BytesIO = io.BytesIO(image_bytes)
        while True:
            data = image_BytesIO.read(DATA_PACKET_SIZE - DATA_HEADER_SIZE)
            if len(data) == 0:
                marker = 2  # 2 for end image
                self._sendRTPPacket(timestamp, pack('ll', height, width), marker, device_ID, video_ID, chunk_ID)
                print("The last DATA_SEQUENCE is ", self.DATA_SEQUENCE)
                break
            self._sendRTPPacket(timestamp, data, marker, device_ID, video_ID, chunk_ID)
            marker = 1  # 1 for normal image
            time.sleep(0.0005)
        image_BytesIO.close()

    def send_image_cloud(self, timestamp, image_bytes, device_ID, video_ID, chunk_ID, height, width):
        marker = 0
        self.DATA_SEQUENCE = 0
        image_BytesIO = io.BytesIO(image_bytes)
        while True:
            data = image_BytesIO.read(DATA_PACKET_SIZE - DATA_HEADER_SIZE)
            if len(data) == 0:
                marker = 2  # 2 for end image
                # print(pack('ll', height, width))
                self._sendRTPPacket(timestamp, pack('ll', height, width), marker, device_ID, video_ID, chunk_ID)
                print("The last DATA_SEQUENCE is ", self.DATA_SEQUENCE)
                break
            self._sendRTPPacket(timestamp, data, marker, device_ID, video_ID, chunk_ID)
            marker = 1  # 1 for normal image
            time.sleep(0.0001)
        image_BytesIO.close()
