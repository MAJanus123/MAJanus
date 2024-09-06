# coding=utf-8
import re
import threading
import subprocess
import time
import numpy as np


# GPU POWER:
# nvidia-smi --format=csv --query-gpu=power.draw -l 1

class EngUbuntuCPU:   # 仅CPU
    def __init__(self, sudoPassword, fq):
        self.all_power = 0

        self.p = subprocess.Popen(
            "echo " + sudoPassword + " | sudo -S sudo /usr/bin/turbostat --Summary --quiet --show PkgWatt --show GFXWatt --show RAMWatt --interval " + str(
                fq), stdout=subprocess.PIPE, shell=True)

        out = self.p.stdout.readline().decode("utf8")
        print('out=', out, 'type(out)=', type(out))
        out1 = self.p.stdout.readline().decode("utf8")
        out1 = re.findall(r'\d+', out1, re.S)
        print('out1=', out1, 'type(out1)=', type(out1))
        pkg = int(out1[0]) + int(out1[1]) / 100
        # cor = int(out1[2]) + int(out1[3]) / 100
        gfx = int(out1[2]) + int(out1[3]) / 100
        ram = int(out1[4]) + int(out1[5]) / 100
        print('pkg=', pkg, 'gfx=', gfx, 'ram=', ram)

        self.read_thread = threading.Thread(target=self._read)
        self.read_thread.start()

    def _read(self):
        while True:
            # t1 = time.time()
            # out = float(self.p.stdout.readline().decode("utf8"))
            out = self.p.stdout.readline().decode("utf8")
            # out = p.stdout.readline().decode("utf8").replace('\n', '').replace('\r', '')
            # print('out=', out, 'type(out)=', type(out))

            out1 = re.findall(r'\d+', out, re.S)
            self.pkg = int(out1[0]) + int(out1[1]) / 100
            # self.cor = int(out1[2]) + int(out1[3]) / 100
            self.gfx = int(out1[2]) + int(out1[3]) / 100
            self.ram = int(out1[4]) + int(out1[5]) / 100
            # print('pkg=', self.pkg, 'cor=', self.cor, 'gfx=', self.gfx, 'ram=', self.ram)

            # print('t=', time.time() - t1)
            out_b = self.pkg + self.gfx + self.ram
            self.all_power += out_b
            # print('all_power=', self.all_power)

    def get(self):
        return round(self.all_power, 2)

    def _get_all(self):
        return round(self.all_power, 3), round(self.pkg, 3), round(self.gfx, 3), round(self.ram, 3)

    def reset(self):
        self.all_power = 0


class EngNX:
    def __init__(self, sudoPassword, fq):
        self.power_NX_list = []
        self.fq = fq
        self.sudoPassword = sudoPassword
        self.p = subprocess.Popen(
            "echo " + self.sudoPassword + " | sudo -S sudo cat /sys/bus/i2c/drivers/ina3221x/7-0040/iio:device0/in_power0_input",
            stdout=subprocess.PIPE, shell=True)
        self.last_power = 10.00
        self.read_thread = threading.Thread(target=self._read)
        self.read_thread.start()

    def _read(self):   # 获取CPU和GPU的功率
        while True:
            time.sleep(self.fq)
            self.p = subprocess.Popen(
                "echo " + self.sudoPassword + " | sudo -S sudo cat /sys/bus/i2c/drivers/ina3221x/7-0040/iio:device0/in_power0_input",
                stdout=subprocess.PIPE, shell=True)

            out = float(self.p.stdout.readline().decode('utf8'))
            # print('out=', out, 'type(out)=', type(out))
            out = round(float(out) / 1000, 4)
            # print('out=', out, 'type(out)=', type(out))

            self.power_NX_list.append(out)

    def get(self):
        if self.power_NX_list == []:
            return self.last_power
        else:
            self.last_power = np.mean(self.power_NX_list)
            return round(self.last_power, 2)

    def reset(self):
        self.power_NX_list = []


class EngUbuntuGPU:   # 包含CPU和GPU和RAM
    def __init__(self, sudoPassword, fq):
        self.all_power = 0
        self.fq = fq
        self.power_cpu_list = []
        self.power_gpu_list = []

        self.p = subprocess.Popen(
            "echo " + sudoPassword + " | sudo -S sudo /usr/bin/turbostat --Summary --quiet --show PkgWatt --show RAMWatt --interval " + str(
                self.fq), stdout=subprocess.PIPE, shell=True)

        out = self.p.stdout.readline().decode("utf8")
        # out1 = self.p.stdout.readline().decode("utf8")
        # out1 = re.findall(r'\d+', out1, re.S)
        # print('out1=', out1, 'type(out1)=', type(out1))
        # pkg = int(out1[0]) + int(out1[1]) / 100
        # ram = int(out1[2]) + int(out1[3]) / 100
        # print('pkg=', pkg, 'ram=', ram)

        self.read_thread = threading.Thread(target=self._read, daemon=True)
        self.read_thread.start()

        # self.read_gpu_thread = threading.Thread(target=self._gpu_read)
        # self.read_gpu_thread.start()

        self._gpu_read()

    def _gpu_read(self):
        """nvidia-smi --format=csv --query-gpu=power.draw -l 1"""
        while True:
            time.sleep(self.fq)
            self.pg = subprocess.Popen('nvidia-smi --format=csv --query-gpu=power.draw', stdout=subprocess.PIPE, shell=True)
            out_g = self.pg.stdout.read().decode('utf8')
            out_g = re.findall(r'\d+', out_g, re.S)
            # print('out_g0=', out_g, 'type(out_g)=', type(out_g))
            out_g = int(out_g[0]) + int(out_g[1])/100
            # print('out_g1=', out_g, 'type(out_g)=', type(out_g))
            # out_gb = out_g * (self.fq/10 + time.time() - tg)
            # print(out_gb, time.time() - tg)
            # self.all_power += out_g
            self.power_gpu_list.append(out_g)

    def _read(self):    # 读cpu
        while True:
            # t1 = time.time()
            # out = float(self.p.stdout.readline().decode("utf8"))
            out = self.p.stdout.readline().decode("utf8")
            # out = p.stdout.readline().decode("utf8").replace('\n', '').replace('\r', '')
            # print('out=', out, 'type(out)=', type(out))

            out1 = re.findall(r'\d+', out, re.S)
            # print(out1)
            try:
                self.pkg = int(out1[0]) + int(out1[1]) / 100
                self.ram = int(out1[2]) + int(out1[3]) / 100
                # print('pkg=', self.pkg, 'cor=', self.cor, 'gfx=', self.gfx, 'ram=', self.ram)

                # print('t=', time.time() - t1)
                out_c = self.pkg + self.ram
                # self.all_power += out_c
                # print('all_power=', self.all_power)1
                # print("out1=================", out1)
            except:
                out_c = 0
                print("power error")
            self.power_cpu_list.append(out_c)

    def get(self):
        return round(np.mean(self.power_cpu_list) + np.mean(self.power_gpu_list), 2)

    def reset(self):
        self.power_cpu_list = []
        self.power_gpu_list = []

if __name__ == '__main__':

    power_monitor = EngUbuntuGPU('1223', fq=0.1)
    while True:
        time.sleep(1)
        power = power_monitor.get()
        print(power)
        power_monitor.reset()


