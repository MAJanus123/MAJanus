#coding:utf-8
# 从abc库中导入ABC, abstractmethod模块
import base64
from abc import ABC, abstractmethod


# 抽象父类
import numpy as np
import pika

from RL_training_shared_PPG_main.command import COMMAND_RESET, COMMAND_ACTION, COMMAND_KILL
from cloud.monitor import Monitor
from res.config_cloud import Config


class BaseEnv(ABC, Monitor):
    def __init__(self, agent_num, monitor_path, weight, reward_seperated,threshold,lt_target, camera_num, acc_weight, eliminate):
        self.config = Config(agent_num)
        super().__init__(self.config, monitor_path, agent_num, eliminate)  # 调用基类的__init__()方法 启动monitor
        self.agent_num = agent_num
        self.camera_num = camera_num
        self.actor_state_dim = 7
        self.critic_state_dim = 8 * agent_num
        self.action_dim = 4
        self.time_interval = self.config.time_interval
        if camera_num == 2 and self.agent_num > 4:
            self.action_old = [1, 0, 1, 0]  # [rate,resolution,bitrate,model]
        else:
            self.action_old = [2, 1, 1, 1]  # [rate,resolution,bitrate,model]
        self.latency_all = []
        # use backup policy, 1 means use
        self.switch_flag = [0] * agent_num
        self.switch_num = [0] * agent_num
        self.w = weight  # 奖励函数系数
        self.T = threshold  # 最大延迟
        self.F = -4  # 负反馈
        self.acc_weight = acc_weight
        self.lt_target = lt_target  # 目标延迟
        self.reward_seperated = reward_seperated
        self.immediate_videoID_chunkID = [[] for _ in range(agent_num)]
        user_info = pika.PlainCredentials(self.config.mquser, self.config.mqpw)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.config.cloudhost, self.config.mqport, credentials=user_info))
        self.control_channel = connection.channel()

        self.action = None
        self.system_average_latency = None
        self.acc = None
        self.queue_wait = None
    # 抽象方法
    @abstractmethod
    def step(self, action, step, episode, switch, flag):
        pass

    @abstractmethod
    def _get_reward(self, agent_id, system_average_latency, acc, all_gap_acc):
        pass

    def _broadcast_control_command(self, message):

        self.control_channel.exchange_declare(exchange='command', exchange_type='fanout')

        self.control_channel.basic_publish(exchange='command', routing_key='', body=message)

    def _reset(self, done=False, first=False, use_agent_ID=None):
        COMMAND_RESET['value']['done'] = done
        COMMAND_RESET['value']['first'] = first
        COMMAND_RESET['episode'] = 0
        COMMAND_ACTION['step'] = 0
        COMMAND_RESET['use_agent_ID'] = use_agent_ID
        COMMAND_ACTION['use_agent_ID'] = use_agent_ID

        print('broadcast reset command: done %s, first %s' % (str(done), str(first)))

        message = str(COMMAND_RESET).encode('ascii')
        message_byte = base64.b64encode(message)
        self._broadcast_control_command(message_byte)

    def _set_action(self, offload_rate, resolution, bitrate, edge_model, step, episode, flag):
        COMMAND_ACTION['value']['offloading_rate'] = offload_rate
        COMMAND_ACTION['value']['resolution'] = resolution
        COMMAND_ACTION['value']['bitrate'] = bitrate
        COMMAND_ACTION['value']['edge_model'] = edge_model

        COMMAND_ACTION['step'] = step
        COMMAND_ACTION['episode'] = episode
        COMMAND_ACTION['flag'] = flag
        message = str(COMMAND_ACTION).encode('ascii')
        message_byte = base64.b64encode(message)

        self._broadcast_control_command(message_byte)

    def kill(self):
        message = str(COMMAND_KILL).encode('ascii')
        message_byte = base64.b64encode(message)

        self._broadcast_control_command(message_byte)

    def reset(self, done=False, first=True, agent_num=2, use_agent_ID=None):
        """
        reset all the system states, start inference from the start of the system.
        """
        actions = []
        for agent_id in range(agent_num):
            actions.append(self.action_old)
        self._reset(done, first, use_agent_ID)
        return actions

