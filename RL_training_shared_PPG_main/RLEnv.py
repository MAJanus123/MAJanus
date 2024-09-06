import sys
import threading
import time
from pathlib import Path
import numpy as np
from res.utils import edge_model_select, cloud_model_select, resolution_select, bitrate_select
from RL_training_shared_PPG_main.Base import BaseEnv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-1])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from res.config_cloud import Config
import pika
import base64
from command import COMMAND_RESET, COMMAND_ACTION
from cloud.monitor import Monitor
"""

System:
    edge
    cloud

RL Environment:
    RLEnv -> 系统的抽像类，它可以控制比如edge端的offloading rate
    RLAgent


"""


class Env(BaseEnv):
    def __init__(self, agent_num, monitor_path, weight, reward_seperated,threshold,lt_target, camera_num, acc_weight, eliminate):
        super().__init__(agent_num, monitor_path, weight, reward_seperated,threshold,lt_target, camera_num, acc_weight, eliminate)  # 调用基类的__init__()方法

        self.offload_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.resolution_list = [0, 0.5, 1]
        self.bitrate_list = [0, 0.5, 1]
        self.model_list = [0, 0.5, 1]
        self.all_actions = [self.offload_list, self.resolution_list, self.bitrate_list, self.model_list]

    def step(self, actions, step, episode, action_mask, flag):
        """
        INPUT:
            action: offloading rate for the edge device

        OUTPUT:
            state: wait for self.time_interval seconds, then return the current system state.
                offloading_queue_size
                edge_inference_queue_size
                cloud_inference_queue_size
                system_average_latency
                action for last round: offloading rate at time t - 1

            reward: normalized system latency

        1. when the environment receivces the offloading rate, the ENV sends an control command to the edge,
        change the offloading rate in the edge side.

        2. wait for one time_interval sleep self.time_interval s.

        3. retrieve the current state from the influxdb
        """
        offload_rate = []
        resolution = []
        bitrate = []
        edge_model = []
        for i, action in enumerate(actions):
            action = [self.all_actions[index][x] for index, x in enumerate(action)]
            action = np.clip(np.array(action, dtype=np.float32), 0, 1).round(2)  # 后处理
            offload_rate.append(action[0])
            resolution.append(resolution_select(action[1]))
            bitrate.append(bitrate_select(action[2]))
            edge_model.append(edge_model_select(action[3]))

        self._set_action(offload_rate, resolution, bitrate, edge_model, step, episode, flag)
        time.sleep(self.time_interval)

        all_inference_queue_size, all_offloading_queue_size = self._get_edge_queue_size(self.agent_num)
        cloud_transmission_queue_size = self._get_cloud_queue_size(self.agent_num)

        all_AVG_latency, all_average_latency_cloud_edge = self._get_system_average_latency(self.agent_num, step)
        all_AVG_latency = [round(x,4) for x in all_AVG_latency]
        all_average_latency_cloud_edge = [[round(x[0],4),round(x[1],4)] for x in all_average_latency_cloud_edge]

        all_acc, all_gap_acc, all_videoID_list, all_chunkID_list = self._get_system_average_acc(self.agent_num, step)
        all_acc = [round(x,4) for x in all_acc]
        all_gap_acc = [round(x,4) for x in all_gap_acc]

        all_queue_wait, all_average_qwait_cloud_edge = self._get_system_average_queue_wait(self.agent_num, step)
        all_queue_wait = [round(x,4) for x in all_queue_wait]
        all_average_qwait_cloud_edge = [[round(x[0],4),round(x[1],4)] for x in all_average_qwait_cloud_edge]

        all_bandwidth = self._get_bandwidth(self.agent_num)
        all_bandwidth = [round(x, 4) for x in all_bandwidth]

        all_throughput = self._get_throughput(self.time_interval, self.agent_num, step)

        actions_old = actions

        # action_mask的启动和关闭
        if action_mask:
            for i in range(self.agent_num):
                if self.switch_flag[i] == 0 and (all_AVG_latency[i] > self.T or all_inference_queue_size[i] > 1.5 or cloud_transmission_queue_size[i] > 10):
                    self.switch_flag[i] = 1
                    self.switch_num[i] += 1
                    print("agent" + str(i+1) + " switch to backup")
                if self.switch_flag[i] == 1 and all_AVG_latency[i] <= self.T and (all_inference_queue_size[i] <= 1.5) and (cloud_transmission_queue_size[i] <= 10):
                    self.switch_flag[i] = 0
                    print("agent" + str(i+1) + " back to RL")

        global_state = []  # [(local state, cloud queue), (local state, cloud queue), ... , (local state, cloud queue)]
        rewards = []
        all_local_states = []
        print('\n')
        for i in range(self.agent_num):
            state = [all_offloading_queue_size[i],   # 归一化
                     all_inference_queue_size[i],
                     all_bandwidth[i]/25.0
                     ]
            state.extend(actions_old[i])
            all_local_states.append(state)
            global_state_part = state[:]
            global_state_part.append(cloud_transmission_queue_size[i])  # 云端多一个 listen queue
            global_state.append(global_state_part)
            reward = self._get_reward(i, all_AVG_latency[i], all_acc[i], all_gap_acc[i])
            rewards.append(reward)
            print("Agent" + str(i+1) + "  " + "edge offload:" + str(all_offloading_queue_size[i]) + ",  " + "edge inference:" + str(all_inference_queue_size[i]) + ",  " + "latency:" + str(all_AVG_latency[i]) + ",  " + "acc:" +str(all_acc[i]) + ",  " + "queue wait:" + str(all_queue_wait[i]) + ",  " + "bandwidth:" + str(all_bandwidth[i]) + ",  " + "all_throughput:" + str(all_throughput[i]) + '\n'
                 + "        " + str(state[3]) + ": " + str(offload_rate[i]) + ", ", str(state[4]) + ": " + str(resolution[i]) + ", ", str(state[5]) + ": " + str(bitrate[i]) + ", ", str(state[6]) + ": " + str(edge_model[i]) + ",  " + "reward: " + str(reward))

            one_step_videoID_chunkID = []
            for j in range(len(all_videoID_list[i])):
                one_step_videoID_chunkID.append((all_videoID_list[i][j], all_chunkID_list[i][j]))
            self.immediate_videoID_chunkID[i].append(one_step_videoID_chunkID)

        print("cloud queue : ", cloud_transmission_queue_size[0])
        if not self.reward_seperated:
            avg_reward = np.mean(rewards)
            rewards = [avg_reward] * self.agent_num
        print("reward:  ", rewards)
        print('\n')
        return np.array(global_state, dtype=np.float32), np.array(all_local_states, dtype=np.float32), rewards, all_AVG_latency, all_queue_wait, all_acc, all_throughput, all_average_latency_cloud_edge, all_average_qwait_cloud_edge, all_gap_acc, cloud_transmission_queue_size[0]

    def _get_state(self, actions,step):
        all_inference_queue_size, all_offloading_queue_size = self._get_edge_queue_size(self.agent_num)
        cloud_transmission_queue_size = self._get_cloud_queue_size(self.agent_num)

        all_bandwidth = self._get_bandwidth(self.agent_num)
        all_bandwidth = [round(x, 4) for x in all_bandwidth]

        all_AVG_latency, all_average_latency_cloud_edge = self._get_system_average_latency(self.agent_num, step)
        all_AVG_latency = [round(x, 4) for x in all_AVG_latency]
        all_average_latency_cloud_edge = [[round(x[0], 4), round(x[1], 4)] for x in all_average_latency_cloud_edge]

        actions_old = actions

        global_state = []  # [(local state, cloud queue), (local state, cloud queue), ... , (local state, cloud queue)]
        all_local_states = []
        rewards = []
        for i in range(self.agent_num):
            state = [all_offloading_queue_size[i],
                     all_inference_queue_size[i],
                     all_bandwidth[i] / 25.0
                     ]
            state.extend(actions_old[i])
            all_local_states.append(state)
            global_state_part = state[:]
            global_state_part.append(cloud_transmission_queue_size[i])  # 云端多一个 listen queue
            global_state.append(global_state_part)

        return np.array(global_state, dtype=np.float32), np.array(all_local_states, dtype=np.float32), all_AVG_latency, all_average_latency_cloud_edge, cloud_transmission_queue_size[0]


    def _get_reward(self, agent_id, system_average_latency, acc, gap_acc):
        if system_average_latency <= self.T:
            reward = np.float32(-1*self.acc_weight*gap_acc - self.w*abs(system_average_latency - self.lt_target))
        else:
            reward = self.F * self.w - (system_average_latency - self.T)
        return reward

    def _get_delayed_feedback(self, start_step, end_step, aux_flag):
        delayed_reward = [[] for _ in range(self.agent_num)]
        delayed_videoID_chunkID = []
        start = time.time()
        all_acc, all_lt, all_gap_acc, all_videoID, all_chunkID = self._get_steps_lt_acc(self.agent_num, start_step, end_step, aux_flag)
        for agent_id in range(self.agent_num):
            for step in range(len(all_acc[0])):
                delayed_reward[agent_id].append(self._get_reward(agent_id, all_lt[agent_id][step], all_acc[agent_id][step], all_gap_acc[agent_id][step]))  # 这里出问题了

        for agent_id in range(self.agent_num):
            one_agent_delayed_videoID_chunkID = [[] for _ in range(len(all_videoID[agent_id]))]
            for step in range(len(all_videoID[agent_id])):
                for j in range(len(all_videoID[agent_id][step])):
                    one_agent_delayed_videoID_chunkID[step].append((all_videoID[agent_id][step][j], all_chunkID[agent_id][step][j]))
            delayed_videoID_chunkID.append(one_agent_delayed_videoID_chunkID)    # [[agent1[step1(v1,c1),(v2,c2)], [step2], ... , [stepN]],[agent2[step1], [step2], ... , [stepN]]]
        return delayed_reward, delayed_videoID_chunkID, all_acc, all_lt, all_gap_acc

if __name__ == "__main__":
    env = Env()
    state_initial = env.reset()
