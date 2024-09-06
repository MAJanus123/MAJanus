import argparse

import threading
import time
import sys

import torch
import tqdm
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='./exp.log', filemode='a')
logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT = str(ROOT).split("/")
ROOT = '/'.join(ROOT[0:-1])
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from RLEnv import Env
import MAPPO as MAPPO

seed = 42
torch.manual_seed(seed)
# 设置CUDA的随机种子
torch.cuda.manual_seed_all(seed)

def RL_train(**kwargs):
    def select_actions(local_states, all_average_latency_cloud_edge, trainer, buffer, cloud_queue):
        actions = []
        for agent_id in range(env.agent_num):
            local_state = local_states[agent_id]
            agent = trainer[0]
            memory = buffer[agent_id]
            edge_inference_queue = local_state[1]
            action = agent.select_actions(local_state, memory, all_average_latency_cloud_edge[agent_id],
                                          env.switch_flag[agent_id], cloud_queue, edge_inference_queue)
            # action = np.array([2,0,0,0])
            actions.append(action)
        return actions

    def insert_immediate_data(data):
        all_throughput, all_queue_wait, all_average_latency_cloud_edge, global_state, rewards, all_acc, all_AVG_latency = data
        epsiode_throughput.append(all_throughput)
        epsiode_queue_wait.append(all_queue_wait)
        epsiode_cloud_edge_latency.append(all_average_latency_cloud_edge)
        epsiode_global_state.append(global_state)
        epsiode_immediate_rewards.append(rewards)
        epsiode_immediate_acc.append(all_acc)
        epsiode_immediate_lt.append(all_AVG_latency)

    def aux_phase_update(env, copy_step, agent, big_buffer, aux_flag):
        start_time = time.time()
        index = copy_step - max_phase_episode * every_epsiode + 1
        start_step = index if index >= 1 else 1
        aux_delayed_reward, aux_delayed_videoID_chunkID, aux_all_acc, aux_all_lt, aux_all_gap_acc = env._get_delayed_feedback(
            start_step, copy_step, True)  # 修正多个episode
        agent.aux_update(aux_delayed_reward, big_buffer, start_step, copy_step, aux_flag)
        print("aux_phase time===", time.time() - start_time)

    rl_betas = (0.9, 0.999)
    rl_gamma = 0  # discount factor
    K_epochs = 50  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    every_epsiode = 13  # 每个episode的step数
    rl_lr = kwargs['rl_lr']  # parameters for Adam optimizer
    model_update = kwargs['model_update']  # Do you need to update the model
    agent_num = kwargs['agent_num']
    max_phase = kwargs['max_phase']  # max training phase
    max_phase_episode = kwargs['max_phase_episode']  # max training episodes in a phase
    model_load = kwargs['model_load']
    load_agent_list = kwargs['load_agent_list']
    save_path = kwargs['save_path']
    load_path = kwargs['load_path']
    monitor_path = kwargs['monitor_path']
    use_agent_ID = kwargs['use_agent_ID']
    action_mask = kwargs['action_mask']
    weight = kwargs['weight']
    reward_seperated = kwargs['reward_seperated']
    threshold = kwargs['threshold']
    lt_target = kwargs['lt_target']
    camera_num = kwargs['camera_num']
    aux_phase = kwargs['aux_phase']
    use_gae = kwargs['use_gae']
    actor_update = kwargs['actor']
    critic_update = kwargs['critic']
    auxiliary_epoch = kwargs['auxiliary_epoch']
    delayed_reward_flag = kwargs['delayed_reward']
    acc_weight = kwargs['acc_weight']
    stream = kwargs['stream']
    mini_batch = kwargs['mini_batch']
    lr2 = kwargs['lr2']   # PPG的学习率
    eliminate = kwargs['eliminate']   # 是否剔除其它step的经验
    sample_way = kwargs['sample_way']
    update_buffer = kwargs['update_buffer']   # 多久更新一次大buffer
    queue_threshold = kwargs['queue_threshold']
    if queue_threshold == 0:  # 最低限制
        cloud_queue_threshold = 5
        edge_queue_threshold = 1
    elif queue_threshold == 1:  # 中等限制
        cloud_queue_threshold = 10
        edge_queue_threshold = 1.5
    else:  #  最大限制
        cloud_queue_threshold = 20
        edge_queue_threshold = 3

    # Creating environment
    env = Env(agent_num, monitor_path, weight, reward_seperated, threshold, lt_target, camera_num, acc_weight, eliminate)

    device = 'cpu'
    # Creating PPO agent
    actor_state_dim = env.actor_state_dim
    critic_state_dim = env.critic_state_dim
    action_dim = env.action_dim
    buffer = []  # store memory
    trainer = []  # store agents
    big_buffer = MAPPO.big_buffer()
    agent = MAPPO.MAPPO(actor_state_dim, critic_state_dim, action_dim, rl_lr, rl_betas, rl_gamma, K_epochs,
                        eps_clip, agent_num, model_update, model_load, load_agent_list, load_path, threshold, aux_phase,
                        use_gae, actor_update, critic_update, auxiliary_epoch, delayed_reward_flag, mini_batch, lr2,save_path, sample_way, cloud_queue_threshold, edge_queue_threshold)
    for agent_id in range(agent_num):
        memory = MAPPO.Memory()
        buffer.append(memory)
        trainer.append(agent)

    # RL training
    logger.debug('==> RL Training Start.')

    # 初始化变量
    auxiliary_phase_flag = False  # 该轮次更新是否要使用Aux
    episode = 1
    local_states = []
    all_average_latency_cloud_edge = []
    all_global_states = []  # Store Global States for a trajectory
    pre_global_state = None
    pre_loacl_state = None
    delayed_reward = []
    aux_delayed_reward = []

    epsiode_throughput = []
    epsiode_queue_wait = []
    epsiode_cloud_edge_latency = []
    epsiode_global_state = []
    epsiode_immediate_rewards = []
    epsiode_immediate_acc = []
    epsiode_immediate_lt = []

    flag = False  # 是否需要启动edge的load

    for step in tqdm.tqdm(range(0, max_phase * max_phase_episode * every_epsiode + 1)):
        done = False  # Flag controling finish of one episode
        if step == 0:
            first = True
            actions = env.reset(done, first, env.agent_num, use_agent_ID)
            time.sleep(env.time_interval)
            global_state, local_states, all_AVG_latency, all_average_latency_cloud_edge, cloud_queue = env._get_state(actions, step)
            pre_global_state = torch.tensor(global_state.copy())
            pre_loacl_state = local_states.copy()
        else:
            actions = select_actions(local_states, all_average_latency_cloud_edge, trainer, buffer, cloud_queue)

            # 是否是更新完的第一步，需要启动edge的load
            if stream == False and step != 1 and step % every_epsiode == 1:
                flag = True
            else:
                flag = False
            global_state, local_states, rewards, all_AVG_latency, all_queue_wait, all_acc, all_throughput, all_average_latency_cloud_edge, all_average_qwait_cloud_edge, all_gap_acc, cloud_queue = env.step(
                actions, step, episode, action_mask, flag)
            data = all_throughput, all_queue_wait, all_average_latency_cloud_edge, global_state, rewards, all_acc, all_AVG_latency
            insert_immediate_data(data)

            for i in range(env.agent_num):
                buffer[i].rewards.append(rewards[i])
            all_global_states.append(torch.tensor(global_state.copy()))  # Storing global state in Tensor format

            # UPDATE
            if step % every_epsiode == 0:
                if not stream:
                    # kill
                    env.kill()
                    time.sleep(30)
                    while True:  # 确保所有edge都kill了
                        if env.start_flag:
                            env.start_flag = False
                            break
                        time.sleep(0.1)

                start_time = time.time()
                delayed_reward, delayed_videoID_chunkID, all_acc, all_lt, all_gap_acc = env._get_delayed_feedback(
                    step - every_epsiode + 1, step, False)

                print("=================update MAPPO==================")
                all_global_states.insert(0, pre_global_state)
                pre_global_state = all_global_states[-1]
                for agent_id in range(agent_num):
                    buffer[agent_id].states.insert(0, torch.FloatTensor(pre_loacl_state[agent_id].reshape(1, -1)))
                pre_loacl_state = local_states

                trainer[0].update(buffer, all_global_states, delayed_reward, big_buffer)

                copy_step = step
                if aux_phase and episode % update_buffer == 0:  # 触发AUX
                    print(episode / max_phase_episode, "=================PPG==================")
                    if episode % max_phase_episode == 0:
                        aux_flag = True
                    else:
                        aux_flag = False
                    aux_thread = threading.Thread(target=aux_phase_update,args=(env, copy_step, trainer[0], big_buffer, aux_flag), daemon=True)
                    aux_thread.start()

                all_global_states = []  # clean
                for agent_id in range(agent_num):
                    buffer[agent_id].clear_memory()

                # 存log  13个step存一次
                for agent_id in range(env.agent_num):
                    for step_id in range(len(epsiode_throughput)):
                        log_content = str(delayed_reward[agent_id][step_id]) + ',  ' + str(
                            all_lt[agent_id][step_id]) + ',  ' + str(all_acc[agent_id][step_id]) + \
                                      ',  ' + str(all_gap_acc[agent_id][step_id]) + ',  ' + str(
                            epsiode_throughput[step_id][agent_id]) \
                                      + ',  ' + str(epsiode_queue_wait[step_id][agent_id]) + ',  ' + str(
                            epsiode_cloud_edge_latency[step_id][agent_id][0]) \
                                      + ',  ' + str(epsiode_cloud_edge_latency[step_id][agent_id][1]) + ',  ' + str(
                            epsiode_global_state[step_id][agent_id][0]) \
                                      + ',  ' + str(epsiode_global_state[step_id][agent_id][1]) + ',  ' + str(
                            epsiode_global_state[step_id][agent_id][2]) \
                                      + ',  ' + str(epsiode_global_state[step_id][agent_id][7]) + ',  ' + str(
                            epsiode_immediate_rewards[step_id][agent_id]) \
                                      + ',  ' + str(epsiode_immediate_acc[step_id][agent_id]) + ',  ' + str(
                            epsiode_immediate_lt[step_id][agent_id]) \
                                      + ',  ' + str(epsiode_global_state[step_id][agent_id][3]) + ',  ' + str(
                            epsiode_global_state[step_id][agent_id][4]) \
                                      + ',  ' + str(epsiode_global_state[step_id][agent_id][5]) + ',  ' + str(
                            epsiode_global_state[step_id][agent_id][6]) + '\n'

                        with open(
                                "/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/experiment/" + save_path + str(
                                    agent_id + 1) + ".txt", "a") as file:
                            file.write(log_content)

                    content = str(delayed_videoID_chunkID[agent_id]) + '\n' + str(env.immediate_videoID_chunkID[agent_id]) + '\n'
                    with open(
                            "/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/verify/" + save_path + str(
                                agent_id + 1) + ".txt", "a") as file:
                        file.write(content)
                env.immediate_videoID_chunkID = [[] for _ in range(agent_num)]

                epsiode_throughput = []
                epsiode_queue_wait = []
                epsiode_cloud_edge_latency = []
                epsiode_global_state = []
                epsiode_immediate_rewards = []
                epsiode_immediate_acc = []
                epsiode_immediate_lt = []

                episode += 1
                print("update time=====", time.time() - start_time)

    print("switch num", env.switch_num)
    env.kill()
    time.sleep(30)
    while True:  # 确保所有edge都kill了
        if env.start_flag:
            env.start_flag = False
            break
        time.sleep(1)


if __name__ == '__main__':
    '''
    model_load: Do you need to load the trained model
    model_update: Do you need to update the RL model
    load_agent_list: Which devices need to load the model, input the agent ID
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_seq", required=False, default=1, type=int,
                        help="experiment sequence")
    args = parser.parse_args()

    if args.experiment_seq == 1:
        RL_train(agent_num=6, camera_num=1, max_phase=50, max_phase_episode=5, rl_lr=0.0001, lr2=0.00001, weight=0.1, threshold=2, queue_threshold=1,
                 lt_target=1,
                 reward_seperated=True, model_load=False,
                 model_update=True, load_agent_list=[],
                 save_path='8.30_num6_w0.1_T2_ablation1_PPG_randomSample780_nosafety_load_small_multidynamic_agent',
                 load_path='',
                 monitor_path='/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/experiment/7.31_num8_w0.1_T1.5_aw1_target1_phase40_episode5_shared_PP0_actor10_critic10_seed42_immediate_reward_df0.9_nosafety_stream_allmax_acc_agent',
                 use_agent_ID=[1, 2, 3, 4, 5, 6], action_mask=False, aux_phase=True, use_gae=False, actor=10,
                 critic=10, auxiliary_epoch=10,
                 delayed_reward=True, acc_weight=1, stream=True, mini_batch=780, eliminate=True, sample_way=0, update_buffer=1)
    elif args.experiment_seq == 2:
        RL_train(agent_num=4, camera_num=1, max_phase=50, max_phase_episode=5, rl_lr=0.0001, lr2=0.00001, weight=0.2, threshold=2, queue_threshold=1,
                 lt_target=1,
                 reward_seperated=True, model_load=False,
                 model_update=True, load_agent_list=[],
                 save_path='8.28_num4_w0.2_T2_tradeoff_PPG_randomSample780_load2_agent',
                 load_path='',
                 monitor_path='/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/experiment/7.31_num8_w0.1_T1.5_aw1_target1_phase40_episode5_shared_PP0_actor10_critic10_seed42_immediate_reward_df0.9_nosafety_stream_allmax_acc_agent',
                 use_agent_ID=[1, 2, 3, 4], action_mask=True, aux_phase=True, use_gae=False, actor=10,
                 critic=10, auxiliary_epoch=10,
                 delayed_reward=True, acc_weight=1, stream=True, mini_batch=780, eliminate=True, sample_way=0, update_buffer=5)
    elif args.experiment_seq == 3:
        RL_train(agent_num=4, camera_num=1, max_phase=30, max_phase_episode=5, rl_lr=0.0001, lr2=0.00001, weight=0.8, threshold=2, queue_threshold=1,
                 lt_target=1,
                 reward_seperated=True, model_load=False,
                 model_update=True, load_agent_list=[],
                 save_path='8.28_num4_w0.8_T2_tradeoff_PPG_randomSample780_load2_agent',
                 load_path='',
                 monitor_path='/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/experiment/7.31_num8_w0.1_T1.5_aw1_target1_phase40_episode5_shared_PP0_actor10_critic10_seed42_immediate_reward_df0.9_nosafety_stream_allmax_acc_agent',
                 use_agent_ID=[1, 2, 3, 4], action_mask=True, aux_phase=True, use_gae=False, actor=10,
                 critic=10, auxiliary_epoch=10,
                 delayed_reward=True, acc_weight=1, stream=True, mini_batch=780, eliminate=True, sample_way=0, update_buffer=5)
    elif args.experiment_seq == 4:
        RL_train(agent_num=4, camera_num=1, max_phase=30, max_phase_episode=5, rl_lr=0.0001, lr2=0.00001, weight=1.5, threshold=2, queue_threshold=1,
                 lt_target=1,
                 reward_seperated=True, model_load=False,
                 model_update=True, load_agent_list=[],
                 save_path='8.28_num4_w1.5_T2_tradeoff_PPG_randomSample780_load2_agent',
                 load_path='',
                 monitor_path='/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/experiment/7.31_num8_w0.1_T1.5_aw1_target1_phase40_episode5_shared_PP0_actor10_critic10_seed42_immediate_reward_df0.9_nosafety_stream_allmax_acc_agent',
                 use_agent_ID=[1, 2, 3, 4], action_mask=True, aux_phase=True, use_gae=False, actor=10,
                 critic=10, auxiliary_epoch=10,
                 delayed_reward=True, acc_weight=1, stream=True, mini_batch=780, eliminate=True, sample_way=0, update_buffer=5)
    elif args.experiment_seq == 5:
        RL_train(agent_num=2, camera_num=1, max_phase=20, max_phase_episode=5, rl_lr=0.0001, lr2=0.00001, weight=0.1,
                 threshold=1.5,queue_threshold=1,
                 lt_target=1,
                 reward_seperated=True, model_load=False,
                 model_update=True, load_agent_list=[],
                 save_path='8.21_num2_w0.1_T1.5_aw1_target1_phase20_episode5_shared_PPG5_actor10_critic10_df0_multiProcess_randomSample390_load1_agent',
                 load_path='',
                 monitor_path='/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/experiment/7.31_num8_w0.1_T1.5_aw1_target1_phase40_episode5_shared_PP0_actor10_critic10_seed42_immediate_reward_df0.9_nosafety_stream_allmax_acc_agent',
                 use_agent_ID=[1, 2, 3, 4], action_mask=True, aux_phase=True, use_gae=False, actor=10,
                 critic=10, auxiliary_epoch=10,
                 delayed_reward=True, acc_weight=1, stream=True, mini_batch=390, eliminate=True, sample_way=0,
                 update_buffer=5)
    elif args.experiment_seq == 6:
        RL_train(agent_num=2, camera_num=1, max_phase=30, max_phase_episode=5, rl_lr=0.0001, lr2=0.00001, weight=0.1,
                 threshold=2,queue_threshold=1,
                 lt_target=1,
                 reward_seperated=True, model_load=False,
                 model_update=True, load_agent_list=[],
                 save_path='8.21_num2_w0.1_T1.5_aw1_target1_phase30_episode5_shared_PPG5_actor10_critic10_df0_multiProcess_randomSample390_load_dynamic_agent',
                 load_path='',
                 monitor_path='/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/experiment/7.31_num8_w0.1_T1.5_aw1_target1_phase40_episode5_shared_PP0_actor10_critic10_seed42_immediate_reward_df0.9_nosafety_stream_allmax_acc_agent',
                 use_agent_ID=[1, 2, 3, 4], action_mask=True, aux_phase=True, use_gae=False, actor=10,
                 critic=10, auxiliary_epoch=10,
                 delayed_reward=True, acc_weight=1, stream=True, mini_batch=390, eliminate=True, sample_way=0,
                 update_buffer=5)
    elif args.experiment_seq == 7:  # MAPPO
        RL_train(agent_num=6, camera_num=1, max_phase=30, max_phase_episode=5, rl_lr=0.0001, lr2=0.00001, weight=0.1, threshold=2,queue_threshold=1,
                 lt_target=1,
                 reward_seperated=True, model_load=False,
                 model_update=True, load_agent_list=[],
                 save_path='8.24_num6_w0.1_T2_target1_phase30_episode5_shared_MAPPO_load2_agent',
                 load_path='',
                 monitor_path='/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/experiment/7.31_num8_w0.1_T1.5_aw1_target1_phase40_episode5_shared_PP0_actor10_critic10_seed42_immediate_reward_df0.9_nosafety_stream_allmax_acc_agent',
                 use_agent_ID=[1, 2, 3, 4, 5, 6], action_mask=False, aux_phase=False, use_gae=False, actor=10,
                 critic=10, auxiliary_epoch=10,
                 delayed_reward=False, acc_weight=1, stream=True, mini_batch=780, eliminate=False, sample_way=0, update_buffer=5)
    elif args.experiment_seq == 8:  # MAPPO
        RL_train(agent_num=4, camera_num=1, max_phase=20, max_phase_episode=5, rl_lr=0.0001, lr2=0.00001, weight=0.1, threshold=1.5,queue_threshold=1,
                 lt_target=1,
                 reward_seperated=True, model_load=False,
                 model_update=True, load_agent_list=[],
                 save_path='8.23_num4_w0.1_T1.5_aw1_target1_phase20_episode5_shared_MAPPO_load1_agent',
                 load_path='',
                 monitor_path='/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/experiment/7.31_num8_w0.1_T1.5_aw1_target1_phase40_episode5_shared_PP0_actor10_critic10_seed42_immediate_reward_df0.9_nosafety_stream_allmax_acc_agent',
                 use_agent_ID=[1, 2, 3, 4], action_mask=False, aux_phase=False, use_gae=False, actor=10,
                 critic=10, auxiliary_epoch=10,
                 delayed_reward=False, acc_weight=1, stream=True, mini_batch=780, eliminate=False, sample_way=0, update_buffer=5)