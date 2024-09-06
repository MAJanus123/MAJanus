import torch
import random
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import torch.distributions as dist
from res.MultiHeadSelfAttention import MultiHeadAttention, PositionalEncoding
import logging
from res.utils import find_closest_index, huber_loss
from res.valuenorm import ValueNorm

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='./exp.log', filemode='a')
logger = logging.getLogger(__name__)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_switch = []  # 是否使用switch机制的backup policy
        self.available_actions_all = [[], [], [], []]  # 存储四个动作的available_actions

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_switch[:]
        for i in range(4):
            self.available_actions_all[i] = []


class big_buffer:
    def __init__(self):
        self.returns = []
        self.local_states = []
        self.global_states = []

    def clear_memory(self):
        del self.returns[:]
        del self.local_states[:]
        del self.global_states[:]


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
                .log_prob(actions.squeeze(-1))
                .view(actions.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
        )


class Heads(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Heads, self).__init__()
        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, x, available_actions=None, detach=False):
        x = self.linear(x)
        if detach:
            x = x.detach()
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)  # 返回给action_out(actor_features)


class ActorCritic(nn.Module):
    def __init__(self, actor_state_dim, critic_state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor_backbone = nn.Sequential(
            nn.Linear(actor_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )

        # 初始化
        self.action_outs = []
        self.action_outs.append(Heads(32, 11))  # offload_rate
        self.action_outs.append(Heads(32, 3))  # resolution
        self.action_outs.append(Heads(32, 3))  # bitrate
        self.action_outs.append(Heads(32, 3))  # model
        self.action_outs = nn.ModuleList(self.action_outs)

        self.auxiliary_value_head = nn.Linear(32, 1)

        # critic
        self.critic = nn.Sequential(
            nn.Linear(critic_state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.offload_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.resolution_list = [0, 0.5, 1]
        self.bitrate_list = [0, 0.5, 1]
        self.model_list = [0, 0.5, 1]
        self.all_actions = [self.offload_list, self.resolution_list, self.bitrate_list, self.model_list]

        self.actor_state_dim = actor_state_dim
        self.critic_state_dim = critic_state_dim
        self.action_dim = action_dim

    def forward(self):
        raise NotImplementedError

    def forward_critic(self, critic_input):
        input = critic_input.reshape(critic_input.shape[0], -1)
        output = self.critic(input)
        return output

    def forward_actor(self, actor_features, available_actions):
        actions = []
        action_log_probs = []
        for index, action_out in enumerate(self.action_outs):
            if available_actions is None:
                action_logit = action_out(actor_features, available_actions)
            else:
                action_logit = action_out(actor_features, (available_actions[index]).view(1, -1))
            action = action_logit.sample()
            action_log_prob = action_logit.log_probs(action)
            actions.append(action)
            action_log_probs.append(action_log_prob)

        actions = torch.cat(actions, -1)  # (1,4)
        action_log_probs = torch.cat(action_log_probs, -1)  # (1,4)
        # action_log_probs = action_log_probs.sum(dim=1)
        return actions, action_log_probs

    def forward_auxiliary(self, local_states):
        output = self.actor_backbone(local_states)
        output = self.auxiliary_value_head(output)
        return output

    def act(self, state, memory, available_actions=None):
        actor_features = self.actor_backbone(state)
        action, action_logprob = self.forward_actor(actor_features, available_actions)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action

    def evaluate_policy(self, state, action, available_actions_all):
        actor_features = self.actor_backbone(state)
        batch = action.shape[0]
        action = torch.transpose(action, 0, 1)
        action_log_probs = []
        dist_entropy = []
        for action_out, act, available_actions in zip(self.action_outs, action, available_actions_all):
            action_logit = action_out(actor_features, torch.stack(available_actions).view(batch, -1))
            action_log_probs.append(action_logit.log_probs(act))
            dist_entropy.append(action_logit.entropy().mean())

        action_log_probs = torch.cat(action_log_probs, -1)
        # action_log_probs = action_log_probs.sum(dim=1)
        dist_entropy = sum(dist_entropy) / len(dist_entropy)
        return action_log_probs, dist_entropy

    def evaluate_value(self, global_state):
        state_value = self.forward_critic(global_state)
        return torch.squeeze(state_value)

    def evaluate_auxiliary(self, local_states):
        auxiliary_state_values = self.forward_auxiliary(local_states)
        return torch.squeeze(auxiliary_state_values)


class MAPPO:
    def __init__(self, actor_state_dim, critic_state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip,
                 agent_num, model_update, model_load, load_agent_list, load_path, threshold, aux_phase, use_gae, actor_update, critic_update,
                 auxiliary_epoch, delayed_reward_flag,mini_batch,lr2,save_path, sample_way, cloud_queue_threshold, edge_queue_threshold):

        self.lr = lr
        self.lr2 = lr2
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.threshold = threshold
        self.agent_num = agent_num
        self.huber_delta = 10.0
        self.gae_lambda = 0.95
        self.aux_phase = aux_phase
        self.use_gae = use_gae
        self.actor_update = actor_update
        self.critic_update = critic_update
        self.auxiliary_epoch = auxiliary_epoch
        self.delayed_reward_flag = delayed_reward_flag
        self.value_normalizer = ValueNorm(1).to(device)
        self.mini_batch = mini_batch
        self.save_path = save_path
        self.sample_way = sample_way
        self.cloud_queue_threshold = cloud_queue_threshold
        self.edge_queue_threshold = edge_queue_threshold

        self.policy = ActorCritic(actor_state_dim, critic_state_dim, action_dim).to(device)
        self.optimizer_actor = torch.optim.Adam([{'params': self.policy.actor_backbone.parameters()},
                                                 {'params': self.policy.action_outs.parameters()},
                                                 ], lr=lr, betas=betas)
        self.optimizer_critic = torch.optim.Adam(
            [{'params': self.policy.critic.parameters()}], lr=lr,
            betas=betas)

        self.optimizer_auxiliary = torch.optim.Adam([{'params': self.policy.actor_backbone.parameters()},
                                                     {'params': self.policy.action_outs.parameters()},
                                                     {'params': self.policy.auxiliary_value_head.parameters()},
                                                     ], lr=lr2, betas=betas)

        if model_load:  # 需要加载已经训练的模型
            PATH_TO_PTH_FILE = '/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/RL_model/' + load_path + '.pth'
            self.policy.load_state_dict(torch.load(PATH_TO_PTH_FILE))

        self.policy_old = ActorCritic(actor_state_dim, critic_state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()


    def select_actions(self, state, memory, average_latency_cloud_edge, is_switch, cloud_queue, edge_inference_queue):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  # (1,7)

        available_actions_all = [torch.ones(11), torch.ones(3), torch.ones(3),
                                 torch.ones(3)]  # rate resolution bitrate model
        if is_switch:
            if cloud_queue > self.cloud_queue_threshold and edge_inference_queue > self.edge_queue_threshold:  # 都炸了
                offload_rate = edge_inference_queue * self.agent_num / 2 / (cloud_queue + edge_inference_queue * self.agent_num)
                offload_rate_index = find_closest_index(offload_rate, self.policy_old.offload_list)
                available_actions_all[0][:offload_rate_index] = 0
                available_actions_all[0][offload_rate_index + 1:] = 0
                # 最小配置
                available_actions_all[1][1:] = 0
                available_actions_all[2][1:] = 0
                available_actions_all[3][1:] = 0
            elif cloud_queue > self.cloud_queue_threshold and edge_inference_queue <= self.edge_queue_threshold:   # 云炸
                available_actions_all[0][3:] = 0  # offload rate = 0 or 0.1 or 0.2
                if self.agent_num >= 6:
                    # 小配置
                    available_actions_all[1][1:] = 0
                    available_actions_all[2][1:] = 0
                    available_actions_all[3][1:] = 0
                else:
                    # 中小配置
                    available_actions_all[1][2] = 0
                    available_actions_all[2][2] = 0
                    available_actions_all[3][2] = 0
            elif cloud_queue <= self.cloud_queue_threshold and edge_inference_queue > self.edge_queue_threshold:   # 边炸
                if self.agent_num >= 6:   # offload rate = 0.4 or 0.5
                    available_actions_all[0][:4] = 0
                    available_actions_all[0][6:] = 0
                    # 小配置
                    available_actions_all[1][1:] = 0
                    available_actions_all[2][1:] = 0
                    available_actions_all[3][1:] = 0
                else:
                    available_actions_all[0][:-3] = 0  # offload rate = 0.8 or 0.9 or 1
                    # 中小配置
                    available_actions_all[1][2] = 0
                    available_actions_all[2][2] = 0
                    available_actions_all[3][2] = 0
            else:   # 两个队列都小，但是延迟超过阈值
                if average_latency_cloud_edge[0] + average_latency_cloud_edge[1] == 0:   # 说明边的传输队列堵了
                    available_actions_all[0][:4] = 0
                    available_actions_all[0][5:] = 0
                else:
                    offload_rate = average_latency_cloud_edge[1] / (
                            average_latency_cloud_edge[0] + average_latency_cloud_edge[1])
                    offload_rate_index = find_closest_index(offload_rate, self.policy_old.offload_list)
                    available_actions_all[0][:offload_rate_index] = 0
                    available_actions_all[0][offload_rate_index + 1:] = 0
                # 中小配置
                available_actions_all[1][2] = 0
                available_actions_all[2][2] = 0
                available_actions_all[3][2] = 0
            action = self.policy_old.act(state, memory, available_actions_all)

        else:
            action = self.policy_old.act(state, memory, available_actions_all)

            # action = self.policy_old.act(state, memory)
        memory.is_switch.append(is_switch)
        memory.available_actions_all[0].append(available_actions_all[0])
        memory.available_actions_all[1].append(available_actions_all[1])
        memory.available_actions_all[2].append(available_actions_all[2])
        memory.available_actions_all[3].append(available_actions_all[3])

        return action.cpu().data.numpy().flatten()

    def update(self, memory, all_global_state, delayed_feedback, big_buffer):
        # Monte Carlo estimate of rewards:
        returns = []
        old_states = []
        old_actions = []
        old_logprobs = []
        old_state_values = []
        global_states = []
        available_actions_all = [[], [], [], []]
        for i in range(self.agent_num):
            global_state_temp = torch.stack(all_global_state).to(device).detach()  # (13 + 1,N,8)
            old_state_values_temp = torch.squeeze(self.policy_old.forward_critic(global_state_temp).detach())  # (13,)

            # if self.delayed_reward_flag:
            #     rewards = delayed_feedback[i]
            # else:
            rewards = memory[i].rewards

            return_temp = []
            if self.use_gae:
                gae = 0
                for step in reversed(range(len(rewards))):
                    delta = rewards[step] + self.gamma * self.value_normalizer.denormalize(
                        old_state_values_temp[step + 1]) - self.value_normalizer.denormalize(
                        old_state_values_temp[step])
                    gae = delta + self.gamma * self.gae_lambda * gae
                    return_temp.insert(0, gae + self.value_normalizer.denormalize(old_state_values_temp[step]))
            else:
                discounted_reward = old_state_values_temp[-1].item()
                for reward in reversed(rewards):
                    discounted_reward = np.float32(reward + (self.gamma * discounted_reward))
                    return_temp.insert(0, discounted_reward)


            available_actions_all[0].extend(memory[i].available_actions_all[0])
            available_actions_all[1].extend(memory[i].available_actions_all[1])
            available_actions_all[2].extend(memory[i].available_actions_all[2])
            available_actions_all[3].extend(memory[i].available_actions_all[3])

            old_state_values.extend(old_state_values_temp[0:-1])
            returns.extend(return_temp)
            old_states.extend(memory[i].states[0:-1])
            old_actions.extend(memory[i].actions)
            old_logprobs.extend(memory[i].logprobs)
            global_states.extend(all_global_state[0:-1])

        if self.aux_phase:
            big_buffer.local_states.extend(old_states)  # local
            big_buffer.global_states.extend(global_states)
            big_buffer.returns.extend(returns)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(old_states).to(device),
                                   1).detach()  # (N*13,7) 将包含N个形状为(1, M)的张量的Python列表 转化为形状为[N,M]的张量
        old_actions = torch.squeeze(torch.stack(old_actions).to(device), 1).detach()  # (N*13,4)
        old_logprobs = torch.squeeze(torch.stack(old_logprobs), 1).to(device).detach()  # (N*13,)
        old_state_values = torch.tensor(old_state_values).detach()
        returns = torch.stack([torch.tensor(arr).squeeze() for arr in returns])
        global_states = torch.stack(global_states).to(device).detach()

        denormal_old_state_values = self.value_normalizer.denormalize(old_state_values)
        advantages = (returns - denormal_old_state_values).tolist()
        advantages_copy = advantages.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        advantages = torch.tensor(advantages).reshape(-1, 1).to(device)  # (12,1)

        # Optimize policy
        actor_update = self.actor_update
        critic_update = self.critic_update
        for _ in range(actor_update):
            # Evaluating old actions:
            logprobs, dist_entropy = self.policy.evaluate_policy(old_states, old_actions,
                                                                 available_actions_all)  # (N*12,)  (N*12,)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())  # (12,)

            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
            self.optimizer_actor.zero_grad()
            (actor_loss - dist_entropy * 0.01).backward()
            self.optimizer_actor.step()

        for _ in range(critic_update):
            state_values = self.policy.evaluate_value(global_states)
            # update critic
            value_pred_clipped = old_state_values + (state_values - old_state_values).clamp(-self.eps_clip,
                                                                                            self.eps_clip)
            self.value_normalizer.update(returns)
            error_clipped = self.value_normalizer.normalize(returns) - value_pred_clipped
            error_original = self.value_normalizer.normalize(returns) - state_values
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
            critic_loss = torch.max(value_loss_original, value_loss_clipped)
            # critic_loss.retain_grad()
            self.optimizer_critic.zero_grad()
            critic_loss.mean().backward()
            self.optimizer_critic.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        try:
            torch.save(self.policy.state_dict(),
                       '/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/RL_model/' + self.save_path + '.pth')
        except PermissionError:
            pass


    def aux_update(self,aux_delayed_reward, big_buffer, start_step, end_step, aux_flag):
        aux_returns = []
        for i in range(self.agent_num):
            aux_return_temp = []
            aux_df_reward = 0
            for reward in reversed(aux_delayed_reward[i]):
                aux_df_reward = np.float32(reward + (self.gamma * aux_df_reward))
                aux_return_temp.insert(0, aux_df_reward)
            aux_returns.extend(aux_return_temp)
        # 修正大buffer
        big_buffer.returns[(start_step-1)*self.agent_num: end_step*self.agent_num] = aux_returns[:]
        if aux_flag:
            # 更新模型
            print("========auxiliary phase==========")
            length = len(big_buffer.local_states)
            all_auxiliary_local_states = big_buffer.local_states
            all_auxiliary_returns = big_buffer.returns
            all_auxiliary_returns = [torch.tensor(arr).squeeze() for arr in all_auxiliary_returns]
            all_auxiliary_global_states = big_buffer.global_states
            # if length >= self.mini_batch:
            #     batch = int(length / self.mini_batch)
            #     if length - batch * self.mini_batch > 0:
            #         batch = batch+1
            #     print("length=", length, "batch=", batch)
            #     for b in range(batch):
            #         if b == batch - 1:
            #             auxiliary_local_states = all_auxiliary_local_states[b*self.mini_batch:]
            #             auxiliary_returns = all_auxiliary_returns[b*self.mini_batch:]
            #             auxiliary_global_states = all_auxiliary_global_states[b*self.mini_batch:]
            #         else:
            #             auxiliary_local_states = all_auxiliary_local_states[b*self.mini_batch: (b+1)*self.mini_batch]
            #             auxiliary_returns = all_auxiliary_returns[b*self.mini_batch: (b+1)*self.mini_batch]
            #             auxiliary_global_states = all_auxiliary_global_states[b*self.mini_batch: (b+1)*self.mini_batch]
            #
            #         auxiliary_local_states = torch.stack(auxiliary_local_states).detach()
            #         auxiliary_global_states = torch.stack(auxiliary_global_states).detach()
            #         auxiliary_returns = auxiliary_returns.detach()
            #
            #         auxiliary_policy_old_state_values = (self.policy.evaluate_auxiliary(auxiliary_local_states)).detach()
            #         auxiliary_value_old_state_values = (self.policy.evaluate_value(auxiliary_global_states)).detach()
            #         auxiliary_old_logits = []
            #         auxiliary_old_features = (self.policy.actor_backbone(auxiliary_local_states)).detach()  # backbone的features
            #         for index, action_out in enumerate(self.policy.action_outs):
            #             action_old_logit = action_out(auxiliary_old_features, detach=True)
            #             auxiliary_old_logits.append(action_old_logit)
            #
            #         for _ in range(self.auxiliary_epoch):
            #             # update value head and actor backbone
            #             auxiliary_logits = []
            #             auxiliary_features = self.policy.actor_backbone(auxiliary_local_states)
            #             for index, action_out in enumerate(self.policy.action_outs):
            #                 action_logit = action_out(auxiliary_features)
            #                 auxiliary_logits.append(action_logit)
            #
            #             auxiliary_policy_state_values = self.policy.evaluate_auxiliary(auxiliary_local_states)
            #             auxiliary_policy_pred_clipped = auxiliary_policy_old_state_values + (
            #                         auxiliary_policy_state_values - auxiliary_policy_old_state_values).clamp(-self.eps_clip, self.eps_clip)
            #             auxiliary_policy_error_clipped = self.value_normalizer.normalize(
            #                 auxiliary_returns) - auxiliary_policy_pred_clipped
            #             auxiliary_policy_error_original = self.value_normalizer.normalize(
            #                 auxiliary_returns) - auxiliary_policy_state_values
            #             auxiliary_policy_value_loss_clipped = huber_loss(auxiliary_policy_error_clipped, self.huber_delta)
            #             auxiliary_policy_value_loss_original = huber_loss(auxiliary_policy_error_original, self.huber_delta)
            #             auxiliary_policy_loss = torch.max(auxiliary_policy_value_loss_original, auxiliary_policy_value_loss_clipped)
            #             # 创建多项分布对象
            #             all_kl = []
            #             for i in range(4):
            #                 p = auxiliary_old_logits[i]
            #                 q = auxiliary_logits[i]
            #                 all_kl.append(dist.kl_divergence(p, q).mean())
            #             all_kl = torch.tensor(all_kl)
            #             kl_mean = torch.mean(all_kl)
            #             auxiliary_policy_loss = auxiliary_policy_loss.mean() + kl_mean
            #
            #             self.optimizer_auxiliary.zero_grad()
            #             auxiliary_policy_loss.mean().backward()
            #
            #             # for name, param in self.policy.named_parameters():
            #             #     # if param.grad is not None:
            #             #     print(f"{name} gradient: {param.grad}")
            #
            #             self.optimizer_auxiliary.step()
            #
            #             # 更新value
            #             auxiliary_value_state_values = self.policy.evaluate_value(auxiliary_global_states)
            #             auxiliary_value_pred_clipped = auxiliary_value_old_state_values + (
            #                         auxiliary_value_state_values - auxiliary_value_old_state_values).clamp(-self.eps_clip,
            #                                                                                                self.eps_clip)
            #             auxiliary_value_error_clipped = self.value_normalizer.normalize(
            #                 auxiliary_returns) - auxiliary_value_pred_clipped
            #             auxiliary_value_error_original = self.value_normalizer.normalize(
            #                 auxiliary_returns) - auxiliary_value_state_values
            #             auxiliary_value_value_loss_clipped = huber_loss(auxiliary_value_error_clipped, self.huber_delta)
            #             auxiliary_value_value_loss_original = huber_loss(auxiliary_value_error_original, self.huber_delta)
            #             auxiliary_value_loss = torch.max(auxiliary_value_value_loss_original,
            #                                              auxiliary_value_value_loss_clipped)
            #             self.optimizer_critic.zero_grad()
            #             auxiliary_value_loss.mean().backward()
            #
            #             self.optimizer_critic.step()
            #
            #     # Copy new weights into old policy:
            #     self.policy_old.load_state_dict(self.policy.state_dict())
            #     try:
            #         torch.save(self.policy.state_dict(),
            #                    '/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/RL_model/' + self.save_path + '.pth')
            #     except PermissionError:
            #         pass

            # 随机sample  取前五个episode的经验
            # if length >= self.mini_batch:
            num = self.mini_batch
            if length < self.mini_batch:
                num = length
            if self.sample_way == 0:
                idx = np.random.randint(0, length, num)
                auxiliary_local_states = [all_auxiliary_local_states[i] for i in idx]
                auxiliary_global_states = [all_auxiliary_global_states[i] for i in idx]
                auxiliary_returns = [all_auxiliary_returns[i] for i in idx]
            elif self.sample_way == 1:
                auxiliary_local_states = all_auxiliary_local_states[-1*num:]
                auxiliary_global_states = all_auxiliary_global_states[-1*num:]
                auxiliary_returns = all_auxiliary_returns[-1*num:]
            else:
                if num == self.mini_batch:
                    num = int(num/2)
                idx = np.random.randint(0, length, num)
                auxiliary_local_states = [all_auxiliary_local_states[i] for i in idx]
                auxiliary_global_states = [all_auxiliary_global_states[i] for i in idx]
                auxiliary_returns = [all_auxiliary_returns[i] for i in idx]
                auxiliary_local_states.extend(all_auxiliary_local_states[-1*num:])
                auxiliary_global_states.extend(all_auxiliary_global_states[-1*num:])
                auxiliary_returns.extend(all_auxiliary_returns[-1*num:])


            auxiliary_local_states = torch.stack(auxiliary_local_states).detach()
            auxiliary_global_states = torch.stack(auxiliary_global_states).detach()
            auxiliary_returns = torch.stack(auxiliary_returns).detach()

            auxiliary_policy_old_state_values = (self.policy.evaluate_auxiliary(auxiliary_local_states)).detach()
            auxiliary_value_old_state_values = (self.policy.evaluate_value(auxiliary_global_states)).detach()
            auxiliary_old_logits = []
            auxiliary_old_features = (self.policy.actor_backbone(auxiliary_local_states)).detach()  # backbone的features
            for index, action_out in enumerate(self.policy.action_outs):
                action_old_logit = action_out(auxiliary_old_features, detach=True)
                auxiliary_old_logits.append(action_old_logit)

            for _ in range(self.auxiliary_epoch):
                # update value head and actor backbone
                auxiliary_logits = []
                auxiliary_features = self.policy.actor_backbone(auxiliary_local_states)
                for index, action_out in enumerate(self.policy.action_outs):
                    action_logit = action_out(auxiliary_features)
                    auxiliary_logits.append(action_logit)

                auxiliary_policy_state_values = self.policy.evaluate_auxiliary(auxiliary_local_states)
                auxiliary_policy_pred_clipped = auxiliary_policy_old_state_values + (
                            auxiliary_policy_state_values - auxiliary_policy_old_state_values).clamp(-self.eps_clip, self.eps_clip)
                auxiliary_policy_error_clipped = self.value_normalizer.normalize(
                    auxiliary_returns) - auxiliary_policy_pred_clipped
                auxiliary_policy_error_original = self.value_normalizer.normalize(
                    auxiliary_returns) - auxiliary_policy_state_values
                auxiliary_policy_value_loss_clipped = huber_loss(auxiliary_policy_error_clipped, self.huber_delta)
                auxiliary_policy_value_loss_original = huber_loss(auxiliary_policy_error_original, self.huber_delta)
                auxiliary_policy_loss = torch.max(auxiliary_policy_value_loss_original, auxiliary_policy_value_loss_clipped)
                # 创建多项分布对象
                all_kl = []
                for i in range(4):
                    p = auxiliary_old_logits[i]
                    q = auxiliary_logits[i]
                    all_kl.append(dist.kl_divergence(p, q).mean())
                all_kl = torch.tensor(all_kl)
                kl_mean = torch.mean(all_kl)
                auxiliary_policy_loss = auxiliary_policy_loss.mean() + kl_mean

                self.optimizer_auxiliary.zero_grad()
                auxiliary_policy_loss.mean().backward()

                # for name, param in self.policy.named_parameters():
                #     # if param.grad is not None:
                #     print(f"{name} gradient: {param.grad}")

                self.optimizer_auxiliary.step()

                # 更新value
                auxiliary_value_state_values = self.policy.evaluate_value(auxiliary_global_states)
                auxiliary_value_pred_clipped = auxiliary_value_old_state_values + (
                            auxiliary_value_state_values - auxiliary_value_old_state_values).clamp(-self.eps_clip,
                                                                                                   self.eps_clip)
                auxiliary_value_error_clipped = self.value_normalizer.normalize(
                    auxiliary_returns) - auxiliary_value_pred_clipped
                auxiliary_value_error_original = self.value_normalizer.normalize(
                    auxiliary_returns) - auxiliary_value_state_values
                auxiliary_value_value_loss_clipped = huber_loss(auxiliary_value_error_clipped, self.huber_delta)
                auxiliary_value_value_loss_original = huber_loss(auxiliary_value_error_original, self.huber_delta)
                auxiliary_value_loss = torch.max(auxiliary_value_value_loss_original,
                                                 auxiliary_value_value_loss_clipped)
                self.optimizer_critic.zero_grad()
                auxiliary_value_loss.mean().backward()

                self.optimizer_critic.step()

            # Copy new weights into old policy:
            self.policy_old.load_state_dict(self.policy.state_dict())
            try:
                torch.save(self.policy.state_dict(),
                           '/home/binqian/xyb/Video-Analytics-Task-Offloading_mobicom_stream/RL_training_shared_PPG_main/RL_model/' + self.save_path + '.pth')
            except PermissionError:
                pass