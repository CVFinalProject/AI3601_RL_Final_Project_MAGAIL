import sys
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import time
import math
from collections import defaultdict
import cv2

# Increase system recursion limit
sys.setrecursionlimit(25000)
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def im_show(image):
    cv2.imshow('husky', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def getlist(list_, idx):
    if idx < 0 or idx >= len(list_) or len(list_) == 0:
        return None
    else:
        return list_[idx]


class trajectory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.probs = []
        self.pol_out = []
        self.pos_x = 0
        self.steps = 0
        self.start_pos = 0


class samples_agents():
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.probs = []
        self.pol_out = []


def selective_dump_trajectory(expert_trajectory, agent_id, batch_samples, counter, done_agents_steps, final_pos,
                              finish_rate):
    batch_samples.states += expert_trajectory[agent_id].states
    batch_samples.probs += expert_trajectory[agent_id].probs
    batch_samples.actions += expert_trajectory[agent_id].actions
    batch_samples.next_states += expert_trajectory[agent_id].next_states
    batch_samples.rewards += expert_trajectory[agent_id].rewards
    batch_samples.dones += expert_trajectory[agent_id].dones
    batch_samples.pol_out += expert_trajectory[agent_id].pol_out
    finish_rate += (1 - (300 - expert_trajectory[agent_id].pos_x) / (300 - expert_trajectory[agent_id].start_pos))
    final_pos += expert_trajectory[agent_id].pos_x
    counter += len(expert_trajectory[agent_id].states)
    done_agents_steps += (len(expert_trajectory[agent_id].states) + expert_trajectory[agent_id].steps)
    return counter, done_agents_steps, final_pos, finish_rate


def dump_all(expert_trajectory, agent_traj):
    for env in expert_trajectory.keys():
        for agent_id in expert_trajectory[env].keys():
            agent_traj.states += expert_trajectory[env][agent_id].states
            agent_traj.probs += expert_trajectory[env][agent_id].probs
            agent_traj.actions += expert_trajectory[env][agent_id].actions
            agent_traj.next_states += expert_trajectory[env][agent_id].next_states
            agent_traj.rewards += expert_trajectory[env][agent_id].rewards
            agent_traj.dones += expert_trajectory[env][agent_id].dones
            agent_traj.pol_out += expert_trajectory[env][agent_id].pol_out
            steps = len(expert_trajectory[env][agent_id].states) + expert_trajectory[env][agent_id].steps
            start = expert_trajectory[env][agent_id].start_pos
            expert_trajectory[env][agent_id] = trajectory()
            expert_trajectory[env][agent_id].steps = steps
            expert_trajectory[env][agent_id].start_pos = start


def trans2tensor(batch):
    for k in batch:
        batch[k] = torch.tensor(batch[k], device=device, dtype=torch.float32)
    return batch


def sampling(psgail, vector_env, batch_size, vec_obs, vec_done, expert_trajectory):
    total_agent_num = 0
    agent_traj = samples_agents()
    counter = 0
    done_agents_steps = 0
    finish_rate = 0
    final_pos = 0
    while True:
        vec_act = []
        obs_vectors_orig = np.zeros((1, 76))
        for idx, obs in enumerate(vec_obs):
            for agent_id in obs.keys():
                if getlist(vec_done, idx) is None or not vec_done[idx].get(agent_id):
                    if agent_id not in expert_trajectory[idx]:
                        expert_trajectory[idx][agent_id] = trajectory()
                        expert_trajectory[idx][agent_id].start_pos = obs[agent_id]['ego_pos'][0][0]
                    obs_vectors_orig = np.vstack((obs_vectors_orig, obs[agent_id]['neighbor'].squeeze()))
                    expert_trajectory[idx][agent_id].pos_x = obs[agent_id]['ego_pos'][0][0]
        obs_vectors = torch.tensor(obs_vectors_orig[1:, :], device=device, dtype=torch.float32)
        with torch.no_grad():
            acts, prob, pol_outs, _ = psgail.get_action(obs_vectors)
        act_idx = 0
        acts = acts.cpu()
        pol_outs = pol_outs.cpu()
        pol_outs = pol_outs.detach().numpy()
        acts = acts.numpy()
        for idx, obs in enumerate(vec_obs):
            act_n = {}
            for agent_id in obs.keys():
                if getlist(vec_done, idx) is None or not vec_done[idx].get(agent_id):
                    act_n[agent_id] = acts[act_idx]
                    expert_trajectory[idx][agent_id].states.append(obs_vectors_orig[act_idx + 1])
                    expert_trajectory[idx][agent_id].probs.append(prob[act_idx])
                    expert_trajectory[idx][agent_id].actions.append(act_n[agent_id])
                    expert_trajectory[idx][agent_id].pol_out.append(pol_outs[act_idx])
                    act_idx += 1
            vec_act.append(act_n)
        vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
        for idx, act_n in enumerate(vec_act):
            for agent_id in act_n.keys():
                # obs_vectors = obs_extractor_new(vec_obs[idx].get(agent_id)).squeeze()
                expert_trajectory[idx][agent_id].rewards.append(vec_rew[idx].get(agent_id))
                expert_trajectory[idx][agent_id].dones.append(vec_done[idx].get(agent_id))
                if vec_done[idx].get(agent_id):
                    expert_trajectory[idx][agent_id].next_states.append(np.zeros(76))
                    counter, done_agents_steps, final_pos, finish_rate = selective_dump_trajectory(
                        expert_trajectory[idx], agent_id,
                        agent_traj, counter,
                        done_agents_steps, final_pos, finish_rate)
                    total_agent_num += 1
                    del expert_trajectory[idx][agent_id]
                else:
                    expert_trajectory[idx][agent_id].next_states.append(vec_obs[idx][agent_id]['neighbor'].squeeze())
        if counter >= batch_size:
            dump_all(expert_trajectory, agent_traj)
            break
    return total_agent_num, done_agents_steps, final_pos, finish_rate, agent_traj.states, agent_traj.next_states, agent_traj.actions, agent_traj.probs, agent_traj.dones, agent_traj.rewards, agent_traj.pol_out, vec_obs, vec_done, expert_trajectory


# def sampling(psgail, vector_env, batch_size):
#     vector_env.seed(random.randint(1, 500))
#     vec_obs = vector_env.reset()
#     vec_done = []
#
#     expert_trajectory = {}
#     total_agent_num = 0
#     for i in range(12):
#         expert_trajectory[i] = {}
#     agent_traj = samples_agents()
#     counter = 0
#     ends = 0
#     final_xs = 0
#     while True:
#         vec_act = []
#         for idx, obs in enumerate(vec_obs):
#             act_n = {}
#             for agent_id in obs.keys():
#                 if agent_id not in expert_trajectory[idx]:
#                     expert_trajectory[idx][agent_id] = trajectory()
#                 elif getlist(vec_done, idx) is not None and vec_done[idx].get(agent_id):
#                     length, end, final_x = dump_trajectory(expert_trajectory[idx], agent_id, agent_traj)
#                     counter += length
#                     ends += end
#                     final_xs += final_x
#                     total_agent_num += 1
#                     del expert_trajectory[idx][agent_id]
#                     continue
#                 obs_vectors_orig = obs[agent_id]['neighbor'].squeeze()
#                 expert_trajectory[idx][agent_id].states.append(obs_vectors_orig)
#                 obs_vectors_orig = torch.tensor([obs_vectors_orig], device=device, dtype=torch.float32)
#                 acts, log_prob = psgail.get_action(obs_vectors_orig)
#                 acts = acts.cpu()
#                 act_n[agent_id] = acts.numpy().squeeze()
#                 expert_trajectory[idx][agent_id].probs.append(log_prob)
#                 expert_trajectory[idx][agent_id].actions.append(act_n[agent_id])
#                 # im_show(obs[agent_id].top_down_rgb.data)
#             vec_act.append(act_n)
#         vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
#         for idx, act_n in enumerate(vec_act):
#             for agent_id in act_n.keys():
#                 if vec_obs[idx].get(agent_id) is None:
#                     expert_trajectory[idx][agent_id].next_states.append(np.zeros(53))
#                 else:
#                     expert_trajectory[idx][agent_id].next_states.append(vec_obs[idx][agent_id]['neighbor'].squeeze())
#                 expert_trajectory[idx][agent_id].rewards.append(vec_rew[idx].get(agent_id))
#                 expert_trajectory[idx][agent_id].dones.append(vec_done[idx].get(agent_id))
#         if counter >= batch_size:
#             total_agent_num, counter, ends, final_xs = dump_all(expert_trajectory, agent_traj, total_agent_num, counter,
#                                                                 ends, final_xs)
#             break
#     return agent_traj.states, agent_traj.next_states, agent_traj.actions, agent_traj.probs, agent_traj.dones, agent_traj.rewards, total_agent_num, counter, ends, final_xs


def _cal_angle(vec):
    if vec[1] < 0:
        base_angle = math.pi
        base_vec = np.array([-1.0, 0.0])
    else:
        base_angle = 0.0
        base_vec = np.array([1.0, 0.0])

    cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
    angle = math.acos(cos)
    return angle + base_angle


def _get_closest_vehicles(ego, neighbor_vehicles, n):
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        if abs(rel_pos_vec[0]) > 60 or abs(rel_pos_vec[1]) > 15:
            continue
        # calculate its partitions
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)
    return groups

# def evaluating(psgail, sap_size=10000, env_num=12, agent_number=10):
#     env_creator = lambda: MATrafficSimV(["./ngsim"], agent_number=agent_number)
#     vector_env = ParallelEnv([env_creator] * env_num, auto_reset=True)
#     vec_obs = vector_env.reset()
#     vec_done = []
#     states = []
#     acts = []
#     rewards = []
#     next_states = []
#     probs = []
#     dones = []
#     while True:
#         vec_act = []
#         for idx, obs in enumerate(vec_obs):
#             act_n = {}
#             obs_vectors = {}
#             for agent_id in obs.keys():
#                 if (getlist(vec_done, idx) is not None and vec_done[idx][agent_id]):
#                     continue
#                 obs_vectors[agent_id] = obs_extractor(obs[agent_id])
#                 states.append(obs_vectors)
#                 log_prob, prob, act_n[agent_id] = psgail.get_action(obs_vectors)
#                 acts.append(act_n[agent_id])
#                 probs.append(prob)
#             vec_act.append(act_n)
#         vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
#         for idx, obs in enumerate(vec_obs):
#             for agent_id in vec_act[idx].keys():
#                 obs_vectors = obs_extractor(vec_obs[idx].get(agent_id))
#                 next_states.append(obs_vectors)
#                 rewards.append(vec_rew[idx].get(agent_id))
#                 dones.append(vec_done[idx].get(agent_id))
#         if len(dones) >= sap_size:
#             break
#     vector_env.close()
#     return states, next_states, acts, probs, dones, rewards

# def assign_neighbors(neighbors, targets, relative_pos, idx):
#     if abs(relative_pos[0]) < abs(targets[1]):
#         targets[1] = relative_pos[0]
#         neighbors[1] = idx
#     elif targets[0] < relative_pos[0] < targets[1]:
#         targets[0] = relative_pos[0]
#         neighbors[0] = idx
#     elif targets[1] < relative_pos[0] < targets[2]:
#         targets[2] = relative_pos[0]
#         neighbors[2] = idx
#
#
# def obs_extractor(obs):
#     if obs is None:
#         return None
#     ego_vehicle_state = obs.ego_vehicle_state
#     neighborhood_vehicle_states = obs.neighborhood_vehicle_states
#     neighbors_up_idx = -np.ones(3).astype(int)
#     neighbors_middle_idx = -np.ones(3).astype(int)
#     neighbors_down_idx = -np.ones(3).astype(int)
#     neighbors_up = np.zeros((3, 4)).astype(float)
#     neighbors_middle = np.zeros((3, 4)).astype(float)
#     neighbors_down = np.zeros((3, 4)).astype(float)
#     center_lane = ego_vehicle_state.lane_index
#     targets_up = np.array([-10000, -10000, 10000])
#     targets_middle = np.array([-10000, 0, 10000])
#     targets_down = np.array([-10000, -10000, 10000])
#     for idx, info in enumerate(neighborhood_vehicle_states):
#         relative_pos = info[1][:-1] - ego_vehicle_state[1][:-1]
#         if info.lane_index == center_lane + 1:
#             assign_neighbors(neighbors_up_idx, targets_up, relative_pos, idx)
#         elif info.lane_index == center_lane:
#             assign_neighbors(neighbors_middle_idx, targets_middle, relative_pos, idx)
#         elif info.lane_index == center_lane - 1:
#             assign_neighbors(neighbors_down_idx, targets_down, relative_pos, idx)
#     for i in range(3):
#         idx_up = neighbors_up_idx[i]
#         idx_down = neighbors_down_idx[i]
#         # relative pos
#         if idx_up != -1:
#             neighbors_up[i, :2] = neighborhood_vehicle_states[idx_up][1][:-1] - ego_vehicle_state[1][:-1]
#             neighbors_up[i, 2] = float(neighborhood_vehicle_states[idx_up][3] - ego_vehicle_state[3])
#             neighbors_up[i, 3] = float(neighborhood_vehicle_states[idx_up][4] - ego_vehicle_state[4])
#         if idx_down != -1:
#             neighbors_down[i, :2] = neighborhood_vehicle_states[idx_down][1][:-1] - ego_vehicle_state[1][:-1]
#             # relative heading
#             neighbors_down[i, 2] = float(neighborhood_vehicle_states[idx_down][3] - ego_vehicle_state[3])
#             # relative speed
#             neighbors_down[i, 3] = float(neighborhood_vehicle_states[idx_down][4] - ego_vehicle_state[4])
#     for i in range(3):
#         if i != 1:
#             idx = neighbors_middle_idx[i]
#             if idx != -1:
#                 neighbors_middle[i, :2] = neighborhood_vehicle_states[idx][1][:-1] - ego_vehicle_state[1][:-1]
#                 neighbors_middle[i, 2] = float(neighborhood_vehicle_states[idx][3] - ego_vehicle_state[3])
#                 neighbors_middle[i, 3] = float(neighborhood_vehicle_states[idx][4] - ego_vehicle_state[4])
#     neighbors_middle = np.delete(neighbors_middle, 1, axis=0)
#     flatten_up = neighbors_up.flatten()
#     flatten_middle = neighbors_middle.flatten()
#     flatten_down = neighbors_down.flatten()
#     ego_v = np.zeros(13)
#     if len(obs.events.collisions) != 0:
#         ego_v[0] = 1
#     ego_v[1] = obs.events.off_road
#     ego_v[2] = obs.events.on_shoulder
#     # pos
#     ego_v[3:5] = ego_vehicle_state[1][:-1]
#     # heading
#     ego_v[5] = ego_vehicle_state[3]
#     # speed
#     ego_v[6] = ego_vehicle_state[4]
#     # linear speed
#     ego_v[7:13] = np.concatenate((ego_vehicle_state[11][:-1], ego_vehicle_state[12][:-1], ego_vehicle_state[13][:-1]))
#     obs_vectors = np.concatenate((flatten_up, flatten_middle, flatten_down, ego_v))
#     return obs_vectors
