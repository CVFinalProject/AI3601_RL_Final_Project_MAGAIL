import numpy as np
from dataclasses import replace
from smarts.core.smarts import SMARTS
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.traffic_history_provider import TrafficHistoryProvider
import gym
from envision.client import Client as Envision
from example_adapter import get_observation_adapter
import json
import os
import signal
import subprocess
import argparse
import pickle as pk
import math
import torch

device1 = "cuda:0"


def _get_closest_vehicles(ego, neighbor_vehicles, n):
    """将周角分成n个区域，获取每个区域最近的车辆"""
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    half_part = math.pi / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        if abs(rel_pos_vec[0]) > 50 or abs(rel_pos_vec[1]) > 10:
            continue
        # calculate its partitions
        angle = np.arctan2(rel_pos_vec[1], rel_pos_vec[0])
        if angle < 0:
            angle = 2 * math.pi + angle
        if 2 * math.pi - half_part > angle >= 0:
            angle += half_part
        else:
            angle = half_part - (2 * math.pi - angle)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)
    return groups


def cal_neighbor(env_obs):
    ego = env_obs.ego_vehicle_state
    neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
    closest_neighbor_num = 8
    # dist, speed, ttc, pos
    features = np.zeros((closest_neighbor_num, 5))
    # fill neighbor vehicles into closest_neighboor_num areas
    surrounding_vehicles = _get_closest_vehicles(
        ego, neighbor_vehicle_states, n=closest_neighbor_num
    )
    husky = {'gnE05b_0': [-1.5713518822932961, -0.1, 180, 2.87],
             'gneE01': [-1.571439578989722, -0.2, 310.92, 8.43, 3.69],
             'gneE51_0': [-1.5715601518831366, -0.1, 130.92, 3.50]}
    curve = [-1.5597124433619645, -1.5563403521948302, -1.5484982609760536, -1.5297118970629224,
             -1.5365798573689613, -1.5674406346927465]
    husky_idx = {'gneE01': 1, 'gnE05b_0': 2, 'gneE51_0': 3, 'gneE05a_0': 4}
    ego_pos = ego.position[:2]
    ego_heading = np.asarray(float(ego.heading))
    for i, v in surrounding_vehicles.items():
        if v[0] is None:
            # existence rel x, rel y, rel heading, speed
            features[i, :] = np.array([-1, 0, 0, 0, 0])
            continue
        else:
            v = v[0]
        pos = v.position[:2]
        heading = np.asarray(float(v.heading))
        speed = np.asarray(v.speed)
        rel0 = pos[0] - ego_pos[0]
        rel1 = pos[1] - ego_pos[1]
        if i == 0 or i == 4:
            if rel0 >= 0:
                rel0 = rel0 - v[2].length / 2 - ego[2].length / 2
            else:
                rel0 = rel0 + v[2].length / 2 + ego[2].length / 2
        if i == 2 or i == 6:
            if rel1 >= 0:
                rel1 = rel1 - v[2].width / 2 - ego[2].width / 2
            else:
                rel1 = rel1 + v[2].width / 2 + ego[2].width / 2
        features[i, :] = np.asarray([1, abs(rel0), abs(rel1), heading - ego_heading, speed - ego.speed])
    features[:, 3] = features[:, 3] * 180 / math.pi
    features = features.reshape((-1,))
    ego_pos = np.zeros(13)
    ego_pos[0] = ego.heading
    ego_pos[1] = ego.speed
    ego_pos[2:4] = ego.angular_velocity[:2]
    ego_pos[4:6] = ego.angular_acceleration[:2]
    ego_pos[6:8] = ego.linear_velocity[:2]
    ego_pos[8:10] = ego.linear_acceleration[:2]
    # 0.00, -1.98       20.75, 0.23 -1.5597124433619645
    # 20.75, -1.75      28.36 0.41  -1.5563403521948302
    # 49.11, -1.34      36.32 0.81  -1.5484982609760536
    # 85.43, -0.53      46.95 1.93  -1.5297118970629224
    # 132.38, 1.40      32.72 1.12  -1.5365798573689613
    # 165.10, 2.52      14.9    0.05    -1.5674406346927465
    # 180.00, 2.57
    if husky_idx.get(ego.lane_id) is not None:
        if husky_idx.get(ego.lane_id) == 4:
            ego_pos[10] = 1
            if ego.position[0] < 20.75:
                ego_pos[11] = ego.heading - curve[0]
                ego_pos[12] = ego.position[0] * 0.23 / 20.75 - ego.position[1] - 1.98
            elif 20.75 <= ego.position[0] < 49.11:
                ego_pos[11] = ego.heading - curve[1]
                ego_pos[12] = (ego.position[0] - 20.75) * 0.41 / 28.36 - ego.position[1] - 1.75
            elif 49.11 <= ego.position[0] < 85.43:
                ego_pos[11] = ego.heading - curve[2]
                ego_pos[12] = (ego.position[0] - 49.11) * 0.81 / 36.32 - ego.position[1] - 1.34
            elif 85.43 <= ego.position[0] < 132.38:
                ego_pos[11] = ego.heading - curve[3]
                ego_pos[12] = (ego.position[0] - 85.43) * 1.93 / 46.95 - ego.position[1] - 0.53
            elif 132.38 <= ego.position[0] < 165.10:
                ego_pos[11] = ego.heading - curve[4]
                ego_pos[12] = (ego.position[0] - 132.38) * 1.12 / 32.72 - ego.position[1] + 1.40
            elif 165.10 <= ego.position[0] <= 180.00:
                ego_pos[11] = ego.heading - curve[5]
                ego_pos[12] = (ego.position[0] - 165.10) * 0.05 / 14.9 - ego.position[1] + 2.52
        elif husky_idx.get(ego.lane_id) == 2:
            ego_pos[10] = 0
            ego_pos[11] = ego.heading - husky['gnE05b_0'][0]
            ego_pos[12] = ego.position[0] * husky['gnE05b_0'][1] / husky['gnE05b_0'][2] + husky['gnE05b_0'][
                3] - ego.position[1]
        elif husky_idx.get(ego.lane_id) == 3:
            ego_pos[10] = 0
            ego_pos[11] = ego.heading - husky['gneE51_0'][0]
            ego_pos[12] = (ego.position[0] - 180) * husky['gneE51_0'][1] / husky['gneE51_0'][2] + \
                          husky['gneE51_0'][3] - ego.position[1]
    elif ego.lane_id is not None and ego.lane_id[:6] in husky_idx:
        ego_pos[10] = 0
        ego_pos[11] = ego.heading - husky['gneE01'][0]
        ego_pos[12] = ego.position[0] * husky['gneE01'][1] / husky['gneE01'][2] + \
                      husky['gneE01'][3] + ego.lane_index * husky['gneE01'][4] - ego.position[1]
    ego_pos[11] = ego_pos[11] * 180 / math.pi
    vecs = np.concatenate((features, ego_pos), axis=0)
    return vecs


def get_action_adapter():
    def action_adapter(model_action):
        assert len(model_action) == 2
        return (model_action[0], model_action[1])

    return action_adapter


class SMARTSImitation(gym.Env):
    def __init__(self, scenarios):
        super(SMARTSImitation, self).__init__()
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._next_scenario()
        self.obs_stacked_size = 1
        self.agent_spec = AgentSpec(
            interface=AgentInterface(
                max_episode_steps=None,
                waypoints=False,
                neighborhood_vehicles=True,
                ogm=False,
                rgb=False,
                lidar=False,
                action=ActionSpaceType.Imitation,
            ),
            action_adapter=get_action_adapter(),
            observation_adapter=get_observation_adapter(self.obs_stacked_size),
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64
        )

        envision_client = Envision(
            endpoint=None,
            sim_name="NGSIM_TEST",
            output_dir='./visual',
            headless=None,
        )
        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=envision_client,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def _convert_obs(self, raw_observations):
        observation = self.agent_spec.observation_adapter(
            raw_observations[self.vehicle_id]
        )
        ego_state = []
        other_info = []
        for feat in observation:
            if feat in ["ego_pos", "speed", "heading"]:
                ego_state.append(observation[feat])
            else:
                other_info.append(observation[feat])
        ego_state = np.concatenate(ego_state, axis=1).reshape(-1)
        other_info = np.concatenate(other_info, axis=1).reshape(-1)
        full_obs = np.concatenate((ego_state, other_info))
        return full_obs

    def step(self, action):
        # action = np.clip(action, -1, 1)
        # Transform the normalized action back to the original range
        # *** Formula for transformation from x in [xmin, xmax] to [ymin, ymax]
        # *** y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
        # action = (self._action_range[1] - self._action_range[0]) * (
        #     action + 1
        # ) / 2 + self._action_range[0]

        raw_observations, rewards, dones, _ = self.smarts.step(
            {self.vehicle_id: self.agent_spec.action_adapter(action)}
        )

        info = {}
        info["reached_goal"] = raw_observations[self.vehicle_id].events.reached_goal
        info["collision"] = len(raw_observations[self.vehicle_id].events.collisions) > 0
        obs = self.agent_spec.observation_adapter(raw_observations[self.vehicle_id])

        return (
            obs,
            rewards[self.vehicle_id],
            dones[self.vehicle_id],
            info,
        )

    def reset(self):
        if self.vehicle_itr >= len(self.vehicle_ids):
            self._next_scenario()

        self.vehicle_id = self.vehicle_ids[self.vehicle_itr]
        vehicle_mission = self.vehicle_missions[self.vehicle_id]

        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        traffic_history_provider.start_time = vehicle_mission.start_time

        modified_mission = replace(vehicle_mission, start_time=0.0)
        self.scenario.set_ego_missions({self.vehicle_id: modified_mission})
        self.smarts.switch_ego_agents({self.vehicle_id: self.agent_spec.interface})

        observations = self.smarts.reset(self.scenario)
        obs = self.agent_spec.observation_adapter(observations[self.vehicle_id])
        self.vehicle_itr += 1
        return obs

    def _next_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.vehicle_ids = list(self.vehicle_missions.keys())
        np.random.shuffle(self.vehicle_ids)
        self.vehicle_itr = 0

    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()


def load_model(model2test):
    with open('./models/' + model2test, "rb") as f:
        models = pk.load(f)
    return models


if __name__ == "__main__":
    envision_proc = subprocess.Popen(
        "scl envision start -s ./ngsim", shell=True
    )

    env = SMARTSImitation(
        scenarios=["./ngsim"],
    )

    filename = 'psgail_1557_gail_2_260.model'
    print(filename)
    modesls = load_model(filename)
    psgail = modesls['model']

    for epoch in range(10):
        observations = env.reset()
        dones = {}
        for step in range(500):
            obs_vectors = observations['neighbor']
            obs_vectors = torch.tensor(obs_vectors, device=device1, dtype=torch.float32)
            agent_actions, _, _1 = psgail.get_action(obs_vectors)
            act_tmp = agent_actions.cpu()
            agent_actions = act_tmp.numpy().squeeze()
            if dones:
                break
            observations, rew, dones, _ = env.step(agent_actions)

    env.destroy()

    os.killpg(os.getpgid(envision_proc.pid), signal.SIGKILL)
    envision_proc.wait()
