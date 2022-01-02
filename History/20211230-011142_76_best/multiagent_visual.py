import subprocess
from dataclasses import replace
import gym
from envision.client import Client as Envision
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from example_adapter import get_observation_adapter
from utils import get_vehicle_start_at_time
import torch
import numpy as np
import pickle as pk
import os
import signal

def get_action_adapter():
    def action_adapter(model_action):
        assert len(model_action) == 2
        return (model_action[0], model_action[1])
    return action_adapter

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class TrafficSimV(gym.Env):
    def __init__(self, scenarios):
        super(TrafficSimV, self).__init__()
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
            sim_name="NGSIM_MAGAIL",
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
        observation = self.agent_spec.observation_adapter(raw_observations[self.vehicle_id])
        ego_state = []
        other_info = []
        for feature in observation:
            if feature in ["ego_pos", "speed", "heading"]:
                ego_state.append(observation[feature])
            else:
                other_info.append(observation[feature])
        ego_state = np.concatenate(ego_state, axis=1).reshape(-1)
        other_info = np.concatenate(other_info, axis=1).reshape(-1)
        full_obs = np.concatenate((ego_state, other_info))
        return full_obs

    def step(self, action):
        raw_observations, rewards, dones, _ = self.smarts.step(
            {self.vehicle_id: self.agent_spec.action_adapter(action)})

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
        self.vehicle_itr = np.random.choice(len(self.vehicle_ids))
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


class MATrafficSimV(gym.Env):
    def __init__(self, scenarios, agent_number=10, obs_stacked_size=1):
        super(MATrafficSimV, self).__init__()
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self.n_agents = agent_number
        self.obs_stacked_size = obs_stacked_size
        self._init_scenario()
        self.agentid_to_vehid = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
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
            observation_adapter=get_observation_adapter(obs_stacked_size),
        )

        envision_client = Envision(
            endpoint=None,
            sim_name="NGSIM_MAGAIL",
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

    def step(self, action):
        for agent_id in self.agent_ids:
            if agent_id not in action.keys():
                continue
            agent_action = action[agent_id]
            action[agent_id] = self.agent_spec.action_adapter(agent_action)

        observations, rewards, dones, _ = self.smarts.step(action)
        info = {}

        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(
                observations[k]
            )

        dones["__all__"] = all(dones.values())

        return (observations,
                rewards,
                dones,
                info,
                )

    def reset(self):
        if self.vehicle_itr + self.n_agents >= (len(self.vehicle_ids) - 1):
            self.vehicle_itr = 0

        self.vehicle_id = self.vehicle_ids[self.vehicle_itr: self.vehicle_itr + self.n_agents]

        traffic_history_provider = self.smarts.get_provider_by_type(TrafficHistoryProvider)
        assert traffic_history_provider

        for i in range(self.n_agents):
            self.agentid_to_vehid[f"agent_{i}"] = self.vehicle_id[i]

        agent_interfaces = {}
        history_start_time = self.vehicle_missions[self.vehicle_id[0]].start_time
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            agent_interfaces[agent_id] = self.agent_spec.interface
            if history_start_time > self.vehicle_missions[vehicle].start_time:
                history_start_time = self.vehicle_missions[vehicle].start_time

        traffic_history_provider.start_time = history_start_time
        ego_missions = {}
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            ego_missions[agent_id] = replace(self.vehicle_missions[vehicle], start_time=self.vehicle_missions[
                                                                                            vehicle].start_time - history_start_time, )
        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(agent_interfaces)

        observations = self.smarts.reset(self.scenario)

        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(observations[k])
        self.vehicle_itr += np.random.choice(len(self.vehicle_ids) - self.n_agents)

        return observations

    def _init_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.veh_start_times = {}
        for v_id, mission in self.vehicle_missions.items():
            self.veh_start_times[v_id] = mission.start_time
        self.vehicle_ids = list(self.vehicle_missions.keys())
        vlist = []
        for vehicle_id, start_time in self.veh_start_times.items():
            vlist.append((vehicle_id, start_time))
        dtype = [("id", int), ("start_time", float)]
        vlist = np.array(vlist, dtype=dtype)
        vlist = np.sort(vlist, order="start_time")
        self.vehicle_ids = list(self.vehicle_missions.keys())
        for id in range(len(self.vehicle_ids)):
            self.vehicle_ids[id] = f"{vlist[id][0]}"
        self.vehicle_itr = np.random.choice(len(self.vehicle_ids) - self.n_agents)

    def close(self):
        if self.smarts is not None:
            self.smarts.destroy()

class MATrafficSim_new:
    def __init__(self, scenarios, agent_number, obs_stacked_size=1):
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._init_scenario()
        self.obs_stacked_size = obs_stacked_size
        self.n_agents = agent_number
        self.agentid_to_vehid = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
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
            observation_adapter=get_observation_adapter(obs_stacked_size),
        )
        envision_client = Envision(
            endpoint=None,
            sim_name="NGSIM_MAGAIL",
            output_dir=None,
            headless=None,
        )
        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=envision_client,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        for agent_id in self.agent_ids:
            if agent_id not in action.keys():
                continue
            agent_action = action[agent_id]
            action[agent_id] = self.agent_spec.action_adapter(agent_action)
        observations, rewards, dones, _ = self.smarts.step(action)
        info = {}

        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(observations[k])

        dones["__all__"] = all(dones.values())

        return (
            observations,
            rewards,
            dones,
            info,
        )

    def reset(self, internal_replacement=False, min_successor_time=5.0):
        if self.vehicle_itr + self.n_agents >= (len(self.vehicle_ids) - 1):
            self.vehicle_itr = 0

        self.vehicle_id = self.vehicle_ids[
            self.vehicle_itr : self.vehicle_itr + self.n_agents
        ]

        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider

        for i in range(self.n_agents):
            self.agentid_to_vehid[f"agent_{i}"] = self.vehicle_id[i]

        history_start_time = self.vehicle_missions[self.vehicle_id[0]].start_time
        agent_interfaces = {a_id: self.agent_spec.interface for a_id in self.agent_ids}

        if internal_replacement:
            # NOTE(zbzhu): we use the first-end vehicle to compute the end time since we want to make sure all vehicles can exist on the map
            history_end_time = min(
                [
                    self.scenario.traffic_history.vehicle_final_exit_time(v_id)
                    for v_id in self.vehicle_id
                ]
            )
            alive_time = history_end_time - history_start_time
            traffic_history_provider.start_time = (
                history_start_time
                + np.random.choice(
                    max(0, round(alive_time * 10) - round(min_successor_time * 10))
                )
                / 10
            )
            traffic_history_provider.start_time = history_start_time + alive_time
        else:
            traffic_history_provider.start_time = history_start_time

        ego_missions = {}
        for agent_id in self.agent_ids:
            vehicle_id = self.agentid_to_vehid[agent_id]
            start_time = max(
                0,
                self.vehicle_missions[vehicle_id].start_time
                - traffic_history_provider.start_time,
            )
            try:
                ego_missions[agent_id] = replace(
                    self.vehicle_missions[vehicle_id],
                    start_time=start_time,
                    start=get_vehicle_start_at_time(
                        vehicle_id,
                        max(traffic_history_provider.start_time, self.vehicle_missions[vehicle_id].start_time),
                        self.scenario.traffic_history,
                    ),
                )
            except AssertionError:
                print(vehicle_id)
                print(traffic_history_provider.start_time)
                print(self.vehicle_missions[vehicle_id].start_time)
                print(self.scenario.traffic_history.vehicle_final_exit_time(vehicle_id))
                raise AssertionError

        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(agent_interfaces)

        observations = self.smarts.reset(self.scenario)
        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(observations[k])
        self.vehicle_itr += self.n_agents

        return observations

    def _init_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.veh_start_times = {}
        for v_id, mission in self.vehicle_missions.items():
            self.veh_start_times[v_id] = mission.start_time
        self.vehicle_ids = list(self.vehicle_missions.keys())
        vlist = []
        for vehicle_id, start_time in self.veh_start_times.items():
            vlist.append((vehicle_id, start_time))
        dtype = [("id", int), ("start_time", float)]
        vlist = np.array(vlist, dtype=dtype)
        vlist = np.sort(vlist, order="start_time")
        self.vehicle_ids = list(self.vehicle_missions.keys())
        for id in range(len(self.vehicle_ids)):
            self.vehicle_ids[id] = f"{vlist[id][0]}"
        self.vehicle_itr = np.random.choice(len(self.vehicle_ids))

    def close(self):
        if self.smarts is not None:
            self.smarts.destroy()

def load_model(model2test):
    with open(model2test, "rb") as f:
        models = pk.load(f)
    return models

def getlist(list_, idx):
    if idx < 0 or idx >= len(list_) or len(list_) == 0:
        return None
    else:
        return list_[idx]


if __name__ == "__main__":
    envision_proc = subprocess.Popen("scl envision start -s ./ngsim", shell=True)

    agent_number = 5
    eval_flag = True

    filename = 'psgail_696_gail_2_874_874_1008.model'
    print(filename)
    modesls = load_model(filename)
    psgail = modesls['model']

    if agent_number == 0:
        env = TrafficSimV(scenarios=["./ngsim"])
        if eval_flag:
            for epoch in range(10):
                observations = env.reset()
                dones = {}
                n_steps = 500
                for step in range(n_steps):
                    if not dones:
                        obs_vectors = observations['neighbor']
                        obs_vectors = torch.tensor(obs_vectors, device=device, dtype=torch.float32)
                        agent_actions, _, _1, _2 = psgail.get_action(obs_vectors)
                        agent_actions = agent_actions.cpu()
                        agent_actions = agent_actions.numpy().squeeze()
                    if dones:
                        break
                    observations, rew, dones, _ = env.step(agent_actions)
        print('finished')

    else:
        env = MATrafficSim_new(scenarios=["./ngsim"], agent_number=agent_number)
        for epoch in range(10):
            observations = env.reset()
            dones = {}
            n_steps = 200 * agent_number
            for step in range(n_steps):
                act_n = {}
                for agent_id in observations.keys():
                    if not dones.get(agent_id):
                        obs_vectors = observations[agent_id]['neighbor']
                        obs_vectors = torch.tensor(obs_vectors, device=device, dtype=torch.float32)
                        agent_actions, _, _1, _2 = psgail.get_action(obs_vectors)
                        agent_actions = agent_actions.cpu()
                        act_n[agent_id] = agent_actions.numpy().squeeze()
                    if step and dones[agent_id]:
                        continue
                observations, rew, dones, _ = env.step(act_n)
            print("finished")
    env.close()
    env.destroy()
    os.killpg(os.getpgid(envision_proc.pid), signal.SIGKILL)
    envision_proc.wait()