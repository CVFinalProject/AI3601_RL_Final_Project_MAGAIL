import pickle as pk
import os
from utils_psgail import sampling
import sys
import numpy as np
from multiagent_visual import MATrafficSimV
from smarts.env.wrappers.parallel_env import ParallelEnv
from utils_psgail import *


# {
#     'model': psgail,
#     'epoch': i_episode,
#     'rewards_log': rewards_log,
#     'episodes_log': episodes_log,
#     'agent_num': agent_num,
#     'stage': stage,
#     'dis_ag_losses': dis_ag_losses,
#     'dis_ex_losses': dis_ex_losses,
#     'dis_gp_losses': dis_gp_losses,
#     'pol_losses': pol_losses,
#     'val_losses': val_losses,
# },

# Increase system recursion limit
sys.setrecursionlimit(25000)
device1 = "cuda:0"
device0 = "cpu"

def load_model(model2test):
    with open('./models/' + model2test, "rb") as f:
        models = pk.load(f)
    return models


def eval(psgail, batch_size=10000, agent_num=10):
    env = MATrafficSimV(["./ngsim"], agent_number=5)
    obs = env.reset()
    done = {}
    n_steps = 10
    for step in range(n_steps):
        act_n = {}
        for agent_id in obs.keys():
            if step and done[agent_id]:
                continue
            obs_vectors = obs_extractor(obs[agent_id])
            obs_vectors = torch.tensor(obs_vectors, device=device1, dtype=torch.float32)
            _0, act_n[agent_id], _1, _2 = psgail.get_action(obs_vectors)
            act_tmp = act_n[agent_id].cpu()
            act_n[agent_id] = act_tmp.numpy().squeeze()
        obs, rew, done, info = env.step(act_n)
        print(rew)
    print("finished")
    env.close()

    return None

if __name__ == "__main__":
    current_path = 'models'
    filename_list = os.listdir(current_path)
    # filename = 'psgail_3_tuning.model'
    # print(filename)
    # modesls = load_model(filename)
    # psgail = modesls['model']
    # eval(psgail, 10000, 10)
    for filename in filename_list:
        filename = 'psgail_1_gail.model'
        print(filename)
        modesls = load_model(filename)
        psgail = modesls['model']
        eval(psgail, 10000, 10)