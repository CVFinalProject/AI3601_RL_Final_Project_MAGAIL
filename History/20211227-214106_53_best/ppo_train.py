import torch

from utils_psgail import *
import pickle as pk
import logging
import os
import time
from multiagent_traffic_simulator import MATrafficSim
from multiagent_traffic_simulator_orig import MATrafficSimOrig, MATrafficSimOrigV
from smarts.env.wrappers.parallel_env import ParallelEnv
from traffic_simulator import TrafficSim
from ppo import *
import re
import shutil
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


def train_BC_GAIL(psgail, experts, i_episode_res, writer, path, stage='gail', num_episode=200000, print_every=1,
                  gamma=0.997,
                  batch_size=4096,
                  agent_num=2, mini_epoch=100):
    logger.info('batch_size {}'.format(batch_size))
    critic_epoch = mini_epoch
    policy_epoch = mini_epoch
    rewards_log = []
    avg_step_log = []
    episodes_log = []
    dis_ag_rew = []
    dis_ex_rew = []
    dis_total_losses = []
    pol_losses = []
    val_losses = []
    kl_log = []
    avg_final_log = []
    best_perform = 150
    logger.info('stage: {}, agents num {}'.format(stage, agent_num))
    env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
    vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
    vector_env.seed(random.randint(1, 500))
    vec_obs = vector_env.reset()
    experts_split = np.array_split(experts, 80, axis=0)
    experts_idx = 0
    vec_done = []
    expert_trajectory = {}
    for i in range(12):
        expert_trajectory[i] = {}
    for i_episode in range(i_episode_res, num_episode):
        if (i_episode + 1) % 2000 == 0:
            agent_num += 5
            vector_env.close()
            logger.info('adding agents to {}'.format(agent_num))
            env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
            vector_env = ParallelEnv([env_creator] * 12, auto_reset=True)
            vector_env.seed(random.randint(1, 500))
            vec_obs = vector_env.reset()
            expert_trajectory = {}
            vec_done = []
            for i in range(12):
                expert_trajectory[i] = {}
        dis_agent_buffer = []
        dis_expert_buffer = []
        dis_total_buffer = []
        pol_buffer = []
        val_buffer = []
        kl_buffer = []
        dist_buffer = []
        # time1 = time.time()
        t_a_n, done_agents_steps, t_final_pos, states, next_states, actions, log_probs, dones, rewards, pol_out, vec_obs, vec_done, expert_trajectory = sampling(
            psgail, vector_env, batch_size, vec_obs, vec_done, expert_trajectory)
        # time2 = time.time()
        # print('sample time {}'.format(time2 - time1))
        rewards_log.append(np.sum(rewards) / t_a_n)
        avg_step_log.append(done_agents_steps / t_a_n)
        avg_final_log.append(t_final_pos / t_a_n)
        writer.add_scalars('Agent Status', {'Reward': rewards_log[-1], 'Avg-Survival Time': avg_step_log[-1],
                                            'Avg-Driving Distance': avg_final_log[-1]}, i_episode)
        # policy_epoch = max(int(mini_epoch * 50 / avg_final_log[-1]), 20)
        episodes_log.append(i_episode)
        batch = trans2tensor({"state": states, "action": actions,
                              "log_prob": log_probs,
                              "next_state": next_states, "done": dones, 'pol_out': pol_out})
        sap_agents = torch.cat((batch["state"], batch["action"]), dim=1)
        sap_agents = sap_agents.detach()
        for j in range(critic_epoch):
            if experts_idx == 80:
                experts_idx = 0
            cur_experts = experts_split[experts_idx]
            experts_idx += 1
            dis_agent_tmp, dis_expert_tmp, dis_total_tmp = psgail.update_discriminator(sap_agents, cur_experts)
            dis_agent_buffer.append(dis_agent_tmp)
            dis_expert_buffer.append(dis_expert_tmp)
            dis_total_buffer.append(dis_total_tmp)
        D_agents = psgail.discriminator(batch["state"], batch["action"])
        batch["agents_rew"] = -torch.log(1 - D_agents.detach())
        batch['adv'], batch['td_target'] = psgail.compute_adv(batch, gamma)
        for j in range(mini_epoch):
            value_tmp = psgail.update_value(batch)
            val_buffer.append(value_tmp)
        for j in range(policy_epoch):
            policy_tmp, kl_div = psgail.update_policy(batch)
            if kl_div > psgail.klmax:
                critic_epoch = i + 1
                break
            else:
                kl_buffer.append(kl_div)
                pol_buffer.append(policy_tmp)
        kl_log.append(np.mean(kl_buffer))
        pol_losses.append(np.mean(pol_buffer))
        val_losses.append(np.mean(val_buffer))
        dis_ag_rew.append(np.mean(dis_agent_buffer))
        dis_ex_rew.append(np.mean(dis_expert_buffer))
        dis_total_losses.append(np.mean(dis_total_buffer))
        writer.add_scalars('Discriminator Scores', {'Agent Score': dis_ag_rew[-1], 'Experts Score': dis_ex_rew[-1]},
                           i_episode)
        writer.add_scalar('Critic Loss', dis_total_losses[-1], i_episode)
        writer.add_scalar('KL Distance', kl_log[-1], i_episode)
        writer.add_scalar('Policy Loss', pol_losses[-1], i_episode)
        writer.add_scalar('Value Loss', val_losses[-1], i_episode)
        writer.flush()
        if (i_episode + 1) % print_every == 0 or i_episode + 1 == num_episode:
            if print_every < 10:
                prt = print_every
            else:
                prt = 10
            logger.info(
                "St: {}, Ep: {}, Rew: {}, ag_num: {}, time: {}, final: {}, pol_l: {}, kl: {}, val_l: {}, ag_rew: {}, ex_rew: {}, dis_l: {}".format(
                    stage,
                    i_episode + 1, round(np.mean(rewards_log[-prt:]), 4), agent_num,
                    round(np.mean(avg_step_log[-prt:]), 4),
                    round(np.mean(avg_final_log[-prt:]), 4),
                    round(np.mean(pol_losses[-prt:]), 4),
                    round(np.mean(kl_log[-prt:]), 4),
                    round(np.mean(val_losses[-prt:]), 4),
                    round(np.mean(dis_ag_rew[-prt:]), 4),
                    round(np.mean(dis_ex_rew[-prt:]), 4),
                    round(np.mean(dis_total_losses[-prt:]), 4)
                ))
        if (i_episode + 1) % 25 == 0 or i_episode + 1 == num_episode or np.mean(avg_final_log[-2:]) > best_perform:
            if np.mean(avg_final_log[-2:]) > best_perform:
                best_perform = np.mean(avg_final_log[-2:])
            logger.info('stage {}, checkpoints establish, episode {}'.format(stage, i_episode + 1))
            with open(
                    path + 'psgail_{}_{}_{}_{}.model'.format(i_episode, stage, agent_num,
                                                             int(np.mean(avg_final_log[-2:]))),
                    "wb") as f:
                pk.dump(
                    {
                        'model': psgail,
                        'epoch': i_episode,
                        'rewards_log': rewards_log,
                        'episodes_log': episodes_log,
                        'agent_num': agent_num,
                        'stage': stage,
                        'dis_ag_rew': dis_ag_rew,
                        'dis_ex_rew': dis_ex_rew,
                        'pol_losses': pol_losses,
                        'val_losses': val_losses,
                    },
                    f,
                )
    infos = {
        "rewards_gail": rewards_log,
        "episodes": episodes_log,
        'pol_loss_gail': pol_losses,
        'val_loss_gail': val_losses,
        'dis_ag_rew_gail': dis_ag_rew,
        'dis_ex_rew_gail': dis_ex_rew,
        'avg_survival_time_gail': avg_step_log,
        'dis_total_losses_gail': dis_total_losses,
    }
    vector_env.close()
    return infos


def load_model(model2test):
    with open(model2test, "rb") as f:
        models = pk.load(f)
    return models


if __name__ == "__main__":
    history = './History/'
    file_path = time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '/'
    log_path = history + file_path
    mods_path = history + file_path + 'models/'
    if not os.path.exists(history):
        os.makedirs(history)
    if not os.path.exists(history + file_path):
        os.makedirs(history + file_path)
        os.makedirs(mods_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    log_file_name = log_path + file_path[:-1] + '.log'
    logfile = log_file_name
    handler = logging.FileHandler(logfile, mode='a+')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Start print log")

    env_name = 'NGSIM SMARTS'
    writer = SummaryWriter(history + file_path)
    # psgail = PSGAIL()
    filename = 'psgail_634_gail_2_183.model'
    model = load_model(filename)
    psgail = model['model']
    dummy_obs = torch.randn(1, 53, device=device)
    dummy_act = torch.randn(1, 2, device=device)
    writer.add_graph(psgail.policy, dummy_obs)
    writer.add_graph(psgail.value, dummy_obs)
    writer.add_graph(psgail.discriminator, (dummy_obs, dummy_act))
    logger.info('policy')
    logger.info(summary(psgail.policy, (1, 53)))
    logger.info('value')
    logger.info(summary(psgail.value, (1, 53)))
    logger.info('discriminator')
    logger.info(summary(psgail.discriminator, [[53], [2]]))

    experts = np.load('experts_53_cleaned.npy')
    filename_list = os.listdir('.')
    expr = '\.py'
    for filename in filename_list:
        if re.search(expr, filename) != None:
            shutil.copyfile('./' + filename, history + file_path + filename)
    infos_1 = train_BC_GAIL(psgail, experts, 550, writer, mods_path, stage='gail')

    for keys in infos_1:
        if keys != "episodes":
            plt.title('Reinforce training ' + keys + ' on {}'.format(env_name))
            plt.ylabel(keys)
            plt.xlabel("episodes")
            labels = ["PS-GAIL"]
            x, y = infos_1["episodes"], infos_1[keys]
            # y, x = moving_average(y, x)
            plt.plot(x, y)
            plt.legend(labels)
            plt.savefig('train_' + keys + '.jpg')
            plt.close()
    writer.close()

    # fine tuning
    # infos_2 = train_BC_GAIL(psgail, experts, 0, stage='gail_tuning', num_episode=1000, print_every=1,
    #                         gamma=0.999, batch_size=10000, agent_num=100, mini_epoch=1)
    #
    # for keys in infos_2:
    #     if keys !="episodes"
    #         plt.title('Reinforce training ' + keys + ' on {}'.format(env_name))
    #         plt.ylabel(keys)
    #         plt.xlabel("Frame")
    #         infos = [infos_1, ]
    #         labels = ["PS-GAIL", ]
    #         for info in infos:
    #             x, y = info["episodes"], info[keys]
    #             # y, x = moving_average(y, x)
    #             plt.plot(x, y)
    #         plt.legend(labels)
    #         plt.savefig('train_' + keys + '.jpg')
    #         plt.close()
