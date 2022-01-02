import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, MultivariateNormal
import math

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class NetCritic(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=2):
        super(NetCritic, self).__init__()
        layers = []
        last_size = input_size

        for i in range(layer_num - 1):
            layers.append(torch.nn.utils.spectral_norm(torch.nn.Linear(last_size, hidden_size[i])))
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Dropout(0.1))
            last_size = hidden_size[i]
        layers.append(torch.nn.Linear(last_size, output_size))
        # self.state_encoder = torch.nn.Linear(53, 16).to(device)
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)
        self.noise = 0.15

    def forward(self, state, action):
        # state_encoded = self.state_encoder(state)
        inputs = torch.cat((state, action), dim=1)
        inputs += torch.normal(0, self.noise, size=inputs.shape, device=device)
        res = self._net(inputs)
        res = torch.sigmoid(res)
        return res


class NetAgent(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size, layer_num=2):
        super(NetAgent, self).__init__()
        layers = []
        last_size = input_size
        for i in range(layer_num - 1):
            layers.append(torch.nn.Linear(last_size, hidden_size[i]))
            layers.append(torch.nn.Tanh())
            last_size = hidden_size[i]
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)

    def forward(self, inputs):
        res = self._net(inputs)
        return res


class PSGAIL():
    def __init__(self,
                 discriminator_lr=5e-5,
                 policy_lr=5e-5,
                 value_lr=5e-5,
                 hidden_size=[512, 256, 128, 64],
                 state_action_space=78,
                 state_space=76,
                 action_space=4,
                 ):
        self._tau = 0.05
        self._clip_range = 0.1
        self.beta = 0.5
        self.klmax = 0.01
        self.discriminator = NetCritic(hidden_size, state_action_space, output_size=1, layer_num=5)
        self.discriminator.to(device)
        self.value = NetAgent(hidden_size, state_space, output_size=1, layer_num=5)
        self.value.to(device)
        self.target_value = NetAgent(hidden_size, state_space, output_size=1, layer_num=5)
        self.target_value.to(device)
        self.policy = NetAgent(hidden_size, state_space, output_size=action_space, layer_num=5)
        self.policy.to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr,
                                                        betas=(0.5, 0.999))
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        self.scheduler_discriminator = torch.optim.lr_scheduler.StepLR(self.discriminator_optimizer,
                                                                       step_size=10000,
                                                                       gamma=0.95)

    def soft_update(self, source, target, tau=None):
        if tau is None:
            tau = self._tau
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, obs, action=None):
        policy_out = self.policy(obs)
        mean, var = torch.chunk(policy_out, 2, dim=-1)
        # print(mean.shape)
        # if len(mean) == 1:
        #     mean[0] = 3 * torch.tanh(mean[0])
        #     mean[1] = 0.3 * torch.tanh(mean[1])
        # else:
        mean[:, 0] = 3 * torch.tanh(mean[:, 0])
        mean[:, 1] = 0.3 * torch.tanh(mean[:, 1])
        var = torch.nn.functional.softplus(var)
        cov_mat = torch.diag_embed(var)
        act_dist = MultivariateNormal(mean, cov_mat)
        if action is None:
            action = act_dist.sample()
            log_prob = act_dist.log_prob(action)
        else:
            log_prob = act_dist.log_prob(action)
        dist_entropy = act_dist.entropy()
        return action, log_prob.reshape(-1, 1), torch.cat((mean, var), dim=1), dist_entropy

    def compute_adv(self, batch, gamma):
        s = batch["state"]
        s1 = batch["next_state"]
        reward = batch['agents_rew']
        done = batch["done"].reshape(-1, 1)
        with torch.no_grad():
            td_target = reward + gamma * self.value(s1) * (1 - done)
            adv = td_target - self.value(s)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, td_target

    def grad_penalty(self, agent_data, experts_data):
        alpha = torch.tensor(np.random.random(size=experts_data.shape), dtype=torch.float32).cuda()
        interpolates = (alpha * experts_data + ((1 - alpha) * agent_data)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = self.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_discriminator(self, sap_agents, sap_experts):
        sap_experts = torch.tensor(sap_experts, device=device, dtype=torch.float32)
        s_exp = sap_experts[:, :-2]
        a_exp = sap_experts[:, -2:]
        sap_agents = sap_agents.detach()
        s_agents = sap_agents[:, :-2]
        a_agents = sap_agents[:, -2:]
        D_expert = self.discriminator(s_exp, a_exp)
        D_agents = self.discriminator(s_agents, a_agents)
        experts_score = -torch.log(D_expert).mean()
        agents_score = -torch.log(1 - D_agents).mean()
        discriminator_loss = (agents_score + experts_score)
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        self.scheduler_discriminator.step()
        return float(agents_score.mean().data), float(-torch.log(1 - D_expert).mean().data), float(discriminator_loss)

    def update_policy(self, batch):
        state = batch["state"]
        action = batch["action"]
        old_log_prob = batch["log_prob"]
        adv = batch["adv"]
        pol_out = batch['pol_out']
        act, log_prob, cur_pol, dist_entropy = self.get_action(state, action)
        old_log_prob = old_log_prob.detach()
        old_log_prob = old_log_prob.unsqueeze(1)
        ip_sp = torch.exp(log_prob - old_log_prob)
        ip_sp_clip = torch.clamp(ip_sp, 1 - self._clip_range, 1 + self._clip_range)
        old_mat = torch.diag_embed(pol_out[:, 2:])
        cur_mat = torch.diag_embed(cur_pol[:, 2:])
        kl_ = torch.distributions.kl_divergence(MultivariateNormal(pol_out[:, :2], old_mat),
                                                MultivariateNormal(cur_pol[:, :2], cur_mat)).mean()
        policy_loss = -torch.mean(
            torch.min(ip_sp * adv.detach(), ip_sp_clip * adv.detach()) + 1e-2 * dist_entropy)
        if kl_ > self.klmax:
            return float(policy_loss.data), float(kl_.data)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return float(policy_loss.data), float(kl_.data)


    def update_value(self, batch):
        state = batch["state"]
        td_target = batch['td_target']
        value_loss = torch.mean(F.mse_loss(self.value(state), td_target.detach()))
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        self.soft_update(self.value, self.target_value, self._tau)
        return float(value_loss.data)

    def behavior_clone(self, sap_experts):
        sap_experts = torch.tensor(sap_experts, device=device, dtype=torch.float32)
        s_experts = sap_experts[:, :-2]
        a_experts = sap_experts[:, -2:]
        _, log_probs = self.get_action(s_experts, a_experts)
        log_probs = - torch.nn.functional.relu(-log_probs)
        policy_loss = -log_probs.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return float(log_probs.mean().data), float(policy_loss.data)
