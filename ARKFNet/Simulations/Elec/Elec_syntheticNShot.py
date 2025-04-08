import math

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Uniform

class SyntheticNShot:

    def __init__(self, args, is_linear=True):
        # 定义模型
        self.args = args
        self.use_cuda = args.use_cuda

        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            self.device = torch.device('cpu')

        self.is_linear = is_linear

        self.x_dim = 5
        self.y_dim = 2

        Rs = torch.tensor(0.18)
        Rr = torch.tensor(0.15)
        self.M = torch.tensor(0.068)
        Ls = torch.tensor(0.0699)
        self.Lr = torch.tensor(0.0699)
        self.J = torch.tensor(0.18)
        Tr = torch.tensor(10.)
        self.p = torch.tensor(1.)
        self.h = torch.tensor(0.0001)
        self.sigma = (1. - self.M ** 2 / (Ls * self.Lr)).float()
        self.k = (self.M / (self.sigma * Ls * self.Lr)).float()
        self.rho = (Rs / (self.sigma * Ls) + Rr * self.M ** 2 / (self.sigma * Ls * self.Lr ** 2)).float()

        temp_1 = 1. - self.rho * self.h
        temp_2 = self.k * self.h / Tr
        temp_3 = 1. - 1. / Tr
        self.const = torch.tensor([[temp_1, 0, temp_2, 0, 0],
                                   [0, temp_1, 0, temp_2, 0],
                                   [temp_2, 0, temp_3, 0, 0],
                                   [0, temp_2, 0, temp_3, 0],
                                   [0, 0, 0, 0, 1]]).float()

        self.B = torch.zeros((self.x_dim, self.x_dim))
        self.B[0, 0] = self.h / (self.sigma * Ls)
        self.B[1, 1] = self.h / (self.sigma * Ls)
        self.B[4, 4] = -Tr * self.h / self.J
        self.times = torch.tensor(0)  # 用于计时
        self.times_copy = self.times.clone()

        # theta = 10 * 2 * math.pi / 360
        theta = 0.
        self.H = torch.tensor([[math.cos(theta), -math.sin(theta), 0., 0., 0.],
                               [math.sin(theta), math.cos(theta), 0., 0., 0.]]).to(self.device)

        self.q2_dB = self.args.q2_dB
        self.q2 = torch.tensor(10 ** (self.q2_dB / 10))  # 将db转换为常数值
        self.v_dB = self.args.v_dB  # 观测噪声和过程噪声的协方差的比值
        self.v = 10 ** (self.v_dB / 10)
        self.r2 = torch.mul(self.q2, self.v)
        self.cov_q = self.q2 * torch.eye(self.x_dim)
        self.cov_r = self.r2 * torch.eye(self.y_dim)

        self.init_state = torch.tensor([0., 0., 0., 0., 0.]).reshape((-1, 1))
        temp = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        self.init_cov = torch.diag(temp)
        self.init_state_filter = torch.tensor([2., 2., 0.5, 0.5, 0.2]).reshape((-1, 1))
        temp = torch.tensor([1e-1, 1e-1, 1e-2, 1e-2, 1e-2])
        self.init_cov_filter = torch.diag(temp)

        temp = torch.tensor([1e-1, 1e-1, 1e-2, 1e-2, 1e-2])
        self.ekf_cov = torch.diag(temp)
        self.ekf_cov_post = self.ekf_cov.detach().clone()
        self.save_path = './Data/Elec/'

        self.inter_value_1 = (self.k * self.p * self.h).to(self.device)
        self.inter_value_2 = (self.p * self.h).to(self.device)
        self.inter_value_3 = (self.p * self.M * self.h / (self.J * self.Lr)).to(self.device)

    def generate_data(self, seq_len, seq_num, mode='train'):

        state_mtx = torch.zeros((seq_num, self.x_dim, seq_len)).to(self.device)  # 初始化状态矩阵
        obs_mtx = torch.zeros((seq_num, self.y_dim, seq_len)).to(self.device)  # 初始化观测矩阵

        # theta_distribution = Uniform(0, 2 * torch.pi)
        # theta = theta_distribution.rsample(torch.Size([seq_num, 1]))
        # velocity_distribution = Uniform(0, 100)
        # velocity = velocity_distribution.rsample(torch.Size([seq_num, 1]))
        # x3 = velocity * torch.cos(theta)
        # x4 = velocity * torch.sin(theta)
        # location_distribution = Uniform(0, 200)
        # x1 = location_distribution.rsample(torch.Size([seq_num, 1]))
        # x2 = location_distribution.rsample(torch.Size([seq_num, 1]))
        # x5 = location_distribution.rsample(torch.Size([seq_num, 1]))
        # x_prev = torch.stack([x1, x2, x3, x4, x5], dim=1).view(seq_num, self.x_dim, -1).to(self.device)
        x_prev = self.init_state.reshape((1, self.x_dim, 1)).repeat(seq_num, 1, 1).to(self.device)

        # 以下代码中，torch的自动求导功能被禁用
        with torch.no_grad():
            for i in range(seq_len):
                # 状态值：
                xt = self.f(x_prev)
                x_mean = torch.zeros(seq_num, self.x_dim)
                distrib = MultivariateNormal(loc=x_mean, covariance_matrix=self.cov_q)
                eq = distrib.rsample().view(seq_num, self.x_dim, 1).to(self.device)
                xt = torch.add(xt, eq)
                # 观测值：
                yt = self.g(xt)
                y_mean = torch.zeros(seq_num, self.y_dim)
                distrib = MultivariateNormal(loc=y_mean, covariance_matrix=self.cov_r)
                er = distrib.rsample().view(seq_num, self.y_dim, 1).to(self.device)
                yt = torch.add(yt, er)

                x_prev = xt.clone()
                state_mtx[:, :, i] = torch.squeeze(xt, 2)
                obs_mtx[:, :, i] = torch.squeeze(yt, 2)

        torch.save(state_mtx, self.save_path + mode + '/state.pt')
        torch.save(obs_mtx, self.save_path + mode + '/obs.pt')
        self.times = self.times_copy.clone()
        return state_mtx, obs_mtx

    def f(self, x):
        F_add = self.get_parameters(x)
        F = torch.add(F_add, self.const.to(x.device))
        u_current = torch.zeros((x.shape[0], self.x_dim, 1))
        u_current[:, 0, :] = 350. * torch.cos(0.003 * self.times)
        u_current[:, 1, :] = 350. * torch.sin(0.003 * self.times)
        u_current[:, 4, :] = 1.
        temp_B = self.B.repeat(x.shape[0], 1, 1).to(x.device)
        x_predict = torch.bmm(F, x) + torch.bmm(temp_B, u_current.to(x.device))
        self.times += 1
        return x_predict

    def f_single(self, x, dt=1):
        x = torch.from_numpy(x).float().reshape(-1, 1)
        F_add = self.get_parameters_single(x)
        F = torch.add(F_add, self.const)
        u_current = torch.zeros((self.x_dim, 1))
        u_current[0] = 350. * torch.cos(0.003 * self.times)
        u_current[1] = 350. * torch.sin(0.003 * self.times)
        u_current[4] = 1.
        x_predict = F @ x + self.B @ u_current
        return x_predict.squeeze().numpy()

    def g_single(self, x):
        return self.H @ x

    def get_parameters_single(self, x):
        F_temp = torch.zeros(self.x_dim, self.x_dim)
        F_temp[0, 4] = self.inter_value_1 * x[3].squeeze()
        F_temp[1, 4] = -self.inter_value_1 * x[2].squeeze()
        F_temp[2, 4] = -self.inter_value_2 * x[3].squeeze()
        F_temp[3, 4] = self.inter_value_2 * x[2].squeeze()
        F_temp[4, 0] = -self.inter_value_3 * x[3].squeeze()
        F_temp[4, 1] = self.inter_value_3 * x[2].squeeze()
        return F_temp

    def g(self, x):
        # return x[:, 0:2, :]
        batched_H = self.H.view(1, self.H.shape[0], self.H.shape[1]).expand(x.shape[0], -1, -1).to(x.device)
        return torch.bmm(batched_H, x)

    def Jacobian_f(self, x):
        F_add = self.get_parameters_Jacobian(x)
        F = torch.add(F_add, self.const.to(x.device))
        return F

    def Jacobian_g(self, x):
        # temp = torch.tensor([[1., 0., 0., 0., 0.],
        #                      [0., 1., 0., 0., 0.]])
        # return temp.reshape(1, temp.shape[0], temp.shape[1]).repeat(x.shape[0], 1, 1).to(x.device)
        temp = self.H.view(1, self.H.shape[0], self.H.shape[1]).expand(x.shape[0], -1, -1).to(x.device)
        return temp

    def get_parameters(self, x):
        F_temp = torch.zeros(x.shape[0], self.x_dim, self.x_dim).to(x.device)
        F_temp[:, 0, 4] = self.inter_value_1 * x[:, 3].squeeze()
        F_temp[:, 1, 4] = -self.inter_value_1 * x[:, 2].squeeze()
        F_temp[:, 2, 4] = -self.inter_value_2 * x[:, 3].squeeze()
        F_temp[:, 3, 4] = self.inter_value_2 * x[:, 2].squeeze()
        F_temp[:, 4, 0] = -self.inter_value_3 * x[:, 3].squeeze()
        F_temp[:, 4, 1] = self.inter_value_3 * x[:, 2].squeeze()
        return F_temp.to(x.device)

    def get_parameters_Jacobian(self, x):
        Jacobian_F_temp = torch.zeros((x.shape[0], self.x_dim, self.x_dim)).to(x.device)
        Jacobian_F_temp[:, 0, 3] = self.inter_value_1 * x[:, 4].squeeze()
        Jacobian_F_temp[:, 0, 4] = self.inter_value_1 * x[:, 3].squeeze()
        Jacobian_F_temp[:, 1, 2] = -self.inter_value_1 * x[:, 4].squeeze()
        Jacobian_F_temp[:, 1, 4] = -self.inter_value_1 * x[:, 2].squeeze()
        Jacobian_F_temp[:, 2, 3] = -self.inter_value_2 * x[:, 4].squeeze()
        Jacobian_F_temp[:, 2, 4] = -self.inter_value_2 * x[:, 3].squeeze()
        Jacobian_F_temp[:, 3, 2] = self.inter_value_2 * x[:, 4].squeeze()
        Jacobian_F_temp[:, 3, 4] = self.inter_value_2 * x[:, 2].squeeze()
        Jacobian_F_temp[:, 4, 0] = -self.inter_value_3 * x[:, 3].squeeze()
        Jacobian_F_temp[:, 4, 1] = self.inter_value_3 * x[:, 2].squeeze()
        Jacobian_F_temp[:, 4, 2] = self.inter_value_3 * x[:, 1].squeeze()
        Jacobian_F_temp[:, 4, 3] = -self.inter_value_3 * x[:, 0].squeeze()
        return Jacobian_F_temp

    def get_H(self):
        return self.H.to(self.device)
