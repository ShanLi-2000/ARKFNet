import torch
import math
from torch import nn
from torch import optim
from state_dict_learner import Learner_KalmanNet, Learner_OutlierNet, Learner_OutlierNet_v2, Learner_Split_KalmanNet
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import chi2


class Filter:
    def __init__(self, args, model):
        self.update_lr = args.update_lr  # 0.4

        self.model = model
        self.x_dim = self.model.x_dim
        self.y_dim = self.model.y_dim
        self.args = args
        if args.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            self.device = torch.device('cpu')

        self.loss_fn = nn.MSELoss()

        self.ekf_cov = self.model.ekf_cov.to(self.device)
        self.ekf_cov_post = self.ekf_cov.detach().clone()
        self.cov_post = self.model.init_cov_filter.to(self.device)

        self.state_history = torch.zeros((args.batch_size, self.x_dim, 1)).to(self.device)
        self.y_predict_history = torch.zeros((args.batch_size, self.y_dim, 1)).to(self.device)
        self.residual_history = torch.zeros((args.batch_size, self.y_dim, 1)).to(self.device)
        self.state_cov_history = torch.zeros((args.batch_size, self.x_dim, self.x_dim, 1)).to(self.device)
        self.obs_cov_history = torch.zeros((args.batch_size, self.y_dim, self.y_dim, 1)).to(self.device)

        self.data_idx = 0
        self.batch_size = args.batch_size
        self.alpha = 1.

    def compute_x_post(self, state, obs, task_net, use_initial=True, switch_loss_fn=False):  # obs:[seq_num, y_dim, seq_len]

        self.reset_net()
        task_net.initialize_hidden()
        seq_num, y_dim, seq_len = obs.shape

        if self.data_idx + self.batch_size >= seq_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(seq_num)
            state = state[shuffle_idx]
            obs = obs[shuffle_idx]
        batch_x = state[self.data_idx:self.data_idx + self.batch_size]
        batch_y = obs[self.data_idx:self.data_idx + self.batch_size]
        # self.temp_batch_y = batch_y  # 用于unsupervised KalmanNet

        if use_initial:
            self.state_post = self.model.init_state_filter.reshape((1, self.x_dim, 1)).repeat(self.batch_size, 1, 1).to(state.device)
        else:
            self.state_post = batch_x[:, :, 0].reshape((batch_x.shape[0], batch_x.shape[1], 1))

        if isinstance(task_net, Learner_KalmanNet):
            temp_filtering = self.filtering_KalmanNet
        elif isinstance(task_net, Learner_Split_KalmanNet):
            temp_filtering = self.filtering_KalmanNet
        elif isinstance(task_net, Learner_OutlierNet):
            temp_filtering = self.filtering_OutlierNet
        elif isinstance(task_net, Learner_OutlierNet_v2):
            temp_filtering = self.filtering_OutlierNet
        else:
            raise NotImplementedError
        for i in range(1, seq_len):
            temp_filtering(batch_y[:, :, i].unsqueeze(dim=2), task_net)
        state_filtering = self.state_history
        state_cov_filtering = self.state_cov_history

        self.reset_net()
        task_net.initialize_hidden()
        if switch_loss_fn:
            # residual_state = state_filtering[:, :, 1:] - batch_x[:, :, 1:]
            # mid_value_x = (self.x_dim * torch.log(torch.tensor(2.) * torch.pi)).to(self.device)
            # loss_x = torch.zeros(self.batch_size, device=self.device)
            # for i in range(int(seq_len - 1)):
            #     temp_x_1 = -mid_value_x + torch.log(torch.linalg.det(state_cov_filtering[:, :, :, i+1]))
            #     temp_x_2 = torch.bmm(residual_state[:, :, i].unsqueeze(-1).permute(0, 2, 1),
            #                          torch.bmm(torch.linalg.inv(state_cov_filtering[:, :, :, i+1]),
            #                                    residual_state[:, :, i].unsqueeze(-1))).squeeze()
            #     loss_x += 0.5 * (temp_x_1 + temp_x_2)
            # loss = self.alpha * torch.mean(loss_x / (seq_len * self.x_dim))

            # loss = self.loss_fn(state_filtering[:, :2, 1:], batch_x[:, :2, 1:])
            loss = self.loss_fn(state_filtering[:, :, 1:], batch_x[:, :, 1:])
        else:
            # loss = self.loss_fn(state_filtering[:, :2, 1:], batch_x[:, :2, 1:])
            loss = self.loss_fn(state_filtering[:, :, 1:], batch_x[:, :, 1:])
        self.data_idx += self.batch_size
        return loss

    def compute_x_post_qry(self, state, obs, task_net, use_initial=True):  # obs:[seq_num, y_dim, seq_len]
        self.reset_net(is_train=False)
        task_net.initialize_hidden(is_train=False)
        seq_num, y_dim, seq_len = obs.shape
        if use_initial:
            self.state_post = self.model.init_state_filter.reshape((1, self.x_dim, 1)).repeat(seq_num, 1, 1).to(state.device)
        else:
            self.state_post = state[:, :, 0].reshape((state.shape[0], state.shape[1], 1))
        if isinstance(task_net, Learner_KalmanNet):
            temp_filtering = self.filtering_KalmanNet
        elif isinstance(task_net, Learner_Split_KalmanNet):
            temp_filtering = self.filtering_KalmanNet
        elif isinstance(task_net, Learner_OutlierNet):
            temp_filtering = self.filtering_OutlierNet
        elif isinstance(task_net, Learner_OutlierNet_v2):
            temp_filtering = self.filtering_OutlierNet
        else:
            raise NotImplementedError
        with torch.no_grad():
            for i in range(1, seq_len):
                temp_filtering(obs[:, :, i].unsqueeze(dim=2), task_net)
        state_filtering = self.state_history
        self.reset_net(is_train=False)
        task_net.initialize_hidden(is_train=False)

        # loss = self.loss_fn(state_filtering[:, :2, 1:], state[:, :2, 1:])
        loss = self.loss_fn(state_filtering[:, :, 1:], state[:, :, 1:])
        self.state_predict = state_filtering
        return loss

    def filtering_KalmanNet(self, observation, task_net):

        if self.training_first:
            self.state_post_past = self.state_post.detach().clone()  # state_post_past = x_{k-1|k-1} 此处的state_post即trainer.py中的self.dnn.state_post

        x_last = self.state_post
        x_predict = self.model.f(x_last)

        if self.training_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.model.g(x_predict)
        residual = observation - y_predict

        # input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        # input 2: residual
        # input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        # input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past

        if isinstance(task_net, Learner_Split_KalmanNet):
            H_jacobian = self.model.Jacobian_g(x_predict)
            linearization_error = y_predict - H_jacobian @ x_predict
            Pk, Sk = task_net(diff_obs, residual, diff_state, state_inno, linearization_error, H_jacobian)
            K_gain = torch.bmm(Pk, torch.bmm(H_jacobian.permute(0, 2, 1), Sk))
        else:
            K_gain = task_net(diff_obs, residual, diff_state, state_inno)

        x_post = x_predict + torch.matmul(K_gain, residual)

        self.training_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), 2)
        # x_predict_k = self.model.f(x_post)  # 下一时刻k+1的预测值
        # y_predict_k = self.model.g(x_predict_k)

    def filtering_OutlierNet(self, observation, task_net):

        x_last = self.state_post
        x_predict = self.model.f(x_last)

        F_jacob = self.model.Jacobian_f(x_last)
        H_jacob = self.model.Jacobian_g(x_predict)

        cov_pred = (torch.bmm(torch.bmm(F_jacob, self.cov_post.to(observation.device)), torch.transpose(F_jacob, 1, 2)))\
                   + self.model.cov_q.reshape((1, self.x_dim, self.x_dim)).repeat(F_jacob.shape[0], 1, 1).to(observation.device)

        y_predict = self.model.g(x_predict)
        residual = observation - y_predict
        if self.training_first:
            self.y_predict_past = y_predict.detach().clone()
            self.obs_past = observation.detach().clone()
            # self.residual_list = residual.repeat(1, 1, 5)

        # input 1: residual
        # input 2: y_k - y_{k-1}
        diff_obs = observation - self.obs_past
        # input 3: hat_{y_k} - hat_{y_{k-1}}
        diff_pre_y = y_predict - self.y_predict_past
        # input 4: cov_pred

        residual_rectify, Sk = task_net(residual, diff_obs, diff_pre_y)

        # self.residual_list = torch.cat((self.residual_list[:, :2, 1:], residual.detach().clone()), 2)
        # residual, Sk = task_net(residual, diff_obs, cov_pred)
        # residual = residual - residual_correct

        # K_gain = torch.bmm(torch.bmm(cov_pred, torch.transpose(H_jacob, 1, 2)), Sk.squeeze())
        K_gain = torch.bmm(torch.bmm(cov_pred, torch.transpose(H_jacob, 1, 2)), Sk)

        x_post = x_predict + torch.matmul(K_gain, residual_rectify)

        cov_post = torch.bmm((torch.eye(self.model.x_dim).to(self.device) - torch.bmm(K_gain, H_jacob)), cov_pred)

        self.training_first = False
        self.cov_post = cov_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.y_predict_past = y_predict.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), 2)
        self.residual_history = torch.cat((self.residual_history, residual_rectify.clone()), 2)
        self.state_cov_history = torch.cat((self.state_cov_history, cov_post.unsqueeze(-1).clone()), 3)
        self.obs_cov_history = torch.cat((self.obs_cov_history, Sk.unsqueeze(-1).clone()), 3)

        # x_predict_k = self.model.f(x_post)  # 下一时刻k+1的预测值
        # y_predict_k = self.model.g(x_predict_k)
        # self.y_predict_history = torch.cat((self.y_predict_history, y_predict_k.clone()), 2)

    def reset_net(self, is_train=True):
        self.training_first = True
        self.cov_post = self.model.init_cov.clone()
        if is_train:
            self.state_post = torch.zeros((self.batch_size, self.x_dim, 1)).to(self.device)
            self.y_predict_history = torch.zeros((self.batch_size, self.y_dim, 1)).to(self.device)
            self.residual_history = torch.zeros((self.batch_size, self.y_dim, 1)).to(self.device)
            self.state_cov_history = torch.zeros((self.batch_size, self.x_dim, self.x_dim, 1)).to(self.device)
            self.obs_cov_history = torch.zeros((self.batch_size, self.y_dim, self.y_dim, 1)).to(self.device)
            if self.cov_post.shape[0] != self.args.batch_size:
                self.cov_post = self.cov_post.repeat(self.args.batch_size, 1, 1).to(self.device)
        else:
            self.state_post = torch.zeros((self.args.valid_seq_num, self.x_dim, 1)).to(self.device)
            self.y_predict_history = torch.zeros((self.args.valid_seq_num, self.y_dim, 1)).to(self.device)
            self.residual_history = torch.zeros((self.args.valid_seq_num, self.y_dim, 1)).to(self.device)
            self.state_cov_history = torch.zeros((self.args.valid_seq_num, self.x_dim, self.x_dim, 1)).to(self.device)
            self.obs_cov_history = torch.zeros((self.args.valid_seq_num, self.y_dim, self.y_dim, 1)).to(self.device)
            if self.cov_post.shape[0] != self.args.valid_seq_num:
                self.cov_post = self.cov_post.repeat(self.args.valid_seq_num, 1, 1).to(self.device)

        self.state_history = self.state_post.clone()

    def EKF(self, state, obs, use_initial=True):
        seq_num, y_dim, seq_len = obs.shape
        with torch.no_grad():
            if use_initial:
                self.ekf_state_post = self.model.init_state_filter.reshape((1, self.x_dim, 1)).repeat(seq_num, 1, 1).to(state.device)
            else:
                self.ekf_state_post = state[:, :, 0].reshape((state.shape[0], self.x_dim, 1))

            state_filtering = torch.zeros_like(state)
            for i in range(1, seq_len):
                state_filtering[:, :, i] = self.ekf_filtering(obs[:, :, i].unsqueeze(dim=2), self.model.cov_q, self.model.cov_r)

            self.reset_ekf()

            # loss = self.loss_fn(state_filtering[:, :2, 1:], state[:, :2, 1:])
            loss = self.loss_fn(state_filtering[:, :, 1:], state[:, :, 1:])
            self.state_EKF = state_filtering  # 用于compute_trajectory绘制EKF的图像

        return 10 * torch.log10(loss)

    def ekf_filtering(self, observation, Q, R):
        x_last = self.ekf_state_post
        x_predict = self.model.f(x_last)

        y_predict = self.model.g(x_predict)
        residual = observation - y_predict

        F_jacob = self.model.Jacobian_f(x_last)
        H_jacob = self.model.Jacobian_g(x_predict)

        if F_jacob.shape[0] != self.ekf_cov_post.shape[0]:
            self.ekf_cov_post = self.ekf_cov_post.reshape((1, self.ekf_cov_post.shape[1], self.ekf_cov_post.shape[1])).repeat(F_jacob.shape[0], 1, 1)
        cov_pred = (torch.bmm(torch.bmm(F_jacob, self.ekf_cov_post), torch.transpose(F_jacob, 1, 2))) \
                   + Q.reshape((1, Q.shape[0], Q.shape[0])).repeat(F_jacob.shape[0], 1, 1).to(observation.device)

        temp = torch.linalg.inv(torch.bmm(torch.bmm(H_jacob, cov_pred), torch.transpose(H_jacob, 1, 2))
                                + R.reshape((1, R.shape[0], R.shape[0])).repeat(F_jacob.shape[0], 1, 1).to(self.device))

        K_gain = torch.bmm(torch.bmm(cov_pred, torch.transpose(H_jacob, 1, 2)), temp)  # torch.linalg.inv 计算矩阵的倒数

        x_post = x_predict + torch.bmm(K_gain, residual)

        cov_post = torch.bmm((torch.eye(self.x_dim, device=observation.device) - torch.bmm(K_gain, H_jacob)), cov_pred)
        self.ekf_state_post = x_post.detach().clone()
        self.ekf_cov_post = cov_post.detach().clone()

        return x_post.squeeze()

    def EKF_chi_squared(self, state, obs, use_initial=True):
        seq_num, y_dim, seq_len = obs.shape
        state_filtering = torch.zeros_like(state)
        with torch.no_grad():
            for j in range(seq_num):
                if use_initial:
                    self.ekf_state_post = self.model.init_state_filter.reshape((1, self.x_dim, 1)).to(state.device)
                else:
                    self.ekf_state_post = state[j, :, 0].reshape((1, self.x_dim, 1))
                for i in range(1, seq_len):
                    state_filtering[j, :, i] = self.ekf_filtering_chi_squared(obs[j, :, i].reshape((1, self.y_dim, 1)), self.model.cov_q, self.model.cov_r)
                    self.ekf_cov_post = self.ekf_cov.detach().clone()
                    if hasattr(self.model, 'times'):
                        if self.model.times != 0:
                            self.model.times = self.model.times_copy.clone()

            # loss = self.loss_fn(state_filtering[:, :2, 1:], state[:, :2, 1:])
            loss = self.loss_fn(state_filtering[:, :, 1:], state[:, :, 1:])
            self.state_EKF = state_filtering  # 用于compute_trajectory绘制EKF的图像

        return 10 * torch.log10(loss)

    def ekf_filtering_chi_squared(self, observation, Q, R):
        x_last = self.ekf_state_post
        x_predict = self.model.f(x_last)

        y_predict = self.model.g(x_predict)
        residual = observation - y_predict

        F_jacob = self.model.Jacobian_f(x_last)
        H_jacob = self.model.Jacobian_g(x_predict)

        if F_jacob.shape[0] != self.ekf_cov_post.shape[0]:
            self.ekf_cov_post = self.ekf_cov_post.reshape((1, self.ekf_cov_post.shape[1], self.ekf_cov_post.shape[1])).repeat(F_jacob.shape[0], 1, 1)
        cov_pred = (torch.bmm(torch.bmm(F_jacob, self.ekf_cov_post), torch.transpose(F_jacob, 1, 2))) \
                   + Q.reshape((1, Q.shape[0], Q.shape[0])).repeat(F_jacob.shape[0], 1, 1).to(observation.device)

        temp = torch.linalg.inv(torch.bmm(torch.bmm(H_jacob, cov_pred), torch.transpose(H_jacob, 1, 2))
                                + R.reshape((1, R.shape[0], R.shape[0])).repeat(F_jacob.shape[0], 1, 1).to(self.device))

        chi_square_bound = chi2.isf(1-0.95, self.y_dim)
        chi_square = residual.squeeze(0).T @ temp.squeeze() @ residual.squeeze(0)
        if chi_square < chi_square_bound:
            K_gain = torch.bmm(torch.bmm(cov_pred, torch.transpose(H_jacob, 1, 2)), temp)  # torch.linalg.inv 计算矩阵的倒数
            x_post = x_predict + torch.bmm(K_gain, residual)
            cov_post = torch.bmm((torch.eye(self.x_dim, device=observation.device) - torch.bmm(K_gain, H_jacob)), cov_pred)
            self.ekf_state_post = x_post.detach().clone()
            self.ekf_cov_post = cov_post.detach().clone()
        else:
            x_post = x_predict.clone()
            cov_post = cov_pred.clone()
            self.ekf_state_post = x_post.detach().clone()
            self.ekf_cov_post = cov_post.detach().clone()

        return x_post.squeeze()

    def reset_ekf(self):
        self.ekf_state_post = torch.zeros((self.args.valid_seq_num, self.x_dim, 1), device=self.device)
        self.ekf_cov_post = self.ekf_cov.detach().clone()

    def IMUNet_compute_x_post(self, state, obs, task_net, use_initial=True):
        seq_num, y_dim, seq_len = obs.shape
        if self.data_idx + self.batch_size >= seq_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(seq_num)
            state = state[shuffle_idx]
            obs = obs[shuffle_idx]
        batch_x = state[self.data_idx:self.data_idx + self.batch_size]
        batch_y = obs[self.data_idx:self.data_idx + self.batch_size]
        if use_initial:
            state_post = self.model.init_state_filter.reshape((1, self.x_dim, 1)).repeat(self.batch_size, 1, 1).to(state.device)
        else:
            state_post = batch_x[:, :, 0].reshape((batch_x.shape[0], batch_x.shape[1], 1))
        cov_post = self.ekf_cov_post
        state_filtering = torch.zeros_like(batch_x)
        obs_list = batch_y[:, :, 0].unsqueeze(-1).repeat(1, 1, 5)
        Q = self.model.cov_q
        R_pre = self.model.cov_r
        for i in range(1, seq_len):
            observation = batch_y[:, :, i].unsqueeze(dim=2)
            x_last = state_post
            x_predict = self.model.f(x_last)
            y_predict = self.model.g(x_predict)
            residual = observation - y_predict
            F_jacobian = self.model.Jacobian_f(x_predict)
            H_jacobian = self.model.Jacobian_g(x_predict)

            if F_jacobian.shape[0] != cov_post.shape[0]:
                cov_post = cov_post.reshape((1, cov_post.shape[1], cov_post.shape[1])).repeat(F_jacobian.shape[0], 1, 1)
            cov_pred = (torch.bmm(torch.bmm(F_jacobian, cov_post), torch.transpose(F_jacobian, 1, 2))) \
                       + Q.reshape((1, Q.shape[0], Q.shape[0])).repeat(F_jacobian.shape[0], 1, 1).to(observation.device)

            obs_list = torch.cat((obs_list[:, :, 1:], observation), 2)
            R = task_net(obs_list, R_pre.to(self.device))

            temp = torch.linalg.inv(torch.bmm(torch.bmm(H_jacobian, cov_pred), torch.transpose(H_jacobian, 1, 2)) + R)

            K_gain = torch.bmm(torch.bmm(cov_pred, torch.transpose(H_jacobian, 1, 2)), temp)

            x_post = x_predict + torch.bmm(K_gain, residual)

            cov_post = torch.bmm((torch.eye(self.x_dim, device=observation.device) - torch.bmm(K_gain, H_jacobian)), cov_pred)

            state_post = x_post.detach().clone()
            state_filtering[:, :, i] = x_post.clone().squeeze()

        # loss = self.loss_fn(state_filtering[:, :2, 1:], batch_x[:, :2, 1:])
        loss = self.loss_fn(state_filtering[:, :, 1:], batch_x[:, :, 1:])
        self.state_predict = state_filtering
        self.data_idx += self.batch_size
        return loss

    def vanilla_compute_x_post(self, state, obs, task_net, use_initial=True):  # obs:[seq_num, y_dim, seq_len]
        task_net.initialize_hidden()
        seq_num, y_dim, seq_len = obs.shape

        if self.data_idx + self.batch_size >= seq_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(seq_num)
            state = state[shuffle_idx]
            obs = obs[shuffle_idx]
        batch_x = state[self.data_idx:self.data_idx + self.batch_size]
        batch_y = obs[self.data_idx:self.data_idx + self.batch_size]

        if use_initial:
            self.state_post = self.model.init_state_filter.reshape((1, self.x_dim, 1)).repeat(self.batch_size, 1, 1).to(state.device)
        else:
            self.state_post = batch_x[:, :, 0].reshape((batch_x.shape[0], batch_x.shape[1], 1))

        state_filtering = torch.zeros_like(batch_x)
        for i in range(1, seq_len):
            state_filtering[:, :, i] = task_net(batch_y[:, :, i].unsqueeze(dim=2))

        task_net.initialize_hidden()
        # loss = self.loss_fn(state_filtering[:, :2, 1:], batch_x[:, :2, 1:])
        loss = self.loss_fn(state_filtering[:, :, 1:], batch_x[:, :, 1:])
        self.data_idx += self.batch_size
        return loss

    def vanilla_compute_x_post_qry(self, state, obs, task_net, use_initial=True):  # obs:[seq_num, y_dim, seq_len]
        task_net.initialize_hidden(is_train=False)
        seq_num, y_dim, seq_len = obs.shape
        if use_initial:
            self.state_post = self.model.init_state_filter.reshape((1, self.x_dim, 1)).repeat(seq_num, 1, 1).to(state.device)
        else:
            self.state_post = state[:, :, 0].reshape((state.shape[0], state.shape[1], 1))

        state_filtering = torch.zeros_like(state)
        with torch.no_grad():
            for i in range(1, seq_len):
                state_filtering[:, :, i] = task_net(obs[:, :, i].unsqueeze(dim=2))
        task_net.initialize_hidden(is_train=False)

        # loss = self.loss_fn(state_filtering[:, :2, 1:], state[:, :2, 1:])
        loss = self.loss_fn(state_filtering[:, :, 1:], state[:, :, 1:])
        return loss

    # def compute_x_post_unsupervised(self, state, obs, task_net, set_init_state=None):  # obs:[seq_num, y_dim, seq_len]
    #     self.reset_net()
    #     task_net.initialize_hidden()
    #     seq_num, y_dim, seq_len = obs.shape
    #
    #     if self.data_idx + self.batch_size >= seq_num:
    #         self.data_idx = 0
    #         shuffle_idx = torch.randperm(seq_num)
    #         state = state[shuffle_idx]
    #         obs = obs[shuffle_idx]
    #     batch_x = state[self.data_idx:self.data_idx + self.batch_size]
    #     batch_y = obs[self.data_idx:self.data_idx + self.batch_size]
    #
    #     if set_init_state is not None:
    #         self.state_post = set_init_state[:, :, 0].reshape((batch_x.shape[0], batch_x.shape[1], 1))
    #     else:
    #         self.state_post = self.model.init_state_filter.reshape((1, self.x_dim, 1)).repeat(self.batch_size, 1, 1).to(state.device)
    #         # self.state_post = batch_x[:, :, 0].reshape((batch_x.shape[0], batch_x.shape[1], 1))
    #     for i in range(1, seq_len):
    #         self.filtering(batch_y[:, :, i].unsqueeze(dim=2), task_net)
    #     loss = self.loss_fn(self.y_predict_history[:, :2, 1:-1], batch_y[:, :, 2:])
    #     self.reset_net()
    #     task_net.initialize_hidden()
    #     self.data_idx += self.batch_size
    #     return loss
