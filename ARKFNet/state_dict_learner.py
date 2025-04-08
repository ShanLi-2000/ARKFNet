import torch
from torch import nn


class Learner_KalmanNet(nn.Module):
    def __init__(self, x_dim, y_dim, args):
        super(Learner_KalmanNet, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        l1_input = x_dim * 2 + y_dim * 2

        l1_hidden = (x_dim + y_dim) * 10 * 4
        self.l1 = nn.Sequential(
            nn.Linear(l1_input, l1_hidden),
            nn.ReLU()
        )

        self.gru_n_layer = 1
        self.gru_hidden_dim = 4 * (x_dim ** 2 + y_dim ** 2)

        self.hn_train_init = torch.randn(self.gru_n_layer, args.batch_size, self.gru_hidden_dim).to(self.device)
        self.hn_qry_init = torch.randn(self.gru_n_layer, args.valid_seq_num, self.gru_hidden_dim).to(self.device)

        self.GRU = nn.GRU(input_size=l1_hidden, hidden_size=self.gru_hidden_dim, num_layers=self.gru_n_layer)

        self.l2 = nn.Sequential(
            nn.Linear(in_features=self.gru_hidden_dim, out_features=x_dim * y_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=x_dim * y_dim * 4, out_features=x_dim * y_dim)
        )

    def forward(self, state_inno, diff_state, residual, diff_obs):

        x = torch.cat((state_inno, diff_state, residual, diff_obs), 1).reshape(1, residual.shape[0], -1)
        l1_out = self.l1(x)
        GRU_in = torch.zeros(1, state_inno.shape[0], (self.x_dim + self.y_dim) * 10 * 4).to(self.device)
        GRU_in[0, :, :] = l1_out.squeeze().clone()  # 使用clone()以避免就地操作
        GRU_out, hn = self.GRU(GRU_in, self.hn.clone())
        self.hn = hn.detach().clone()  # 更新self.hn
        l2_out = self.l2(GRU_out)
        kalman_gain = torch.reshape(l2_out, (state_inno.shape[0], self.x_dim, self.y_dim))

        return kalman_gain

    def initialize_hidden(self, is_train=True):
        if is_train:
            self.hn = self.hn_train_init.detach().clone()
        else:
            self.hn = self.hn_qry_init.detach().clone()


class Learner_Split_KalmanNet(nn.Module):
    def __init__(self, x_dim, y_dim, args):
        super(Learner_Split_KalmanNet, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.args = args

        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        l1_input = self.x_dim * 2 + self.y_dim + self.x_dim * self.y_dim
        l2_input = self.y_dim * 2 + self.y_dim + self.x_dim * self.y_dim
        output_dim_1 = self.x_dim * self.x_dim
        output_dim_2 = self.y_dim * self.y_dim

        H1 = (x_dim + y_dim) * 10 * 8
        H2 = (x_dim * y_dim) * 1 * (4)
        self.l1 = nn.Sequential(
            nn.Linear(l1_input, H1),
            nn.ReLU()
        )

        # GRU
        self.gru_input_dim = H1
        self.gru_hidden_dim = round(2 * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim)))
        self.gru_n_layer = 1
        self.seq_len_input = 1

        self.hn_train_init_1 = torch.randn(self.gru_n_layer, self.args.batch_size, self.gru_hidden_dim).to(self.device)
        self.hn_qry_init_1 = torch.randn(self.gru_n_layer, self.args.valid_seq_num, self.gru_hidden_dim).to(self.device)

        self.GRU1 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, output_dim_1)
        )

        # input layer
        self.l3 = nn.Sequential(
            nn.Linear(l2_input, H1),
            nn.ReLU()
        )
        # GRU
        self.hn_train_init_2 = torch.randn(self.gru_n_layer, self.args.batch_size, self.gru_hidden_dim).to(self.device)
        self.hn_qry_init_2 = torch.randn(self.gru_n_layer, self.args.valid_seq_num, self.gru_hidden_dim).to(self.device)

        self.GRU2 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        self.l4 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, output_dim_2)
        )

    def initialize_hidden(self, is_train=True):
        if is_train:
            self.hn1 = self.hn_train_init_1.detach().clone().to(self.device)
            self.hn2 = self.hn_train_init_2.detach().clone().to(self.device)
        else:
            self.hn1 = self.hn_qry_init_1.detach().clone().to(self.device)
            self.hn2 = self.hn_qry_init_2.detach().clone().to(self.device)

    def forward(self, diff_obs, observation_inno, diff_state, state_inno, linearization_error, Jacobian):
        task_num = state_inno.shape[0]
        input1 = torch.cat((state_inno, diff_state, linearization_error, Jacobian.reshape((task_num, -1, 1))), 1).permute(2, 0, 1)
        input2 = torch.cat((observation_inno, diff_obs, linearization_error, Jacobian.reshape((task_num, -1, 1))), 1).permute(2, 0, 1)
        l1_out = self.l1(input1)
        GRU_in = torch.zeros(self.seq_len_input, task_num, self.gru_input_dim).to(input1.device)
        GRU_in[0, :, :] = l1_out.squeeze().clone()  # 使用clone()以避免就地操作
        GRU_out, hn1 = self.GRU1(GRU_in, self.hn1.clone())
        self.hn1 = hn1.detach().clone()  # 更新self.hn
        l2_out = self.l2(GRU_out)
        Pk = torch.reshape(l2_out, (task_num, self.x_dim, self.x_dim))

        # forward computing Sk
        l3_out = self.l3(input2)
        GRU_in = torch.zeros(self.seq_len_input, task_num, self.gru_input_dim).to(input1.device)
        GRU_in[0, :, :] = l3_out.squeeze().clone()
        GRU_out, hn2 = self.GRU2(GRU_in, self.hn2.clone())
        self.hn2 = hn2.detach().clone()
        l4_out = self.l4(GRU_out)
        Sk = torch.reshape(l4_out, (task_num, self.y_dim, self.y_dim))

        return Pk, Sk

class Learner_OutlierNet(nn.Module):
    def __init__(self, x_dim, y_dim, args):
        super(Learner_OutlierNet, self).__init__()
        self.args = args
        if args.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            self.device = torch.device('cpu')

        self.x_dim = x_dim
        self.y_dim = y_dim
        H1 = (x_dim + y_dim) * 8 * 4
        H2 = (x_dim + y_dim) * 10 * 8
        H3 = (x_dim + y_dim) * 10 * 4
        H4 = (x_dim + y_dim) * 4 * 8

        self.input_dim_1 = self.y_dim * 3
        self.input_dim_2 = self.y_dim * 3
        self.output_dim_1 = self.y_dim
        self.output_dim_2 = self.y_dim * self.y_dim
        # self.output_dim_2 = self.y_dim

        # architecture for computing residual

        # input layer
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim_1, H1),
            nn.ReLU()
        )

        # GRU
        self.gru_input_dim_1 = H1
        self.gru_input_dim_2 = H2
        self.gru_hidden_dim = round(2 * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim)))
        self.gru_n_layer = 1
        self.seq_len_input = 1

        self.hn_train_init_1 = torch.randn(self.gru_n_layer, self.args.batch_size, self.gru_hidden_dim).to(self.device)
        self.hn_qry_init_1 = torch.randn(self.gru_n_layer, self.args.valid_seq_num, self.gru_hidden_dim).to(self.device)

        self.GRU1 = nn.GRU(self.gru_input_dim_1, self.gru_hidden_dim, self.gru_n_layer)

        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H3),
            nn.ReLU(),
            nn.Linear(H3, self.output_dim_1)
        )

        # architecture for computing Sk

        # input layer
        self.l3 = nn.Sequential(
            nn.Linear(self.input_dim_2, H2),
            nn.ReLU()
        )

        # GRU
        self.hn_train_init_2 = torch.randn(self.gru_n_layer, self.args.batch_size, self.gru_hidden_dim).to(self.device)
        self.hn_qry_init_2 = torch.randn(self.gru_n_layer, self.args.valid_seq_num, self.gru_hidden_dim).to(self.device)

        self.GRU2 = nn.GRU(self.gru_input_dim_2, self.gru_hidden_dim, self.gru_n_layer)

        self.l4 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H4),
            nn.ReLU(),
            nn.Linear(H4, self.output_dim_2)
            # nn.Softplus()
        )

    def initialize_hidden(self, is_train=True):
        if is_train:
            self.hn1 = self.hn_train_init_1.detach().clone().to(self.device)
            self.hn2 = self.hn_train_init_2.detach().clone().to(self.device)
        else:
            self.hn1 = self.hn_qry_init_1.detach().clone().to(self.device)
            self.hn2 = self.hn_qry_init_2.detach().clone().to(self.device)
            # self.hn1 = self.hn_train_init_1.detach().clone().to(self.device)
            # self.hn2 = self.hn_train_init_2.detach().clone().to(self.device)

    def forward(self, residual_pre, diff_obs, diff_pre_y):
        task_num = residual_pre.shape[0]
        input1 = torch.cat((residual_pre, diff_obs, diff_pre_y), 1).permute(2, 0, 1)

        # forward computing residual
        l1_out = self.l1(input1)
        GRU_in = torch.zeros(self.seq_len_input, task_num, self.gru_input_dim_1).to(input1.device)
        GRU_in[0, :, :] = l1_out.squeeze().clone()  # 使用clone()以避免就地操作
        GRU_out, hn1 = self.GRU1(GRU_in, self.hn1.clone())
        self.hn1 = hn1.detach().clone()  # 更新self.hn
        l2_out = self.l2(GRU_out)
        residual = torch.reshape(l2_out, (task_num, self.y_dim, 1))

        # residual = residual_pre - residual_correct
        input2 = torch.cat((residual, residual_pre, diff_obs), 1).permute(2, 0, 1)
        # forward computing Sk
        l3_out = self.l3(input2)
        GRU_in = torch.zeros(self.seq_len_input, task_num, self.gru_input_dim_2).to(input1.device)
        GRU_in[0, :, :] = l3_out.squeeze().clone()
        GRU_out, hn2 = self.GRU2(GRU_in, self.hn2.clone())
        self.hn2 = hn2.detach().clone()
        l4_out = self.l4(GRU_out)

        Sk = torch.reshape(l4_out, (task_num, self.y_dim, self.y_dim))
        eigvals, eigvecs = torch.linalg.eigh(Sk)
        eigvals_positive = torch.log(1 + torch.exp(eigvals))
        Sk = torch.matmul(eigvecs, eigvals_positive.unsqueeze(-1) * eigvecs.transpose(-1, -2))
        # Sk = torch.bmm(Sk, Sk.permute(0, 2, 1))

        return residual, Sk


class Learner_OutlierNet_v2(nn.Module):
    def __init__(self, x_dim, y_dim, args):
        super(Learner_OutlierNet_v2, self).__init__()
        self.args = args
        if args.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            self.device = torch.device('cpu')

        self.x_dim = x_dim
        self.y_dim = y_dim
        H1 = y_dim * 20
        H2 = y_dim * 20
        H3 = y_dim * 8 * 4

        self.residual_input_dim_1 = self.y_dim * 2
        self.residual_output_dim_1 = self.y_dim
        self.residual_input_dim_2 = self.y_dim
        self.residual_output_dim_2 = self.y_dim

        # architecture for computing residual

        # input layer
        self.l1 = nn.Sequential(
            nn.Linear(self.residual_input_dim_1, H1),
            nn.ReLU()
        )

        # GRU
        self.gru_n_layer = 1
        self.seq_len_input = 1
        self.residual_gru_input_dim_1 = H1
        self.residual_gru_hidden_dim_1 = self.residual_output_dim_1

        self.hn_train_init_residual_1 = torch.randn(self.gru_n_layer, self.args.batch_size, self.residual_gru_hidden_dim_1).to(self.device)
        self.hn_qry_init_residual_1 = torch.randn(self.gru_n_layer, self.args.valid_seq_num, self.residual_gru_hidden_dim_1).to(self.device)

        self.GRU1 = nn.GRU(self.residual_gru_input_dim_1, self.residual_gru_hidden_dim_1, self.gru_n_layer)

        self.l2 = nn.Sequential(
            nn.Linear(self.residual_input_dim_2, H2),
            nn.ReLU()
        )
        self.residual_gru_input_dim_2 = H2 + self.residual_gru_hidden_dim_1
        self.residual_gru_hidden_dim_2 = round(20 * (2 * self.y_dim))

        self.hn_train_init_residual_2 = torch.randn(self.gru_n_layer, self.args.batch_size, self.residual_gru_hidden_dim_2).to(self.device)
        self.hn_qry_init_residual_2 = torch.rand(self.gru_n_layer, self.args.valid_seq_num, self.residual_gru_hidden_dim_2).to(self.device)

        self.GRU2 = nn.GRU(self.residual_gru_input_dim_2, self.residual_gru_hidden_dim_2, self.gru_n_layer)

        self.l3 = nn.Sequential(
            nn.Linear(self.residual_gru_hidden_dim_2, H3),
            nn.ReLU(),
            nn.Linear(H3, self.residual_output_dim_2),
        )

        # architecture for computing Sk

        H4 = (x_dim + y_dim) * 10 * 8
        H5 = (x_dim + y_dim) * 4 * 8
        self.Sk_input_dim = self.y_dim * 3
        self.Sk_output_dim = self.y_dim * self.y_dim
        self.Sk_gru_hidden_dim = round(20 * (self.y_dim * 3 + self.x_dim ** 2))
        # self.Sk_gru_hidden_dim = round(2 * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim)))
        # input layer
        self.l4 = nn.Sequential(
            nn.Linear(self.Sk_input_dim, H4),
            nn.ReLU()
        )

        # GRU
        self.Sk_gru_input_dim = H4
        self.hn_train_init_Sk = torch.randn(self.gru_n_layer, self.args.batch_size, self.Sk_gru_hidden_dim).to(self.device)
        self.hn_qry_init_Sk = torch.randn(self.gru_n_layer, self.args.valid_seq_num, self.Sk_gru_hidden_dim).to(self.device)

        self.GRU3 = nn.GRU(self.Sk_gru_input_dim, self.Sk_gru_hidden_dim, self.gru_n_layer)

        self.l5 = nn.Sequential(
            nn.Linear(self.Sk_gru_hidden_dim, H5),
            nn.ReLU(),
            nn.Linear(H5, self.Sk_output_dim)
            # nn.Softplus()
        )

    def initialize_hidden(self, is_train=True):
        if is_train:
            self.hn1 = self.hn_train_init_residual_1.detach().clone().to(self.device)
            self.hn2 = self.hn_train_init_residual_2.detach().clone().to(self.device)
            self.hn3 = self.hn_train_init_Sk.detach().clone().to(self.device)
        else:
            self.hn1 = self.hn_qry_init_residual_1.detach().clone().to(self.device)
            self.hn2 = self.hn_qry_init_residual_2.detach().clone().to(self.device)
            self.hn3 = self.hn_qry_init_Sk.detach().clone().to(self.device)

    def forward(self, residual_pre, diff_obs, diff_pre_y):
        task_num = residual_pre.shape[0]
        input1 = torch.cat((diff_obs, diff_pre_y), 1).permute(2, 0, 1)
        # input1 = residual_list.reshape(task_num, -1, 1)
        # input1 = input1.permute(2, 0, 1)

        # forward computing residual_predict
        l1_out = self.l1(input1)
        GRU_in = torch.zeros(self.seq_len_input, task_num, self.residual_gru_input_dim_1).to(self.device)
        GRU_in[0, :, :] = l1_out.squeeze().clone()
        GRU_out, hn1 = self.GRU1(GRU_in, self.hn1.clone())
        self.hn1 = hn1.detach().clone()  # 更新self.hn
        # residual_predict = torch.reshape(GRU_out, (task_num, self.y_dim, 1))

        # forward correcting residual
        input2 = residual_pre.permute(2, 0, 1)
        l2_out = self.l2(input2)

        temp_input = torch.cat((l2_out, GRU_out), 2)
        GRU_in = torch.zeros(self.seq_len_input, task_num, self.residual_gru_input_dim_2).to(self.device)
        GRU_in[0, :, :] = temp_input.squeeze().clone()
        GRU_out, hn2 = self.GRU2(GRU_in, self.hn2.clone())
        self.hn2 = hn2.detach().clone()  # 更新self.hn
        l3_out = self.l3(GRU_out)
        residual = torch.reshape(l3_out, (task_num, self.y_dim, 1))

        # forward computing Sk
        input3 = torch.cat((residual, residual_pre, diff_obs), 1).permute(2, 0, 1)
        l4_out = self.l4(input3)
        GRU_in = torch.zeros(self.seq_len_input, task_num, self.Sk_gru_input_dim).to(input1.device)
        GRU_in[0, :, :] = l4_out.squeeze().clone()
        GRU_out, hn3 = self.GRU3(GRU_in, self.hn3.clone())
        self.hn3 = hn3.detach().clone()
        l5_out = self.l5(GRU_out)

        Sk = torch.reshape(l5_out, (task_num, self.y_dim, self.y_dim))
        eigvals, eigvecs = torch.linalg.eigh(Sk)
        eigvals_positive = torch.log(1 + torch.exp(eigvals))
        Sk = torch.matmul(eigvecs, eigvals_positive.unsqueeze(-1) * eigvecs.transpose(-1, -2))

        return residual, Sk

# class Learner_OutlierNet_v2(nn.Module):
#     def __init__(self, x_dim, y_dim, args):
#         super(Learner_OutlierNet_v2, self).__init__()
#         self.args = args
#         if args.use_cuda:
#             if torch.cuda.is_available():
#                 self.device = torch.device('cuda')
#             else:
#                 raise Exception("No GPU found, please set args.use_cuda = False")
#         else:
#             self.device = torch.device('cpu')
#
#         self.x_dim = x_dim
#         self.y_dim = y_dim
#         H1 = y_dim * 20
#         H2 = y_dim * 2 * 20
#         H3 = y_dim * 8 * 10
#
#         self.residual_input_dim_1 = self.y_dim * 2
#         self.residual_output_dim_1 = self.y_dim
#         self.residual_input_dim_2 = self.y_dim + self.residual_output_dim_1
#         self.residual_output_dim_2 = self.y_dim
#
#         # architecture for computing residual
#
#         # input layer
#         self.l1 = nn.Sequential(
#             nn.Linear(self.residual_input_dim_1, H1),
#             nn.ReLU()
#         )
#
#         # GRU
#         self.gru_n_layer = 1
#         self.seq_len_input = 1
#         self.residual_gru_input_dim_1 = H1
#         self.residual_gru_hidden_dim_1 = self.residual_output_dim_1
#
#         self.hn_train_init_residual_1 = torch.randn(self.gru_n_layer, self.args.batch_size, self.residual_gru_hidden_dim_1).to(self.device)
#         self.hn_qry_init_residual_1 = torch.randn(self.gru_n_layer, self.args.valid_seq_num, self.residual_gru_hidden_dim_1).to(self.device)
#
#         self.GRU1 = nn.GRU(self.residual_gru_input_dim_1, self.residual_gru_hidden_dim_1, self.gru_n_layer)
#
#         self.l2 = nn.Sequential(
#             nn.Linear(self.residual_input_dim_2, H2),
#             nn.ReLU()
#         )
#         self.residual_gru_input_dim_2 = H2
#         self.residual_gru_hidden_dim_2 = round(20 * self.y_dim)
#
#         self.hn_train_init_residual_2 = torch.randn(self.gru_n_layer, self.args.batch_size, self.residual_gru_hidden_dim_2).to(self.device)
#         self.hn_qry_init_residual_2 = torch.rand(self.gru_n_layer, self.args.valid_seq_num, self.residual_gru_hidden_dim_2).to(self.device)
#
#         self.GRU2 = nn.GRU(self.residual_gru_input_dim_2, self.residual_gru_hidden_dim_2, self.gru_n_layer)
#
#         self.l3 = nn.Sequential(
#             nn.Linear(self.residual_gru_hidden_dim_2, H3),
#             nn.ReLU(),
#             nn.Linear(H3, self.residual_output_dim_2),
#         )
#
#         # architecture for computing Sk
#
#         H4 = (x_dim + y_dim) * 10 * 8
#         H5 = (x_dim + y_dim) * 4 * 8
#         self.Sk_input_dim = self.y_dim * 3
#         self.Sk_output_dim = self.y_dim * self.y_dim
#         self.Sk_gru_hidden_dim = round(2 * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim)))
#         # input layer
#         self.l4 = nn.Sequential(
#             nn.Linear(self.Sk_input_dim, H4),
#             nn.ReLU()
#         )
#
#         # GRU
#         self.Sk_gru_input_dim = H4
#         self.hn_train_init_Sk = torch.randn(self.gru_n_layer, self.args.batch_size, self.Sk_gru_hidden_dim).to(self.device)
#         self.hn_qry_init_Sk = torch.randn(self.gru_n_layer, self.args.valid_seq_num, self.Sk_gru_hidden_dim).to(self.device)
#
#         self.GRU3 = nn.GRU(self.Sk_gru_input_dim, self.Sk_gru_hidden_dim, self.gru_n_layer)
#
#         self.l5 = nn.Sequential(
#             nn.Linear(self.Sk_gru_hidden_dim, H5),
#             nn.ReLU(),
#             nn.Linear(H5, self.Sk_output_dim)
#             # nn.Softplus()
#         )
#
#     def initialize_hidden(self, is_train=True):
#         if is_train:
#             self.hn1 = self.hn_train_init_residual_1.detach().clone().to(self.device)
#             self.hn2 = self.hn_train_init_residual_2.detach().clone().to(self.device)
#             self.hn3 = self.hn_train_init_Sk.detach().clone().to(self.device)
#         else:
#             self.hn1 = self.hn_qry_init_residual_1.detach().clone().to(self.device)
#             self.hn2 = self.hn_qry_init_residual_2.detach().clone().to(self.device)
#             self.hn3 = self.hn_qry_init_Sk.detach().clone().to(self.device)
#
#     def forward(self, residual_pre, diff_obs, diff_pre_y):
#         task_num = residual_pre.shape[0]
#         input1 = torch.cat((diff_obs, diff_pre_y), 1).permute(2, 0, 1)
#         # input1 = residual_list.reshape(task_num, -1, 1)
#         # input1 = input1.permute(2, 0, 1)
#
#         # forward computing residual_predict
#         l1_out = self.l1(input1)
#         GRU_in = torch.zeros(self.seq_len_input, task_num, self.residual_gru_input_dim_1).to(self.device)
#         GRU_in[0, :, :] = l1_out.squeeze().clone()
#         GRU_out, hn1 = self.GRU1(GRU_in, self.hn1.clone())
#         self.hn1 = hn1.detach().clone()  # 更新self.hn
#         residual_predict = torch.reshape(GRU_out, (task_num, self.y_dim, 1))
#
#         # forward correcting residual
#         input2 = torch.cat((residual_predict, residual_pre), 1).permute(2, 0, 1)
#         l2_out = self.l2(input2)
#
#         GRU_in = torch.zeros(self.seq_len_input, task_num, self.residual_gru_input_dim_2).to(self.device)
#         GRU_in[0, :, :] = l2_out.squeeze().clone()
#         GRU_out, hn2 = self.GRU2(GRU_in, self.hn2.clone())
#         self.hn2 = hn2.detach().clone()  # 更新self.hn
#         l3_out = self.l3(GRU_out)
#         residual = torch.reshape(l3_out, (task_num, self.y_dim, 1))
#
#         # forward computing Sk
#         input3 = torch.cat((residual, residual_pre, diff_obs), 1).permute(2, 0, 1)
#         l4_out = self.l4(input3)
#         GRU_in = torch.zeros(self.seq_len_input, task_num, self.Sk_gru_input_dim).to(input1.device)
#         GRU_in[0, :, :] = l4_out.squeeze().clone()
#         GRU_out, hn3 = self.GRU3(GRU_in, self.hn3.clone())
#         self.hn3 = hn3.detach().clone()
#         l5_out = self.l5(GRU_out)
#
#         Sk = torch.reshape(l5_out, (task_num, self.y_dim, self.y_dim))
#         eigvals, eigvecs = torch.linalg.eigh(Sk)
#         eigvals_positive = torch.log(1 + torch.exp(eigvals))
#         Sk = torch.matmul(eigvecs, eigvals_positive.unsqueeze(-1) * eigvecs.transpose(-1, -2))
#
#         return residual, Sk


class Learner_IMUNet(nn.Module):
    def __init__(self, x_dim, y_dim, args):
        super(Learner_IMUNet, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        input_dim = y_dim
        output_dim = y_dim
        self.cov_net = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.cov_lin = nn.Sequential(
            nn.Linear(in_features=32, out_features=output_dim),
            nn.Tanh()
        )
        self.cov_lin[0].bias.data[:] /= 100
        self.cov_lin[0].weight.data[:] /= 100

    def forward(self, obs_list, R_pre):

        y_cov = self.cov_net(obs_list).permute(2, 0, 1)
        z_cov = self.cov_lin(y_cov)
        temp = torch.diag_embed((10**z_cov).squeeze())
        R = R_pre.repeat(obs_list.shape[0], 1, 1)
        R = temp * R
        return R


class Learner_Vanilla_RNN(nn.Module):
    def __init__(self, x_dim, y_dim, args):
        super(Learner_Vanilla_RNN, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        l1_input = y_dim

        l1_hidden = (x_dim + y_dim) * 10 * 4
        self.l1 = nn.Sequential(
            nn.Linear(l1_input, l1_hidden),
            nn.ReLU()
        )

        self.gru_n_layer = 1
        self.gru_hidden_dim = 4 * (x_dim ** 2 + y_dim ** 2)

        self.hn_train_init = torch.randn(self.gru_n_layer, args.batch_size, self.gru_hidden_dim).to(self.device)
        self.hn_qry_init = torch.randn(self.gru_n_layer, args.valid_seq_num, self.gru_hidden_dim).to(self.device)

        self.GRU = nn.GRU(input_size=l1_hidden, hidden_size=self.gru_hidden_dim, num_layers=self.gru_n_layer)

        self.l2 = nn.Sequential(
            nn.Linear(in_features=self.gru_hidden_dim, out_features=x_dim * y_dim * 4),
            nn.ReLU(),
            nn.Linear(in_features=x_dim * y_dim * 4, out_features=x_dim)
        )

    def forward(self, obs):

        x = obs.permute(2, 0, 1)
        l1_out = self.l1(x)
        GRU_in = torch.zeros(1, obs.shape[0], (self.x_dim + self.y_dim) * 10 * 4).to(self.device)
        GRU_in[0, :, :] = l1_out.squeeze().clone()  # 使用clone()以避免就地操作
        GRU_out, hn = self.GRU(GRU_in, self.hn.clone())
        self.hn = hn.detach().clone()  # 更新self.hn
        l2_out = self.l2(GRU_out)
        x_predict = torch.reshape(l2_out, (obs.shape[0], self.x_dim, 1))

        return x_predict.squeeze()

    def initialize_hidden(self, is_train=True):
        if is_train:
            self.hn = self.hn_train_init.detach().clone()
        else:
            self.hn = self.hn_train_init.detach().clone()
