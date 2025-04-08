import torch
import math
import numpy as np
import argparse
from filter import Filter
from state_dict_learner import Learner_OutlierNet, Learner_KalmanNet, Learner_OutlierNet_v2, \
    Learner_Split_KalmanNet, Learner_IMUNet
from Simulations.Elec.Elec_syntheticNShot import SyntheticNShot
from matplotlib import pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from DANSE.DANSE_filter import DANSE_Filter


def generate_addFDI(obs):
    y_mean = torch.zeros(obs.shape[0], model.y_dim)
    y_mean_FDI = torch.ones_like(y_mean) * args.attack_mean_add
    distrib_FDI = MultivariateNormal(loc=y_mean_FDI,
                                     covariance_matrix=args.attack_covariance_add * torch.eye(model.y_dim))
    # 批量生成随机噪声
    er_FDI = distrib_FDI.rsample((obs.shape[2],)).permute(1, 2, 0)  # Shape: (batch_size, y_dim, seq_len)
    # 批量生成随机掩码
    probabilities = torch.rand(obs.shape[0], 1, obs.shape[2])
    mask = (probabilities < args.attack_probability_add).expand(-1, model.y_dim, -1)  # Expand to match er_FDI
    # 应用掩码
    er_FDI = (er_FDI * mask).to(device)
    # # 加入干扰生成 FDIA 数据
    # obs_FDIA = obs + er_FDI
    return er_FDI


def generate_multiFDI(obs):
    y_mean = torch.zeros(obs.shape[0], model.y_dim)
    y_mean_FDI = torch.ones_like(y_mean) * (args.attack_mean_multi - 1)
    distrib_FDI = MultivariateNormal(loc=y_mean_FDI, covariance_matrix=args.attack_covariance_multi * torch.eye(model.y_dim))
    # 批量生成随机噪声
    er_FDI = distrib_FDI.rsample((obs.shape[2],)).permute(1, 2, 0)  # Shape: (batch_size, y_dim, seq_len)
    # 批量生成随机掩码
    probabilities = torch.rand(obs.shape[0], 1, obs.shape[2])
    mask = (probabilities < args.attack_probability_multi).expand(-1, model.y_dim, -1)  # Expand to match er_FDI
    # 应用掩码
    er_FDI = (er_FDI * mask).to(device)
    # 加入干扰生成 FDIA 数据
    er_FDI = torch.multiply(obs, er_FDI)
    return er_FDI


argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
argparser.add_argument('--valid_seq_num', type=int, help='valid seq number', default=32)
argparser.add_argument('--batch_size', type=int, help='batch size for training FDI-Net', default=32)
argparser.add_argument('--valid_period', type=int, help='step size for computing valid loss', default=25)
argparser.add_argument('--update_lr', type=float, help='update learning rate', default=2e-4)
argparser.add_argument('--update_lr_danse', type=float, help='update learning rate for danse', default=3e-4)

argparser.add_argument('--Is_add_attack', type=int, help='Is aggravated assault', default=True)  # 加性攻击
argparser.add_argument('--attack_probability_add', type=float, help='the probability of add attack', default=0.1)
argparser.add_argument('--attack_mean_add', type=float, help='the mean of the add attack', default=6)
argparser.add_argument('--attack_covariance_add', type=float, help='the covariance of the add attack', default=5)

argparser.add_argument('--Is_multi_attack', type=int, help='Is multiplicative attack', default=False)  # 乘性攻击
argparser.add_argument('--attack_probability_multi', type=float, help='the probability of multiplicative attack',
                       default=0.05)
argparser.add_argument('--attack_mean_multi', type=float, help='the mean of the multiplicative attack', default=0.95)
argparser.add_argument('--attack_covariance_multi', type=float, help='the covariance of the multiplicative attack',
                       default=0.05)

argparser.add_argument('--q2_dB', type=float, help='process noise (dB)', default=-20.)
argparser.add_argument('--v_dB', type=float, help='r2 / q2 (dB)', default=0.)
argparser.add_argument('--use_cuda', type=int, help='use GPU to accelerate training', default=True)

args = argparser.parse_args()

print(args)

Generate_data = False
TRAIN = True

if args.use_cuda:
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda')
    else:
        raise Exception("No GPU found, please set args.use_cuda = False")
else:
    print("Using CPU")
    device = torch.device('cpu')

model = SyntheticNShot(args, is_linear=True)
my_filter = Filter(args, model)

if Generate_data:
    state_train, obs_train = model.generate_data(seq_len=40, seq_num=800, mode='train')  # 生成数据
    state_valid, obs_valid = model.generate_data(seq_len=40, seq_num=args.valid_seq_num, mode='valid')
    state_test, obs_test = model.generate_data(seq_len=80, seq_num=args.valid_seq_num, mode='test')
else:
    state_train = torch.load(model.save_path + 'train/state.pt', map_location=device)
    obs_train = torch.load(model.save_path + 'train/obs.pt', map_location=device)
    state_valid = torch.load(model.save_path + 'valid/state.pt', map_location=device)
    obs_valid = torch.load(model.save_path + 'valid/obs.pt', map_location=device)
    state_test = torch.load(model.save_path + 'test/state.pt', map_location=device)
    obs_test = torch.load(model.save_path + 'test/obs.pt', map_location=device)

# Generating outlier data (FDIAs)
if args.Is_add_attack is True and args.Is_multi_attack is False:
    train_FDI = generate_addFDI(obs_train)
    obs_train_FDI = obs_train + train_FDI
    valid_FDI = generate_addFDI(obs_valid)
    obs_valid_FDI = obs_valid + valid_FDI
    test_FDI = generate_addFDI(obs_test)
    obs_test_FDI = obs_test + test_FDI
elif args.Is_add_attack is False and args.Is_multi_attack is True:
    train_FDI = generate_multiFDI(obs_train)
    obs_train_FDI = obs_train + train_FDI
    valid_FDI = generate_multiFDI(obs_valid)
    obs_valid_FDI = obs_valid + valid_FDI
    test_FDI = generate_multiFDI(obs_test)
    obs_test_FDI = obs_test + test_FDI
elif args.Is_add_attack is True and args.Is_multi_attack is True:
    train_FDI_1 = generate_addFDI(obs_train)
    train_FDI_2 = generate_multiFDI(obs_train)
    obs_train_FDI = obs_train + train_FDI_1 + train_FDI_2
    valid_FDI_1 = generate_addFDI(obs_valid)
    valid_FDI_2 = generate_multiFDI(obs_valid)
    obs_valid_FDI = obs_valid + valid_FDI_1 + valid_FDI_2
    test_FDI_1 = generate_addFDI(obs_test)
    test_FDI_2 = generate_multiFDI(obs_test)
    obs_test_FDI = obs_test + test_FDI_1 + test_FDI_2
else:
    obs_train_FDI = obs_train
    obs_valid_FDI = obs_valid
    obs_test_FDI = obs_test


train_iter = args.epoch  # 迭代次数：500

test_list = [str(train_iter)]

test_loss_OutlierNet = []
test_loss_KalmanNet = []
test_loss_Split_KalmanNet = []
test_loss_SDANSE = []
test_loss_IMUNet = []
valid_loss_OutlierNet = []
valid_loss_KalmanNet = []
valid_loss_Split_KalmanNet = []
valid_loss_SDANSE = []
valid_loss_IMUNet = []

# torch.manual_seed(3333)
# np.random.seed(3333)

losses_dB_EKF_valid = my_filter.EKF(state_valid, obs_valid, use_initial=True)
# 更新 时间梯度
if hasattr(model, 'times'):
    if model.times != 0:
        model.times = model.times_copy.clone()
print('q2= ' + str(model.q2) + ' r2= ' + str(model.r2) + ' EKF (no attacked) loss(dB): ' + str(losses_dB_EKF_valid.item()))

losses_dB_EKF_valid_attacked = my_filter.EKF(state_valid, obs_valid_FDI, use_initial=True)
if hasattr(model, 'times'):
    if model.times != 0:
        model.times = model.times_copy.clone()
print('q2= ' + str(model.q2) + ' r2= ' + str(model.r2) + ' EKF (attacked) loss(dB): ' + str(losses_dB_EKF_valid_attacked.item()))

# losses_dB_EKF_valid_chi_squared = my_filter.EKF_chi_squared(state_valid, obs_valid_FDI, use_initial=True)
# if hasattr(model, 'times'):
#     if model.times != 0:
#         model.times = model.times_copy.clone()
# print('q2= ' + str(model.q2) + ' r2= ' + str(model.r2) + ' EKF (Chi-squared) loss(dB): ' + str(losses_dB_EKF_valid_chi_squared.item()))

if TRAIN:
    # Training OutlierNet
    task_model_OutlierNet = Learner_OutlierNet(model.x_dim, model.y_dim, args).to(device)
    task_model_OutlierNet.initialize_hidden(is_train=True)

    # use for Learner_OutlierNet
    network1 = [task_model_OutlierNet.l1, task_model_OutlierNet.GRU1, task_model_OutlierNet.l2]
    network2 = [task_model_OutlierNet.l3, task_model_OutlierNet.GRU2, task_model_OutlierNet.l4]

    # # use for Learner_OutlierNet_v2
    # network1 = [task_model_OutlierNet.l1, task_model_OutlierNet.GRU1, task_model_OutlierNet.l2, task_model_OutlierNet.GRU2, task_model_OutlierNet.l3]
    # network2 = [task_model_OutlierNet.l4, task_model_OutlierNet.GRU3, task_model_OutlierNet.l5]

    tmp = filter(lambda x: x.requires_grad, task_model_OutlierNet.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    param_group_1 = []
    for elem in network1:
        param_group_1 += [{'params': elem.parameters()}]
    param_group_2 = []
    for elem in network2:
        param_group_2 += [{'params': elem.parameters()}]
    param_group_list = [param_group_1, param_group_2]
    optimizer_list = [torch.optim.Adam(param_group_1, lr=args.update_lr, weight_decay=0),
                      torch.optim.Adam(param_group_2, lr=args.update_lr, weight_decay=0)]

    for i in range(args.epoch):

        if i % 2 == 0:
            inner_optimizer = optimizer_list[0]
            switch_model = False
        else:
            inner_optimizer = optimizer_list[1]
            switch_model = True

        loss = my_filter.compute_x_post(state_train, obs_train_FDI, task_model_OutlierNet, use_initial=True, switch_loss_fn=switch_model)
        inner_optimizer.zero_grad()
        loss.backward()
        parameters_to_clip = []
        for group in inner_optimizer.param_groups:
            parameters_to_clip += group['params']
        torch.nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=2)
        # torch.nn.utils.clip_grad_norm_(task_model_OutlierNet.parameters(), 5)
        inner_optimizer.step()
        if hasattr(model, 'times'):
            if model.times != 0:
                model.times = model.times_copy.clone()

        if i % args.valid_period == 0:
            print('OutlierNet train num: ' + str(i))
            print('Training loss(dB): ' + str((10 * torch.log10(loss)).item()))

            loss_qry = my_filter.compute_x_post_qry(state_valid, obs_valid_FDI, task_model_OutlierNet, use_initial=True)
            if hasattr(model, 'times'):
                if model.times != 0:
                    model.times = model.times_copy.clone()

            print('Validating loss(dB): ' + str((10 * torch.log10(loss_qry)).item()))
            valid_loss_OutlierNet.append((10 * torch.log10(loss_qry)).item())
    torch.save(task_model_OutlierNet, './Model/Elec/OutlierNet.pt')

    # Training KalmanNet
    task_model_KalmanNet = Learner_KalmanNet(model.x_dim, model.y_dim, args).to(device)
    temp_net = torch.load('./Model/Elec/initial_KalmanNet.pt')
    # temp_net = torch.load('./Model/Elec/initial_KalmanNet_multi.pt')
    task_model_KalmanNet.load_state_dict(temp_net.state_dict())
    task_model_KalmanNet.initialize_hidden(is_train=True)
    inner_optimizer = torch.optim.Adam(task_model_KalmanNet.parameters(), lr=args.update_lr)
    for i in range(args.epoch):
        loss = my_filter.compute_x_post(state_train, obs_train_FDI, task_model_KalmanNet, use_initial=True)
        inner_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task_model_KalmanNet.parameters(), 1)
        inner_optimizer.step()
        if hasattr(model, 'times'):
            if model.times != 0:
                model.times = model.times_copy.clone()

        if i % args.valid_period == 0:
            print('KalmanNet train num: ' + str(i))
            print('Training loss(dB): ' + str((10 * torch.log10(loss)).item()))

            loss_qry = my_filter.compute_x_post_qry(state_valid, obs_valid_FDI, task_model_KalmanNet, use_initial=True)
            if hasattr(model, 'times'):
                if model.times != 0:
                    model.times = model.times_copy.clone()

            print('Validating loss(dB): ' + str((10 * torch.log10(loss_qry)).item()))
            valid_loss_KalmanNet.append((10 * torch.log10(loss_qry)).item())
    torch.save(task_model_KalmanNet, './Model/Elec/KalmanNet.pt')

    # Training Split-KalmanNet
    task_model_SKNet = Learner_Split_KalmanNet(model.x_dim, model.y_dim, args).to(device)
    task_model_SKNet.initialize_hidden(is_train=True)

    network1 = [task_model_SKNet.l1, task_model_SKNet.GRU1, task_model_SKNet.l2]
    network2 = [task_model_SKNet.l3, task_model_SKNet.GRU2, task_model_SKNet.l4]

    param_group_1 = []
    for elem in network1:
        param_group_1 += [{'params': elem.parameters()}]
    param_group_2 = []
    for elem in network2:
        param_group_2 += [{'params': elem.parameters()}]
    param_group_list = [param_group_1, param_group_2]
    optimizer_list = [torch.optim.Adam(param_group_1, lr=args.update_lr, weight_decay=0),
                      torch.optim.Adam(param_group_2, lr=args.update_lr, weight_decay=0)]

    for i in range(args.epoch):

        if i % 2 == 0:
            inner_optimizer = optimizer_list[0]
        else:
            inner_optimizer = optimizer_list[1]

        loss = my_filter.compute_x_post(state_train, obs_train_FDI, task_model_SKNet, use_initial=True)
        inner_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task_model_SKNet.parameters(), 1)
        inner_optimizer.step()
        if hasattr(model, 'times'):
            if model.times != 0:
                model.times = model.times_copy.clone()

        if i % args.valid_period == 0:
            print('Split-KalmanNet train num: ' + str(i))
            print('Training loss(dB): ' + str((10 * torch.log10(loss)).item()))

            loss_qry = my_filter.compute_x_post_qry(state_valid, obs_valid_FDI, task_model_SKNet, use_initial=True)
            if hasattr(model, 'times'):
                if model.times != 0:
                    model.times = model.times_copy.clone()

            print('Validating loss(dB): ' + str((10 * torch.log10(loss_qry)).item()))
            valid_loss_Split_KalmanNet.append((10 * torch.log10(loss_qry)).item())
    torch.save(task_model_SKNet, './Model/Elec/Split-KalmanNet.pt')

    # Training supervised DANSE
    data_idx = 0
    data_num = state_train.shape[0]
    state_train_danse = state_train.permute(0, 2, 1)
    obs_train_FDI_danse = obs_train_FDI.permute(0, 2, 1)
    state_valid_danse = state_valid.permute(0, 2, 1)
    obs_valid_FDI_danse = obs_valid_FDI.permute(0, 2, 1)
    sdanse_filter = DANSE_Filter(args, model)
    for i in range(args.epoch):
        if data_idx + args.batch_size >= data_num:
            data_idx = 0
            shuffle_idx = torch.randperm(data_num)
            state_train_danse = state_train_danse[shuffle_idx]
            obs_train_FDI_danse = obs_train_FDI_danse[shuffle_idx]
        batch_y = obs_train_FDI_danse[data_idx:data_idx + args.batch_size]
        batch_x = state_train_danse[data_idx:data_idx + args.batch_size]
        loss = -sdanse_filter.DANSE_filtering(batch_y, model.cov_r, batch_x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sdanse_filter.danse_net.parameters(), 10)
        sdanse_filter.optimizer.step()
        if hasattr(model, 'times'):
            if model.times != 0:
                model.times = model.times_copy.clone()
        data_idx += args.batch_size

        if i % args.valid_period == 0:
            print('SDANSE train num: ' + str(i))
            # print('Training loss(dB): ' + str((10 * torch.log10(loss)).item()))
            with torch.no_grad():
                te_mu_X_predictions_batch, te_var_X_predictions_batch, te_mu_X_filtered_batch, te_var_X_filtered_batch = \
                    sdanse_filter.compute_predictions(obs_valid_FDI_danse, model.cov_r)
                log_pY_test_batch = -sdanse_filter.DANSE_filtering(obs_valid_FDI_danse, model.cov_r, state_valid_danse)
                loss_qry = my_filter.loss_fn(state_valid_danse[:, :, 1:], te_mu_X_filtered_batch[:, :, 1:])
                if hasattr(model, 'times'):
                    if model.times != 0:
                        model.times = model.times_copy.clone()
                print('Validating loss(dB): ' + str((10 * torch.log10(loss_qry)).item()))
            valid_loss_SDANSE.append((10 * torch.log10(loss_qry)).item())
    torch.save(sdanse_filter.danse_net, './Model/Elec/SDANSE.pt')

    # Training IMUNet
    task_model_IMUNet = Learner_IMUNet(model.x_dim, model.y_dim, args).to(device)
    inner_optimizer = torch.optim.Adam(task_model_IMUNet.parameters(), lr=args.update_lr)
    for i in range(args.epoch):
        loss = my_filter.IMUNet_compute_x_post(state_train, obs_train_FDI, task_model_IMUNet, use_initial=True)
        inner_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(task_model_IMUNet.parameters(), 5)
        inner_optimizer.step()
        if hasattr(model, 'times'):
            if model.times != 0:
                model.times = model.times_copy.clone()
        if i % args.valid_period == 0:
            print('IMUNet train num: ' + str(i))
            print('Training loss(dB): ' + str((10 * torch.log10(loss)).item()))

            with torch.no_grad():
                loss_qry = my_filter.IMUNet_compute_x_post(state_valid, obs_valid_FDI, task_model_IMUNet, use_initial=True)
            if hasattr(model, 'times'):
                if model.times != 0:
                    model.times = model.times_copy.clone()

            print('Validating loss(dB): ' + str((10 * torch.log10(loss_qry)).item()))
            valid_loss_IMUNet.append((10 * torch.log10(loss_qry)).item())
        torch.save(task_model_IMUNet, './Model/Elec/IMUNet.pt')

    np.save('./Result/Elec/valid_loss_OutlierNet.npy', np.array(valid_loss_OutlierNet))
    np.save('./Result/Elec/valid_loss_KalmanNet.npy', np.array(valid_loss_KalmanNet))
    np.save('./Result/Elec/valid_loss_SDANSE.npy', np.array(valid_loss_SDANSE))
    np.save('./Result/Elec/valid_loss_Split_KalmanNet.npy', np.array(valid_loss_Split_KalmanNet))
    np.save('./Result/Elec/valid_loss_IMUNet.npy', np.array(valid_loss_IMUNet))
    np.save('./Result/Elec/valid_loss_EKF.npy', np.array(losses_dB_EKF_valid_attacked.cpu()))

with torch.no_grad():

    task_model = torch.load('./Model/Elec/KalmanNet.pt')
    loss_test = my_filter.compute_x_post_qry(state_test, obs_test_FDI, task_model, use_initial=True)
    test_loss_KalmanNet.append((10 * torch.log10(loss_test)).item())
    if hasattr(model, 'times'):
        if model.times != 0:
            model.times = model.times_copy.clone()

    task_model = torch.load('./Model/Elec/Split-KalmanNet.pt')
    loss_test = my_filter.compute_x_post_qry(state_test, obs_test_FDI, task_model, use_initial=True)
    test_loss_Split_KalmanNet.append((10 * torch.log10(loss_test)).item())
    if hasattr(model, 'times'):
        if model.times != 0:
            model.times = model.times_copy.clone()

    sdanse_filter = DANSE_Filter(args, model)
    task_model = torch.load('./Model/Elec/SDANSE.pt')
    sdanse_filter.danse_net.load_state_dict(task_model.state_dict())
    state_test_danse = state_test.permute(0, 2, 1)
    obs_test_FDI_danse = obs_test_FDI.permute(0, 2, 1)
    te_mu_X_predictions_batch, te_var_X_predictions_batch, te_mu_X_filtered_batch, te_var_X_filtered_batch = \
        sdanse_filter.compute_predictions(obs_test_FDI_danse, model.cov_r)
    log_pY_test_batch = -sdanse_filter.DANSE_filtering(obs_test_FDI_danse, model.cov_r, state_test_danse)
    loss_test = my_filter.loss_fn(state_test_danse[:, :2, 1:], te_mu_X_filtered_batch[:, :2, 1:])
    if hasattr(model, 'times'):
        if model.times != 0:
            model.times = model.times_copy.clone()
    test_loss_SDANSE.append((10 * torch.log10(loss_test)).item())

    task_model = torch.load('./Model/Elec/IMUNet.pt')
    loss_test = my_filter.IMUNet_compute_x_post(state_test, obs_test_FDI, task_model, use_initial=True)
    test_loss_IMUNet.append((10 * torch.log10(loss_test)).item())
    if hasattr(model, 'times'):
        if model.times != 0:
            model.times = model.times_copy.clone()

    task_model = torch.load('./Model/Elec/OutlierNet.pt')
    loss_test = my_filter.compute_x_post_qry(state_test, obs_test_FDI, task_model, use_initial=True)
    test_loss_OutlierNet.append((10 * torch.log10(loss_test)).item())
    if hasattr(model, 'times'):
        if model.times != 0:
            model.times = model.times_copy.clone()

loss_test_EKF = my_filter.EKF(state_test, obs_test, use_initial=True)
if hasattr(model, 'times'):
    if model.times != 0:
        model.times = model.times_copy.clone()

loss_test_EKF_FDI = my_filter.EKF(state_test, obs_test_FDI, use_initial=True)
if hasattr(model, 'times'):
    if model.times != 0:
        model.times = model.times_copy.clone()

print("Test_loss_EKF (no attacked) (dB): " + str(loss_test_EKF))
print("Test_loss_EKF (attacked) (dB): " + str(loss_test_EKF_FDI))
print("Test_loss_KalmanNet (dB): " + str(test_loss_KalmanNet))
print("Test_loss_Split-KalmanNet (dB): " + str(test_loss_Split_KalmanNet))
print("Test_loss_SDANSE (dB): " + str(test_loss_SDANSE))
print("Test_loss_IMUNet (dB): " + str(test_loss_IMUNet))
print("Test_loss_OutlierNet (dB): " + str(test_loss_OutlierNet))

#  绘制loss曲线

x1 = args.valid_period  # valid_period = 25
x2 = args.epoch  # train_iter = 500

plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体

fontsize = 18
labelsize = fontsize
linewidth = 3

x = np.arange(x1, x2 + 1, x1)
selection = np.arange(0, x.shape[0], 1)

fig, ax = plt.subplots()

# 绘制收敛曲线
loss_val = np.load('./Result/Elec/valid_loss_EKF.npy')
ax.plot(x[selection], np.ones(x[selection].shape) * loss_val,
        label='EKF',
        linewidth=linewidth, color='#000000', linestyle='dashed')

loss_val = np.load('./Result/Elec/valid_loss_KalmanNet.npy')
ax.plot(x[selection], loss_val[selection],
        label='KalmanNet',
        linewidth=linewidth, color='#0000ff', linestyle='solid')

loss_val = np.load('./Result/Elec/valid_loss_Split_KalmanNet.npy')
ax.plot(x[selection], loss_val[selection],
        label='Split-KalmanNet',
        linewidth=linewidth, color='#FF7F01', linestyle='solid')

loss_val = np.load('./Result/Elec/valid_loss_SDANSE.npy')
ax.plot(x[selection], loss_val[selection],
        label='DANSE',
        linewidth=linewidth, color='#BE0E98', linestyle='solid')

loss_val = np.load('./Result/Elec/valid_loss_IMUNet.npy')
ax.plot(x[selection], loss_val[selection],
        label='AI-IMU',
        linewidth=linewidth, color='#9467BD', linestyle='solid')

loss_val = np.load('./Result/Elec/valid_loss_OutlierNet.npy')
ax.plot(x[selection], loss_val[selection],
        label='ARKFNet',
        linewidth=linewidth, color='#00ff00', linestyle='solid')

ax.xaxis.set_tick_params(labelsize=labelsize)
ax.yaxis.set_tick_params(labelsize=labelsize)
ax.set_xlabel('Training round', fontsize=fontsize, fontweight='bold')
ax.set_ylabel('MSE [dB]', fontsize=fontsize, fontweight='bold')

ax.legend(fontsize=fontsize, prop={'size': 12})
# ax.legend(fontsize=fontsize, prop={'size': 12}, loc='center right',bbox_to_anchor=(1, 0.35))
# ax.legend(fontsize=fontsize, loc='lower left')
ax.grid()
plt.tight_layout()  # 确保元素值不超过plot中设置横纵坐标的范围
# plt.savefig('E:/Is_shanli/实验结果/ARKFNet/add_attack.eps', format='eps')

plt.show()
