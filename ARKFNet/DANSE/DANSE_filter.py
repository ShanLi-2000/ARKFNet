import torch
import math
from DANSE.DANSE_learner import DANSE_Learner
from torch import nn
from torch import optim

class DANSE_Filter:
    def __init__(self, args, model):

        self.update_lr = args.update_lr_danse
        self.model = model
        self.args = args

        self.danse_net = DANSE_Learner(self.model.y_dim, self.model.x_dim, n_hidden=30, n_layers=1, use_cuda=args.use_cuda)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.danse_net.parameters(), lr=self.update_lr)

    def compute_posterior_mean_vars(self, Yi_batch, cov_r):

        model_H = self.model.get_H()
        Re_t_inv = torch.inverse(model_H @ self.L_xt_yt_prev @ torch.transpose(model_H, 0, 1) + cov_r.to(Yi_batch.device))
        self.K_t = (self.L_xt_yt_prev @ (model_H.T @ Re_t_inv))
        self.mu_xt_yt_current = self.mu_xt_yt_prev + torch.einsum('ntij,ntj->nti', self.K_t, (Yi_batch - torch.einsum('ij,ntj->nti',model_H, self.mu_xt_yt_prev)))
        self.L_xt_yt_current = self.L_xt_yt_prev - (torch.einsum('ntij,ntjk->ntik',
                            self.K_t, model_H @ self.L_xt_yt_prev @ torch.transpose(model_H, 0, 1) + cov_r.to(Yi_batch.device)) @ torch.transpose(self.K_t, 2, 3))
        return self.mu_xt_yt_current, self.L_xt_yt_current

    def compute_predictions(self, Y_test_batch, cov_r):

        mu_x_given_Y_test_batch, vars_x_given_Y_test_batch = self.danse_net.forward(x=Y_test_batch)
        mu_xt_yt_prev_test, L_xt_yt_prev_test = self.compute_prior_mean_vars(
            mu_xt_yt_prev=mu_x_given_Y_test_batch,
            L_xt_yt_prev=vars_x_given_Y_test_batch
            )
        mu_xt_yt_current_test, L_xt_yt_current_test = self.compute_posterior_mean_vars(Yi_batch=Y_test_batch, cov_r=cov_r)
        return mu_xt_yt_prev_test, L_xt_yt_prev_test, mu_xt_yt_current_test, L_xt_yt_current_test

    def compute_prior_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev):

        self.mu_xt_yt_prev = mu_xt_yt_prev
        self.L_xt_yt_prev = torch.diag_embed(L_xt_yt_prev)
        return self.mu_xt_yt_prev, self.L_xt_yt_prev

    def compute_marginal_mean_vars(self, mu_xt_yt_prev, L_xt_yt_prev, cov_r):
        model_H = self.model.get_H()
        self.mu_yt_current = torch.einsum('ij,ntj->nti', model_H, mu_xt_yt_prev)
        self.L_yt_current = model_H @ L_xt_yt_prev @ torch.transpose(model_H, 0, 1) + cov_r.to(mu_xt_yt_prev.device)

    def compute_logpdf_Gaussian(self, Y):
        _, T, _ = Y.shape
        logprob = 0.5 * self.model.y_dim * T * math.log(math.pi * 2) - 0.5 * torch.logdet(self.L_yt_current).sum(1) \
                  - 0.5 * torch.einsum('nti,nti->nt',
                                       (Y - self.mu_yt_current),
                                       torch.einsum('ntij,ntj->nti', torch.inverse(self.L_yt_current),
                                                    (Y - self.mu_yt_current))).sum(1)

        return logprob

    def compute_supervised_logpdf_Gaussian(self, X):
        _, T, _ = X.shape
        logprob = 0.5 * self.model.x_dim * T * math.log(math.pi * 2) - 0.5 * torch.logdet(self.L_xt_yt_current).sum(1) \
                  - 0.5 * torch.einsum('nti,nti->nt',
                                       (X - self.mu_xt_yt_current),
                                       torch.einsum('ntij,ntj->nti', torch.inverse(self.L_xt_yt_current),
                                                    (X - self.mu_xt_yt_current))).sum(1)
        # logprob = nn.MSELoss()(X, self.mu_xt_yt_current)

        return logprob

    def DANSE_filtering(self, Yi_batch, cov_r, Xi_batch):
        # observation: column vector

        mu_batch, vars_batch = self.danse_net.forward(x=Yi_batch)
        mu_xt_yt_prev, L_xt_yt_prev = self.compute_prior_mean_vars(mu_xt_yt_prev=mu_batch, L_xt_yt_prev=vars_batch)
        self.compute_marginal_mean_vars(mu_xt_yt_prev=mu_xt_yt_prev, L_xt_yt_prev=L_xt_yt_prev, cov_r=cov_r)
        _ = self.compute_posterior_mean_vars(Yi_batch=Yi_batch, cov_r=cov_r)
        # logprob_batch = self.compute_logpdf_Gaussian(Y=Yi_batch) / (Yi_batch.shape[1] * Yi_batch.shape[2])  # Per dim. and per sequence length
        logprob_batch = self.compute_supervised_logpdf_Gaussian(X=Xi_batch) / (Xi_batch.shape[1] * Xi_batch.shape[2])
        # logprob_batch = self.compute_supervised_logpdf_Gaussian(X=Xi_batch)
        log_pYT_batch_avg = logprob_batch.mean(0)

        return log_pYT_batch_avg