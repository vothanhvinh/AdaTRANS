# -*- coding: utf-8 -*-
import torch
import gpytorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FLAGS_NO_TRANSFER = 0
FLAGS_FULL_TRANSFER = 1
FLAGS_LEARN_TRANSFER = 2
#=============================================================================================================================================
""" Approximate $P(W\,|\,X)$ """

class ModelW(torch.nn.Module):
  def __init__(self, xobs, n_domains, domain_ranges, transfer_flag=FLAGS_LEARN_TRANSFER):
    super().__init__()
    self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    self.kernel = gpytorch.kernels.MaternKernel(5/2)
    self.transfer_factor_logit = torch.nn.Parameter(torch.zeros(int(((n_domains-1)*n_domains)/2)))
    self.sigmoid = torch.nn.Sigmoid()
    self.n_domains = n_domains
    self.domain_ranges = domain_ranges
    self.xobs = xobs
    self.alpha = torch.nn.Parameter(torch.rand((xobs.shape[0],1)))
    self.transfer_flag = transfer_flag

  def forward(self, x, domain_ranges, return_K=False):
    transfer_factor = self.sigmoid(self.transfer_factor_logit)

    M = self.kernel(self.xobs, x).evaluate()
    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      K = M
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      ones_matrices = [torch.ones((self.domain_ranges[i][1]-self.domain_ranges[i][0], domain_ranges[i][1]-domain_ranges[i][0]), device=device)
                        for i in range(self.n_domains)]
      C = torch.block_diag(*ones_matrices)
      K = C*M
    elif self.transfer_flag == FLAGS_LEARN_TRANSFER:
      K = M
      n = 0
      for i in range(self.n_domains):
        for j in range(i+1,self.n_domains):
          marker1_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker1_obs[self.domain_ranges[i][0]:self.domain_ranges[i][1],:] = -1
          marker1 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + x.shape[0], device=device).reshape(-1,1)
          marker1[domain_ranges[j][0]:domain_ranges[j][1],:] = -1
          C1 = (marker1_obs==marker1.t())*1.0

          marker2_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker2_obs[self.domain_ranges[j][0]:self.domain_ranges[j][1],:] = -1
          marker2 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + x.shape[0], device=device).reshape(-1,1)
          marker2[domain_ranges[i][0]:domain_ranges[i][1],:] = -1
          C2 = (marker2_obs==marker2.t())*1.0

          C = C1 + C2
          
          K = (transfer_factor[n]*C + 1-C)*K
          n += 1

    if return_K==False:
      return self.alpha.t().matmul(K)
    else:
      return self.alpha.t().matmul(K), K

  def trans_factor(self):
    return self.sigmoid(self.transfer_factor_logit)

  def pred(self, x, domain_ranges):
    f_preds = self.forward(x, domain_ranges=domain_ranges)
    return f_preds

  def sample(self, x, n_samples):
    domain_ranges = [(0,0)]*(self.n_domains-1) + [(0,x.shape[0])]
    f_preds = self.forward(x, domain_ranges=domain_ranges).reshape(-1)
    m = torch.distributions.bernoulli.Bernoulli(logits=f_preds)
    w_samples = m.sample((n_samples,)).t()
    return w_samples

  def loss(self, x, w, domain_ranges, reg_alpha=0.1):
    wpred_logit, K = self.forward(x, domain_ranges, return_K=True)
    return self.loss_bce(wpred_logit.reshape(-1), w.reshape(-1)) + reg_alpha*self.alpha.t().matmul(K).matmul(self.alpha)

#=============================================================================================================================================
""" Approximate $P(Y\,|\,W,X)$ """

class ModelY(torch.nn.Module):
  def __init__(self, xobs, wobs, n_domains, domain_ranges, transfer_flag=FLAGS_LEARN_TRANSFER):
    super().__init__()
    self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    self.loss_mse = torch.nn.MSELoss(reduction='sum')
    self.kernel = gpytorch.kernels.MaternKernel(5/2)
    self.transfer_factor_logit = torch.nn.Parameter(torch.zeros(int(((n_domains-1)*n_domains)/2)))
    self.sigma_y_logit = torch.nn.Parameter(torch.tensor(0.0))
    self.sigmoid = torch.nn.Sigmoid()
    self.n_domains = n_domains
    self.domain_ranges = domain_ranges
    self.xobs = xobs
    self.wobs = wobs
    self.xwobs = torch.cat((xobs,wobs),dim=1)
    self.alpha0 = torch.nn.Parameter(torch.rand((xobs.shape[0],1)))
    self.alpha1 = torch.nn.Parameter(torch.rand((xobs.shape[0],1)))
    self.transfer_flag = transfer_flag # 0: no transfer, 1: full transfer, 2: learn transfer factor

  def forward(self, x, w, domain_ranges, return_K=False):
    transfer_factor = self.sigmoid(self.transfer_factor_logit)

    # xw = torch.cat((x,w),dim=1)
    M = self.kernel(self.xobs, x).evaluate()
    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      K = M
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      ones_matrices = [torch.ones((self.domain_ranges[i][1]-self.domain_ranges[i][0], domain_ranges[i][1]-domain_ranges[i][0]), device=device)
                        for i in range(self.n_domains)]
      C = torch.block_diag(*ones_matrices)
      K = C*M
    else:
      K = M
      n = 0
      for i in range(self.n_domains):
        for j in range(i+1,self.n_domains):
          marker1_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker1_obs[self.domain_ranges[i][0]:self.domain_ranges[i][1],:] = -1
          marker1 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + x.shape[0], device=device).reshape(-1,1)
          marker1[domain_ranges[j][0]:domain_ranges[j][1],:] = -1
          C1 = (marker1_obs==marker1.t())*1.0

          marker2_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker2_obs[self.domain_ranges[j][0]:self.domain_ranges[j][1],:] = -1
          marker2 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + x.shape[0], device=device).reshape(-1,1)
          marker2[domain_ranges[i][0]:domain_ranges[i][1],:] = -1
          C2 = (marker2_obs==marker2.t())*1.0

          C = C1 + C2
          
          K = (transfer_factor[n]*C + 1-C)*K
          n += 1

    f = w.reshape(-1)*(self.alpha0.t().matmul(K)).reshape(-1) + (1-w).reshape(-1)*(self.alpha1.t().matmul(K)).reshape(-1)
    if return_K==False:
      return f
    else:
      return f, K

  def trans_factor(self):
    return self.sigmoid(self.transfer_factor_logit)

  def pred(self, x, w, domain_ranges):
    f_preds = self.forward(x, w, domain_ranges=domain_ranges)
    return f_preds

  def sample(self, x, w_samples, n_samples, is_binary=False):
    y_samples = []
    domain_ranges = [(0,0)]*(self.n_domains-1) + [(0,x.shape[0])]
    for i in range(n_samples):
      f_preds = self.forward(x, w_samples[:,i:i+1], domain_ranges=domain_ranges).reshape(-1)
      if is_binary == True:
        m = torch.distributions.bernoulli.Bernoulli(logits=f_preds)
      else:
        m = torch.distributions.normal.Normal(f_preds, torch.exp(self.sigma_y_logit))
      y_sample = m.sample((1,)).t()
      y_samples.append(y_sample)
    return torch.cat(y_samples, dim=1)

  def loss(self, x, y, w, domain_ranges, is_binary=False, reg_alpha=0.1, reg_noise=0.1):
    ypred_logit, K = self.forward(x, w, domain_ranges, return_K=True)
    reg = reg_alpha*self.alpha0.t().matmul(K).matmul(self.alpha0) + reg_alpha*self.alpha1.t().matmul(K).matmul(self.alpha1)
    if is_binary == True:
      return self.loss_bce(ypred_logit.reshape(-1), y.reshape(-1)) + reg
    else:
      reg = reg + reg_noise*self.sigma_y_logit**2
      return x.shape[0]*self.sigma_y_logit + (0.5/torch.exp(2*self.sigma_y_logit))*self.loss_mse(ypred_logit.reshape(-1), y.reshape(-1)) + reg

#=============================================================================================================================================
""" Approximate $P(Z\,|\, X, Y, W)$ and $P(Y\,|\, Z, W)$ """

class ModelZY(torch.nn.Module):
  def __init__(self, xobs, yobs, wobs, n_domains, domain_ranges, dim_z=16, transfer_flag=FLAGS_LEARN_TRANSFER):
    super().__init__()
    self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
    self.loss_mse = torch.nn.MSELoss(reduction='sum')
    self.kernel = gpytorch.kernels.MaternKernel(5/2)
    self.transfer_factor_logit = torch.nn.Parameter(torch.zeros(int(((n_domains-1)*n_domains)/2)))
    self.sigma_y_logit = torch.nn.Parameter(torch.tensor(0.0))
    self.sigmoid = torch.nn.Sigmoid()
    self.n_domains = n_domains
    self.domain_ranges = domain_ranges
    self.xobs = xobs
    self.yobs = yobs
    self.wobs = wobs
    self.xyobs = torch.cat((xobs,yobs),dim=1)
    self.dim_z = dim_z
    self.alphaz0 = torch.nn.Parameter(torch.rand((xobs.shape[0],dim_z)))
    self.alphaz1 = torch.nn.Parameter(torch.rand((xobs.shape[0],dim_z)))
    self.sigmaz_logit = torch.nn.Parameter(torch.rand(dim_z))
    self.alphay0 = torch.nn.Parameter(torch.rand((xobs.shape[0],1)))
    self.alphay1 = torch.nn.Parameter(torch.rand((xobs.shape[0],1)))
    self.alphaw = torch.nn.Parameter(torch.rand((xobs.shape[0],1)))
    self.alphax = torch.nn.Parameter(torch.rand((xobs.shape[0],xobs.shape[1])))
    self.transfer_flag = transfer_flag # 0: no transfer, 1: full transfer, 2: learn transfer factor


  def fz(self, x, y, w, domain_ranges, return_K=False):
    transfer_factor = self.sigmoid(self.transfer_factor_logit)

    xy = torch.cat((x,y),dim=1)
    Mz = self.kernel(self.xyobs, xy).evaluate()
    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      Kz = Mz
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      ones_matrices = [torch.ones((self.domain_ranges[i][1]-self.domain_ranges[i][0], domain_ranges[i][1]-domain_ranges[i][0]), device=device)
                        for i in range(self.n_domains)]
      C = torch.block_diag(*ones_matrices)
      Kz = C*Mz
    else:
      Kz = Mz
      n = 0
      for i in range(self.n_domains):
        for j in range(i+1,self.n_domains):
          marker1_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker1_obs[self.domain_ranges[i][0]:self.domain_ranges[i][1],:] = -1
          marker1 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + x.shape[0], device=device).reshape(-1,1)
          marker1[domain_ranges[j][0]:domain_ranges[j][1],:] = -1
          C = (marker1_obs==marker1.t())*1.0

          marker2_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker2_obs[self.domain_ranges[j][0]:self.domain_ranges[j][1],:] = -1
          marker2 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + x.shape[0], device=device).reshape(-1,1)
          marker2[domain_ranges[i][0]:domain_ranges[i][1],:] = -1
          C = C + (marker2_obs==marker2.t())*1.0

          Kz = (transfer_factor[n]*C + 1-C)*Kz
          n += 1

    f = (w.reshape(-1)*(self.alphaz0.t().matmul(Kz)) + (1-w).reshape(-1)*(self.alphaz1.t().matmul(Kz))).t()
    if return_K==False:
      return f, None
    else:
      return f, Kz

  def fy(self, w, z, z_samples, domain_ranges, return_K=False):
    transfer_factor = self.sigmoid(self.transfer_factor_logit)

    xzsamples = z_samples #torch.cat((self.xobs,z_samples),dim=1)
    xz = z #torch.cat((x,z),dim=1)
    My = self.kernel(xzsamples, xz).evaluate()
    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      Ky = My
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      ones_matrices = [torch.ones((self.domain_ranges[i][1]-self.domain_ranges[i][0], domain_ranges[i][1]-domain_ranges[i][0]), device=device)
                        for i in range(self.n_domains)]
      C = torch.block_diag(*ones_matrices)
      Ky = C*My
    else:
      Ky = My
      n = 0
      for i in range(self.n_domains):
        for j in range(i+1,self.n_domains):
          marker1_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker1_obs[self.domain_ranges[i][0]:self.domain_ranges[i][1],:] = -1
          marker1 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + z.shape[0], device=device).reshape(-1,1)
          marker1[domain_ranges[j][0]:domain_ranges[j][1],:] = -1
          C = (marker1_obs==marker1.t())*1.0

          marker2_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker2_obs[self.domain_ranges[j][0]:self.domain_ranges[j][1],:] = -1
          marker2 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + z.shape[0], device=device).reshape(-1,1)
          marker2[domain_ranges[i][0]:domain_ranges[i][1],:] = -1
          C = C + (marker2_obs==marker2.t())*1.0
          
          Ky = (transfer_factor[n]*C + 1-C)*Ky
          n += 1

    f = w.reshape(-1)*(self.alphay0.t().matmul(Ky)).reshape(-1) + (1-w).reshape(-1)*(self.alphay1.t().matmul(Ky)).reshape(-1)
    if return_K==False:
      return f, None
    else:
      return f, Ky

  def fw(self, z, z_samples, domain_ranges, return_K=False):
    transfer_factor = self.sigmoid(self.transfer_factor_logit)

    Mw = self.kernel(z_samples, z).evaluate()
    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      Kw = Mw
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      ones_matrices = [torch.ones((self.domain_ranges[i][1]-self.domain_ranges[i][0], domain_ranges[i][1]-domain_ranges[i][0]), device=device)
                        for i in range(self.n_domains)]
      C = torch.block_diag(*ones_matrices)
      Kw = C*Mw
    else:
      Kw = Mw
      n = 0
      for i in range(self.n_domains):
        for j in range(i+1,self.n_domains):
          marker1_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker1_obs[self.domain_ranges[i][0]:self.domain_ranges[i][1],:] = -1
          marker1 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + z.shape[0], device=device).reshape(-1,1)
          marker1[domain_ranges[j][0]:domain_ranges[j][1],:] = -1
          C = (marker1_obs==marker1.t())*1.0

          marker2_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker2_obs[self.domain_ranges[j][0]:self.domain_ranges[j][1],:] = -1
          marker2 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + z.shape[0], device=device).reshape(-1,1)
          marker2[domain_ranges[i][0]:domain_ranges[i][1],:] = -1
          C = C + (marker2_obs==marker2.t())*1.0
          
          Kw = (transfer_factor[n]*C + 1-C)*Kw
          n += 1

    f = self.alphaw.t().matmul(Kw)
    if return_K==False:
      return f, None
    else:
      return f, Kw

  def fx(self, z, z_samples, domain_ranges, return_K=False):
    transfer_factor = self.sigmoid(self.transfer_factor_logit)

    Mx = self.kernel(z_samples, z).evaluate()

    if self.transfer_flag == FLAGS_FULL_TRANSFER:
      Kx = Mx
    elif self.transfer_flag == FLAGS_NO_TRANSFER:
      ones_matrices = [torch.ones((self.domain_ranges[i][1]-self.domain_ranges[i][0], domain_ranges[i][1]-domain_ranges[i][0]), device=device)
                        for i in range(self.n_domains)]
      C = torch.block_diag(*ones_matrices)
      Kx = C*Mx
    else:
      Kx = Mx
      n = 0
      for i in range(self.n_domains):
        for j in range(i+1,self.n_domains):
          marker1_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker1_obs[self.domain_ranges[i][0]:self.domain_ranges[i][1],:] = -1
          marker1 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + z.shape[0], device=device).reshape(-1,1)
          marker1[domain_ranges[j][0]:domain_ranges[j][1],:] = -1
          C = (marker1_obs==marker1.t())*1.0

          marker2_obs = torch.arange(0,self.xobs.shape[0], device=device).reshape(-1,1)
          marker2_obs[self.domain_ranges[j][0]:self.domain_ranges[j][1],:] = -1
          marker2 = torch.arange(self.xobs.shape[0], self.xobs.shape[0] + z.shape[0], device=device).reshape(-1,1)
          marker2[domain_ranges[i][0]:domain_ranges[i][1],:] = -1
          C = C + (marker2_obs==marker2.t())*1.0
          
          Kx = (transfer_factor[n]*C + 1-C)*Kx
          n += 1

    f = (self.alphax.t().matmul(Kx)).t()

    if return_K==False:
      return f, None
    else:
      return f, Kx

  def forward(self, x, y, w, domain_ranges, return_K=False):
    fz, Kz = self.fz(x, y, w, domain_ranges, return_K=return_K)
    sigmaz = torch.exp(self.sigmaz_logit)
    z_samples = fz + sigmaz*torch.rand((x.shape[0], self.dim_z), device=device)
    fy, Ky = self.fy(w, z_samples, z_samples, domain_ranges, return_K=return_K)
    fw, Kw = self.fw(z_samples, z_samples, domain_ranges, return_K=return_K)
    fx, Kx = self.fx(z_samples, z_samples, domain_ranges, return_K=return_K)
    return fz, Kz, fy, Ky, fw, Kw, fx, Kx
    # return fz, Kz, fy, Ky

  def trans_factor(self):
    return self.sigmoid(self.transfer_factor_logit).detach()

  def pred_z(self, x, y, w, domain_ranges):
    f_preds,_ = self.fz(x, y, w, domain_ranges=domain_ranges)
    return f_preds.detach().clone()

  def pred_y(self, x, w, z, zobs_pred, domain_ranges):
    f_preds,_ = self.fy(w, z, zobs_pred, domain_ranges=domain_ranges)
    return f_preds.detach().clone()

  def sample_z(self, x, y_samples, w_samples, n_samples):
    sigmaz = torch.exp(self.sigmaz_logit)
    z_samples = []
    domain_ranges = [(0,0)]*(self.n_domains-1) + [(0,x.shape[0])]
    for i in range(n_samples):
      fz = self.pred_z(x, y_samples[:,i:i+1], w_samples[:,i:i+1], domain_ranges=domain_ranges)
      z_sample = fz + sigmaz*torch.rand((x.shape[0], self.dim_z), device=device)
      z_samples.append(z_sample)

    return z_samples

  def sample_y(self, x, do_w, z_sample, zobs_pred, n_samples, is_binary=False):
    domain_ranges = [(0,0)]*(self.n_domains-1) + [(0,x.shape[0])]
    f_preds = self.pred_y(x, do_w, z_sample, zobs_pred, domain_ranges=domain_ranges)

    if is_binary == True:
      m = torch.distributions.bernoulli.Bernoulli(logits=f_preds)
    else:
      m = torch.distributions.normal.Normal(f_preds, 1)
    y_samples = m.sample((n_samples,)).t()
    return y_samples

  def sample(self, x, do_w, y_samples, w_samples, n_samples, is_binary=False):
    zobs_pred = self.pred_z(self.xobs, self.yobs, self.wobs, domain_ranges=self.domain_ranges)
    
    z_samples = self.sample_z(x, y_samples, w_samples, n_samples)
    y_samples = []
    for z_sample in z_samples:
      y_sample = self.sample_y(x, do_w, z_sample=z_sample, zobs_pred=zobs_pred, n_samples=1, is_binary=is_binary)
      y_samples.append(y_sample)

    return  torch.cat(y_samples,dim=1), z_samples

  def sample_v2(self, x, do_w, y_samples, w_samples, n_samples, is_binary=False):
    zobs_pred = self.pred_z(self.xobs, self.yobs, self.wobs, domain_ranges=self.domain_ranges)
    
    z_samples = self.sample_z(x, y_samples, w_samples, n_samples)
    y_samples = []
    domain_ranges = [(0,0)]*(self.n_domains-1) + [(0,x.shape[0])]
    for z_sample in z_samples:
      if is_binary==False:
        y_sample = self.pred_y(x, do_w, z_sample, zobs_pred, domain_ranges=domain_ranges)
      else:
        y_sample = self.sigmoid(self.pred_y(x, do_w, z_sample, zobs_pred, domain_ranges=domain_ranges))
      y_samples.append(y_sample.reshape(-1,1))

    return  torch.cat(y_samples,dim=1), z_samples

  def loss(self, x, y, w, domain_ranges, feats_binary=None, feats_continuous=None, is_binary=False, reg_alpha=0.1):
    zpred, Kz, ypred_logit, Ky, wpred_logit, Kw, xpred, Kx = self.forward(x, y, w, domain_ranges, return_K=True)
    KL = 0.5*torch.sum(torch.exp(2*self.sigmaz_logit) - 2*self.sigmaz_logit)*x.shape[0] + 0.5*torch.sum(zpred**2)
    reg = reg_alpha*torch.sum(self.alphaz0.t().matmul(Kz)*self.alphaz0.t()) + reg_alpha*torch.sum(self.alphaz1.t().matmul(Kz)*self.alphaz1.t()) \
          + reg_alpha*self.alphay0.t().matmul(Ky).matmul(self.alphay0) + reg_alpha*self.alphay1.t().matmul(Ky).matmul(self.alphay1) \
          + reg_alpha*self.alphaw.t().matmul(Kw).matmul(self.alphaw) \
          + reg_alpha*torch.sum(self.alphax.t().matmul(Kx)*self.alphax.t())
        
        
    if feats_continuous==None and feats_binary==None: # default is bce for binary features
        loss_x = self.loss_bce(xpred, x)
    elif feats_continuous==None: # for binary features
        loss_x = self.loss_bce(xpred, x)
    elif feats_binary==None: # for continuous features
        loss_x = self.loss_mse(xpred, x)
    else: # for both binary and continuous features
        loss_x = self.loss_bce(xpred[:,feats_binary], x[:,feats_binary]) + self.loss_mse(xpred[:,feats_continuous], x[:,feats_continuous])
    
    if is_binary == True: # binary outcomes
      return self.loss_bce(ypred_logit.reshape(-1), y.reshape(-1)) \
             + self.loss_bce(wpred_logit.reshape(-1), w.reshape(-1)) \
             + loss_x \
             + KL + reg
    else: # continuous outcomes
      return self.loss_mse(ypred_logit.reshape(-1), y.reshape(-1)) \
             + self.loss_bce(wpred_logit.reshape(-1), w.reshape(-1)) \
             + loss_x \
             + KL + reg
