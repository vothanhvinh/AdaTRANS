# -*- coding: utf-8 -*-
import numpy as np
import torch
from models import *

#======================================================================================================================
''' ## Approximate $P(W\,|\,X)$ '''
def trainW(train_x, train_w, n_domains, domain_ranges, training_iter=5000, display_per_iter=100,
           transfer_flag=FLAGS_LEARN_TRANSFER, reg_alpha=1e-3, lr=1e-3):


  model = ModelW(xobs=train_x, n_domains=n_domains, domain_ranges=domain_ranges, transfer_flag=transfer_flag).to(device)

  # Use the adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  for i in range(training_iter):
    loss = model.loss(train_x, train_w, domain_ranges, reg_alpha=reg_alpha)
    if (i+1)%display_per_iter==0:
      print('Iter %d/%d - Loss: %.3f   Transfer Ratio: %s' % (i + 1, training_iter, loss.item(),
                                                                np.round(model.trans_factor().cpu().detach().numpy(),3)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return model

#======================================================================================================================
''' ## Train $P(Y\,|\,W,X)$ '''
def trainY(train_x, train_y, train_w, n_domains, domain_ranges, is_binary=False, training_iter=5000, display_per_iter=100,
           transfer_flag=FLAGS_LEARN_TRANSFER, reg_alpha=1e-2, reg_noise=1e-1, lr=1e-3):

  model = ModelY(xobs=train_x, wobs=train_w, n_domains=n_domains, domain_ranges=domain_ranges,
                 transfer_flag=transfer_flag).to(device)

  # Use the adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  for i in range(training_iter):
    loss = model.loss(train_x, train_y, train_w, domain_ranges, is_binary=is_binary, reg_alpha=reg_alpha, reg_noise=reg_noise)
    if (i+1)%display_per_iter==0:
      print('Iter %d/%d - Loss: %.3f   Transfer Ratio: %s' % (i + 1, training_iter, loss.item(),
                                                                np.round(model.trans_factor().cpu().detach().numpy(),3)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return model

#======================================================================================================================
''' ## Train $P(Z\,|\, X, Y, W)$ and $P(Y\,|\, Z, W)$ '''
def trainZY(train_x, train_y, train_w, n_domains, domain_ranges, feats_binary=None, feats_continuous=None,
            is_binary=False, dim_z=16, training_iter=5000, display_per_iter=100,
            transfer_flag=FLAGS_LEARN_TRANSFER, reg_alpha=1e-1, lr=1e-2):

  model = ModelZY(xobs=train_x, yobs=train_y, wobs=train_w, dim_z=dim_z,
                  n_domains=n_domains, domain_ranges=domain_ranges, transfer_flag=transfer_flag).to(device)

  # Use the adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  for i in range(training_iter):
    loss = model.loss(train_x, train_y, train_w, domain_ranges, feats_binary=feats_binary, feats_continuous=feats_continuous,
                      is_binary=is_binary, reg_alpha=reg_alpha)
    if (i+1)%display_per_iter==0:
      print('Iter %d/%d - Loss: %.3f   Transfer Ratio: %s' % (i + 1, training_iter, loss.item(),
                                                                np.round(model.trans_factor().cpu().detach().numpy(),3)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return model