# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# Synthetic dataset multi-source
class SynDataMultiSrc:
  def __init__(self, source_size=1000):
    data = np.load('data/synthetic/data-multi-source-F.npz', allow_pickle=True)
    self.T = data['T']
    self.Tt = data['Tt']
    self.Ts = data['Ts']
    self.data_lst = data['data_lst'][0]
    self.n_replicates = data['n_replicates']
    self.n_sources = data['n_sources']
    self.source_size = source_size
    
  def get_source_target(self, m_sources=1):
    for i in range(self.n_replicates):
      data = self.data_lst[i]
      t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]

      isrc = range(0,m_sources*self.source_size)
      itar = range(4000,5000)
      src = (x[isrc], t[isrc], y[isrc]), (y_cf[isrc], mu_0[isrc], mu_1[isrc])
      tar = (x[itar], t[itar], y[itar]), (y_cf[itar], mu_0[itar], mu_1[itar])
      yield src, tar

# Synthetic dataset one source
class SynDataOneSrc:
  def __init__(self):
    data = np.load('data/synthetic/data-one-source-F.npz', allow_pickle=True)
    self.T = data['T']
    self.Tt = data['Tt']
    self.datas_lst = data['datas_lst']
    self.datat_lst = data['datat_lst']
    self.n_replicates = data['n_replicates']
    
  def get_source_target(self, discrepancy_idx):
    for i in range(self.n_replicates):
      datat = self.datat_lst[i]
      datas = self.datas_lst[i][discrepancy_idx]
      data = np.concatenate((datas, datat),axis=0)

      t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]

      isrc = range(0,1000)
      itar = range(1000,2000)
      src = (x[isrc], t[isrc], y[isrc]), (y_cf[isrc], mu_0[isrc], mu_1[isrc])
      tar = (x[itar], t[itar], y[itar]), (y_cf[itar], mu_0[itar], mu_1[itar])
      yield src, tar

# TWINS dataset
class TWINS:
  def __init__(self):
    self.k_folds = 9
  def get_source_target(self, delta=0.5):
    for i in range(self.k_folds):
      data_source = pd.read_csv('data/TWINS/data_discrepancy_w_source_{}.csv'.format(delta),header=None).values
      data_target = pd.read_csv('data/TWINS/data_discrepancy_w_target_{}.csv'.format(delta),header=None).values

      # put training samples first and then testing and validation
      idx_target_train = range(i*100, (i+1)*100)
      data_target1 = data_target[idx_target_train,:]
      idx_target_test_val = np.sort(list(set(range(0,data_target.shape[0])) - set(idx_target_train)))
      data_target2 = data_target[idx_target_test_val,:]
      data_target = np.concatenate((data_target1,data_target2),axis=0)

      source_ranges = [(0,data_source.shape[0])]
      yield data_source, data_target, source_ranges#list(map(tuple, source_ranges))