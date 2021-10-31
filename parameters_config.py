# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:59:34 2020

@author: Yanzhen Liu
"""
import numpy as np
import torch
class Config(object):
    def __init__(self):
        self.USE_GPU = False
        if self.USE_GPU == False:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')
        self.dtype = torch.float64
        self.K = 2 #uplink user number
        self.L = 2 #downlink user number
        self.learning_rate = 0.01
        self.num_total_layer = 8
        
        self.num_BS_receive_antenna = 32; 
        self.num_BS_transmit_antenna = 32; 
        self.num_up_user_antenna = 4; 
        self.num_up_user_symbol = 4; 
        self.num_down_user_antenna = 4;
        self.num_down_user_symbol = 4; 
        self.num_reflecting_element = 200; 
        
        self.alpha = 1; 
        self.belta = 1;
        
        self.user_power = 1*np.ones(self.K);
        self.BS_power = 100;
        self.sigma_up = 1;
        self.sigma_down = 1;
        