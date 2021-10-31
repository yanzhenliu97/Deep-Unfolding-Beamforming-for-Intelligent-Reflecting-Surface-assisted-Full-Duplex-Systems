# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:07:47 2021

@author: Yanzhen Liu
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
import random
from operators import *
from parameters_config import Config
import os
import torch.optim as optim
import scipy.io  as scio 
from my_rician_channel import my_rician_channel
from UnfoldingWithStochasticTheta import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_default_dtype(torch.float64)



def compute_loss(HUbar,P,Htilde,F,HDbar,Jbar):
    uplink_sum_rate = 0
    downlink_sum_rate = 0
    AU = AU_update(HUbar,P,Htilde,F,HDbar,Jbar,K,L,N_r,N_t,sigmaU,sigmaD)
    AD = AD_update(HUbar,P,Htilde,F,HDbar,Jbar,K,L,N_D,N_t,sigmaU,sigmaD)
    for k in range(K):
        AU_temp = cmul(HUbar[:,:,:,k],cmul(P[:,:,:,k],cmul(conjT(P[:,:,:,k]),conjT(HUbar[:,:,:,k])))) 
        uplink_user_sinr = alpha*MyLogDet.apply(ceye(N_r) + cmul(AU_temp,MyInv.apply(AU[:,:,:,k] - AU_temp)))
        #uplink_user_sinr = alpha * MyLogDet.apply(ceye(N_r) + cmul(AU_temp, cinv(AU[:, :, :, k] - AU_temp)))
        #uplink_user_sinr = alpha * torch.log(complex_det(ceye(N_r) + cmul(AU_temp, cinv(AU[:, :, :, k] - AU_temp)))[0])
        uplink_sum_rate = uplink_sum_rate + uplink_user_sinr
    
    for l in range(L):
        AD_temp = cmul(HDbar[:,:,:,l],cmul(F[:,:,:,l],cmul(conjT(F[:,:,:,l]),conjT(HDbar[:,:,:,l])))) 
        downlink_user_sinr = belta*MyLogDet.apply(ceye(N_D) + cmul(AD_temp,MyInv.apply(AD[:,:,:,l] - AD_temp)))
        #downlink_user_sinr = belta * MyLogDet.apply(ceye(N_D) + cmul(AD_temp, cinv(AD[:, :, :, l] - AD_temp)))
        #downlink_user_sinr = belta * torch.log(complex_det(ceye(N_D) + cmul(AD_temp, cinv(AD[:, :, :, l] - AD_temp)))[0])
        downlink_sum_rate = downlink_sum_rate + downlink_user_sinr
    
    return uplink_sum_rate + downlink_sum_rate

def compute_seperate_loss(HUbar,P,Htilde,F,HDbar,Jbar):
    uplink_sum_rate = 0
    downlink_sum_rate = 0
    AU = AU_update(HUbar,P,Htilde,F,HDbar,Jbar,K,L,N_r,N_t,sigmaU,sigmaD)
    AD = AD_update(HUbar,P,Htilde,F,HDbar,Jbar,K,L,N_D,N_t,sigmaU,sigmaD)
    for k in range(K):
        AU_temp = cmul(HUbar[:,:,:,k],cmul(P[:,:,:,k],cmul(conjT(P[:,:,:,k]),conjT(HUbar[:,:,:,k])))) 
        uplink_user_sinr = alpha*MyLogDet.apply(ceye(N_r) + cmul(AU_temp,MyInv.apply(AU[:,:,:,k] - AU_temp)))
        #uplink_user_sinr = alpha * MyLogDet.apply(ceye(N_r) + cmul(AU_temp, cinv(AU[:, :, :, k] - AU_temp)))
        #uplink_user_sinr = alpha * torch.log(complex_det(ceye(N_r) + cmul(AU_temp, cinv(AU[:, :, :, k] - AU_temp)))[0])
        uplink_sum_rate = uplink_sum_rate + uplink_user_sinr
    
    for l in range(L):
        AD_temp = cmul(HDbar[:,:,:,l],cmul(F[:,:,:,l],cmul(conjT(F[:,:,:,l]),conjT(HDbar[:,:,:,l])))) 
        downlink_user_sinr = belta*MyLogDet.apply(ceye(N_D) + cmul(AD_temp,MyInv.apply(AD[:,:,:,l] - AD_temp)))
        #downlink_user_sinr = belta * MyLogDet.apply(ceye(N_D) + cmul(AD_temp, cinv(AD[:, :, :, l] - AD_temp)))
        #downlink_user_sinr = belta * torch.log(complex_det(ceye(N_D) + cmul(AD_temp, cinv(AD[:, :, :, l] - AD_temp)))[0])
        downlink_sum_rate = downlink_sum_rate + downlink_user_sinr
    # print('uplink_sum_rate')
    # print(uplink_sum_rate/alpha)
    # print('downlink_sum_rate')
    # print(downlink_sum_rate/belta)
    return uplink_sum_rate/alpha, downlink_sum_rate/belta

def produce_data(position_AP,position_IRS,position_down_users,position_up_users):
    scale_factor = 1e5
    HU,HD,VU,VD,J,GU,GD,Z = my_rician_channel(K,L,N_t,N_r,N_U,N_D,M_fai,position_AP,position_IRS,position_down_users,position_up_users)
    HU = HU*scale_factor 
    HD = HD*scale_factor
    VU = VU*np.sqrt(1e1)
    VD = VD*np.sqrt(1e1)
    J = J*scale_factor
    GU = GU*scale_factor
    GD = GD*scale_factor
    Z = Z*np.sqrt(1e1)
    
    Htilde = np.random.randn(N_r,N_t) + 1j*np.random.randn(N_r,N_t)
    Htilde = Htilde*0.01
    P = np.random.randn(N_U,M_U,K) + 1j*np.random.randn(N_U,M_U,K)    
    F = np.random.randn(N_t,M_D,L) + 1j*np.random.randn(N_t,M_D,L)
    for l in range(L):
        P[:,:,l] = P[:,:,l]/np.linalg.norm(P[:,:,l])*np.sqrt(P_t[l])

    F = F/np.linalg.norm(F)*np.sqrt(P_max)
    return P,F,HU,HD,VU,VD,J,GU,GD,Z,Htilde


def produce_data_set(num_data):
    P_set = []
    F_set = []
    HU_set = []
    HD_set = []
    VU_set = []
    VD_set = []
    J_set = []
    GU_set = []
    GD_set = []
    Z_set = []
    Htilde_set = []
    
    position_AP = np.zeros([3,1])
    y_IRS = 80
    z_IRS = 3
    position_IRS = np.expand_dims(np.array([0,80,z_IRS]),1)
    position_up_users = np.zeros([3,K])
    position_down_users = np.zeros([3,L])
    
    for i in range(K):
        position_up_users[0,i]= np.random.rand(1)*10;
        position_up_users[1,i]=(np.random.rand(1)-0.5)*10 + y_IRS
        position_up_users[2,i]=0
        # """fixed location"""
        position_up_users[0,i]= -10
        position_up_users[1,i]= -15+i*10 + y_IRS
        
        position_up_users[0,i]= -5*2
        position_up_users[1,i]= (-5+i*10)*2 + y_IRS        
    
    for i in range(L):
        position_down_users[0,i]= np.random.rand(1)*10;
        position_down_users[1,i]=(np.random.rand(1)-0.5)*10 + y_IRS
        position_down_users[2,i]=0
        # """fixed location"""
        position_down_users[0,i]= 10
        position_down_users[1,i]= -15+i*10 + y_IRS
        
        position_down_users[0,i]= 5*2
        position_down_users[1,i]= (-5+i*10)*2 + y_IRS
        
    for i in range(num_data):
       
        P,F,HU,HD,VU,VD,J,GU,GD,Z,Htilde = produce_data(position_AP,position_IRS,position_down_users,position_up_users)
        """numpy to tensor"""        
        HU = np2tensor(HU).to(dtype=dtype,device=device)
        P = np2tensor(P).to(dtype=dtype,device=device)
        VU = np2tensor(VU).to(dtype=dtype,device=device)
        GU = np2tensor(GU).to(dtype=dtype,device=device)
        Htilde = np2tensor(Htilde).to(dtype=dtype,device=device)
        HD = np2tensor(HD).to(dtype=dtype,device=device)
        F = np2tensor(F).to(dtype=dtype,device=device)
        VD = np2tensor(VD).to(dtype=dtype,device=device)
        GD = np2tensor(GD).to(dtype=dtype,device=device)
        J = np2tensor(J).to(dtype=dtype,device=device)
        Z = np2tensor(Z).to(dtype=dtype,device=device)
                 
        #P,F,Theta = BCD(Theta, HU, HD, VU, VD, J, GU, GD, Z, Htilde,P,F)
        
        P_set.append(P)
        F_set.append(F)      
        HU_set.append(HU)
        HD_set.append(HD)
        VU_set.append(VU)
        VD_set.append(VD)
        J_set.append(J)
        GU_set.append(GU)
        GD_set.append(GD)
        Z_set.append(Z)
        Htilde_set.append(Htilde)

    
    return P_set,F_set,HU_set,HD_set,VU_set,VD_set,J_set,GU_set,GD_set,Z_set,Htilde_set


def stackInputChannel(HU,HD,J,GU,GD,VU,VD,Z,Htilde):
    H_cat = torch.zeros(0)
    H_cat = torch.cat((H_cat,HU.reshape(-1),HD.reshape(-1),J.reshape(-1),GU.reshape(-1),GD.reshape(-1)),0)
    H_cat = torch.cat((H_cat,VU.reshape(-1),VD.reshape(-1),Z.reshape(-1),Htilde.reshape(-1)),0)
    totalElements = H_cat.numel()
    roundNum = math.ceil(totalElements/(N_r*N_U))
    zeroNum = roundNum*(N_r*N_U) - totalElements
    zeroTensor = torch.zeros(zeroNum)
    H_cat = torch.cat((H_cat,zeroTensor),0)
    
    H_cat = H_cat.reshape(-1,N_r,N_U)
    return H_cat.unsqueeze(0)

def getLayerNum():
    P_set,F_set,HU_set,HD_set,VU_set,VD_set,J_set,GU_set,GD_set,Z_set,Htilde_set = produce_data_set(1)
    for HU,HD,VU,VD,J,GU,GD,Z,Htilde in zip(HU_set,HD_set,VU_set,VD_set,J_set,GU_set,GD_set,Z_set,Htilde_set):
        H_cat = stackInputChannel(HU,HD,J,GU,GD,VU,VD,Z,Htilde)
    return H_cat.size(1)
    
def trainNetwork(model,optimizer,epochs=10):
    model = model.to(device=device,dtype=dtype)  # move the model parameters to CPU/GPU
    i = 0
    loss = 0
    total_loss = 0
    rho_t = 0
    gamma_t = 0
    varpi = 0.4
    theta_bar_t = torch.zeros(2,M_fai)
    theta_grad = torch.zeros(2,M_fai)
    print("start generating channels ============")
    P_set,F_set,HU_set,HD_set,VU_set,VD_set,J_set,GU_set,GD_set,Z_set,Htilde_set = produce_data_set(800)
    gradient_value = 0
    
    loss_list = []
    test_list = []
    for e in range(epochs):
        print("epochs %d ============"%e)

        for P,F,HU,HD,VU,VD,J,GU,GD,Z,Htilde in zip(P_set,F_set,HU_set,HD_set,VU_set,VD_set,J_set,GU_set,GD_set,Z_set,Htilde_set):
            P_final,F_final,Theta_final,UU_final,UD_final = model(P,F,HU,HD,VU,VD,J,GU,GD,Z,Htilde)
            HUbar,HDbar,Jbar = H_J_bar_update(HU,HD,J,Theta_final,GU,GD,VU,VD,Z,K,L)                        
            loss = -compute_loss(HUbar, P_final, Htilde, F_final, HDbar, Jbar)
            uprate_loss,downrate_loss = compute_seperate_loss(HUbar, P_final, Htilde, F_final, HDbar, Jbar)
            balance_loss = (uprate_loss-downrate_loss)**2
            balance_factor = 0.00
            new_loss = loss+balance_factor*balance_loss
            optimizer.zero_grad()
            new_loss.backward()
            loss_list.append(np.array(loss.detach()))

            layer_norm = torch.zeros(config.num_total_layer)
            layer_numel = torch.zeros(config.num_total_layer)
            for name, parms in model.named_parameters():
                if parms.grad is not None:
                    if name[:10] == 'MidLayer_f':
                        layer_index = int(name[11])                        
                        layer_norm[layer_index] += torch.norm(parms.grad.detach())**2
                        layer_numel[layer_index] += torch.numel(parms.grad)
                    # if i%100 ==0:
                    #         print(name)
                    #         print(torch.norm(parms.grad.detach())/torch.numel(parms.grad))
            if i%100 ==0:                
                print( torch.sqrt(layer_norm))
                
            layer_norm =  torch.sqrt(layer_norm)/layer_numel 
            
            for name, parms in model.named_parameters():
                if parms.grad is not None:                    
                    if name == "theta":
                           
                        parms.grad = parms.grad/(torch.norm(parms.grad)+1e-32)*torch.numel(parms.grad)
                            
                        # parms.grad = 0 * parms.grad
                        #parms.grad = parms.grad / (torch.norm(parms.grad)+1e-32)
                    if name[:10] == 'MidLayer_f':
                        layer_index = int(name[11])
                        
                        parms.grad = parms.grad / layer_norm[layer_index]
                        #parms.grad = parms.grad/(torch.norm(parms.grad)+1e-32)*torch.numel(parms.grad)
                        #print(parms.grad)
                        # if i<500:
                        #     parms.grad = parms.grad / layer_norm[layer_index]
                        # else:
                        #     parms.grad = parms.grad/torch.norm(layer_norm)*torch.numel(layer_norm)
                        #print(torch.norm(parms.grad)/torch.numel(parms.grad))
            
                    
            optimizer.step()
            total_loss += loss
            
            """The validation results are printed every 100 batches"""
            if i%100 ==0:
                #decaying learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.90
                    print('learning rate is')
                    print(param_group['lr'])
                
                print('loss is')
                print(total_loss/100)
                total_loss = 0
                result = 0
                uprate = 0
                downrate = 0                  
                model.eval()
                P_val,F_val,HU_val,HD_val,VU_val,VD_val,J_val,GU_val,GD_val,Z_val,Htilde_val = produce_data_set(50)
                for P,F,HU,HD,VU,VD,J,GU,GD,Z,Htilde in zip(P_val,F_val,HU_val,HD_val,VU_val,VD_val,J_val,GU_val,GD_val,Z_val,Htilde_val):
                    P_final,F_final,Theta_final,UU_final,UD_final = model(P,F,HU,HD,VU,VD,J,GU,GD,Z,Htilde)
                    HUbar,HDbar,Jbar = H_J_bar_update(HU,HD,J,Theta_final,GU,GD,VU,VD,Z,K,L)
                    result += -compute_loss(HUbar, P_final, Htilde, F_final, HDbar, Jbar)
                    uprate_tmp,downrate_tmp = compute_seperate_loss(HUbar, P_final, Htilde, F_final, HDbar, Jbar)
                    uprate += uprate_tmp
                    downrate +=downrate_tmp                   
                print('uplink rate is')
                print(uprate/50)
                print('downlink rate is')
                print(downrate/50)
                print('sum rate on the validation set is')
                print(result/50)
                
            i += 1 

                
"""parameters initialization"""
config = Config()
dtype = config.dtype # we will be using float
device = config.device
print('using device:', device)

K = config.K #uplink user number
L = config.L #downlink user number

N_r = config.num_BS_receive_antenna 
N_t = config.num_BS_transmit_antenna 
N_U = config.num_up_user_antenna 
N_D = config.num_down_user_antenna 
M_U = config.num_up_user_symbol
M_D = config.num_down_user_symbol
M_fai = config.num_reflecting_element

alpha = config.alpha
belta = config.belta

P_t = config.user_power;
P_max = config.BS_power;
sigmaU = config.sigma_up;
sigmaD = config.sigma_down;


seed = 2021
torch.manual_seed(seed)  
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) 
learning_rate = config.learning_rate
channel_layer = getLayerNum()

#train
channel_layer = getLayerNum()
MyThetaDNN = InitialThetaModel(config,channel_layer).to(device=device,dtype=dtype)
optimizer = optim.SGD(MyThetaDNN.parameters(),lr=learning_rate)
trainNetwork(MyThetaDNN, optimizer, epochs=int(5000/1000))
   
