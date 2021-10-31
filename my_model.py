# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:52:04 2020

@author: Yanzhen Liu
"""


import numpy as np
import torch
from operators import *
from parameters_config import Config
import torch.nn as nn
import math
import torch.nn.functional as Func
import scipy

class model(nn.Module):
    def __init__(self,config):
        super(model,self).__init__()
        self.n = config.num_total_layer
        self.K = config.K #uplink user number
        self.L = config.L #downlink user number        
        self.N_r = config.num_BS_receive_antenna 
        self.N_t = config.num_BS_transmit_antenna 
        self.N_U = config.num_up_user_antenna 
        self.N_D = config.num_down_user_antenna 
        self.M_U = config.num_up_user_symbol
        self.M_D = config.num_down_user_symbol
        self.M_fai = config.num_reflecting_element
        self.MidLayer_f = {}
        self.device = config.device
        
        for i in range(self.n):
            self.MidLayer_f["%d"%(i)] = MidLayer(config)
        self.MidLayer_f = nn.ModuleDict(self.MidLayer_f)
        self.FinalLayer_f = FinalLayer(config)
        
    def forward(self,P,F,Theta,HU,HD,VU,VD,J,GU,GD,Z,Htilde):
        P_c = {}
        P_c['0'] = P
        F_c = {}
        F_c['0'] = F
        Theta_c = {}
        Theta_c['0'] = Theta
        for i in range(self.n):
            #print("midder layer==== ",i)
            P_c['%d'%(i+1)],F_c['%d'%(i+1)],Theta_c['%d'%(i+1)] = self.MidLayer_f["%d"%(i)](P_c['%d'%(i)],F_c['%d'%(i)],Theta_c['%d'%(i)],HU,HD,VU,VD,J,GU,GD,Z,Htilde)
       
        return self.FinalLayer_f(P_c['%d'%(self.n)],F_c['%d'%(self.n)],Theta_c['%d'%(self.n)],HU,HD,VU,VD,J,GU,GD,Z,Htilde)
#中间层

class MidLayer(nn.Module):
    def __init__(self,config):
        super(MidLayer,self).__init__()
        self.K = config.K #uplink user number
        self.L = config.L #downlink user number        
        self.N_r = config.num_BS_receive_antenna 
        self.N_t = config.num_BS_transmit_antenna 
        self.N_U = config.num_up_user_antenna 
        self.N_D = config.num_down_user_antenna 
        self.M_U = config.num_up_user_symbol
        self.M_D = config.num_down_user_symbol
        self.M_fai = config.num_reflecting_element
        self.sigmaU = config.sigma_up
        self.sigmaD = config.sigma_down
        self.MidLayer_f = {}
        self.device = config.device
        self.alpha = config.alpha
        self.belta = config.belta
        self.P_t = config.user_power
        self.Pmax = config.BS_power
        
        self.UU_f = U_unfolding(self.N_r,self.M_U,self.K)
        self.UD_f = U_unfolding(self.N_D,self.M_D,self.L)
        self.WU_f = W_unfolding(self.M_U,self.M_U,self.K)
        self.WD_f = W_unfolding(self.M_D,self.M_D,self.L)
        self.P_f = PF_unfolding(self.N_U,self.M_U,self.K)
        self.F_f = PF_unfolding(self.N_t,self.M_D,self.L)
        #self.Theta_f = Theta_black_box(self.M_fai)
        self.Theta_f = Theta_unfold(self.M_fai)
        
    def forward(self,P,F,Theta,HU,HD,VU,VD,J,GU,GD,Z,Htilde):                    
        HUbar,HDbar,Jbar = H_J_bar_update(HU,HD,J,Theta,GU,GD,VU,VD,Z,self.K,self.L)
        AU_new = AU_update(HUbar,P,Htilde,F,HDbar,Jbar,self.K,self.L,self.N_r,self.N_t,self.sigmaU,self.sigmaD)
        AD_new = AD_update(HUbar,P,Htilde,F,HDbar,Jbar,self.K,self.L,self.N_D,self.N_t,self.sigmaU,self.sigmaD)
        UU_new = self.UU_f(AU_new,HUbar,P)
        UD_new = self.UD_f(AD_new,HDbar,F)
        EU_new = EU_update(UU_new,HUbar,P,self.M_U,self.K)
        ED_new = ED_update(UD_new,HDbar,F,self.M_D,self.L)
        # EU_new = compute_E_directly(AU_new, UU_new, HUbar, P, self.M_U, self.K)
        # ED_new = compute_E_directly(AD_new, UD_new, HDbar, F, self.M_D, self.L)
        WU_new = self.WU_f(EU_new)
        WD_new = self.WD_f(ED_new)
        BU_new = BU_update(HUbar,self.alpha,UU_new,WU_new,self.belta,Jbar,UD_new,WD_new,self.K,self.L,self.N_U,self.N_r)
        BD_new = BD_update(HDbar,self.belta,UD_new,WD_new,self.alpha,Htilde,UU_new,WU_new,self.K,self.L,self.N_t)
        P_new = self.P_f(BU_new,HUbar,UU_new,WU_new,self.alpha)
        F_new = self.F_f(BD_new,HDbar,UD_new,WD_new,self.belta)
        P_new = P_normalize(P_new,self.P_t,self.K,self.N_U,self.M_U)
        F_new = F_normalize(F_new,self.Pmax)

        Theta_A1 = Theta_A1_update(VU,UU_new,WU_new,self.M_fai,self.K)
        Theta_B1 = Theta_B1_update(VU,UU_new,WU_new,HU,P_new,GU,self.M_fai,self.N_r,self.K)
        Theta_C1 = Theta_C1_update(GU,P_new,self.M_fai,self.K)
        Theta_A2 = Theta_A2_update(GD,UD_new,WD_new,self.M_fai,self.L)
        Theta_B2 = Theta_B2_update(GD,UD_new,WD_new,F_new,VD,HD,self.M_fai,self.N_t,self.L)
        Theta_C2 = Theta_C2_update(VD,F_new,self.M_fai,self.L)
        Theta_A3 = Theta_A3_update(Z,UD_new,WD_new,self.M_fai,self.L)
        Theta_B3 = Theta_B3_update(Z,UD_new,WD_new,J,P_new,GU,self.L,self.K,self.M_fai)
        Theta_C3 = Theta_C3_update(GU,P_new,self.M_fai,self.K)
        Theta_new = self.Theta_f(Theta,Theta_A1,Theta_B1,Theta_C1,Theta_A2,Theta_B2,Theta_C2,Theta_A3,Theta_B3,Theta_C3,self.alpha,self.belta)
        #theta = self.Theta_f(Theta_A1,Theta_B1,Theta_C1,Theta_A2,Theta_B2,Theta_C2,Theta_A3,Theta_B3,Theta_C3)
        # ThetaRe = torch.diag(torch.cos(theta[0,:]))
        # ThetaIm = torch.diag(torch.sin(theta[1,:]))
        # Theta_new = torch.cat((ThetaRe.unsqueeze(0),ThetaIm.unsqueeze(0)),0)
        # Theta_new = one_iteration_BCD(Theta,Theta_A1,Theta_B1,Theta_C1,Theta_A2,Theta_B2,Theta_C2,Theta_A3,Theta_B3,Theta_C3,self.alpha,self.belta,self.M_fai)
        return P_new,F_new,Theta_new
    
class FinalLayer(nn.Module):
    def __init__(self,config):
        super(FinalLayer,self).__init__()
        self.K = config.K #uplink user number
        self.L = config.L #downlink user number        
        self.N_r = config.num_BS_receive_antenna 
        self.N_t = config.num_BS_transmit_antenna 
        self.N_U = config.num_up_user_antenna 
        self.N_D = config.num_down_user_antenna 
        self.M_U = config.num_up_user_symbol
        self.M_D = config.num_down_user_symbol
        self.M_fai = config.num_reflecting_element
        self.sigmaU = config.sigma_up
        self.sigmaD = config.sigma_down
        self.MidLayer_f = {}
        self.device = config.device
        self.alpha = config.alpha
        self.belta = config.belta
        self.P_t = config.user_power
        self.Pmax = config.BS_power
        
        self.UU_f = U_unfolding(self.N_r,self.M_U,self.K)
        self.UD_f = U_unfolding(self.N_D,self.M_D,self.L)
        self.WU_f = W_unfolding(self.M_U,self.M_U,self.K)
        self.WD_f = W_unfolding(self.M_D,self.M_D,self.L)
        self.Theta_f = Theta_black_box(self.M_fai)

        
    def forward(self,P,F,Theta,HU,HD,VU,VD,J,GU,GD,Z,Htilde):                    
        HUbar,HDbar,Jbar = H_J_bar_update(HU,HD,J,Theta,GU,GD,VU,VD,Z,self.K,self.L)
        AU_new = AU_update(HUbar,P,Htilde,F,HDbar,Jbar,self.K,self.L,self.N_r,self.N_t,self.sigmaU,self.sigmaD)
        AD_new = AD_update(HUbar,P,Htilde,F,HDbar,Jbar,self.K,self.L,self.N_D,self.N_t,self.sigmaU,self.sigmaD)
        #UU_new = self.UU_f(AU_new,HUbar,P)
        #UD_new = self.UD_f(AD_new,HDbar,F)
        UU_new = U_update(AU_new,HUbar,P,self.N_r,self.M_U,self.K)
        UD_new = U_update(AD_new,HDbar,F,self.N_D,self.M_D,self.L)

        # EU_new = EU_update(UU_new,HUbar,P,self.M_U,self.K)
        # ED_new = ED_update(UD_new,HDbar,F,self.M_D,self.L)
        EU_new = compute_E_directly(AU_new, UU_new, HUbar, P, self.M_U, self.K)
        ED_new = compute_E_directly(AD_new, UD_new, HDbar, F, self.M_D, self.L)
        #WU_new = self.WU_f(EU_new)
        #WD_new = self.WD_f(ED_new)
        WU_new = W_update(EU_new, self.M_U, self.K)
        WD_new = W_update(ED_new, self.M_D, self.L)
        BU_new = BU_update(HUbar,self.alpha,UU_new,WU_new,self.belta,Jbar,UD_new,WD_new,self.K,self.L,self.N_U,self.N_r)
        BD_new = BD_update(HDbar,self.belta,UD_new,WD_new,self.alpha,Htilde,UU_new,WU_new,self.K,self.L,self.N_t)
        P_new = final_P_F_update(BU_new,HUbar,UU_new,WU_new,self.alpha,self.K,self.N_U,self.M_U)
        F_new = final_P_F_update(BD_new,HDbar,UD_new,WD_new,self.belta,self.L,self.N_t,self.M_D)
        P_new = P_normalize(P_new,self.P_t,self.K,self.N_U,self.M_U)
        F_new = F_normalize(F_new,self.Pmax)
        # print("normilized_F_norm")
        # print(torch.norm(F_new))
        Theta_new = Theta.clone()
        # Theta_A1 = Theta_A1_update(VU,UU_new,WU_new,self.M_fai,self.K)
        # Theta_B1 = Theta_B1_update(VU,UU_new,WU_new,HU,P_new,GU,self.M_fai,self.N_r,self.K)
        # Theta_C1 = Theta_C1_update(GU,P_new,self.M_fai,self.K)
        # Theta_A2 = Theta_A2_update(GD,UD_new,WD_new,self.M_fai,self.L)
        # Theta_B2 = Theta_B2_update(GD,UD_new,WD_new,F_new,VD,HD,self.M_fai,self.N_t,self.L)
        # Theta_C2 = Theta_C2_update(VD,F_new,self.M_fai,self.L)
        # Theta_A3 = Theta_A3_update(Z,UD_new,WD_new,self.M_fai,self.L)
        # Theta_B3 = Theta_B3_update(Z,UD_new,WD_new,J,P_new,GU,self.L,self.K,self.M_fai)
        # Theta_C3 = Theta_C3_update(GU,P_new,self.M_fai,self.K)
        # theta = self.Theta_f(Theta_A1,Theta_B1,Theta_C1,Theta_A2,Theta_B2,Theta_C2,Theta_A3,Theta_B3,Theta_C3)
        # ThetaRe = torch.diag(torch.cos(theta[0,:]))
        # ThetaIm = torch.diag(torch.sin(theta[1,:]))
        # Theta_new = torch.cat((ThetaRe.unsqueeze(0),ThetaIm.unsqueeze(0)),0)
        # Theta_new = one_iteration_BCD(Theta,Theta_A1,Theta_B1,Theta_C1,Theta_A2,Theta_B2,Theta_C2,Theta_A3,Theta_B3,Theta_C3,self.alpha,self.belta,self.M_fai)
        return P_new,F_new,Theta_new,UU_new,UD_new
    
    
class unfolding_matrix(nn.Module):
    def __init__(self,M,N,num_users,Xgain,Ygain,Zgain,Ogain):
        super(unfolding_matrix,self).__init__()
        self.M = M 
        self.N = N 
        self.num_users = num_users
        self.params1 = {}
        self.params2 = {}
        self.params3 = {}
        self.params4 = {}
        for i in range(self.num_users):
            self.params1["%d"%(i)] = nn.Parameter(torch.zeros(2,M,M))
            torch.nn.init.xavier_uniform_(self.params1["%d"%(i)],gain = Xgain)   #X
            self.params2["%d"%(i)] = nn.Parameter(torch.zeros(2,M,M))
            torch.nn.init.xavier_uniform_(self.params2["%d"%(i)],gain = Ygain)   #Y
            self.params3["%d"%(i)] = nn.Parameter(torch.zeros(2,M,M))
            torch.nn.init.xavier_uniform_(self.params3["%d"%(i)],gain = Zgain)   #Z
            self.params4["%d"%(i)] = nn.Parameter(torch.zeros(2,M,N))
            torch.nn.init.xavier_uniform_(self.params4["%d"%(i)],gain = Ogain)   #O
        self.params1 = nn.ParameterDict(self.params1)
        self.params2 = nn.ParameterDict(self.params2)
        self.params3 = nn.ParameterDict(self.params3)
        self.params4 = nn.ParameterDict(self.params4)
    
    def forward(self):
        print("please overriding this method")

class U_unfolding(unfolding_matrix):
    def __init__(self,M,N,num_users):
        power = 0.01
        super(U_unfolding,self).__init__(M,N,num_users,power,power,power,power)
        
    def forward(self,A,Hbar,P):
        U_out = torch.zeros(2,self.M,self.N,self.num_users)        
        A_inv = torch.zeros_like(A)
        for i in range(self.num_users):
            A_inv[:,:,:,i] = cmul(self.params1["%d"%(i)],cdiv(A[:,:,:,i])) + self.params3["%d"%(i)] #+ 0.01*cmul(self.params2["%d"%(i)],A[:,:,:,i])
            # a=cmul(self.params1["%d" % (i)] * 10, cdiv(A[:, :, :, i]))
            # print("test U unfolding")
            # b= self.params3["%d"%(i)]*0.01
        for i in range(self.num_users):
            U_out[:,:,:,i] = cmul(cmul(A_inv[:,:,:,i],Hbar[:,:,:,i]),P[:,:,:,i]) + self.params4["%d"%(i)]
            # print("test U unfolding")
            # b = self.params4["%d"%(i)]*0.1
        return U_out

class W_unfolding(unfolding_matrix):
    def __init__(self,M,N,num_users):
        power = 0.02
        super(W_unfolding,self).__init__(M, N, num_users,power,power,power,power)
        
    def forward(self,E):
        W_out = torch.zeros((2,self.M,self.N,self.num_users))
        for i in range(self.num_users):
            # print("test W unfolding parameters")
            # a=self.params1["%d"%(i)]
            temp  = cmul(self.params1["%d"%(i)],cdiv(E[:,:,:,i])) + self.params3["%d"%(i)] #+ 0.01*cmul(self.params2["%d"%(i)],E[:,:,:,i])
            #W_out[:,:,:,i] =temp
            W_out[:,:,:,i] = (temp + conjT(temp))/2
            #W_out[:, :, :, i] = cmul(temp,conjT(temp))
        return W_out

class PF_unfolding(unfolding_matrix):
    def __init__(self,M,N,num_users):
        power = 0.01
        super(PF_unfolding,self).__init__(M, N, num_users,power,power,power,power)
    
    def forward(self,B,Hbar,U,W,alpha):
        PF_out = torch.zeros(2,self.M,self.N,self.num_users)
        B_inv = torch.zeros_like(B)
        for i in range(self.num_users):
            B_inv[:,:,:,i] = cmul(self.params1["%d"%(i)],cdiv(B[:,:,:,i])) + self.params3["%d"%(i)] #+ 0.01*cmul(self.params2["%d"%(i)],B[:,:,:,i])
        for i in range(self.num_users):
            PF_out[:,:,:,i] = alpha*cmul(cmul(B_inv[:,:,:,i],conjT(Hbar[:,:,:,i])),cmul(U[:,:,:,i],W[:,:,:,i])) + self.params4["%d"%(i)]
        return PF_out
    
def F_normalize(F,Pmax):
    total_F_norm = torch.norm(F)
    # print("F norm is")
    # print(torch.norm(F))
    new_F = F/total_F_norm*np.sqrt(Pmax)
    
    return new_F

def P_normalize(P,P_t,K,NU,MU):
    new_P = torch.zeros(2,NU,MU,K)
    # print("P============")
    # print(P)
    # print("P_t")
    # print(P_t)
    for k in range(K):
        if torch.norm(P[:,:,:,k])>np.sqrt(P_t[k]):    
            new_P[:,:,:,k] = P[:,:,:,k]/torch.norm(P[:,:,:,k])*np.sqrt(P_t[k])
        else:
            #new_P[:,:,:,k] = P[:,:,:,k]
            new_P[:,:,:,k] = P[:,:,:,k]/torch.norm(P[:,:,:,k])*np.sqrt(P_t[k])    
    return new_P

def U_update(A,Hbar,P,M,N,num_users):
    U_out = torch.zeros(2,M,N,num_users)
    for i in range(num_users):
        U_out[:,:,:,i] = cmul(cmul(MyInv.apply(A[:,:,:,i]),Hbar[:,:,:,i]),P[:,:,:,i])
        #U_out[:, :, :, i] = cmul(cmul(cinv(A[:, :, :, i]), Hbar[:, :, :, i]), P[:, :, :, i])
    return U_out

def W_update(E,M,num_users):
    W_out = torch.zeros(2,M,M,num_users)
    for i in range(num_users):
        W_out[:,:,:,i] = MyInv.apply(E[:,:,:,i])
        W_out[:,:,:,i] = (W_out[:,:,:,i]+conjT(W_out[:,:,:,i]))/2
        #W_out[:, :, :, i] = cinv(E[:, :, :, i])
        # print("test inv W ")
        # a = cmul(MyInv.apply(E[:, :, :, i]), E[:, :, :, i])
    return W_out

def final_P_F_update(BU,HUbar,UU,WU,alpha,K,NU,MU):
    P = torch.zeros(2,NU,MU,K)
    for k in range(K):
        P[:,:,:,k] = cmul(MyInv.apply(BU[:,:,:,k]),cmul(alpha*conjT(HUbar[:,:,:,k]),cmul(UU[:,:,:,k],WU[:,:,:,k])))
        #P[:, :, :, k] = cmul(cinv(BU[:, :, :, k]),cmul(alpha * conjT(HUbar[:, :, :, k]), cmul(UU[:, :, :, k], WU[:, :, :, k])))
        # print("test inv final layer P_F ")
        # a = cmul(MyInv.apply(BU[:,:,:,k]),BU[:,:,:,k])
    return P

def H_J_bar_update(HU,HD,J,Theta,GU,GD,VU,VD,Z,K,L):
    HUbar = torch.zeros_like(HU)
    HDbar = torch.zeros_like(HD)
    Jbar = torch.zeros_like(J)
    for k in range(K):
        HUbar[:,:,:,k] = HU[:,:,:,k] + cmul(cmul(VU,Theta),GU[:,:,:,k])
    for l in range(L):
        HDbar[:,:,:,l] = HD[:,:,:,l] + cmul(cmul(GD[:,:,:,l],Theta),VD)
        for k in range(K):
            Jbar[:,:,:,k,l] = J[:,:,:,k,l] + cmul(cmul(Z[:,:,:,l],Theta),GU[:,:,:,k])
    return HUbar,HDbar,Jbar
    
class Theta_black_box(nn.Module):
    def __init__(self,M_fai):    
        super(Theta_black_box, self).__init__()
        self.M_fai = M_fai
        self.conv1 = nn.Conv2d(18,12,2,stride =2)
        self.conv2 = nn.Conv2d(12,8,2,stride = 2)
        self.conv3 = nn.Conv2d(8,6,2,stride=2)
        self.layer1 = nn.Linear(864, 1000)
        self.layer2 = nn.Linear(1000, 700)
        self.layer3 = nn.Linear(700, M_fai*2)
        
    def forward(self,A1,B1,C1,A2,B2,C2,A3,B3,C3):
        x = torch.cat((A1,B1,C1,A2,B2,C2,A3,B3,C3),0)
        x = x.unsqueeze(0)
        x = Func.relu(self.conv1(x))
        x = Func.relu(self.conv2(x))
        x = Func.relu(self.conv3(x)) 
        x = x.view(-1,864)
        x = Func.relu(self.layer1(x))
        x = Func.relu(self.layer2(x))
        x = Func.relu(self.layer3(x))
        x = torch.reshape(x,(2,self.M_fai))
        return x

class Theta_unfold(nn.Module):
    def __init__(self,M_fai):
        super(Theta_unfold,self).__init__()
        self.M_fai = M_fai
        self.param1 = nn.Parameter(torch.zeros(2,M_fai))
        torch.nn.init.xavier_uniform_(self.param1,gain = 0.01)
        self.param2 = nn.Parameter(torch.zeros(2,M_fai))
        torch.nn.init.xavier_uniform_(self.param2,gain = 0.01)
        self.param3 = nn.Parameter(torch.zeros(2,M_fai))
        torch.nn.init.xavier_uniform_(self.param3,gain = 0.01)
        self.param4 = nn.Parameter(torch.zeros(2,M_fai))
        torch.nn.init.xavier_uniform_(self.param4,gain = 0.01)
        self.param5 = nn.Parameter(torch.zeros(2,M_fai))
        torch.nn.init.xavier_uniform_(self.param5,gain = 0.01)        
        
    def forward(self,Theta,A1,B1,C1,A2,B2,C2,A3,B3,C3,alpha,belta):
        theta_new = torch.zeros(2,self.M_fai)
        approximateTheta = cdiag(self.param1)
        Q1 = cmul(A1,cmul(approximateTheta,C1))
        Q2 = cmul(A2,cmul(approximateTheta,C2))
        Q3 = cmul(A2,cmul(approximateTheta,C3))
        
        for i in range(self.M_fai):
            b1 = alpha*cmul(A1[:,i,i].unsqueeze(-1),cmul(Theta[:,i,i].unsqueeze(-1),C1[:,i,i].unsqueeze(-1))) - Q1[:,i,i].unsqueeze(-1) + B1[:,i,i].unsqueeze(-1)
            b2 = belta*cmul(A2[:,i,i].unsqueeze(-1),cmul(Theta[:,i,i].unsqueeze(-1),C2[:,i,i].unsqueeze(-1))) - Q2[:,i,i].unsqueeze(-1) + B2[:,i,i].unsqueeze(-1)
            b3 = belta*cmul(A3[:,i,i].unsqueeze(-1),cmul(Theta[:,i,i].unsqueeze(-1),C3[:,i,i].unsqueeze(-1))) - Q3[:,i,i].unsqueeze(-1) + B3[:,i,i].unsqueeze(-1)
            b = b1 + b2 + b3
            b = b + self.param2[:,i].unsqueeze(-1)
            x = b/torch.norm(b)
            theta_new[:,i] = x.squeeze(-1)
        
        Theta_new = cdiag(theta_new)
        return Theta_new

def one_iteration_BCD(Theta,A1,B1,C1,A2,B2,C2,A3,B3,C3,alpha,belta,M_fai):
    Theta1 = Theta.clone()
    Q1 = cmul(A1,cmul(Theta1,C1))
    Q2 = cmul(A2,cmul(Theta1,C2))
    Q3 = cmul(A3,cmul(Theta1,C3))
    
    for i in range(M_fai):
        b1 = alpha*cmul(A1[:,i,i].unsqueeze(-1),cmul(Theta1[:,i,i].unsqueeze(-1),C1[:,i,i].unsqueeze(-1))) - Q1[:,i,i].unsqueeze(-1) + B1[:,i,i].unsqueeze(-1)
        b2 = belta*cmul(A2[:,i,i].unsqueeze(-1),cmul(Theta1[:,i,i].unsqueeze(-1),C2[:,i,i].unsqueeze(-1))) - Q2[:,i,i].unsqueeze(-1) + B2[:,i,i].unsqueeze(-1)
        b3 = belta*cmul(A3[:,i,i].unsqueeze(-1),cmul(Theta1[:,i,i].unsqueeze(-1),C3[:,i,i].unsqueeze(-1))) - Q3[:,i,i].unsqueeze(-1) + B3[:,i,i].unsqueeze(-1)
        b = b1 + b2 + b3
        x = b/torch.norm(b)
        Q1 = Q1 + cmul(x - Theta1[:,i,i].unsqueeze(-1),cmul(A1[:,:,i].unsqueeze(-1),C1[:,i,:].unsqueeze(1)))       
        Q2 = Q2 + cmul(x - Theta1[:,i,i].unsqueeze(-1),cmul(A2[:,:,i].unsqueeze(-1),C2[:,i,:].unsqueeze(1)))
        Q3 = Q3 + cmul(x - Theta1[:,i,i].unsqueeze(-1),cmul(A3[:,:,i].unsqueeze(-1),C3[:,i,:].unsqueeze(1)))
        Theta1[:,i,i] = x.squeeze(-1)
    return Theta1
        
def AU_update(HUbar,P,Htilde,F,HDbar,Jbar,K,L,N_r,N_t,sigmaU,sigmaD):
    AU = torch.zeros(2,N_r,N_r)
    sum_HUbarP = torch.zeros(2,N_r,N_r);
    sum_F = torch.zeros(2,N_t,N_t);
    # print("HUbar=======")
    # print(HUbar.size())
    # print("P==========")
    # print(P.size())
    # print("N_r========")
    # print(N_r)
    # print("N_t=======")
    # print(N_t)
    for k in range(K):
        sum_HUbarP = sum_HUbarP + cmul(cmul(HUbar[:,:,:,k],cmul(P[:,:,:,k],conjT(P[:,:,:,k]))),conjT(HUbar[:,:,:,k]))
    for l in range(L):
        sum_F = sum_F + cmul(F[:,:,:,l],conjT(F[:,:,:,l]))
    # print("test_whether_conjT_AU_update")
    # print("sum_HUbarP")
    # print(test_whether_conjT(sum_HUbarP.unsqueeze(-1),1))
    # print("sum_F")
    # print(test_whether_conjT(sum_F.unsqueeze(-1),1))
    AU = sum_HUbarP + cmul(Htilde,cmul(sum_F,conjT(Htilde))) + sigmaU**2*ceye(N_r)
    return AU.unsqueeze(-1).repeat(1,1,1,K)
    
def AD_update(HUbar,P,Htilde,F,HDbar,Jbar,K,L,N_D,N_t,sigmaU,sigmaD):
    AD = torch.zeros(2,N_D,N_D,L)
    sum_F = torch.zeros(2,N_t,N_t);
    for l in range(L):
        sum_F = sum_F + cmul(F[:,:,:,l],conjT(F[:,:,:,l]))
    for l in range(L):
        sum_JbarP = torch.zeros(2,N_D,N_D);
        for k in range(K):
            sum_JbarP = sum_JbarP + cmul(Jbar[:,:,:,k,l],cmul(P[:,:,:,k],cmul(conjT(P[:,:,:,k]),conjT(Jbar[:,:,:,k,l]))))
        AD[:,:,:,l] = cmul(HDbar[:,:,:,l],cmul(sum_F,conjT(HDbar[:,:,:,l]))) + sum_JbarP + sigmaD**2*ceye(N_D)
    return AD

def EU_update(UU,HUbar,P,M_U,K):
    EU = torch.zeros(2,M_U,M_U,K)
    for k in range(K):
        EU[:,:,:,k] = ceye(M_U) - cmul(conjT(UU[:,:,:,k]),cmul(HUbar[:,:,:,k],P[:,:,:,k]))
    return EU

def ED_update(UD,HDbar,F,M_D,L):
    ED = torch.zeros(2,M_D,M_D,L)
    for l in range(L):
        ED[:,:,:,l] = ceye(M_D) - cmul(conjT(UD[:,:,:,l]),cmul(HDbar[:,:,:,l],F[:,:,:,l]))
    return ED

def compute_E_directly(A,U,Hbar,P_F,M,num_users):
    E_out = torch.zeros(2,M,M,num_users)
    for i in range(num_users):
        temp = cmul(conjT(P_F[:,:,:,i]),cmul(conjT(Hbar[:,:,:,i]),U[:,:,:,i]))
        E_out[:,:,:,i] = cmul(conjT(U[:,:,:,i]),cmul(A[:,:,:,i],U[:,:,:,i])) - temp - conjT(temp)+ ceye(M)
    return E_out

def BU_update(HUbar,alpha,UU,WU,belta,Jbar,UD,WD,K,L,N_U,N_r,lamb = None):
    if lamb == None:
        lamb = 1.6*torch.ones(K)
        
    BU = torch.zeros(2,N_U,N_U,K)
    UUWU = torch.zeros(2,N_r,N_r)

    for l in range(L):
        UUWU = UUWU + alpha*cmul(UU[:,:,:,l],cmul(WU[:,:,:,l],conjT(UU[:,:,:,l])))
    for k in range(K):
        JbarUD = torch.zeros(2,N_U,N_U)
        for l in range(L):
            JbarUD = JbarUD + belta*cmul(conjT(Jbar[:,:,:,k,l]),cmul(UD[:,:,:,l],cmul(WD[:,:,:,l],cmul(conjT(UD[:,:,:,l]),Jbar[:,:,:,k,l]))))
        BU[:,:,:,k] = cmul(conjT(HUbar[:,:,:,k]),cmul(UUWU,HUbar[:,:,:,k])) + JbarUD + lamb[k]*ceye(N_U)
    return BU

def BD_update(HDbar,belta,UD,WD,alpha,Htilde,UU,WU,K,L,N_t, mu = 0.02):
    BD = torch.zeros(2,N_t,N_t)
    A1 = torch.zeros(2,N_t,N_t)
    A2 = torch.zeros(2,N_t,N_t)
    for j in range(L):
        A1 = A1 + belta*cmul(conjT(HDbar[:,:,:,j]),cmul(UD[:,:,:,j],cmul(WD[:,:,:,j],cmul(conjT(UD[:,:,:,j]),HDbar[:,:,:,j]))))
    for k in range(K):
        A2 = A2 + alpha*cmul(conjT(Htilde),cmul(UU[:,:,:,k],cmul(WU[:,:,:,k],cmul(conjT(UU[:,:,:,k]),Htilde))))
    BD = A1 + A2 + mu*ceye(N_t)
    
    return BD.unsqueeze(-1).repeat(1,1,1,L)

def searchLambdaP(HUbar,alpha,UU,WU,belta,Jbar,UD,WD,K,L,N_U,N_r,P_t):
    delta = 1e-3
    
    A = torch.zeros(2,N_r,N_r)
    
    allLamb = torch.ones(K)
    for l in range(L):
        A = A + alpha*cmul(UU[:,:,:,l],cmul(WU[:,:,:,l],conjT(UU[:,:,:,l])))
    for k in range(K):
        B = torch.zeros(2,N_U,N_U)
        for l in range(L):
            B = B + belta*cmul(conjT(Jbar[:,:,:,k,l]),cmul(UD[:,:,:,l],cmul(WD[:,:,:,l],cmul(conjT(UD[:,:,:,l]),Jbar[:,:,:,k,l])))) 
        
        #start search
        C =  cmul(conjT(HUbar[:,:,:,k]),cmul(A,HUbar[:,:,:,k])) + B
        D = alpha*cmul(conjT(HUbar[:,:,:,k]),cmul(UU[:,:,:,k],WU[:,:,:,k]))
        P = cmul(MyInv.apply(C),D)
        I = ceye(N_U)
        lamb = 0
        
        if torch.norm(P)**2 > P_t[k]:
            lamb_min = 1e-6
            lamb_max = 10
            lamb = 10
            
            P =  cmul(MyInv.apply(C+lamb*I),D)
            while torch.norm(P)**2 > P_t[k]:
                lamb_min = lamb
                lamb = 2*lamb
                P = cmul(MyInv.apply(C+lamb*I),D)
            
            lamb_max = lamb
            while(abs(lamb_min-lamb_max) >delta ):
                lamb = (lamb_min + lamb_max)/2
                P = cmul(MyInv.apply(C+lamb*I),D)
                if torch.norm(P)**2 > P_t[k]:
                    lamb_min = lamb
                else:
                    lamb_max = lamb
        allLamb[k] = lamb
        # print('P_t is')
        # print(P_t[k])
        # print('P norm is')
        # print(torch.norm(P)**2)
        # print('lamb')
        # print(lamb)        
    return allLamb.detach()

def searchMuF(HDbar,belta,UD,WD,alpha,Htilde,UU,WU,K,L,N_t,N_D,P_max):
    delta = 1e-6
    A1 = torch.zeros(2,N_t,N_t)
    A2 = torch.zeros(2,N_t,N_t)
    for j in range(L):
        A1 = A1 + belta*cmul(conjT(HDbar[:,:,:,j]),cmul(UD[:,:,:,j],cmul(WD[:,:,:,j],cmul(conjT(UD[:,:,:,j]),HDbar[:,:,:,j]))))
    for k in range(K):
        A2 = A2 + alpha*cmul(conjT(Htilde),cmul(UU[:,:,:,k],cmul(WU[:,:,:,k],cmul(conjT(UU[:,:,:,k]),Htilde))))
    A = A1+A2
    A = (A+conjT(A))/2
    A = tensor2np(A)
    [a,U] = scipy.linalg.eig(A)
    U = np2tensor(U)
    #U = conjT(U)
    b = torch.zeros(2,L,N_t)
    B = torch.zeros(2,N_t,N_D,L)
    for l in range(L):
        B[:,:,:,l] = belta*cmul(conjT(HDbar[:,:,:,l]),cmul(UD[:,:,:,l],WD[:,:,:,l]))
        b[:,l,:] = cdiag(cmul(conjT(U),cmul(B[:,:,:,l],cmul(conjT(B[:,:,:,l]),U))))
    b = np.abs(tensor2np(b))
    a = np.abs(a)
    
    #start search
    mu = 1e-8
    c = (a+mu)**2
    c = np.tile(c,[L,1])
    if np.sum(b/c)>P_max:
        mu_min = 1e-6
        mu_max = 10
        mu = 10
        
        c = (a+mu)**2
        c = np.tile(c,[L,1])
        
        while np.sum(b/c)>P_max:
            mu_min = mu
            mu = 2*mu
            c = (a+mu)**2
            c = np.tile(c,[L,1])
        
        max_mu = mu
        while (abs(mu_min-mu_max) >delta ):
            mu = (mu_min + mu_max)/2
            c = (a+mu)**2
            c = np.tile(c,[L,1])
            
            if np.sum(b/c)>P_max:
                mu_min = mu
            else:
                mu_max = mu
    norm_counter = 0
    A = np2tensor(A)
    for l in range(L):
        norm_counter += torch.norm(cmul(MyInv.apply(A + mu*ceye(N_t)),B[:,:,:,l]))**2
    # print('F norm is')
    # print(norm_counter)
        
    
    return mu

def Theta_A1_update(VU,UU,WU,M_fai,K):
    A1 = torch.zeros(2,M_fai,M_fai)
    for k in range(K):
        A1 = A1 + cmul(conjT(VU),cmul(UU[:,:,:,k],cmul(WU[:,:,:,k],cmul(conjT(UU[:,:,:,k]),VU))))
    return A1

def Theta_C1_update(GU,P,M_fai,K):
    C1 = torch.zeros(2,M_fai,M_fai)
    for i in range(K):
        C1 = C1 + cmul(GU[:,:,:,i],cmul(P[:,:,:,i],cmul(conjT(P[:,:,:,i]),conjT(GU[:,:,:,i]))))
    return C1

def Theta_B1_update(VU,UU,WU,HU,P,GU,M_fai,N_r,K):
    B1_1 = torch.zeros(2,M_fai,N_r)
    B1_2 = torch.zeros(2,N_r,M_fai)
    B1_3 = torch.zeros(2,M_fai,M_fai)
    B1 = torch.zeros(2,M_fai,M_fai)
    for i in range(K):
        B1_1 = B1_1 + cmul(conjT(VU),cmul(UU[:,:,:,i],cmul(WU[:,:,:,i],conjT(UU[:,:,:,i]))))
        B1_2 = B1_2 + cmul(HU[:,:,:,i],cmul(P[:,:,:,i],cmul(conjT(P[:,:,:,i]),conjT(GU[:,:,:,i]))))
        B1_3 = B1_3 + cmul(conjT(VU),cmul(UU[:,:,:,i],cmul(conjT(WU[:,:,:,i]),cmul(conjT(P[:,:,:,i]),conjT(GU[:,:,:,i])))))
    B1 = B1_3 - cmul(B1_1,B1_2)
    return B1

def Theta_A2_update(GD,UD,WD,M_fai,L):
    A2 = torch.zeros(2,M_fai,M_fai)
    for l in range(L):
        A2 = A2 + cmul(conjT(GD[:,:,:,l]),cmul(UD[:,:,:,l],cmul(WD[:,:,:,l],cmul(conjT(UD[:,:,:,l]),GD[:,:,:,l]))))
    return A2

def Theta_C2_update(VD,F,M_fai,L):
    C2 = torch.zeros(2,M_fai,M_fai)
    for i in range(L):
        C2 = C2 + cmul(VD,cmul(F[:,:,:,i],cmul(conjT(F[:,:,:,i]),conjT(VD))))
    return C2

def Theta_B2_update(GD,UD,WD,F,VD,HD,M_fai,N_t,L):
    B2_1 = torch.zeros(2,M_fai,N_t)
    B2_2 = torch.zeros(2,N_t,M_fai)
    B2_3 = torch.zeros(2,M_fai,M_fai)
    B2 = torch.zeros(2,M_fai,M_fai)
    for i in range(L):
        B2_1 = B2_1 + cmul(conjT(GD[:,:,:,i]),cmul(UD[:,:,:,i],cmul(WD[:,:,:,i],cmul(conjT(UD[:,:,:,i]),HD[:,:,:,i]))))
        B2_2 = B2_2 + cmul(F[:,:,:,i],cmul(conjT(F[:,:,:,i]),conjT(VD)))
        B2_3 = B2_3 + cmul(conjT(GD[:,:,:,i]),cmul(UD[:,:,:,i],cmul(conjT(WD[:,:,:,i]),cmul(conjT(F[:,:,:,i]),conjT(VD)))))
    B2 = B2_3 - cmul(B2_1,B2_2)
    return B2

def Theta_A3_update(Z,UD,WD,M_fai,L):
    A3 = torch.zeros(2,M_fai,M_fai)
    for l in range(L):
        A3 = A3 + cmul(conjT(Z[:,:,:,l]),cmul(UD[:,:,:,l],cmul(WD[:,:,:,l],cmul(conjT(UD[:,:,:,l]),Z[:,:,:,l]))))
    return A3

def Theta_C3_update(GU,P,M_fai,K):
    C3 = torch.zeros(2,M_fai,M_fai)
    for k in range(K):
        C3 = C3 + cmul(GU[:,:,:,k],cmul(P[:,:,:,k],cmul(conjT(P[:,:,:,k]),conjT(GU[:,:,:,k]))))
    return C3

def Theta_B3_update(Z,UD,WD,J,P,GU,L,K,M_fai):
    B3 = torch.zeros(2,M_fai,M_fai)
    for l in range(L):
        for k in range(K):
            B3 = B3 - cmul(conjT(Z[:,:,:,l]),cmul(UD[:,:,:,l],cmul(WD[:,:,:,l],cmul(conjT(UD[:,:,:,l]),cmul(J[:,:,:,k,l],cmul(P[:,:,:,k],cmul(conjT(P[:,:,:,k]),conjT(GU[:,:,:,k]))))))))
    return B3
    