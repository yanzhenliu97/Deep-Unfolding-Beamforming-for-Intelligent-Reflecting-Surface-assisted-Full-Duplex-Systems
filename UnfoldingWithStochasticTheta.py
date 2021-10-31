# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:21:38 2021

@author: Yanzhen Liu
"""
from my_model import *

class InitialThetaModel(nn.Module):
    def __init__(self,config,channel_layer):
        super(InitialThetaModel,self).__init__()
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
        
        #self.initialThetalayer = BlackBoxThetaNet(channel_layer,self.M_fai)
        self.theta = nn.Parameter(torch.rand(1,self.M_fai)*np.pi)
        for i in range(self.n):
            self.MidLayer_f["%d"%(i)] = MidLayer2(config)
        self.MidLayer_f = nn.ModuleDict(self.MidLayer_f)
        self.FinalLayer_f = FinalLayer2(config)
        
    def forward(self,P,F,HU,HD,VU,VD,J,GU,GD,Z,Htilde):
        P_c = {}
        P_c['0'] = P
        F_c = {}
        F_c['0'] = F        
        Theta_c = {}
        # H_cat = stackInputChannel(HU,HD,J,GU,GD,VU,VD,Z,Htilde,self.N_r,self.N_U)
        # theta = self.initialThetalayer(H_cat)
        ThetaRe = torch.diag(torch.cos(self.theta[0,:]))
        ThetaIm = torch.diag(torch.sin(self.theta[0,:]))
        Theta = torch.cat((ThetaRe.unsqueeze(0),ThetaIm.unsqueeze(0)),0)
        Theta_c['0'] = Theta
        for i in range(self.n):
            #print("midder layer==== ",i)
            P_c['%d'%(i+1)],F_c['%d'%(i+1)],Theta_c['%d'%(i+1)] = self.MidLayer_f["%d"%(i)](P_c['%d'%(i)],F_c['%d'%(i)],Theta_c['%d'%(i)],HU,HD,VU,VD,J,GU,GD,Z,Htilde)

        return self.FinalLayer_f(P_c['%d'%(self.n)],F_c['%d'%(self.n)],Theta_c['%d'%(self.n)],HU,HD,VU,VD,J,GU,GD,Z,Htilde)    
    
    def detTheta(self):
        ThetaRe = torch.diag(torch.cos(self.theta[0,:]))
        ThetaIm = torch.diag(torch.sin(self.theta[0,:]))
        Theta = torch.cat((ThetaRe.unsqueeze(0),ThetaIm.unsqueeze(0)),0)
        return Theta.detach()

class InitialThetaModel2(InitialThetaModel):
    def __init__(self,config,channel_layer):
        super(InitialThetaModel2,self).__init__(config,channel_layer)
        #self.FinalLayer_f_middle = FinalLayer2(config)
        
    def forward(self,P,F,HU,HD,VU,VD,J,GU,GD,Z,Htilde):
        P_c = {}
        P_c['0'] = P
        F_c = {}
        F_c['0'] = F        
        Theta_c = {}
        # H_cat = stackInputChannel(HU,HD,J,GU,GD,VU,VD,Z,Htilde,self.N_r,self.N_U)
        # theta = self.initialThetalayer(H_cat)
        ThetaRe = torch.diag(torch.cos(self.theta[0,:]))
        ThetaIm = torch.diag(torch.sin(self.theta[0,:]))
        Theta = torch.cat((ThetaRe.unsqueeze(0),ThetaIm.unsqueeze(0)),0)
        Theta_c['0'] = Theta
        N = int(self.n/2)
        for i in range(N):
            #print("midder layer==== ",i)
            P_c['%d'%(i+1)],F_c['%d'%(i+1)],Theta_c['%d'%(i+1)] = self.MidLayer_f["%d"%(i)](P_c['%d'%(i)],F_c['%d'%(i)],Theta_c['%d'%(i)],HU,HD,VU,VD,J,GU,GD,Z,Htilde)

        P_c['%d'%(N+1)],F_c['%d'%(N+1)],Theta_c['%d'%(N+1)],U_temp,W_temp = self.FinalLayer_f(P_c['%d'%(N)],F_c['%d'%(N)],Theta_c['%d'%(N)],HU,HD,VU,VD,J,GU,GD,Z,Htilde)
        for i in range(N):
            #print("midder layer==== ",i)
            i=i+N+1
            P_c['%d'%(i+1)],F_c['%d'%(i+1)],Theta_c['%d'%(i+1)] = self.MidLayer_f["%d"%(i-1)](P_c['%d'%(i)],F_c['%d'%(i)],Theta_c['%d'%(i)],HU,HD,VU,VD,J,GU,GD,Z,Htilde)

            
        return self.FinalLayer_f(P_c['%d'%(self.n+1)],F_c['%d'%(self.n+1)],Theta_c['%d'%(self.n+1)],HU,HD,VU,VD,J,GU,GD,Z,Htilde)    
  
    
    
        
        
class MidLayer2(MidLayer):
    def forward(self,P,F,Theta,HU,HD,VU,VD,J,GU,GD,Z,Htilde):   
        Theta_new = Theta.clone()  
        Theta_detach = Theta.detach()
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
        # print("AU_new=======")
        # print(AU_new)
        # print(AD_new)
        # print("UU_new=======")
        # print(UU_new)
        # print(UD_new)
        # print("EU_new=======")
        # print(EU_new)
        # print(ED_new)
        # print("WU_new=======")
        # print(WU_new)
        # print(WD_new)
        # print("PF===========")
        # print(F_new)
        # print(P_new)
        
        P_new = P_normalize(P_new,self.P_t,self.K,self.N_U,self.M_U)
        F_new = F_normalize(F_new,self.Pmax)
        
        return P_new,F_new,Theta

class FinalLayer2(FinalLayer):
    def forward(self,P,F,Theta,HU,HD,VU,VD,J,GU,GD,Z,Htilde):
        Theta_detach = Theta.detach()                   
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
        lamb = searchLambdaP(HUbar,self.alpha,UU_new,WU_new,self.belta,Jbar,UD_new,WD_new,self.K,self.L,self.N_U,self.N_r,self.P_t)      
        mu = searchMuF(HDbar,self.belta,UD_new,WD_new,self.alpha,Htilde,UU_new,WU_new,self.K,self.L,self.N_t,self.N_D,self.Pmax)     
        
        BU_new = BU_update(HUbar,self.alpha,UU_new,WU_new,self.belta,Jbar,UD_new,WD_new,self.K,self.L,self.N_U,self.N_r,lamb)
        BD_new = BD_update(HDbar,self.belta,UD_new,WD_new,self.alpha,Htilde,UU_new,WU_new,self.K,self.L,self.N_t,mu)
        P_new = final_P_F_update(BU_new,HUbar,UU_new,WU_new,self.alpha,self.K,self.N_U,self.M_U)
        F_new = final_P_F_update(BD_new,HDbar,UD_new,WD_new,self.belta,self.L,self.N_t,self.M_D)
        P_new = P_normalize(P_new,self.P_t,self.K,self.N_U,self.M_U)
        F_new = F_normalize(F_new,self.Pmax)
        # print("normilized_F_norm")
        # print(torch.norm(F_new))
        
        return P_new,F_new,Theta,UU_new,UD_new

class BlackBoxThetaNet(nn.Module):
    def __init__(self,channel_layer,M_fai):
        super(BlackBoxThetaNet,self).__init__()
        self.channel_layer = channel_layer
        self.M_fai = M_fai
        self.conv1 = nn.Conv2d(channel_layer,channel_layer+2,2)
        self.conv2 = nn.Conv2d(channel_layer+2,channel_layer+4,2)
        self.conv3 = nn.Conv2d(channel_layer+4,channel_layer+2,2)
        self.fc1 = nn.Linear(4785,600)
        self.fc2 = nn.Linear(600, 550)
        self.fc3 = nn.Linear(550,2*self.M_fai)
    
    def forward(self,x):
        x = Func.relu(self.conv1(x))
        x = Func.relu(self.conv2(x))
        x = Func.relu(self.conv3(x)) 
        x = x.view(-1, 4785)
        x = Func.relu(self.fc1(x)) 
        x = Func.relu(self.fc2(x)) 
        x = self.fc3(x)  
        theta = torch.reshape(x,(2,self.M_fai))
        return theta        

def stackInputChannel(HU,HD,J,GU,GD,VU,VD,Z,Htilde,N_r,N_U):
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