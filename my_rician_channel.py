# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 19:13:55 2020

@author: Yanzhen Liu
"""
import numpy as np
from scipy.linalg import sqrtm 
import scipy.special as spl
import math

def correlation_matrix(M,r):
    Phi = np.zeros([M,M])
    for i in range(M):
        for j in range(M):
            if i <= j:
                Phi[i,j] = r**(j-i)
            else:
                Phi[i,j] = Phi[j,i]
    return Phi

def rician_channel(m,n,Phi1,Phi2, K_factor, h_bar):
    g_random = 2**(-0.5)*(np.random.randn(m,n) + 1j*np.random.randn(m,n))
    g = h_bar*(K_factor/(K_factor+1))**(0.5) + (1/(K_factor+1))**(0.5)*np.matmul(sqrtm(Phi1),np.matmul(g_random,sqrtm(Phi2)))
    return g



def my_rician_channel(K,L,N_t,N_r,N_U,N_D,M_fai,position_AP,position_IRS,position_down_users,position_up_users,return_los = False):
    #K,L,N_t,N_r,N_U,N_D,M_fai = 2,2,16,16,4,4,16
    Ny=np.sqrt(M_fai)
    
  
    
    """compute the distance and angle"""
    """AP_IRS"""
    distance_AP_IRS = np.linalg.norm(position_AP-position_IRS)
    azi_AOD_AP_IRS = 0
    eve_AOD_AP_IRS = np.arctan((position_IRS[2,0]-position_AP[2,0])/np.linalg.norm(position_IRS[0:2,0]-position_AP[0:2,0]))
    a_t = np.exp(1j*2*np.pi/4*(np.floor((np.arange(M_fai)+1)/Ny)*np.cos(np.pi-eve_AOD_AP_IRS)+((np.arange(M_fai)+1)-np.floor((np.arange(M_fai)+1)/Ny)*Ny)*np.sin(np.pi-eve_AOD_AP_IRS)*np.cos(np.pi-azi_AOD_AP_IRS)))
    a_r = np.exp(1j*2*np.pi/4*(np.arange(N_t))*np.cos(eve_AOD_AP_IRS))
    H_AP_IRS_bar = np.matmul(np.expand_dims(a_t,1),np.conj(np.expand_dims(a_r,0)))
    #print(H_AP_IRS_bar)
    
    """uplink user--AP and uplink user--IRS"""
    distance_up_AP = np.zeros(K)
    eve_AOD_up_AP = np.zeros(K)
    H_up_AP_bar = np.zeros([N_r,N_U,K])+1j*np.zeros([N_r,N_U,K])
    
    distance_up_IRS = np.zeros(K)
    eve_AOD_up_IRS = np.zeros(K)
    azi_AOD_up_IRS = np.zeros(K)
    H_up_IRS_bar = np.zeros([M_fai,N_U,K])+1j*np.zeros([M_fai,N_U,K])
    
    for k in range(K):
        distance_up_AP[k] = np.linalg.norm(position_up_users[:,k]-position_AP[:,0])
        eve_AOD_up_AP[k] = np.pi/2
        a_t = np.exp(1j*2*np.pi*(np.arange(N_r))*np.cos(np.pi-eve_AOD_up_AP[k])/2)
        a_r = np.exp(1j*2*np.pi*(np.arange(N_U))*np.cos(eve_AOD_up_AP[k])/2)
        H_up_AP_bar[:,:,k] = np.matmul(np.expand_dims(a_t,1),np.conj(np.expand_dims(a_r,0)))
        
        distance_up_IRS[k] = np.linalg.norm(position_up_users[:,k]-position_IRS[:,0])
        eve_AOD_up_IRS[k] = np.arctan((position_IRS[2]-position_up_users[2,k])/np.linalg.norm(position_up_users[0:2,k]-position_IRS[0:2]))
        azi_AOD_up_IRS[k] = np.arctan((position_IRS[0]-position_up_users[0,k])/(position_IRS[1]-position_up_users[1,k]+1e-20))   
        a_t = np.exp(1j*2*np.pi/4*(np.floor((np.arange(M_fai)+1)/Ny)*np.cos(np.pi-eve_AOD_up_IRS[k])+((np.arange(M_fai)+1)-np.floor((np.arange(M_fai)+1)/Ny)*Ny)*np.sin(np.pi-eve_AOD_up_IRS[k])*np.cos(np.pi-azi_AOD_up_IRS[k])))
        a_r = np.exp(1j*2*np.pi/4*(np.arange(N_U))*np.cos(eve_AOD_up_IRS[k]))
        H_up_IRS_bar[:,:,k] = np.matmul(np.expand_dims(a_t,1),np.conj(np.expand_dims(a_r,0)))
    
    """downlink user--AP and downlink user--IRS"""
    distance_AP_down = np.zeros(L)
    eve_AOD_AP_down = np.zeros(L)
    H_AP_down_bar = np.zeros([N_D,N_t,L])+1j*np.zeros([N_D,N_t,L])
    
    distance_IRS_down = np.zeros(L)
    eve_AOD_IRS_down = np.zeros(L)
    azi_AOD_IRS_down = np.zeros(L)
    H_IRS_down_bar = np.zeros([N_D,M_fai,L])+1j*np.zeros([N_D,M_fai,L])
    
    for l in range(L):
        distance_AP_down[l] = np.linalg.norm(position_AP[:,0]-position_down_users[:,l])
        eve_AOD_AP_down[l] = np.pi/2;
        a_t = np.exp(1j*2*np.pi*(np.arange(N_D))*np.cos(np.pi-eve_AOD_AP_down[l])/2)
        a_r = np.exp(1j*2*np.pi*(np.arange(N_t))*np.cos(eve_AOD_AP_down[l])/2)    
        H_AP_down_bar[:,:,l] = np.matmul(np.expand_dims(a_t,1),np.conj(np.expand_dims(a_r,0)))  
        
        distance_IRS_down[l] = np.linalg.norm(position_IRS[:,0]-position_down_users[:,l])
        eve_AOD_IRS_down[l] = np.arctan((position_down_users[2,l]-position_IRS[2])/np.linalg.norm(position_down_users[0:2,l]-position_IRS[0:2]))
        azi_AOD_IRS_down[l] = np.arctan((position_down_users[0,l]-position_IRS[0])/(position_IRS[1]-position_down_users[1,l]+1e-20))
        a_t = np.exp(1j*2*np.pi/4*(np.arange(N_D))*np.cos(np.pi-eve_AOD_IRS_down[l]))
        a_r = np.exp(1j*2*np.pi/4*(np.floor((np.arange(M_fai)+1)/Ny)*np.cos(eve_AOD_IRS_down[l])+((np.arange(M_fai)+1)-np.floor((np.arange(M_fai)+1)/Ny)*Ny)*np.sin(eve_AOD_IRS_down[l])*np.cos(azi_AOD_IRS_down[l])))                               
        H_IRS_down_bar[:,:,l] = np.matmul(np.expand_dims(a_t,1),np.conj(np.expand_dims(a_r,0))) 
    
    """uplink--downlink user""" 
    distance_up_down = np.zeros([K,L])
    H_up_down_bar = np.zeros([N_D,N_U,K,L])+1j*np.zeros([N_D,N_U,K,L])
    for i in range(L):
        for j in range(K):
            distance_up_down[i,j] =np.linalg.norm(position_up_users[:,i]-position_down_users[:,j])
            eve_AOD_up_down = np.pi/2
            a_t = np.exp(1j*2*np.pi/4*(np.arange(N_D))*np.cos(np.pi-eve_AOD_up_down))
            a_r = np.exp(1j*2*np.pi/4*(np.arange(N_U))*np.cos(np.pi-eve_AOD_up_down))
            H_up_down_bar[:,:,i,j] = np.matmul(np.expand_dims(a_t,1),np.conj(np.expand_dims(a_r,0)))
    
    """generate correlation matrix"""
    r_irs = 0.7
    r_AP = 0.4
    r_user = 0.7
    
    Phi_r = correlation_matrix(M_fai,r_irs)
    Phi_Nr = correlation_matrix(N_r,r_AP)
    Phi_Nt = correlation_matrix(N_t,r_AP)
    Phi_NU = correlation_matrix(N_U,r_user)
    Phi_ND = correlation_matrix(N_D,r_user)      
    
    """rician factor"""
    base_factor = 5
    beta_AP_user = 10**((-3+base_factor)/10)
    beta_IRS_user = 10**((3+base_factor)/10)
    beta_user = 10**((0+base_factor)/10)
    beta_AP_IRS = 10**((3+base_factor)/10)
    
    """pathloss"""
    alpha_AP_user = 3.8
    alpha_IRS_user = 2.2
    alpha_user = 3.0
    alpha_AP_IRS = 2.4
    
    C0 = 10**(-3) 
    
    VU = rician_channel(N_r,M_fai,Phi_Nr,Phi_r,beta_AP_IRS,H_AP_IRS_bar.T)*np.sqrt(C0*distance_AP_IRS**(-alpha_AP_IRS))
    VD = rician_channel(M_fai,N_t,Phi_r,Phi_Nt,beta_AP_IRS,H_AP_IRS_bar)*np.sqrt(C0*distance_AP_IRS**(-alpha_AP_IRS))
    HU = np.random.randn(N_r,N_U,K) + 1j*np.random.randn(N_r,N_U,K)
    GU = np.random.randn(M_fai,N_U,K) + 1j*np.random.randn(M_fai,N_U,K)
    
    HD = np.random.randn(N_D,N_t,L) + 1j*np.random.randn(N_D,N_t,L)
    GD = np.random.randn(N_D,M_fai,L) + 1j*np.random.randn(N_D,M_fai,L)
    
    J = np.random.randn(N_D,N_U,K,L) + 1j*np.random.randn(N_D,N_U,K,L)
    Z = np.random.randn(N_D,M_fai,L) + 1j*np.random.randn(N_D,M_fai,L)
    
    
    
    for k in range(K):
        HU[:,:,k] = rician_channel(N_r,N_U,Phi_Nr,Phi_NU,beta_AP_user,H_up_AP_bar[:,:,k])*np.sqrt(C0*distance_up_AP[k]**(-alpha_AP_user))
        GU[:,:,k] = rician_channel(M_fai,N_U,Phi_r,Phi_NU,beta_IRS_user,H_up_IRS_bar[:,:,k])*np.sqrt(C0*distance_up_IRS[k]**(-alpha_IRS_user))
    
    for l in range(L):
        HD[:,:,l] = rician_channel(N_D,N_t,Phi_ND,Phi_Nt,beta_AP_user,H_AP_down_bar[:,:,l])*np.sqrt(C0*distance_AP_down[l]**(-alpha_AP_user))
        GD[:,:,l] = rician_channel(N_D,M_fai,Phi_ND,Phi_r,beta_IRS_user,H_IRS_down_bar[:,:,l])*np.sqrt(C0*distance_IRS_down[l]**(-alpha_IRS_user))
    
    for l in range(L):
        Z[:,:,l] = rician_channel(N_D,M_fai,Phi_ND,Phi_r,beta_IRS_user,H_IRS_down_bar[:,:,l])*np.sqrt(C0*distance_IRS_down[l]**(-alpha_IRS_user))
        for k in range(K):
            J[:,:,k,l] = rician_channel(N_D,N_U,Phi_ND,Phi_NU,beta_user,H_up_down_bar[:,:,k,l])*np.sqrt(C0*distance_up_down[k,l]**(-alpha_user))
            
    VU_ =  H_AP_IRS_bar.T*np.sqrt(C0*distance_AP_IRS**(-alpha_AP_IRS))
    VD_ =  H_AP_IRS_bar*np.sqrt(C0*distance_AP_IRS**(-alpha_AP_IRS))
    HU_ = np.random.randn(N_r,N_U,K) + 1j*np.random.randn(N_r,N_U,K)
    GU_ = np.random.randn(M_fai,N_U,K) + 1j*np.random.randn(M_fai,N_U,K)
    
    HD_ = np.random.randn(N_D,N_t,L) + 1j*np.random.randn(N_D,N_t,L)
    GD_ = np.random.randn(N_D,M_fai,L) + 1j*np.random.randn(N_D,M_fai,L)
    
    J_ = np.random.randn(N_D,N_U,K,L) + 1j*np.random.randn(N_D,N_U,K,L)
    Z_ = np.random.randn(N_D,M_fai,L) + 1j*np.random.randn(N_D,M_fai,L)
    
    
    
    for k in range(K):
        HU_[:,:,k] = H_up_AP_bar[:,:,k]*np.sqrt(C0*distance_up_AP[k]**(-alpha_AP_user))
        GU_[:,:,k] = H_up_IRS_bar[:,:,k]*np.sqrt(C0*distance_up_IRS[k]**(-alpha_IRS_user))
    
    for l in range(L):
        HD_[:,:,l] = H_AP_down_bar[:,:,l]*np.sqrt(C0*distance_AP_down[l]**(-alpha_AP_user))
        GD_[:,:,l] = H_IRS_down_bar[:,:,l]*np.sqrt(C0*distance_IRS_down[l]**(-alpha_IRS_user))
    
    for l in range(L):
        Z_[:,:,l] = H_IRS_down_bar[:,:,l]*np.sqrt(C0*distance_IRS_down[l]**(-alpha_IRS_user))
        for k in range(K):
            J_[:,:,k,l] = H_up_down_bar[:,:,k,l]*np.sqrt(C0*distance_up_down[k,l]**(-alpha_user))
                
            
    if return_los:    
        return HU_,HD_,VU_,VD_,J_,GU_,GD_,Z_
    else:
        return HU,HD,VU,VD,J,GU,GD,Z

def addDelayToChannel(H,tau):
    T = 2.2e-3
    fd=5*10**9*1000/3600/(3*10**8)
    tau_d=math.floor(tau/T)
    
    v = spl.j0(2*np.pi*fd*1*T)
    Rx = spl.j0(2*np.pi*fd*0*T)
    
    a = -v/Rx
    sigma_p = spl.j0(0)
    sigma_p += a*spl.j0(-2*np.pi*fd*1*T)
    
    Hdelay = H
    for i in range(tau_d): 
        Hdelay = -a*Hdelay + math.sqrt(sigma_p)*np.ones_like(H)*np.random.randn(1)
    
    return Hdelay

 
