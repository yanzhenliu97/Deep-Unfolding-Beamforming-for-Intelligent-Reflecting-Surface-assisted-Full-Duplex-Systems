# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:12:51 2020

@author: Yanzhen Liu
"""
import numpy as np
import torch
from scipy import linalg 
"""define complex matrix operators for pytorch tensor object"""

def cmul(A,B):
    #complex matrix multiplication C = A * B
    #only accept 2 dimension matrix mul or matrix mul a scalar or scalar mul a scalar
    #please do not input vectors, unsqueeze to n*1(1*n) matrix instead
    if A.size(0) != 2 or B.size(0) != 2:
        raise Exception("The first dimension of input tensor must be 2")
       
    if A.dim() >= 4 or B.dim() >= 4:
        raise Exception("Input tensor dimension must be smaller than 4")
    
    if A.dim() == 3 and B.dim() == 3:
        Cre = torch.mm(A[0,:],B[0,:]) - torch.mm(A[1,:],B[1,:])
        Cim = torch.mm(A[0,:],B[1,:]) + torch.mm(A[1,:],B[0,:])
        return torch.cat((Cre.unsqueeze(0),Cim.unsqueeze(0)),0)
    
    elif A.dim() <= 1 or B.dim() <= 1 :   
        raise Exception("Input tensor dimension must be bigger than 1")
        
    elif A.dim() ==2:
        if A.size(1) !=1:
            raise Exception("Input scalar size must be 2*1")
        else:
            if B.dim() ==2:
                if B.size(1) !=1:
                    raise Exception("Input scalar size must be 2*1")
                else:
                    return cmul2(A,B)
            else:
                return cmul1(A,B)
    else:
        if B.size(1) !=1:
            raise Exception("Input scalar size must be 2*1")
        else:
            return cmul1(B,A)
   
                    

def cmul1(a,B):
    #scalar mul matrix Dim(a) = 1
    Cre = a[0] * B[0, :] - a[1] * B[1, :]
    Cim = a[0] * B[1, :] + a[1] * B[0, :]

    return torch.cat((Cre.unsqueeze(0),Cim.unsqueeze(0)),0)

def cmul2(a,b):
    #scalar mul scalar Dim(a)=Dim(b)=1
    cre = a[0] * b[0] - a[1] * b[1]
    cim = a[0] * b[1] + a[1] * b[0]

    return torch.cat((cre.unsqueeze(0),cim.unsqueeze(0)),0)

def conjT(A):
    #conjugate transpose of a matrix 
    if A.size(0) != 2:
        raise Exception("The first input tensor dimension must be 2")
        
    if A.dim() >= 2:
        Bre = A[0,:].T
        Bim = -A[1,:].T
        return torch.cat((Bre.unsqueeze(0),Bim.unsqueeze(0)),0)

    else:
        return conj(A)    

def conj(a):
    #conjuecture of a
    bre = a[0]
    bim = -a[1]
    return torch.cat((bre.unsqueeze(0),bim.unsqueeze(0)),0)


def cdiag(A):
    #diagnolize 
    if A.size(0) != 2:
        raise Exception("The first input tensor dimension must be 2")
        
    if A.dim() >= 2:            
        Are = torch.diag(A[0,:])
        Aim = torch.diag(A[1,:])
        return torch.cat((Are.unsqueeze(0), Aim.unsqueeze(0)), 0)
    
    else:
        raise Exception("The input dimension must be 2 or 3")
        
def np2tensor(A):
    """convert a complex type numpy array to a tensor object"""
    return torch.cat((torch.from_numpy(A.real).unsqueeze(0),torch.from_numpy(A.imag).unsqueeze(0)),0)

def tensor2np(A):
    Are = A[0,:].detach()
    Aim = A[1,:].detach()
    
    return Are.numpy() + 1j*Aim.numpy()

def ceye(dim):
    Are = torch.eye(dim)
    Aim = torch.zeros(dim,dim)
    return torch.cat((Are.unsqueeze(0), Aim.unsqueeze(0)), 0)

def cdiv(A):
    #activation function
    Are = torch.diag(torch.div(torch.diag(A[0,:]),torch.diag(A[0,:])**2+torch.diag(A[1,:])**2))
    Aim = torch.diag(torch.div(-torch.diag(A[1,:]),torch.diag(A[0,:])**2+torch.diag(A[1,:])**2))
    return torch.cat((Are.unsqueeze(0), Aim.unsqueeze(0)), 0)

def cinv(A):
    #matrix incerse
    Cre = torch.inverse(A[0,:]+torch.mm(A[1,:],torch.mm(torch.inverse(A[0,:]),A[1,:])))
    Cim = -torch.mm(torch.mm(torch.inverse(A[0,:]),A[1,:]),Cre)
    return torch.cat((Cre.unsqueeze(0), Cim.unsqueeze(0)), 0)

def laplace_temp(A, idx, jdx):
    n = A.size(1)-1
    A_real = torch.zeros((n,n))
    A_imag = torch.zeros((n,n))
    index_x = 0
    index_y = 0
    for i in range(n+1):
        index_y = 0
        for j in range(n+1):        
            if (i!=idx)&(j!=jdx):
                A_real[index_x,index_y] = A[0,i,j]
                A_imag[index_x,index_y] = A[1,i,j]
                index_y += 1
        if (i!=idx):
            index_x += 1
    return torch.cat((A_real.unsqueeze(0),A_imag.unsqueeze(0)),0)

def complex_det(A):
    #det of a matrix
    n = A.size(1)
    if n == 1:
        return torch.cat((A[0,0,0].unsqueeze(0),A[1,0,0].unsqueeze(0)),0)
    else: 
        det = torch.tensor([0., 0.])
        for j in range(n):
            det += (-1)**((2+j)%2)*cmul2(mcat(A[0,0,j],A[1,0,j]),complex_det(laplace_temp(A,0,j)))
        return det

def mcat(A,B):
    Cre = A
    Cim = B
    return torch.cat((Cre.unsqueeze(0),Cim.unsqueeze(0)),0)

def test_det(n):
    A = torch.eye(n) 
    B = torch.zeros([n,n])
    C = mcat(A,B)
    return complex_det(C)

def test_whether_conjT(A,num):
    loss = 0
    for i in range(num):
        loss += torch.norm(A[:,:,:,i]-conjT(A[:,:,:,i]))
    return loss

#override the inverse of a complex matrix
class MyInv(torch.autograd.Function):
    @staticmethod    
    def forward(ctx, A):    
        ctx.save_for_backward(A)
        A_np = tensor2np(A)
        A_np_inv = linalg.pinv(A_np)
        A_inv_torch = np2tensor(A_np_inv)
        return A_inv_torch
        
    @staticmethod    
    def backward(ctx, grad_output):
        A, = ctx.saved_tensors    
        A_np = tensor2np(A)   
        A_inv_np = linalg.pinv(A_np)
        
        grad_output_np = tensor2np(grad_output)
        #print(grad_output_np)
        
        grad_back_np = np.dot(np.dot(A_inv_np,grad_output_np),A_inv_np)
        grad_back = np2tensor(grad_back_np)
        if torch.isfinite(grad_back.mean()) ==False:
            print("forward A is====")
            print(A)
            print("backward A is====")
            print(grad_back)
        #print("inv_grad_back")
        #print(grad_back)
        
        return -grad_back

#override the det of a complex matrix
class MyLogDet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        ctx.save_for_backward(A)
        A_np = tensor2np(A)
        A_np_det = linalg.det(A_np)
        A_np_log_det = np.log(A_np_det)
 
        A_np_log_det = np.array(A_np_log_det)
        A_log_det_torch = np2tensor(A_np_log_det)
        return A_log_det_torch[0]

    @staticmethod
    def backward(ctx, grad_output):
        A, = ctx.saved_tensors
        A_np = tensor2np(A)
        A_inv_np = linalg.pinv(A_np)

        grad_back = np2tensor(A_inv_np) * grad_output
        if torch.isfinite(grad_back.mean()) ==False:
            print("forward A is====")
            print(A)
            print("backward A is====")
            print(grad_back)
        #print("logdet_grad_back")
        #print(grad_back)
        return conjT(grad_back)