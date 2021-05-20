import numpy as np
import torch
import scipy.stats as ss

import scipy.special as scsp

import pandas as pd



def cal_mu_quad(w,eta = -1,gamma = 7,tau = -2 ):
    f = eta + gamma*w + tau*(w**2)
    mu_1 = 1/(1 + np.exp(-f))
    return mu_1


def cal_mu_lin(w,delta = -2,theta = 5):
    g = delta + theta * w
    mu_2 = 1/(1 + np.exp(-g))
    return mu_2


# def cal_prod(a,b):
#     return np.array([v1*b for v1 in a])
def cal_prod(a,b):
    k = a.shape[0]
    res = torch.zeros((k,b.shape[0],a.shape[1]),dtype = torch.float64)
    for i in range(k):
        res[i] = a[i] * b
    return res

def cut_off(a,threshold = 1e-6):
    a[ a <= threshold ] = threshold
    a[ a>= 1 - threshold ] = 1 - threshold
    return a


def totensor(a, precision = torch.float64):
    a = torch.from_numpy(a)
    a = a.type(precision)
    return a


def generate_sim_data( Xprobs1 = [1/3]*3, Xprobs2 = [1/3]*3, Xprobs3 = [1/3]*3
                    , n_atac = 900, n_rna=1100, n_met=1000,p = 1000, prop = 0.05, 
                    shape_rna = [7,1], scale_rna = [0.5,1],
                    a_met = [0.5,1], b_met = [0.5,10],
                    qi = 0.1,pi_rna = [0.7,0.3], pi_met = [0.4,0.7],cutoff=10^-6,
                    alpha1 = 2, beta1 = 2,phi_1 = 10,phi_2 = 10,omega = 0.8,
                    eta = -1,gamma = 7,tau = -2,
                    delta = -2,theta = 5,):
    '''
    n_atac: sample size of ATAC
    n_rna: sample size of RNA
    n_met: sample size of MET
    p: number of features
    '''                                       
    K = len(Xprobs1) #lengths of the three lists are the same
    w_rna = np.random.beta(alpha1,beta1,size = (1,p))
    add_0 = np.zeros((K,p))
    w_rna = w_rna + add_0 #employ ndarray broadcasting here
    
    slot = int(p * prop)
    
    div = K // 2 
    if K % 2:
    # K is odd
        tmp_list = np.linspace(0, omega, div +1).tolist()
        omega_seq1 = tmp_list + [0.5] + (1 - np.array(tmp_list)).tolist()
    else:
        omega_seq1 = tmp_list + (1 - np.array(tmp_list)).tolist()
    omega_seq1.sort(reverse = True)
    
    omega_seq = omega_seq1[1:-1] # drop duplicate
    for i in range(K):
        w_rna[:,i*slot:((i+1) * slot)] = np.roll(omega_seq,-i)[:,None]
    
    # handle missing clusters
    Xprobs2 = np.array(Xprobs2)
    rna_miss = np.where(Xprobs2 == 0.0)[0]
    w_rna[rna_miss] = 0 # need to drop these rows before output
    
    n_diff = int(p * prop * K)
    
    #x atac, y rna, t met
    
    # w_atac
    mu_atac = cal_mu_quad(w_rna,eta = eta,gamma = gamma,tau = tau )
    alpha_atac = mu_atac * phi_1
    beta_atac = -mu_atac * phi_1 + phi_1
    w_atac = np.random.beta(alpha_atac,beta_atac)
    w_atac[1:,n_diff:]  = w_atac[0,n_diff:]
    #handle missing, ops here seem useless
    Xprobs1 = np.array(Xprobs1)
    atac_miss = np.where(Xprobs1 == 0.0)[0]
    w_atac[atac_miss] = 0
    
    
    #w_met
    mu_met = cal_mu_lin(w_rna,delta = delta,theta = theta)
    alpha_met = mu_met * phi_2



    beta_met = -mu_met * phi_2 + phi_2
    w_met = np.random.beta(alpha_met,beta_met)
    w_met[1:,n_diff:] = w_met[0,n_diff:]
    #handle missing, ops here seem useless
    Xprobs3 = np.array(Xprobs3)
    met_miss = np.where(Xprobs3 == 0.0)[0]
    w_met[met_miss] = 0
    
    
    #sample z, Xprobs has zeors and those clusters won't be sampled using np.random.choice
    z_rna = np.random.choice(K,n_rna , p=Xprobs2)
    
    z_atac = np.random.choice(K,n_atac,p = Xprobs1)
    
    z_met = np.random.choice(K,n_met, p = Xprobs3)
    
    
    #sample x atac
    
    #sample u_x
    u_atac = np.ones((n_atac,p),dtype = int)
    bernou_para = w_atac[z_atac,:]
    u_x = np.random.binomial(u_atac,bernou_para)
    
    #sample o_x
    o_atac = np.ones((n_atac,p),dtype = int)
    bernou_para = np.zeros((n_atac,p))
    bernou_para[u_x == 1] = qi
    o_x = np.random.binomial(o_atac,bernou_para)
    
    #generate x,f_1, f_0
    x = o_x.copy()
    f_1 = o_x.copy()
    f_0 = -(o_x - 1) # 0-->1, 1-->0
    
    
    
    #sample y rna
    
    #sample u_y
    u_rna = np.ones((n_rna,p),dtype = int)
    bernou_para = w_rna[z_rna,:]
    u_y = np.random.binomial(u_rna,bernou_para)
    
    
    #sample v_y
    v_rna = np.ones((n_rna,p),dtype = int) 
    bernou_para = np.ones((n_rna,p))* pi_rna[-1]
    bernou_para[u_y == 1] = pi_rna[0]
    v_y = np.random.binomial(v_rna,bernou_para)
    
    
    #generate rna data
    shape_rna_all =  np.ones((n_rna,p)) * shape_rna[-1]
    scale_rna_all = np.ones((n_rna,p)) * scale_rna[-1]
    mask = (v_y == 1)
    shape_rna_all[ mask ] = shape_rna[0]
    scale_rna_all[ mask ] = scale_rna[0]
    y = np.random.gamma(shape_rna_all,scale_rna_all)
    g_1 = ss.gamma.pdf(y,a = shape_rna[0],scale = scale_rna[0])
    g_0 = ss.gamma.pdf(y,a = shape_rna[1],scale = scale_rna[1])
    
    
    #sample u_t
    u_met = np.ones((n_met,p),dtype = int)
    bernou_para = w_met[z_met,:]
    u_t = np.random.binomial(u_met,bernou_para)
    
    
    #sample v_t
    v_met = np.ones((n_met,p),dtype = int)
    bernou_para = np.ones((n_met,p)) * pi_met[-1]
    bernou_para[u_t == 1] = pi_met[0]
    v_t = np.random.binomial(v_met,bernou_para)
    
    
    #generate met data
    
    a_met_all = np.ones((n_met,p)) * a_met[-1]
    b_met_all = np.ones((n_met,p)) * b_met[-1]
    mask = (v_met == 1)
    a_met_all[ mask ] = a_met[0]
    b_met_all[ mask ] = b_met[0]
    t = np.random.beta(a_met_all,b_met_all)
    h_1 = ss.beta.pdf(t,a = a_met[0],b = b_met[0])
    h_0 = ss.beta.pdf(t,a = a_met[1],b = b_met[1])
    
    
    #drop the missing rows
    w_atac = np.delete(w_atac,atac_miss,axis = 0)
    w_rna = np.delete(w_rna,rna_miss,axis = 0)
    w_met = np.delete(w_met,met_miss,axis = 0)
    
    w_atac = cut_off(w_atac,cutoff)
    w_rna = cut_off(w_rna,cutoff)
    w_met = cut_off(w_met,cutoff)
    
    
    #convert to torch.tensor
    w_atac = totensor(w_atac)
    w_rna = totensor(w_rna)
    w_met = totensor(w_met)
    
    u_x = totensor(u_x)
    o_x = totensor(o_x)
    x = totensor(x)
    f_1 = totensor(f_1)
    f_0 = totensor(f_0)
    
    
    u_y = totensor(u_y)
    v_y = totensor(v_y)
    y = totensor(y)
    g_1 = totensor(g_1)
    g_0 = totensor(g_0)
    
    
    u_t = totensor(u_t)
    v_t = totensor(v_t)
    t = totensor(t)
    h_1 = totensor(h_1)
    h_0 = totensor(h_0)
    
    
    return (w_atac, z_atac, u_x, o_x, x, f_1, f_0,
           w_rna, z_rna, u_y, v_y, y, g_1, g_0,
            w_met, z_met,u_t, v_t, t, h_1, h_0)