import numpy as np
import torch
import scipy.stats as ss
import warnings
import scipy.special as scsp

import pandas as pd


# simulation dataset generate
def cal_mu_quad(w, eta=-1, gamma=7, tau=-2):
    f = eta + gamma * w + tau * (w ** 2)
    mu_1 = 1 / (1 + torch.exp(-f))
    return mu_1


def cal_mu_lin(w, delta=-2, theta=5):
    g = delta + theta * w
    mu_2 = 1 / (1 + torch.exp(-g))
    return mu_2


def cal_prod(a, b):
    k = a.shape[0]
    res = torch.zeros((k, b.shape[0], a.shape[1]), dtype=torch.float64)
    for i in range(k):
        res[i] = a[i] * b
    return res


def cut_off(a, threshold=1e-6):
    a[a <= threshold] = threshold
    a[a >= 1 - threshold] = 1 - threshold
    return a



def cal_E(g_1, g_0, phi_rna, w_rna, pi_rna):
    # calculate E_z_lk   k*l'
    add_1 = pi_rna[0][:, None] * g_1 + (1 - pi_rna[0][:, None]) * g_0
    add_2 = pi_rna[1][:, None] * g_1 + (1 - pi_rna[1][:, None]) * g_0

    prod_1 = cal_prod(w_rna, add_1)
    prod_2 = cal_prod(1 - w_rna, add_2)
    Log_E_z_rna = torch.sum(torch.log(prod_1 + prod_2), dim=2)
    Log_E_z_rna = Log_E_z_rna - torch.max(Log_E_z_rna, dim=0)[0] + 20
    Log_E_z_rna = torch.log(phi_rna[:, None]) + Log_E_z_rna
    E_z_rna = torch.exp(Log_E_z_rna)
    E_z_rna = E_z_rna / torch.sum(E_z_rna, dim=0)[None, :]

    # calculate E_z_lk_u_lg k*l*g
    divi_part = prod_1 / (prod_1 + prod_2)
    P_z_rna = E_z_rna[:, :, None]
    E_z_u_rna = divi_part * P_z_rna

    # calculate E_z_lk_(1-u_lg) k*l*g
    E_z_1_u_rna = E_z_rna[:, :, None] - E_z_u_rna

    # calculate E_z_lk_u_lg_v_lg k*l*g
    div_1 = pi_rna[0][:, None] * g_1
    # div_1 = cut_off(div_1,threshold = 1e-24)
    div_2 = (1 - pi_rna[0][:, None]) * g_0
    c = div_1 + div_2
    min_c = torch.min(c)
    if min_c == 0.0:
        # print(min_c)
        c[c <= 1e-300] = 1e-300
    tmp_E = div_1 / c
    E_z_u_v_rna = tmp_E[None, :, :] * E_z_u_rna

    # div_2 = cut_off(div_2,threshold = 1e-24)
    # tmp_E = 1 + np.exp(np.log(div_2) - np.log(div_1))
    # print(pi_rna)
    # print(np.min(div_2))
    # c = np.min(div_1 + div_2)
    # if c < 1e-6:
    #     print(c)
    #     print(pi_rna)
    #     print('\n')
    # tmp_E = div_1 /(div_1 + div_2)
    # E_z_u_v_rna = E_z_u_rna / tmp_E[None,:,:]

    # calculate E_z_lk_(1-u_lg)_v_lg
    div_1 = pi_rna[1][:, None] * g_1
    div_2 = (1 - pi_rna[1][:, None]) * g_0
    c = div_1 + div_2
    min_c = torch.min(c)
    if min_c == 0.0:
        # print(min_c)
        c[c <= 1e-300] = 1e-300
    # print(np.min(1-pi_rna[1]))
    tmp_E = div_1 / c
    E_z_1_u_v_rna = tmp_E[None, :, :] * E_z_1_u_rna

    # calculate E_z_lk_v_lg
    E_z_v_rna = E_z_u_v_rna + E_z_1_u_v_rna
    return (E_z_rna, E_z_u_rna, E_z_1_u_rna, E_z_u_v_rna,
            E_z_1_u_v_rna, E_z_1_u_v_rna)


def E_step(f_1, f_0, phi_atac, w_atac, qi,
           g_1, g_0, phi_rna, w_rna, pi_rna,
           h_1, h_0, phi_met, w_met, pi_met):
    # f_1,f_0 dimension n_atac * p (i*r)
    # qi dimension i*1

    # ----------ATAC
    # calculate E_z_ik result dimension k * n_atac, which is also
    # P_old(z_ik = 1)
    n_atac = f_0.shape[0]
    #     if type(qi) is float:
    #         qi = np.repeat(qi,n_atac)
    qi = qi[:, None]
    qf = qi * f_1 + (1 - qi) * f_0
    prod_1 = cal_prod(w_atac, qf)
    prod_2 = cal_prod(1 - w_atac, f_0)
    Log_E_z_atac = torch.sum(torch.log(prod_1 + prod_2), dim=2)  # result dimension k * n_atac
    Log_E_z_atac = Log_E_z_atac - torch.max(Log_E_z_atac, dim=0)[0] + 20
    Log_E_z_atac = torch.log(phi_atac[:, None]) + Log_E_z_atac
    E_z_atac = torch.exp(Log_E_z_atac)
    E_z_atac = E_z_atac / torch.sum(E_z_atac, dim=0)[None, :]

    # calculate  E_z_ik_u_ir  dimension k*i*r
    divi_part = prod_1 / (prod_1 + prod_2)
    P_z_atac = E_z_atac[:, :, None]
    E_z_u_atac = divi_part * P_z_atac

    # calculate E_z_u_u_t dimension k*i*r
    div_1 = qi * f_1
    div_2 = (1 - qi) * f_0
    tmp_E = div_1 / (div_1 + div_2)
    E_z_u_u_t_atac = tmp_E[None, :, :] * E_z_u_atac  # 1 * i *r  multiply k*i*r

    # -------------RNA

    E_z_rna, E_z_u_rna, E_z_1_u_rna, E_z_u_v_rna, E_z_1_u_v_rna, E_z_1_u_v_rna = cal_E(g_1, g_0, phi_rna,
                                                                                       w_rna, pi_rna)

    E_z_met, E_z_u_met, E_z_1_u_met, E_z_u_v_met, E_z_1_u_v_met, E_z_1_u_v_met = cal_E(h_1, h_0, phi_met,
                                                                                       w_met, pi_met)

    return (E_z_atac, E_z_u_atac, E_z_u_u_t_atac,
            E_z_rna, E_z_u_rna, E_z_1_u_rna, E_z_u_v_rna, E_z_1_u_v_rna, E_z_1_u_v_rna,
            E_z_met, E_z_u_met, E_z_1_u_met, E_z_u_v_met, E_z_1_u_v_met, E_z_1_u_v_met)


def update_w_rna_linked(K, p, E_z_u_rna, E_z_1_u_rna, w_atac, w_met,
                        alpha_1=2, beta_1=2, phi_1=10, phi_2=10,
                        grid_n=100, eta=-1, gamma=7, tau=-2,
                        delta=-2, theta=5):
    #     print(K)
    #     print(n_rna)
    # w_rna = np.array([np.linspace(0,1,grid_n + 1) for i in range(K*p)])
    w_rna = torch.stack([torch.linspace(0, 1, grid_n + 1) for i in range(K * p)])
    w_rna = w_rna.reshape((K, p, grid_n + 1))
    w_rna = w_rna.type(torch.float64)
    w_rna = cut_off(w_rna)
    # k*g
    to_max = torch.sum(E_z_u_rna, dim=1)[:, :, None] * torch.log(w_rna)
    to_max += torch.sum(E_z_1_u_rna, dim=1)[:, :, None] * torch.log(1 - w_rna)
    to_max += (alpha_1 - 1) * torch.log(w_rna) + (beta_1 - 1) * torch.log(1 - w_rna)

    mu_1 = cal_mu_quad(w_rna, eta=eta, gamma=gamma, tau=tau)
    tmp_c = mu_1 * phi_1
    to_max += (tmp_c - 1) * torch.log(w_atac)[:, :, None] + (phi_1 - tmp_c - 1) * torch.log(1 - w_atac)[:, :, None]
    to_max -= torch.log(scsp.beta(tmp_c, phi_1 - tmp_c))

    mu_2 = cal_mu_lin(w_rna, delta=delta, theta=theta)
    tmp_c = mu_2 * phi_2
    to_max += (tmp_c - 1) * torch.log(w_met)[:, :, None] + (phi_2 - tmp_c - 1) * torch.log(1 - w_met)[:, :, None]
    to_max -= torch.log(scsp.beta(tmp_c, phi_2 - tmp_c))

    i = torch.max(to_max, dim=2)[1]
    # a = torch.stack([np.repeat(i,p) for i in range (K)])
    a = torch.stack([torch.tensor(i).repeat(p) for i in range(K)])
    b = torch.stack([torch.arange(p) for i in range(K)])

    res = w_rna[a, b, i]
    return res


def M_step(E_z_atac, E_z_u_atac, E_z_u_u_t_atac,
           E_z_rna, E_z_u_rna, E_z_1_u_rna, E_z_u_v_rna, E_z_1_u_v_rna,
           E_z_met, E_z_u_met, E_z_1_u_met, E_z_u_v_met, E_z_1_u_v_met,
           w_atac, w_met, alpha_qi=2, beta_qi=2,
           phi_1=10, phi_2=10,
           eta=-1, gamma=7, tau=-2,
           delta=-2, theta=5,alpha_1=2, beta_1=2):
    K = E_z_atac.shape[0]
    p = E_z_u_atac.shape[2]

    n_atac = E_z_atac.shape[1]
    phi_atac = (1 + torch.sum(E_z_atac, dim=1)) / (K + n_atac)

    n_rna = E_z_rna.shape[1]
    phi_rna = (1 + torch.sum(E_z_rna, dim=1)) / (K + n_rna)

    n_met = E_z_met.shape[1]
    phi_met = (1 + torch.sum(E_z_met, dim=1)) / (K + n_met)

    # qi is an vector not scalar
    qi = (torch.sum(E_z_u_u_t_atac, dim=(0, 2)) + alpha_qi - 1) / (torch.sum(E_z_u_atac, dim=(0, 2))
                                                                    + alpha_qi + beta_qi - 2)
    # print('qi',qi)
    pi_rna = []  # vector
    pi_rna.append(torch.sum(E_z_u_v_rna, dim=(0, 2)) / torch.sum(E_z_u_rna, dim=(0, 2)))
    pi_rna.append(torch.sum(E_z_1_u_v_rna, dim=(0, 2)) / (torch.sum(E_z_1_u_rna, dim=(0, 2)) - 1))
    pi_rna = torch.stack(pi_rna)


    pi_met = []  # vector
    pi_met.append(torch.sum(E_z_u_v_met, dim=(0, 2)) / torch.sum(E_z_u_met, dim=(0, 2)))
    pi_met.append((torch.sum(E_z_1_u_v_met, dim=(0, 2)) - 1) / (torch.sum(E_z_1_u_met, dim=(0, 2)) - 1))
    pi_met = torch.stack(pi_met)

    w_rna = update_w_rna_linked(K, p, E_z_u_rna, E_z_1_u_rna, w_atac, w_met,
                                phi_1=phi_1, phi_2=phi_2,
                                eta=eta, gamma=gamma, tau=tau,
                                delta=delta, theta=theta,alpha_1=alpha_1, beta_1=beta_1)


    mu_1 = cal_mu_quad(w_rna, eta=eta, gamma=gamma, tau=tau)
    w_atac = (torch.sum(E_z_u_atac, dim=1) + mu_1 * phi_1 - 1) / (torch.sum(E_z_atac, dim=1)[:, None]
                                                                   + phi_1 - 2)

    w_atac = cut_off(w_atac)

    mu_2 = cal_mu_lin(w_rna, delta=delta, theta=theta)
    w_met = (torch.sum(E_z_u_met, dim=1) + mu_2 * phi_2 - 1) / (torch.sum(E_z_met, dim=1)[:, None]
                                                                 + phi_2 - 2)
    w_met = cut_off(w_met)

    return phi_atac, phi_rna, phi_met, qi, pi_rna, pi_met, w_rna, w_atac, w_met


# calculate posterior p( theta | X )
def cal_pst(phi_atac, f_1, f_0,
            phi_rna, g_1, g_0,
            phi_met, h_1, h_0,
            qi, pi_rna, pi_met, w_rna, w_atac, w_met
            , phi_1=10, phi_2=10,
            eta=-1, gamma=7, tau=-2,
            delta=-2, theta=5,alpha_qi=2,beta_qi=2,alpha_1=2,beta_1=2):
    mu_1 = cal_mu_quad(w_rna, eta=eta, gamma=gamma, tau=tau)
    mu_2 = cal_mu_lin(w_rna, delta=delta, theta=theta)

    K = phi_atac.shape[0]
    p = f_1.shape[1]
    pst = []

    # posterior for atac data
    a = cal_prod(w_atac, qi[:, None] * f_1)
    b = cal_prod(w_atac, qi[:, None] * f_0)
    res = a + f_0 - b
    c = phi_atac[:, None] * torch.prod(res, dim=2)
    tmp_pst = torch.sum(torch.log(torch.sum(c, dim=0)))
    pst.append(tmp_pst)

    # for rna data
    a = cal_prod(w_rna, pi_rna[0][:, None] * g_1) + cal_prod(1 - w_rna, pi_rna[1][:, None] * g_1)
    a += cal_prod(w_rna, (1 - pi_rna[0])[:, None] * g_0)
    a += cal_prod(1 - w_rna, (1 - pi_rna[1])[:, None] * g_0)
    b = torch.sum(torch.log(a), dim=2) + torch.log(phi_rna)[:, None]
    b = torch.exp(b)
    tmp_pst = torch.sum(torch.log(torch.sum(b, dim=0)))
    pst.append(tmp_pst)

    # for met data
    n_met = h_1.shape[0]
    a = cal_prod(w_met, pi_met[0][:, None] * h_1) + cal_prod(1 - w_met, pi_met[1][:, None] * h_1)
    a += cal_prod(w_met, (1 - pi_met[0])[:, None] * h_0)
    a += cal_prod(1 - w_met, (1 - pi_met[1])[:, None] * h_0)

    # solve overflow problem
    product = torch.sum(torch.log(a), dim=2)
    product = product.type(torch.float64)
    d = torch.max(product - product[0, :])
    d = d.type(torch.float64)

    c = phi_met[:, None] * torch.exp(product - product[0, :] - d / 2)
    # print('c',c)
    # c =  c.type(torch.float64)
    # print(torch.log(torch.sum(c,dim = 0 )))
    # tmp =torch.log(torch.sum(c,dim = 0 ))
    # print(torch.max(torch.sum(c,dim =0)))
    # print(tmp)
    # print(tmp.shape)
    # print(torch.max(tmp))
    # print(torch.sum(tmp))
    tmp_pst = torch.sum(torch.log(torch.sum(c, dim=0))) + torch.sum(product[0]) + d / 2 * n_met
    # print('met',tmp_pst)
    pst.append(tmp_pst)

    # for prior
    tmp_pst = torch.sum(torch.log(phi_atac)) + torch.sum(torch.log(phi_rna)) + torch.sum(torch.log(phi_met))
    tmp_pst += torch.sum((alpha_qi-1) * torch.log(qi) + (beta_qi-1) * torch.log(1 - qi))
    tmp_pst -= torch.sum(torch.log(1 - pi_rna[1]))
    tmp_pst -= torch.sum(torch.log(pi_met[1]))

    tmp_pst += torch.sum((alpha_1-1) * torch.log(w_rna) + (beta_1-1) * torch.log(1 - w_rna))
    tmp_c = mu_1 * phi_1
    tmp_pst += torch.sum((tmp_c - 1) * torch.log(w_atac) + (phi_1 - tmp_c - 1) * torch.log(1 - w_atac))
    tmp_pst -= torch.sum(torch.log(scsp.beta(tmp_c, phi_1 - tmp_c)))
    tmp_c = mu_2 * phi_2
    tmp_pst += torch.sum((tmp_c - 1) * torch.log(w_met) + (phi_2 - tmp_c - 1) * torch.log(1 - w_met))
    tmp_pst -= torch.sum(torch.log(scsp.beta(tmp_c, phi_2 - tmp_c)))
    pst.append(tmp_pst)
    # print(pst)

    return torch.sum(torch.stack(pst))


# def EM(f_1,f_0,g_1,g_0,h_1,h_0,K = 3,max_iter = 100):
def EM(f_1, f_0, g_1, g_0, h_1, h_0,
       phi_atac, phi_rna, phi_met, pi_rna, pi_met, w_rna, w_met, w_atac, qi,
       phi_1=10, phi_2=10,
       eta=-1, gamma=7, tau=-2,
       delta=-2, theta=5,
       alpha_qi=2, beta_qi=2, alpha_1=2, beta_1=2,
       max_iter=100, flag=True):
    pst = []
    # likely = []
    # initialize
    # p = f_1.shape[1]
    # n_met = h_1.shape[0]
    n_rna = g_1.shape[0]

    for i in range(max_iter):
        (E_z_atac, E_z_u_atac, E_z_u_u_t_atac,
         E_z_rna, E_z_u_rna, E_z_1_u_rna, E_z_u_v_rna, E_z_1_u_v_rna, E_z_1_u_v_rna,
         E_z_met, E_z_u_met, E_z_1_u_met, E_z_u_v_met, E_z_1_u_v_met, E_z_1_u_v_met) = E_step(f_1, f_0, phi_atac,
                                                                                              w_atac, qi, g_1, g_0,
                                                                                              phi_rna, w_rna, pi_rna,
                                                                                              h_1, h_0, phi_met, w_met,
                                                                                              pi_met)

        # E_z_l.append(E_z_rna)

        phi_atac_n, phi_rna_n, phi_met_n, qi_n, pi_rna_n, pi_met_n, w_rna_n, w_atac_n, w_met_n = M_step(E_z_atac,
                                                                                                        E_z_u_atac,
                                                                                                        E_z_u_u_t_atac,
                                                                                                        E_z_rna,
                                                                                                        E_z_u_rna,
                                                                                                        E_z_1_u_rna,
                                                                                                        E_z_u_v_rna,
                                                                                                        E_z_1_u_v_rna,
                                                                                                        E_z_met,
                                                                                                        E_z_u_met,
                                                                                                        E_z_1_u_met,
                                                                                                        E_z_u_v_met,
                                                                                                        E_z_1_u_v_met,
                                                                                                        w_atac, w_met,
                                                                                                        phi_1=phi_1,
                                                                                                        phi_2=phi_2,
                                                                                                        eta=eta,
                                                                                                        gamma=gamma,
                                                                                                        tau=tau,
                                                                                                        delta=delta,
                                                                                                        theta=theta,
                                                                                                        alpha_qi=alpha_qi,
                                                                                                        beta_qi=beta_qi,
                                                                                                        alpha_1=alpha_1,
                                                                                                        beta_1=beta_1)


        phi_rna = phi_rna_n
        phi_atac = phi_atac_n
        phi_met = phi_met_n

        w_rna = w_rna_n

        w_met = w_met_n
        # w_met = w_met

        w_atac = w_atac_n
        # w_atac = w_atac
        # pi_rna = pi_rna_n
        qi = qi_n

        # pi_met = pi_met_n

        ind = pi_met_n[0] < pi_met_n[1]
        pi_met[0][ind] = pi_met_n[0][ind]
        pi_met[1][ind] = pi_met_n[1][ind]

        ind = pi_rna_n[0] > pi_rna_n[1]
        pi_rna[0][ind] = pi_rna_n[0][ind]
        pi_rna[1][ind] = pi_rna_n[1][ind]

        if n_rna > 100:
            print(i, 'done')

        if flag == True:

            tmp_pst = cal_pst(phi_atac, f_1, f_0,
                              phi_rna, g_1, g_0,
                              phi_met, h_1, h_0,
                              qi, pi_rna, pi_met, w_rna, w_atac, w_met,
                              phi_1=phi_1, phi_2=phi_2, eta=eta,
                              gamma=gamma, tau=tau,
                              delta=delta, theta=theta,
                              alpha_qi=alpha_qi, beta_qi=beta_qi, alpha_1=alpha_1, beta_1=beta_1)
            pst.append(tmp_pst)
            if n_rna > 100:
                print(i, 'pst done')

        # tmp_pst=[None,None]
        res = {'pst': pst, 'phi_atac': phi_atac, 'phi_rna': phi_rna,
               'phi_met': phi_met, 'qi': qi, 'pi_rna': pi_rna, 'pi_met': pi_met,
               'w_rna': w_rna, 'w_atac': w_atac, 'w_met': w_met, 'E_z_rna': E_z_rna,
               'E_z_atac': E_z_atac, 'E_z_met': E_z_met}
        # likely.append(tmp_pst[1])
    return res
    # return (pst,phi_atac_n,phi_rna_n,phi_met_n,
    #         qi_n,pi_rna_n,pi_met_n, w_rna_n,w_atac_n,
    #         w_met_n,E_z_rna,E_z_atac,E_z_met)

