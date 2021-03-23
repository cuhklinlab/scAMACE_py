from model_based_torch_cpu import *


def pst_rna(phi_rna, g_1, g_0, w_rna, pi_rna,alpha_1=2, beta_1=2):
    res = 0
    # rna data
    a = cal_prod(w_rna, pi_rna[0][:, None] * g_1) + cal_prod(1 - w_rna, pi_rna[1][:, None] * g_1)
    a += cal_prod(w_rna, (1 - pi_rna[0])[:, None] * g_0)
    a += cal_prod(1 - w_rna, (1 - pi_rna[1])[:, None] * g_0)
    b = torch.sum(torch.log(a), dim=2) + torch.log(phi_rna)[:, None]
    # b = b.type(torch.float64)
    # print(b)
    b = torch.exp(b)
    # print(b)
    # b = phi_rna[:,None] * np.prod(a,dim = 2)
    tmp_pst = torch.sum(torch.log(torch.sum(b, dim=0)))
    res += tmp_pst

    # prior
    res += torch.sum(torch.log(phi_rna))
    res -= torch.sum(torch.log(1 - pi_rna[1]))
    res += torch.sum((alpha_1-1) * torch.log(w_rna) + (beta_1-1) * torch.log(1 - w_rna))
    return res


def EM_rna(g_1, g_0, phi_rna, w_rna, pi_rna, alpha_1=2, beta_1=2, max_iter=200, flag=True):
    pst = []
    for i in range(max_iter):
        E_z_rna, E_z_u_rna, E_z_1_u_rna, E_z_u_v_rna, E_z_1_u_v_rna, E_z_1_u_v_rna = cal_E(g_1, g_0, phi_rna, w_rna,
                                                                                           pi_rna)
        K = E_z_rna.shape[0]
        n_rna = E_z_rna.shape[1]
        phi_rna_n = (1 + torch.sum(E_z_rna, dim=1)) / (K + n_rna)

        pi_rna_n = []  # vector
        pi_rna_n.append(torch.sum(E_z_u_v_rna, dim=(0, 2)) / torch.sum(E_z_u_rna, dim=(0, 2)))
        pi_rna_n.append(torch.sum(E_z_1_u_v_rna, dim=(0, 2)) / (torch.sum(E_z_1_u_rna, dim=(0, 2)) - 1))
        pi_rna_n = torch.stack(pi_rna_n)

        # update w_rna
        w_rna_n = (torch.sum(E_z_u_rna, dim=1) + alpha_1 - 1) / (torch.sum(E_z_rna, dim=1)[:, None] + alpha_1 + beta_1 - 2)
        w_rna_n = cut_off(w_rna_n)

        phi_rna = phi_rna_n
        w_rna = w_rna_n

        ind = pi_rna_n[0] > pi_rna_n[1]
        pi_rna[0][ind] = pi_rna_n[0][ind]
        pi_rna[1][ind] = pi_rna_n[1][ind]
        print(i, 'done')
        if flag == True:
            tmp_pst = pst_rna(phi_rna, g_1, g_0, w_rna, pi_rna,alpha_1, beta_1)
            pst.append(tmp_pst)
            print(i, 'pst done')

    if flag == False:
        pst.append(pst_rna(phi_rna, g_1, g_0, w_rna, pi_rna,alpha_1, beta_1))

    return {'pst': pst, 'phi_rna': phi_rna,
            'w_rna': w_rna, 'pi_rna': pi_rna, 'E_z_rna': E_z_rna}





def pst_atac(f_1, f_0, phi_atac, w_atac, qi,alpha_1=2, beta_1=2,alpha_qi=2, beta_qi=2):
    res = 0
    K = phi_atac.shape[0]
    p = f_1.shape[1]
    # posterior for atac data
    # print(qi[:,None] * f_1)
    a = cal_prod(w_atac, qi[:, None] * f_1)
    b = cal_prod(w_atac, qi[:, None] * f_0)
    d = a + f_0 - b
    c = phi_atac[:, None] * torch.prod(d, dim=2)
    res += torch.sum(torch.log(torch.sum(c, dim=0)))
    # #     n_atac = f_1.shape[0]

    # prior
    res += torch.sum(torch.log(phi_atac)) + torch.sum((alpha_qi-1) * torch.log(qi) + (beta_qi-1) * torch.log(1 - qi))
    res += torch.sum((alpha_1-1) * torch.log(w_atac) + (beta_1-1) * torch.log(1 - w_atac))

    return res


def cal_E_atac(f_1, f_0, phi_atac, w_atac, qi):
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
    # print('E_z_u_atac',E_z_u_atac)

    # calculate E_z_u_u_t dimension k*i*r
    div_1 = qi * f_1
    div_2 = (1 - qi) * f_0
    tmp_E = div_1 / (div_1 + div_2)
    E_z_u_u_t_atac = tmp_E[None, :, :] * E_z_u_atac  # 1 * i *r  multiply k*i*r

    return E_z_atac, E_z_u_atac, E_z_u_u_t_atac


def EM_atac(f_1, f_0, phi_atac, w_atac, qi, alpha_1=2, beta_1=2,
            alpha_qi=2, beta_qi=2, max_iter=200,
            flag=True):
    pst = []
    for i in range(max_iter):
        E_z_atac, E_z_u_atac, E_z_u_u_t_atac = cal_E_atac(f_1, f_0, phi_atac, w_atac, qi)

        # M step
        K = E_z_atac.shape[0]

        n_atac = E_z_atac.shape[1]
        phi_atac_n = (1 + torch.sum(E_z_atac, dim=1)) / (K + n_atac)

        qi_n = (torch.sum(E_z_u_u_t_atac, dim=(0, 2)) + alpha_qi - 1) / (torch.sum(E_z_u_atac, dim=(0, 2))
                                                                         + alpha_qi + beta_qi - 2)

        # update w_atac
        w_atac_n = (torch.sum(E_z_u_atac, dim=1) + alpha_1 - 1) / (
                    torch.sum(E_z_atac, dim=1)[:, None] + alpha_1 + beta_1 - 2)
        w_atac_n = cut_off(w_atac_n)

        qi = qi_n
        phi_atac = phi_atac_n
        w_atac = w_atac_n

        print(i, 'done')
        if flag == True:
            tmp_pst = pst_atac(f_1, f_0, phi_atac, w_atac, qi,alpha_1, beta_1,alpha_qi, beta_qi)
            pst.append(tmp_pst)
            # print(tmp_pst)
            print(i, 'pst done')

    if flag == False:
        pst.append(pst_atac(f_1, f_0, phi_atac, w_atac, qi,alpha_1, beta_1,alpha_qi, beta_qi))

    return {'pst': pst, 'phi_atac': phi_atac, 'w_atac': w_atac,
            'qi': qi, 'E_z_atac': E_z_atac}


def pst_met(h_1, h_0, phi_met, w_met, pi_met,alpha_1=2, beta_1=2):
    res = 0
    n_met = h_1.shape[0]
    a = cal_prod(w_met, pi_met[0][:, None] * h_1) + cal_prod(1 - w_met, pi_met[1][:, None] * h_1)
    a += cal_prod(w_met, (1 - pi_met[0])[:, None] * h_0)
    a += cal_prod(1 - w_met, (1 - pi_met[1])[:, None] * h_0)

    product = torch.sum(torch.log(a), dim=2)
    product = product.type(torch.float64)
    d = torch.max(product - product[0, :])
    d = d.type(torch.float64)
    # print(d)

    c = phi_met[:, None] * torch.exp(product - product[0, :] - d / 2)

    res += torch.sum(torch.log(torch.sum(c, dim=0))) + torch.sum(product[0]) + d / 2 * n_met

    # prior
    res += torch.sum(torch.log(phi_met))
    res -= torch.sum(torch.log(pi_met[1]))
    res += torch.sum((alpha_1-1) * torch.log(w_met) + (beta_1-1) * torch.log(1 - w_met))

    return res


def EM_met(h_1, h_0, phi_met, w_met, pi_met, alpha_1=2, beta_1=2, max_iter=200,
           flag=True):
    pst = []

    for i in range(max_iter):

        E_z_met, E_z_u_met, E_z_1_u_met, E_z_u_v_met, E_z_1_u_v_met, E_z_1_u_v_met = cal_E(h_1, h_0, phi_met,
                                                                                           w_met, pi_met)

        n_met = E_z_met.shape[1]
        K = E_z_met.shape[0]
        phi_met_n = (1 + torch.sum(E_z_met, dim=1)) / (K + n_met)

        pi_met_n = []  # vector
        pi_met_n.append(torch.sum(E_z_u_v_met, dim=(0, 2)) / torch.sum(E_z_u_met, dim=(0, 2)))
        pi_met_n.append((torch.sum(E_z_1_u_v_met, dim=(0, 2)) - 1) / (torch.sum(E_z_1_u_met, dim=(0, 2)) - 1))
        pi_met_n = torch.stack(pi_met_n)

        # updata w_met
        w_met_n = (torch.sum(E_z_u_met, dim=1) + alpha_1 - 1) / (
                    torch.sum(E_z_met, dim=1)[:, None] + alpha_1 + beta_1 - 2)

        w_met_n = cut_off(w_met_n)

        print(i, 'done')

        w_met = w_met_n
        phi_met = phi_met_n

        ind = pi_met_n[0] < pi_met_n[1]
        pi_met[0][ind] = pi_met_n[0][ind]
        pi_met[1][ind] = pi_met_n[1][ind]

        if flag == True:
            tmp_pst = pst_met(h_1, h_0, phi_met, w_met, pi_met,alpha_1,beta_1)
            pst.append(tmp_pst)
            print(i, 'pst done')

    if flag == False:
        pst.append(pst_met(h_1, h_0, phi_met, w_met, pi_met,alpha_1,beta_1))

    return {'pst': pst, 'phi_met': phi_met, 'w_met': w_met, 'pi_met': pi_met, 'E_z_met': E_z_met}


