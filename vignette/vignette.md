# A quick guide to scAMACE_py

#### We present the usage of scAMACE_py in the following sections:
#### [Section 1: Introduction to datasets (Application 1: K562-GM12878 dataset)] (#section1)
#### Section 2: scAMACE CPU version
#### Section 3: scAMACE GPU version
#### Section 4: scAMACE(seperate) CPU version
#### Section 5: scAMACE(seperate) GPU version



## <a name="section1"></a> 1. Datasets
### Application 1: K562-GM12878 dataset
`Feb7_2021_3Types_Data_rna_mean_1000_ratio_f1.csv`: the known pdf f1 of scCAS data, cells by genes.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_f0.csv`: the known pdf f0 of scCAS data, cells by genes.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_g1.csv`: the known pdf g1 of scRNA-Seq data, cells by genes.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_g0.csv`: the known pdf g0 of scRNA-Seq data, cells by genes.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_data.csv`: sc-methylation data, cells by genes.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_atac_cell_lb.csv`: the true cell labels for scCAS data, 1: HL60 cell line, 2: K562 cell line.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_rna_cell_lb.csv`: the true cell labels for scRNA-Seq data, 1: HL60 cell line, 2: K562 cell line.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_cell_lb.csv`: the true cell labels for sc-methylation data, 1: HL60 cell line, 2: K562 cell line.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_acc.csv`: the initialization for omega_acc in scCAS data.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_exp.csv`: the initialization for omega_rna in scRNA-Seq data.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_met.csv`: the initialization for omega_met in sc-methylation data.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_qi.csv`: the initialization of the probability that scCAS data has high gene score when gene g is active in cell i (i.e. pi_{i1}).

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_rna.csv`: the initialization of the probability for scRNA-Seq data that gene g is expressed in cell l.

`Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_met.csv`: the initialization of the probability for sc-methylation data that gene g is methylated in cell d.


## 2. Example of scAMACE CPU version
#### Remarks: We demostrate usage of scAMACE_py through Application 1.
### 2.1 Load data and prepare for EM algorithm
```{python}
import scAMACE_py

folder = '/your_folder/'
f1 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_f1.csv',index_col=0).values
f0 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_f0.csv',index_col=0).values
g1 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_g1.csv',index_col=0).values
g0 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_g0.csv',index_col=0).values
met_data = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_data.csv',index_col=0).values


atac_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_atac_cell_lb.csv',index_col=0).values[:,0]
rna_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_rna_cell_lb.csv',index_col=0).values[:,0]
met_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_cell_lb.csv',index_col=0).values[:,0]


K = 2

rij = f1/(f1 + f0)
replacef0 = 1 - rij

rij = torch.from_numpy(rij)
rij = rij.type(torch.float64)
replacef0 = torch.from_numpy(replacef0)
replacef0 = replacef0.type(torch.float64)


rlg = g1/(g1 + g0)
replaceg0 = 1 - rlg

rlg = torch.from_numpy(rlg)
rlg = rlg.type(torch.float64)
replaceg0 = torch.from_numpy(replaceg0)
replaceg0 = replaceg0.type(torch.float64)


m = met_data
# m[m=="NA"] = np.nan
# m = np.array(m, dtype=float)
ratio = m/(1-m)
c = ratio[np.where(~np.isnan(ratio))]
cc = c[np.where(c!=0)]

sf = np.percentile(cc,50)
sf
rdm = ratio/sf
rdm = rdm**2 # power
rdm[np.isnan(rdm)] = np.mean(~np.isnan(rdm))
rdm[np.isinf(rdm)] = 3000
rdm[1:5,1:5]
replaceh0 = np.ones((rdm.shape[0],rdm.shape[1]))

rdm = torch.from_numpy(rdm)
rdm = rdm.type(torch.float64)
replaceh0 = torch.from_numpy(replaceh0)
replaceh0 = replaceh0.type(torch.float64)



alpha_1 = 2
beta_1 = 2

alpha_qi = 1
beta_qi = 1

alpha_1  = torch.tensor(alpha_1,dtype = torch.float64)
beta_1 = torch.tensor(beta_1,dtype = torch.float64)
alpha_qi = torch.tensor(alpha_qi,dtype = torch.float64)
beta_qi = torch.tensor(beta_qi,dtype = torch.float64)


##-------------------------------------------------------------------------##
### beta regression
phi_1 = 2.683904
eta = -1.190329    
gamma = 4.376499
tau = -3.036440


phi_2 = 3.18635
delta = 0.1167477
theta = 0.7307160


phi_1  = torch.tensor(phi_1,dtype = torch.float64)
phi_2 = torch.tensor(phi_2,dtype = torch.float64)
eta = torch.tensor(eta,dtype = torch.float64)
gamma = torch.tensor(gamma,dtype = torch.float64)
tau = torch.tensor(tau,dtype = torch.float64)
delta = torch.tensor(delta,dtype = torch.float64)
theta = torch.tensor(theta,dtype = torch.float64)


# load initialization
###----------------------------------------------------------------------###
phi_atac = np.array([1/K]*K)
phi_rna = np.array([1/K]*K)
phi_met =np.array([1/K]*K)


w_exp = pd.read_csvpd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_exp.csv',index_col=0).values 
w_acc = w_exp.copy()
w_met = w_exp.copy()


pi_rna = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_rna.csv',index_col=0).values
pi_met = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_met.csv',index_col=0).values
qi = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_square_qi.csv',index_col=0).values[:,0]


phi_atac = torch.from_numpy(phi_atac)
phi_atac = phi_atac.type(torch.float64)

phi_rna = torch.from_numpy(phi_rna)
phi_rna = phi_rna.type(torch.float64)

phi_met = torch.from_numpy(phi_met)
phi_met = phi_met.type(torch.float64)

w_exp = torch.from_numpy(w_exp)
w_exp = w_exp.type(torch.float64)

w_met = torch.from_numpy(w_met)
w_met = w_met.type(torch.float64)

w_acc = torch.from_numpy(w_acc)
w_acc = w_acc.type(torch.float64)

qi = torch.from_numpy(qi)
qi = qi.type(torch.float64)

pi_rna = torch.from_numpy(pi_rna)
pi_rna = pi_rna.type(torch.float64)

pi_met = torch.from_numpy(pi_met)
pi_met = pi_met.type(torch.float64)


```



### 2.2 run the EM algorithm
```{python}
init = EM(rij,replacef0,rlg,replaceg0,rdm,replaceh0,
          phi_atac,phi_rna,phi_met,pi_rna,pi_met,w_exp,w_met,w_acc,qi,
          phi_1 = phi_1, phi_2 = phi_2, eta = eta,gamma = gamma,tau = tau, delta = delta, theta = theta,
          alpha_qi=alpha_qi,beta_qi=beta_qi,alpha_1=alpha_1,beta_1=beta_1,
          max_iter = 1)



start = time.time()
res = EM(rij,replacef0,rlg,replaceg0,rdm,replaceh0,
         init['phi_atac'], init['phi_rna'], init['phi_met'], init['pi_rna'], init['pi_met'],
         init['w_rna'], init['w_met'], init['w_atac'], init['qi'], phi_1=phi_1, phi_2=phi_2,
         eta=eta, gamma=gamma, tau=tau, delta=delta, theta=theta,
         alpha_qi=alpha_qi,beta_qi=beta_qi,alpha_1=alpha_1,beta_1=beta_1,
         max_iter=200, flag=False)

end = time.time()

print('done')

print(end - start)

```


### 2.3 Summary clustering result (get the cluster assignments)
```{python}
(E_z_atac,E_z_u_atac,E_z_u_u_t_atac,
            E_z_rna,E_z_u_rna,E_z_1_u_rna,E_z_u_v_rna,E_z_1_u_v_rna,E_z_1_u_v_rna,
                   E_z_met,E_z_u_met,E_z_1_u_met,E_z_u_v_met,
     E_z_1_u_v_met,E_z_1_u_v_met) = E_step(rij,replacef0,res['phi_atac'], res['w_atac'],res['qi'],rlg,replaceg0,
                        res['phi_rna'],res['w_rna'],res['pi_rna'],rdm,replaceh0,res['phi_met']
                                       ,res['w_met'],res['pi_met'])

# amp, rmp and mmp are the cluster assignments offered by scAMACE
# scCAS data
amp = torch.argmax(E_z_atac, dim=0)

# scRNA-Seq data
rmp = torch.argmax(E_z_rna, dim=0)

# sc-methylation data
mmp = torch.argmax(E_z_met, dim=0)

pd.crosstab(atac_cell_lb,amp)
pd.crosstab(rna_cell_lb,rmp)
pd.crosstab(met_cell_lb,mmp)

```



## 3. Example of scAMACE GPU version
#### Remarks: We demostrate usage of scAMACE_py through Application 1.
### 3.1 Load data and prepare for EM algorithm
```{python}
import scAMACE_py

folder = '/your_folder/'
f1 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_f1.csv',index_col=0).values
f0 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_f0.csv',index_col=0).values
g1 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_g1.csv',index_col=0).values
g0 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_g0.csv',index_col=0).values
met_data = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_data.csv',index_col=0).values


atac_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_atac_cell_lb.csv',index_col=0).values[:,0]
rna_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_rna_cell_lb.csv',index_col=0).values[:,0]
met_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_cell_lb.csv',index_col=0).values[:,0]


K = 2

rij = f1/(f1 + f0)
replacef0 = 1 - rij

rij = torch.from_numpy(rij)
rij = rij.type(torch.float64)
replacef0 = torch.from_numpy(replacef0)
replacef0 = replacef0.type(torch.float64)


rlg = g1/(g1 + g0)
replaceg0 = 1 - rlg

rlg = torch.from_numpy(rlg)
rlg = rlg.type(torch.float64)
replaceg0 = torch.from_numpy(replaceg0)
replaceg0 = replaceg0.type(torch.float64)


m = met_data
# m[m=="NA"] = np.nan
# m = np.array(m, dtype=float)
ratio = m/(1-m)
c = ratio[np.where(~np.isnan(ratio))]
cc = c[np.where(c!=0)]

sf = np.percentile(cc,50)
sf
rdm = ratio/sf
rdm = rdm**2 # power
rdm[np.isnan(rdm)] = np.mean(~np.isnan(rdm))
rdm[np.isinf(rdm)] = 3000
rdm[1:5,1:5]
replaceh0 = np.ones((rdm.shape[0],rdm.shape[1]))

rdm = torch.from_numpy(rdm)
rdm = rdm.type(torch.float64)
replaceh0 = torch.from_numpy(replaceh0)
replaceh0 = replaceh0.type(torch.float64)



alpha_1 = 2
beta_1 = 2

alpha_qi = 1
beta_qi = 1

alpha_1  = torch.tensor(alpha_1,dtype = torch.float64)
beta_1 = torch.tensor(beta_1,dtype = torch.float64)
alpha_qi = torch.tensor(alpha_qi,dtype = torch.float64)
beta_qi = torch.tensor(beta_qi,dtype = torch.float64)


##-------------------------------------------------------------------------##
### beta regression
phi_1 = 2.683904
eta = -1.190329    
gamma = 4.376499
tau = -3.036440


phi_2 = 3.18635
delta = 0.1167477
theta = 0.7307160


phi_1  = torch.tensor(phi_1,dtype = torch.float64)
phi_2 = torch.tensor(phi_2,dtype = torch.float64)
eta = torch.tensor(eta,dtype = torch.float64)
gamma = torch.tensor(gamma,dtype = torch.float64)
tau = torch.tensor(tau,dtype = torch.float64)
delta = torch.tensor(delta,dtype = torch.float64)
theta = torch.tensor(theta,dtype = torch.float64)


# load initialization
###----------------------------------------------------------------------###
phi_atac = np.array([1/K]*K)
phi_rna = np.array([1/K]*K)
phi_met =np.array([1/K]*K)


w_exp = pd.read_csvpd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_exp.csv',index_col=0).values 
w_acc = w_exp.copy()
w_met = w_exp.copy()


pi_rna = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_rna.csv',index_col=0).values
pi_met = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_met.csv',index_col=0).values
qi = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_square_qi.csv',index_col=0).values[:,0]


phi_atac = torch.from_numpy(phi_atac)
phi_atac = phi_atac.type(torch.float64)

phi_rna = torch.from_numpy(phi_rna)
phi_rna = phi_rna.type(torch.float64)

phi_met = torch.from_numpy(phi_met)
phi_met = phi_met.type(torch.float64)

w_exp = torch.from_numpy(w_exp)
w_exp = w_exp.type(torch.float64)

w_met = torch.from_numpy(w_met)
w_met = w_met.type(torch.float64)

w_acc = torch.from_numpy(w_acc)
w_acc = w_acc.type(torch.float64)

qi = torch.from_numpy(qi)
qi = qi.type(torch.float64)

pi_rna = torch.from_numpy(pi_rna)
pi_rna = pi_rna.type(torch.float64)

pi_met = torch.from_numpy(pi_met)
pi_met = pi_met.type(torch.float64)



# adjust for GPU
rij = rij.cuda()
replacef0 = replacef0.cuda()
rlg = rlg.cuda()
replaceg0 = replaceg0.cuda()
rdm = rdm.cuda()
replaceh0 = replaceh0.cuda()

phi_atac = phi_atac.cuda()
phi_rna = phi_rna.cuda()
phi_met = phi_met.cuda()
pi_rna = pi_rna.cuda()
pi_met = pi_met.cuda()
w_exp = w_exp.cuda()
w_met = w_met.cuda()
w_acc = w_acc.cuda()
qi = qi.cuda()

phi_1 = phi_1.cuda()
phi_2 = phi_2.cuda()
eta = eta.cuda()
gamma = gamma.cuda()
tau = tau.cuda()
delta = delta.cuda()
theta = theta.cuda()

alpha_1 = alpha_1.cuda()
beta_1 = beta_1.cuda()
alpha_qi = alpha_qi.cuda()
beta_qi = beta_qi.cuda()



```



### 3.2 run the EM algorithm
```{python}
init = EM_gpu(rij,replacef0,rlg,replaceg0,rdm,replaceh0,
          phi_atac,phi_rna,phi_met,pi_rna,pi_met,w_exp,w_met,w_acc,qi,
          phi_1 = phi_1, phi_2 = phi_2, eta = eta,gamma = gamma,tau = tau, delta = delta, theta = theta,
          alpha_qi=alpha_qi,beta_qi=beta_qi,alpha_1=alpha_1,beta_1=beta_1,
          max_iter = 1)



start = time.time()
res = EM_gpu(rij,replacef0,rlg,replaceg0,rdm,replaceh0,
         init['phi_atac'], init['phi_rna'], init['phi_met'], init['pi_rna'], init['pi_met'],
         init['w_rna'], init['w_met'], init['w_atac'], init['qi'], phi_1=phi_1, phi_2=phi_2,
         eta=eta, gamma=gamma, tau=tau, delta=delta, theta=theta,
         alpha_qi=alpha_qi,beta_qi=beta_qi,alpha_1=alpha_1,beta_1=beta_1,
         max_iter=200, flag=False)

end = time.time()

print('done')

print(end - start)

```



### 3.3 Summary clustering result (get the cluster assignments)
```{python}
(E_z_atac,E_z_u_atac,E_z_u_u_t_atac,
            E_z_rna,E_z_u_rna,E_z_1_u_rna,E_z_u_v_rna,E_z_1_u_v_rna,E_z_1_u_v_rna,
                   E_z_met,E_z_u_met,E_z_1_u_met,E_z_u_v_met,
     E_z_1_u_v_met,E_z_1_u_v_met) = E_step_gpu(rij,replacef0,res['phi_atac'], res['w_atac'],res['qi'],rlg,replaceg0,
                        res['phi_rna'],res['w_rna'],res['pi_rna'],rdm,replaceh0,res['phi_met']
                                       ,res['w_met'],res['pi_met'])

# amp, rmp and mmp are the cluster assignments offered by scAMACE
# scCAS data
amp = torch.argmax(E_z_atac, dim=0)

# scRNA-Seq data
rmp = torch.argmax(E_z_rna, dim=0)

# sc-methylation data
mmp = torch.argmax(E_z_met, dim=0)

pd.crosstab(atac_cell_lb,amp)
pd.crosstab(rna_cell_lb,rmp)
pd.crosstab(met_cell_lb,mmp)

```


## 4. Example of scAMACE (seperate) CPU version
#### Remarks: This section is the demostration of implementing scAMACE seperately on the three datasets using Application 1. 

#### We set K=1 to get $\omega^{acc}_{kg}$, $\omega^{rna}_{kg}$ and $\omega^{met}_{kg}$, then we can further obtain \{$\hat{\eta}, \hat{\gamma}, \hat{\tau}, \hat{\delta}, \hat{\theta}, \hat{\phi}^{acc}, \hat{\phi}^{met}$\} by beta regression.  (Beta regression is implemented by package 'betareg' in R, please refer to https://github.com/cuhklinlab/scAMACE/blob/main/vignette/vignette.md for more details.)

### 4.1 Load data and prepare for EM algorithm
```{python}
import scAMACE_py

folder = '/lustre/project/Stat/s1155116622/Real_data_final/Application1_1000features_rna_mean/'
f1 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_f1.csv',index_col=0).values
f0 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_f0.csv',index_col=0).values
g1 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_g1.csv',index_col=0).values
g0 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_g0.csv',index_col=0).values
met_data = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_data.csv',index_col=0).values


atac_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_atac_cell_lb.csv',index_col=0).values[:,0]
rna_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_rna_cell_lb.csv',index_col=0).values[:,0]
met_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_cell_lb.csv',index_col=0).values[:,0]


rij = f1/(f1 + f0)
replacef0 = 1 - rij

rij = torch.from_numpy(rij)
rij = rij.type(torch.float64)
replacef0 = torch.from_numpy(replacef0)
replacef0 = replacef0.type(torch.float64)


rlg = g1/(g1 + g0)
replaceg0 = 1 - rlg

rlg = torch.from_numpy(rlg)
rlg = rlg.type(torch.float64)
replaceg0 = torch.from_numpy(replaceg0)
replaceg0 = replaceg0.type(torch.float64)


m = met_data
# m[m=="NA"] = np.nan
# m = np.array(m, dtype=float)
ratio = m/(1-m)
c = ratio[np.where(~np.isnan(ratio))]
cc = c[np.where(c!=0)]

sf = np.percentile(cc,50)
sf
rdm = ratio/sf
rdm = rdm**2 # power
rdm[np.isnan(rdm)] = np.mean(~np.isnan(rdm))
rdm[np.isinf(rdm)] = 3000
rdm[1:5,1:5]
replaceh0 = np.ones((rdm.shape[0],rdm.shape[1]))

rdm = torch.from_numpy(rdm)
rdm = rdm.type(torch.float64)
replaceh0 = torch.from_numpy(replaceh0)
replaceh0 = replaceh0.type(torch.float64)



alpha_1 = 2
beta_1 = 2

alpha_qi = 1
beta_qi = 1

alpha_1  = torch.tensor(alpha_1,dtype = torch.float64)
beta_1 = torch.tensor(beta_1,dtype = torch.float64)
alpha_qi = torch.tensor(alpha_qi,dtype = torch.float64)
beta_qi = torch.tensor(beta_qi,dtype = torch.float64)



# load initialization
###----------------------------------------------------------------------###
w_exp = pd.read_csvpd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_exp.csv',index_col=0).values 
w_acc = pd.read_csvpd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_acc.csv',index_col=0).values
w_met = pd.read_csvpd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_met.csv',index_col=0).values

pi_rna = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_rna.csv',index_col=0).values
pi_met = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_met.csv',index_col=0).values
qi = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_square_qi.csv',index_col=0).values[:,0]


phi_atac = torch.from_numpy(phi_atac)
phi_atac = phi_atac.type(torch.float64)

phi_rna = torch.from_numpy(phi_rna)
phi_rna = phi_rna.type(torch.float64)

phi_met = torch.from_numpy(phi_met)
phi_met = phi_met.type(torch.float64)

w_exp = torch.from_numpy(w_exp)
w_exp = w_exp.type(torch.float64)

w_met = torch.from_numpy(w_met)
w_met = w_met.type(torch.float64)

w_acc = torch.from_numpy(w_acc)
w_acc = w_acc.type(torch.float64)

qi = torch.from_numpy(qi)
qi = qi.type(torch.float64)

pi_rna = torch.from_numpy(pi_rna)
pi_rna = pi_rna.type(torch.float64)

pi_met = torch.from_numpy(pi_met)
pi_met = pi_met.type(torch.float64)


```


### 4.2 scCAS data
```{python}
### ------------------- scCAS ---------------------------###
K = 2 # change to K=1 if you want to get w_acc for beta regression
phi_atac = np.array([1/K]*K)
phi_atac = torch.from_numpy(phi_atac)
phi_atac = phi_atac.type(torch.float64)


init = EM_atac(rij, replacef0, phi_atac, w_acc, qi, alpha_1, beta_1, alpha_qi, beta_qi, max_iter=1)

start = time.time()
res = EM_atac(rij, replacef0, init['phi_atac'], init['w_atac'], init['qi'], alpha_1, beta_1, alpha_qi, beta_qi, max_iter=200, flag=False)

end = time.time()

print('atac done')
print(end-start)



(E_z_atac,E_z_u_atac,E_z_u_u_t_atac) = cal_E_atac(rij,replacef0,res['phi_atac'], res['w_atac'],res['qi'])

amp = torch.argmax(E_z_atac, dim=0)

pd.crosstab(atac_cell_lb,amp)

```


### 4.3 scRNA-Seq data
```{python}
### ------------------- scRNA-Seq ---------------------------###
K = 2 # change to K=1 if you want to get w_rna for beta regression
phi_rna = np.array([1/K]*K)

phi_rna = torch.from_numpy(phi_rna)
phi_rna = phi_rna.type(torch.float64)


init = EM_rna(rlg,replaceg0, phi_rna, w_exp, pi_rna, alpha_1, beta_1, max_iter=1)

start = time.time()
res = EM_rna(rlg,replaceg0, init['phi_rna'], init['w_rna'], init['pi_rna'],alpha_1, beta_1,  max_iter=200, flag=False)

end = time.time()

print('rna done')

print(end - start)

(E_z_rna, E_z_u_rna, E_z_1_u_rna, E_z_u_v_rna, E_z_1_u_v_rna, E_z_1_u_v_rna) = cal_E(rlg,replaceg0,
                        res['phi_rna'],res['w_rna'],res['pi_rna'])

rmp = torch.argmax(E_z_rna, dim=0)

pd.crosstab(rna_cell_lb,rmp)

```



### 4.4 sc-methylation data
```{python}
### ------------------- sc-methylation ---------------------------###
K = 2 # change to K=1 if you want to get w_met for beta regression

phi_met =np.array([1/K]*K)

phi_met = torch.from_numpy(phi_met)
phi_met = phi_met.type(torch.float64)

init = EM_met(rdm,replaceh0, phi_met, w_met, pi_met,  alpha_1, beta_1, max_iter=1)

start = time.time()
res = EM_met(rdm,replaceh0, init['phi_met'], init['w_met'], init['pi_met'],  alpha_1, beta_1, max_iter=200, flag=True)

end = time.time()

print('met done')

print(end - start)


(E_z_met, E_z_u_met, E_z_1_u_met, E_z_u_v_met, E_z_1_u_v_met, E_z_1_u_v_met) = cal_E(rdm,replaceh0,res['phi_met']
                                       ,res['w_met'],res['pi_met'])

mmp = torch.argmax(E_z_met, dim=0)

pd.crosstab(met_cell_lb,mmp)

```




## 5. Example of scAMACE (seperate) GPU version
#### Remarks: This section is the demostration of implementing scAMACE seperately on the three datasets using Application 1. 

#### We set K=1 to get $\omega^{acc}_{kg}$, $\omega^{rna}_{kg}$ and $\omega^{met}_{kg}$, then we can further obtain \{$\hat{\eta}, \hat{\gamma}, \hat{\tau}, \hat{\delta}, \hat{\theta}, \hat{\phi}^{acc}, \hat{\phi}^{met}$\} by beta regression.  (Beta regression is implemented by package 'betareg' in R, please refer to https://github.com/cuhklinlab/scAMACE/blob/main/vignette/vignette.md for more details.)

### 5.1 Load data and prepare for EM algorithm
```{python}
import scAMACE_py

folder = '/lustre/project/Stat/s1155116622/Real_data_final/Application1_1000features_rna_mean/'
f1 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_f1.csv',index_col=0).values
f0 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_f0.csv',index_col=0).values
g1 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_g1.csv',index_col=0).values
g0 = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_g0.csv',index_col=0).values
met_data = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_data.csv',index_col=0).values


atac_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_atac_cell_lb.csv',index_col=0).values[:,0]
rna_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_rna_cell_lb.csv',index_col=0).values[:,0]
met_cell_lb = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_met_cell_lb.csv',index_col=0).values[:,0]


rij = f1/(f1 + f0)
replacef0 = 1 - rij

rij = torch.from_numpy(rij)
rij = rij.type(torch.float64)
replacef0 = torch.from_numpy(replacef0)
replacef0 = replacef0.type(torch.float64)


rlg = g1/(g1 + g0)
replaceg0 = 1 - rlg

rlg = torch.from_numpy(rlg)
rlg = rlg.type(torch.float64)
replaceg0 = torch.from_numpy(replaceg0)
replaceg0 = replaceg0.type(torch.float64)


m = met_data
# m[m=="NA"] = np.nan
# m = np.array(m, dtype=float)
ratio = m/(1-m)
c = ratio[np.where(~np.isnan(ratio))]
cc = c[np.where(c!=0)]

sf = np.percentile(cc,50)
sf
rdm = ratio/sf
rdm = rdm**2 # power
rdm[np.isnan(rdm)] = np.mean(~np.isnan(rdm))
rdm[np.isinf(rdm)] = 3000
rdm[1:5,1:5]
replaceh0 = np.ones((rdm.shape[0],rdm.shape[1]))

rdm = torch.from_numpy(rdm)
rdm = rdm.type(torch.float64)
replaceh0 = torch.from_numpy(replaceh0)
replaceh0 = replaceh0.type(torch.float64)



alpha_1 = 2
beta_1 = 2

alpha_qi = 1
beta_qi = 1

alpha_1  = torch.tensor(alpha_1,dtype = torch.float64)
beta_1 = torch.tensor(beta_1,dtype = torch.float64)
alpha_qi = torch.tensor(alpha_qi,dtype = torch.float64)
beta_qi = torch.tensor(beta_qi,dtype = torch.float64)



# load initialization
###----------------------------------------------------------------------###
w_exp = pd.read_csvpd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_exp.csv',index_col=0).values 
w_acc = pd.read_csvpd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_acc.csv',index_col=0).values
w_met = pd.read_csvpd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_w_met.csv',index_col=0).values


pi_rna = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_rna.csv',index_col=0).values
pi_met = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_pi_met.csv',index_col=0).values
qi = pd.read_csv(folder + 'Feb7_2021_3Types_Data_rna_mean_1000_ratio_mcmc_ini_square_qi.csv',index_col=0).values[:,0]


w_exp = torch.from_numpy(w_exp)
w_exp = w_exp.type(torch.float64)

w_met = torch.from_numpy(w_met)
w_met = w_met.type(torch.float64)

w_acc = torch.from_numpy(w_acc)
w_acc = w_acc.type(torch.float64)

qi = torch.from_numpy(qi)
qi = qi.type(torch.float64)

pi_rna = torch.from_numpy(pi_rna)
pi_rna = pi_rna.type(torch.float64)

pi_met = torch.from_numpy(pi_met)
pi_met = pi_met.type(torch.float64)



# adjust for GPU
rij = rij.cuda()
replacef0 = replacef0.cuda()
rlg = rlg.cuda()
replaceg0 = replaceg0.cuda()
rdm = rdm.cuda()
replaceh0 = replaceh0.cuda()

pi_rna = pi_rna.cuda()
pi_met = pi_met.cuda()
w_exp = w_exp.cuda()
w_met = w_met.cuda()
w_acc = w_acc.cuda()
qi = qi.cuda()


alpha_1 = alpha_1.cuda()
beta_1 = beta_1.cuda()
alpha_qi = alpha_qi.cuda()
beta_qi = beta_qi.cuda()



```


### 5.2 scCAS data
```{python}
### ------------------- scCAS ---------------------------###
K = 2 # change to K=1 if you want to get w_acc for beta regression

phi_atac = np.array([1/K]*K)
phi_atac = torch.from_numpy(phi_atac)
phi_atac = phi_atac.type(torch.float64)
phi_atac = phi_atac.cuda()


init = EM_atac_gpu(rij, replacef0, phi_atac, w_acc, qi, alpha_1, beta_1, alpha_qi, beta_qi, max_iter=1)

start = time.time()
res = EM_atac_gpu(rij, replacef0, init['phi_atac'], init['w_atac'], init['qi'], alpha_1, beta_1, alpha_qi, beta_qi, max_iter=200, flag=False)

end = time.time()

print('atac done')
print(end-start)


(E_z_atac,E_z_u_atac,E_z_u_u_t_atac) = cal_E_atac_gpu(rij,replacef0,res['phi_atac'], res['w_atac'],res['qi'])

amp = torch.argmax(E_z_atac, dim=0)

pd.crosstab(atac_cell_lb,amp)


```


### 5.3 scRNA-Seq data
```{python}
### ------------------- scRNA-Seq ---------------------------###
K = 2 # change to K=1 if you want to get w_rna for beta regression
phi_rna = np.array([1/K]*K)

phi_rna = torch.from_numpy(phi_rna)
phi_rna = phi_rna.type(torch.float64)
phi_rna = phi_rna.cuda()


init = EM_rna_gpu(rlg,replaceg0, phi_rna, w_exp, pi_rna, alpha_1, beta_1, max_iter=1)

start = time.time()
res = EM_rna_gpu(rlg,replaceg0, init['phi_rna'], init['w_rna'], init['pi_rna'],alpha_1, beta_1,  max_iter=200, flag=False)

end = time.time()

print('rna done')

print(end - start)

(E_z_rna, E_z_u_rna, E_z_1_u_rna, E_z_u_v_rna, E_z_1_u_v_rna, E_z_1_u_v_rna) = cal_E_gpu(rlg,replaceg0,
                        res['phi_rna'],res['w_rna'],res['pi_rna'])

rmp = torch.argmax(E_z_rna, dim=0)

pd.crosstab(rna_cell_lb,rmp)

```



### 5.4 sc-methylation data
```{python}
### ------------------- sc-methylation ---------------------------###
K = 2 # change to K=1 if you want to get w_met for beta regression

phi_met =np.array([1/K]*K)

phi_met = torch.from_numpy(phi_met)
phi_met = phi_met.type(torch.float64)
phi_met = phi_met.cuda()


init = EM_met_gpu(rdm,replaceh0, phi_met, w_met, pi_met,  alpha_1, beta_1, max_iter=1)


start = time.time()
res = EM_met_gpu(rdm,replaceh0, init['phi_met'], init['w_met'], init['pi_met'],  alpha_1, beta_1, max_iter=200, flag=False)

end = time.time()

print('met done')

print(end - start)


(E_z_met, E_z_u_met, E_z_1_u_met, E_z_u_v_met, E_z_1_u_v_met, E_z_1_u_v_met) = cal_E_gpu(rdm,replaceh0,res['phi_met']
                                       ,res['w_met'],res['pi_met'])

mmp = torch.argmax(E_z_met, dim=0)

pd.crosstab(met_cell_lb,mmp)

```





