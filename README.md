# scAMACE_py (python implementation)

scAMACE (integrative Analysis of single-cell Methylation, chromatin ACcessibility, and gene Expression)

Python implementation (both CPU and GPU version) to a model-based approach to the joint analysis of single-cell data on chromatin accessibility, gene expression and methylation.

## 1. Installation

You can install the released version of scAMACE_py from Github:

```{python}
pip install git+https://github.com/cuhklinlab/scAMACE_py

```


## 2. Main Functions

`EM`: Expectation-maximization (EM) implementation on CPU of scAMACE.

`E_step`: Perform E-step (i.e. calculate the expectations of missing data) for one iteration in the EM algorithm on CPU.


`EM_gpu`: Expectation-maximization (EM) implementation on GPU of scAMACE.

`E_step_gpu`: Perform E-step (i.e. calculate the expectations of missing data) for one iteration in the EM algorithm on GPU.

`generate_sim_data`: Generate simulation data x, y and t.


## 3. Datasets and Examples
Please refer to the [vigenette](https://github.com/cuhklinlab/scAMACE_py/blob/main/vignette/vignette.md) with several examples for a quick guide to scAMACE_py package.

## 4. Reference
Jiaxuan Wangwu, Zexuan Sun, Zhixiang Lin: [scAMACE: Model-based approach to the joint analysis of single-cell data on chromatin accessibility, gene expression and methylation.](https://www.biorxiv.org/content/10.1101/2021.03.29.437485v2)
