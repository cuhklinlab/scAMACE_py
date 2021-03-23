# scAMACE_py
scAMACE (integrative Analysis of single-cell Methylation, chromatin ACcessibility, and gene Expression)

Python implementation (both CPU and GPU version) to a model-based approach to the joint analysis of single-cell data on chromatin accessibility, gene expression and methylation.



## Installation

You can install the released version of scAMACE from Github:

```{python}
pip install git+https://github.com/WWJiaxuan/scAMACE_py.git#egg=scAMACE_py

```
## Main Functions

`EM`: Expectation-maximization (EM) implementation on CPU of scAMACE.

`E_step`: Perform E-step (i.e. calculate the expectations of missing data) for one iteration in the EM algorithm on CPU.


`EM_gpu`: Expectation-maximization (EM) implementation on GPU of scAMACE.

`E_step_gpu`: Perform E-step (i.e. calculate the expectations of missing data) for one iteration in the EM algorithm on GPU.


## Example
Please refer to the [vigenette](https://https://github.com/cuhklinlab/scACE/tree/master/vignette) with several examples for a quick guide to scAMACE package.
