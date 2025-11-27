# PPCA
Probabilistic PCA (PPCA) written in torch.

## Overview

`ppca-torch` implements in PyTorch the method Probabilistic Principal Component Analysis (PPCA) as described by Tipping & Bishop (1999).



## Quickstart

### Dev mode

First, create (`python3 -m venv .venv`) or activate a virtualenv `source .venv/bin/activate`.

Then clone the repository and install:
```
git clone https://github.com/siemovit/ppca.git
cd ppca
pip install -e .
```

## Examples

Go in the examples directory, and find scripts and notebooks. For instance:

```
python examples/quickstart.py
```

## References
[Tipping, Michael E., and Christopher M. Bishop. "Probabilistic principal component analysis." Journal of the Royal Statistical Society Series B: Statistical Methodology 61.3 (1999): 611-622.](https://www.di.ens.fr/~fbach/courses/fall2005/Bishop_Tipping_1999_Probabilistic_PCA.pdf)


## To do
### Implementation
P1
- [x] implement PPCA in torch with eig decomposition method
- [x] implement PPCA EM

P2
- [ ] big datasets
  - [x] estimate parameters via gradient descent
  - [x] implement online EM
  - [ ] Use Mnist to show examples
- [ ] improve the model
  - [ ] Sparse ppca
  - [ ] MPPCA
- [ ] study specific property of the model
  - [ ] Missing values
  - [ ] Convergence of algorithms
- [ ] refining
  - [ ] calculate explained_variance ratio (similar to sklearn PCA)
  - [ ] make a function to display pca axes

P3
- [x] implement basic pca in torch 

### Experiments
P1
- [ ] generate sample from model and comment how it follows the distribution
- [ ] corrupt dataset (missing samples) compare PCA vs PPCA

P2
- [ ] use something else than sklearn datasets
- [ ] plot at incrementing iterations
- [ ] plot digits (incrementing iterations)
- [ ] plot axes
- [ ] plot 
