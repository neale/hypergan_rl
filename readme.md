# Intro

The code here is for reproducing the numbers from the ICLR submission. The changes that matter from the ICLR submission are the following

* Changed intrinsic reward in `SimpleVarianceUtility()` in `utilities.py`. The calculation was changed from `utility = means.var(-1).sum(-1)` to `utility = means.var(1).sum(-1)` which correctly computes the variance over the dynmaic model predictions instead of the state space. 

* Instead of sampling new states according to a Gaussian sample in `imagination.py`, we directly use the means as the new states. 

* Forward calculation of data through sampled weights is faster, and done in a batch mode


#### Ant

* HyperGAN:
```
python main.py with max_explore env_name=MagellanAnt-v2 env_noise_stdev=0.02 eval_freq=100 checkpoint_frequency=100 ant_coverage=True
```

For the purposes of reproduction, the utility is calculated incorrectly in the code. To use the correct version, comment out line 89 in `utilities.py` and uncomment line 90

