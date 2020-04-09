# Intro

The code here is for reproducing the numbers from the ICML submission. 

* Changed intrinsic reward in `SimpleVarianceUtility()` in `utilities.py`. The calculation was changed from `utility = means.var(-1).sum(-1)` to `utility = means.var(1).sum(-1)` which correctly computes the variance over the dynmaic model predictions instead of the state space. 

* Instead of sampling new states according to a Gaussian sample in `imagination.py`, we directly use the means as the new states. 

* Forward calculation of data through sampled weights is faster, and done in a batch mode

# Directions

For the time being, checkout the branch titled `external_reward_update/sac_warmup_working` to see how we do the policy transfer experiments. 
There we can see the two main ways of performing exploration with implicit distributions. 

  * A 2 stage process where we first warm up by training the exploration policy for 10k steps. Then we clear the agent's buffer so it cannot benefit from that experience, finally, SAC is trained as normal on the environment with external reward. 

  * A 3 stage process as given in the supplementary material. We extract the exploration policy, clear the buffer, and start a warm-up period before training with standard external reward. The warm up period consists of sampling directly from the exploration policy to fill the buffer. 

#### Ant

* HyperGAN:
```
python main.py with max_explore env_name=MagellanAnt-v2 env_noise_stdev=0.02 eval_freq=100 checkpoint_frequency=100 ant_coverage=True
```
