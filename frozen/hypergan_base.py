from abc import ABC, abstractmethod
import torch.nn as nn

class HyperGAN_Base(ABC):

    def __init__(self, device, d_action, d_state):
        self.sample_size = 512
        self.latent_width = 256
        self.ngen = 4
        self.ensemble_size = 100

    @abstractmethod
    class Generator(object):
        def __init__(self, device, d_action, d_state):
            raise NotImplementedError

    @abstractmethod
    def eval_f(self):
        raise NotImplementedError

