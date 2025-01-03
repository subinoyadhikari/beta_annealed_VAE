#OWNER=SUBINOY ADHIKARI
#EMAIL=subinoy.adhk@gmail.com

import sys
import os
import math
import torch 
import numpy as np 
from torch import nn
from torch.nn import functional as F

class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """
    def __init__(self, total_steps, shape, baseline=0.0, final_value=1.0, cyclical=False, fraction=1.0):
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        self.current_cycle = 0
        self.final_value = final_value
        self.fraction = fraction  # Fraction of steps to reach the final value
        self.anneal_steps = int(self.fraction * self.total_steps)  # Steps for reaching final value

    def __call__(self):
        return self.slope()

    def slope(self):
        if self.current_step >= self.anneal_steps:
            y = self.final_value  # After reaching the fraction, keep at final value
        else:
            # Compute annealing based on current step
            normalized_step = self.current_step / (self.anneal_steps - 1)  # Scale to fraction steps
            if self.shape == 'linear':
                y = normalized_step * self.final_value
            elif self.shape == 'cosine':
                y = ((math.cos(math.pi * (normalized_step - 1)) + 1) / 2) * self.final_value
            elif self.shape == 'logistic':
                midpoint = (self.anneal_steps - 1) / 2
                k = 10 / (self.anneal_steps - 1)
                exponent = -k * (self.current_step - midpoint)
                y = (1 / (1 + math.exp(exponent))) * self.final_value
            elif self.shape == 'none':
                y = 1.0
            else:
                raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if not isinstance(value, bool):
            raise ValueError('Cyclical_setter method requires boolean argument (True/False)')
        else:
            self.cyclical = value
        return 

    def reset(self):
        self.current_step = 0
        self.current_cycle = 0		
		
		
		
		
		
		
