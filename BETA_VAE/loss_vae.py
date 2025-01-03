#OWNER=SUBINOY ADHIKARI
#EMAIL=subinoy.adhk@gmail.com

"""
Loss function for training GMVAE model
"""

import sys
import os
import math
import torch 
import numpy as np 
from torch import nn
from torch.nn import functional as F

class loss_functions:
	
	def mse_loss(self, true, predicted):			
		"""
		Mean Squared Error between the true and predicted values
		"""
		loss_mse = torch.square((predicted - true)).sum(-1).mean()
		
		return loss_mse

	def bce_loss(self, true, predicted):
		"""
		Binary Cross Entropy loss between the true and predicted values
		"""
		loss_bce = F.binary_cross_entropy(predicted, true, reduction='none').sum(-1).mean()
		
		return loss_bce
	
	def kl_loss(self, z_mu, z_var):

		z_var = z_var+1e-10
		kld_loss=-0.5*torch.sum(1 + torch.log(z_var) - z_mu.pow(2) - z_var, dim=-1).mean() 		
		
		return kld_loss


     
        		
	

