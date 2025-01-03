#OWNER=SUBINOY ADHIKARI
#EMAIL=subinoy.adhk@gmail.com

"""
DENSE VAE model
"""


import sys
import os
import math
import torch
import numpy as np
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F

class encoder(nn.Module):
	def __init__(self, encoder_neurons, encoder_activation, latent_dim):
		super(encoder, self).__init__()
		
		self.enc = nn.ModuleList()			
		for i in range(len(encoder_neurons)-1):
			linear_layer=nn.Linear(encoder_neurons[i], encoder_neurons[i+1])
			batch_norm_layer=nn.BatchNorm1d(encoder_neurons[i+1])
			if encoder_activation[i] != 'Linear':		
				activation=getattr(nn, encoder_activation[i])()
				self.enc.extend([linear_layer, batch_norm_layer, activation])
			else:
				self.enc.extend([linear_layer, batch_norm_layer])

		# Output layers for mean and var
		self.mean_layer = nn.Linear(encoder_neurons[-1], latent_dim)
		self.var_layer = nn.Linear(encoder_neurons[-1], latent_dim)


	def forward(self, x):
		for layer in self.enc:
			x = layer(x)

		# Output mean and var
		mean = self.mean_layer(x)
		var = F.softplus(self.var_layer(x))

		encoder_output = {'z_mu':mean, 'z_var':var}
		return encoder_output
        

class decoder(nn.Module):
	def __init__(self, decoder_neurons, decoder_activation, latent_dim):
		super(decoder, self).__init__()
		
		self.dec = nn.ModuleList()
		linear_layer=nn.Linear(latent_dim, decoder_neurons[0])
		batch_norm_layer=nn.BatchNorm1d(decoder_neurons[0])
		if decoder_activation[0] != 'Linear':
			activation=getattr(nn, decoder_activation[0])()
			self.dec.extend([linear_layer, batch_norm_layer, activation])			
		else:
			self.dec.extend([linear_layer, batch_norm_layer])
			
		for i in range(len(decoder_neurons)-1):
			linear_layer=nn.Linear(decoder_neurons[i], decoder_neurons[i+1])
			batch_norm_layer=nn.BatchNorm1d(decoder_neurons[i+1])
			if decoder_activation[i+1] != 'Linear':
				activation=getattr(nn, decoder_activation[i+1])()
				self.dec.extend([linear_layer, batch_norm_layer, activation])
			else:
				self.dec.extend([linear_layer, batch_norm_layer])
			
	def forward(self, x):
		for layer in self.dec:
			x = layer(x)
		
		decoder_output = {'x_rec':x}	
		return decoder_output

class vae_network(nn.Module):
	def __init__(self, encoder_neurons, encoder_activation, decoder_neurons, decoder_activation, latent_dim):
		super(vae_network, self).__init__()
		
		self.encoder=encoder(encoder_neurons, encoder_activation, latent_dim)
		self.decoder=decoder(decoder_neurons, decoder_activation, latent_dim)	

		# weight initialization
		for m in self.modules():
			if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
				torch.nn.init.xavier_uniform_(m.weight)
				if m.bias.data is not None:
					init.constant_(m.bias, 0)
										
	def print_model_architecture(self):
		print("VAE Network Architecture:")
		print("Encoder:")
		print(self.encoder)

		print("Decoder:")
		print(self.decoder)	
		
	def reparametrize(self, mean, var):
		std = torch.sqrt(var+1e-10)
		eps = torch.randn_like(std)
		z = mean + (std*eps) 
		return z
	
		
	def forward(self, x):
		x=x.view(x.size(0), -1) # Dimension = batch_size, number_of_features
		encoder_output = self.encoder(x)
		mean = encoder_output['z_mu']
		var = encoder_output['z_var']		
		z = self.reparametrize(mean, var)
		decoder_output = self.decoder(z)
		
		output = encoder_output
		for key, value in decoder_output.items():			
			output[key]=value
		return output

# END OF FILE #

