#OWNER=SUBINOY ADHIKARI
#EMAIL=subinoy.adhk@gmail.com

"""
TRAIN DENSE VAE model
"""

import sys
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from dense_vae import *
from loss_vae import *
from annealer import Annealer
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler

class dense_vae:
	def __init__(self, args):

		# Model parameters #
		self.encoder_neurons=args['encoder_neurons']
		self.encoder_activation=args['encoder_activation']
		self.decoder_neurons=args['decoder_neurons']
		self.decoder_activation=args['decoder_activation']
		self.latent_dim=args['latent_dim']
		
		# Training parameters #
		self.cuda=args['cuda']
		self.gpu_id=args['gpu_id']
		self.train_batch_size=args['train_batch_size']
		self.val_batch_size=args['val_batch_size']
		self.num_epochs=args['num_epochs']
		self.learning_rate=args['learning_rate']
		self.alpha=args['alpha']			
		self.beta=args['beta']
		self.rec_loss_type=args['rec_loss_type']
		self.optimizer=args['optimizer']
		self.model_name=args['model_name']
		
		# Annealing parameters #
		self.use_annealing=args['use_annealing']
		self.num_cycles=args['num_cycles']
		self.shape=args['shape']
		self.baseline=args['baseline']
		self.cyclical=args['cyclical']	
		self.final_value=args['final_value']
		self.fraction=args['fraction']	
		
		if self.use_annealing:
			self.annealed_beta = Annealer(total_steps=self.num_epochs // self.num_cycles,
							shape=self.shape,
							baseline=self.baseline,
							final_value=self.final_value,
							cyclical=self.cyclical,
							fraction=self.fraction)
		
		# Build the Model #
		self.network=vae_network(self.encoder_neurons,
					  self.encoder_activation,
					  self.decoder_neurons, 
					  self.decoder_activation, 
					  self.latent_dim)
		
		# Transfer the model to GPU if required
		if self.cuda:
			self.network = self.network.to(self.gpu_id)
		
		# Print the model
		self.print_model()
						
		# Compute the Loss #
		self.loss=loss_functions()
		
		
	def compute_loss(self, data, output):
		
		z_mu=output['z_mu']
		z_var=output['z_var']
		x_rec=output['x_rec']

		# Reconstruction loss #
		if self.rec_loss_type=="mse":
			recon_loss=self.loss.mse_loss(data, x_rec)
		
		elif self.rec_loss_type=="bce":
			recon_loss=self.loss.bce_loss(data, x_rec)
		
		else:
			print("Wrong reconstruction loss type")

		kl_loss=self.loss.kl_loss(z_mu, z_var)
		if self.use_annealing:
			beta_value = self.annealed_beta()
		else:
			beta_value = self.beta
			
		total_loss = self.alpha*recon_loss + beta_value*kl_loss	
		loss_dict={'loss_recon':recon_loss, 'loss_gaussian':kl_loss,  'loss_total':total_loss}

		return loss_dict


	def train_one_epoch(self, optimizer, data_loader):

		self.network.train()
		
		num_batches=0
		recon_loss=0
		kl_loss=0
		total_loss=0

		for data in tqdm(data_loader, ascii=' >='):
			if self.cuda:	
				data=data.to(self.gpu_id)
				
		
			optimizer.zero_grad()

			output=self.network(data)
			loss_dict=self.compute_loss(data, output)
			total=loss_dict['loss_total']

			total_loss += total.item()
			recon_loss += loss_dict['loss_recon'].item()
			kl_loss += loss_dict['loss_gaussian'].item()

			# BACKPROPAGATION #
			total.backward()
			optimizer.step()

			num_batches +=1		
		
		recon_loss = recon_loss/num_batches
		kl_loss = kl_loss/num_batches
		total_loss = total_loss/num_batches	
		
		if self.use_annealing:
			self.annealed_beta.step()	
			
		return recon_loss, kl_loss, total_loss

	def test(self, data_loader):
		
		self.network.eval()
		
		num_batches=0
		recon_loss=0
		kl_loss=0
		total_loss=0
		fve_scores = []
		
		with torch.no_grad():

			for data in data_loader:
				if self.cuda:
					data=data.to(self.gpu_id)
					

				output=self.network(data)
				loss_dict=self.compute_loss(data, output)
				total=loss_dict['loss_total']

				recon_loss += loss_dict['loss_recon'].item()
				kl_loss += loss_dict['loss_gaussian'].item()
				total_loss += total.item()
				
				# FVE calculation
				x_rec = output['x_rec'].cpu().numpy()  # Get reconstructed data
				data_np = data.cpu().numpy()  # Convert the original data to numpy
				
				data_scaled = (data_np - np.mean(data_np, axis=0))  # Mean center the data
				recon_data_scaled = (x_rec - np.mean(x_rec, axis=0))
				
				total_variance = np.sum(data_scaled ** 2)
				reconstruction_error = np.sum((data_scaled - recon_data_scaled) ** 2)
				
				fve_score = 1 - (reconstruction_error / total_variance)
				fve_scores.append(fve_score)				 
				
				num_batches +=1
				
			recon_loss = recon_loss/num_batches
			kl_loss = kl_loss/num_batches
			total_loss = total_loss/num_batches
			fve_mean = np.mean(fve_scores)			                

			return recon_loss, kl_loss, total_loss, fve_mean

	def train(self, train_data, validation_data):

		optimizer=self.optimizer(self.network.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-5)

		train_recon_loss=[]
		val_recon_loss=[]

		train_kl_loss=[]
		val_kl_loss=[]

		train_total_loss=[]
		val_total_loss=[]

		val_fve_scores = []
		
		annealed_beta_values = []
		
		for epoch in range(1, self.num_epochs+1):
			if self.use_annealing:
				annealed_beta_values.append(self.annealed_beta())
				print(f"beta={self.annealed_beta()}")

			trn_recon_loss, trn_kl_loss, trn_total_loss = self.train_one_epoch(optimizer, train_data)
			vl_recon_loss, vl_kl_loss, vl_total_loss, fve_score = self.test(validation_data)

			train_recon_loss.append(trn_recon_loss)
			val_recon_loss.append(vl_recon_loss)

			train_kl_loss.append(trn_kl_loss)
			val_kl_loss.append(vl_kl_loss)

			train_total_loss.append(trn_total_loss)
			val_total_loss.append(vl_total_loss)
			
			val_fve_scores.append(fve_score)

			print("(Epoch %d / %d)" % (epoch, self.num_epochs))
			print("Training   :: recon_loss=%f; kl_loss=%f; total_loss=%f" \
				% (trn_recon_loss, trn_kl_loss, trn_total_loss))
			print("Validation :: recon_loss=%f; kl_loss=%f; total_loss=%f; FVE=%f"\
				% (vl_recon_loss, vl_kl_loss, vl_total_loss, fve_score))

				
		train_recon_loss=np.array(train_recon_loss)
		val_recon_loss=np.array(val_recon_loss)

		train_kl_loss=np.array(train_kl_loss)
		val_kl_loss=np.array(val_kl_loss)

		train_total_loss=np.array(train_total_loss)
		val_total_loss=np.array(val_total_loss)
		
		if self.use_annealing:
			np.save('annealed_beta_values.npy', annealed_beta_values)
		np.save('fve_scores.npy', val_fve_scores)
		
		output = {'train_recon_loss':train_recon_loss, 'train_kl_loss':train_kl_loss, 'train_total_loss':train_total_loss,
			 'val_recon_loss':val_recon_loss, 	'val_kl_loss':val_kl_loss, 'val_total_loss':val_total_loss, 'val_fve_scores': val_fve_scores}
		return output

	def print_model(self):
		self.network.print_model_architecture()		
		
		
	def compute_latent(self, data_loader):
		
		self.network.eval()
		
		latent_data=[]

		with torch.no_grad():
			for data in data_loader:
				if self.cuda:
					data=data.to(self.gpu_id)
				encoder_output=self.network.encoder(data)
				latent_data.append(encoder_output['z_mu'].cpu().detach().numpy())
			
			return latent_data
			
			
	def reconstruct_data(self, data_loader):
	
		self.network.eval()		

		recon_data=[]
		
		with torch.no_grad():
			for data in data_loader:
				if self.cuda:
					data=data.to(self.gpu_id)
				decoder_output=self.network(data)
				recon_data.append(decoder_output['x_rec'].cpu().detach().numpy())
				
		return recon_data

		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
