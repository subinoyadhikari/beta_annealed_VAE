#!/home/subinoy/Softwares/anaconda3/envs/tch/bin/python

#OWNER=SUBINOY ADHIKARI
#EMAIL=subinoy.adhk@gmail.com

"""
TRAIN DENSE VAE model
"""

import os
import sys
import math
import time
import torch
import random
import pickle
import shutil
import argparse
import numpy as np
from train_vae import *
from torch import nn, optim
import torch.nn.init as init
from torch.nn import functional as F
from humanfriendly import format_timespan
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def time_format(seconds: int) -> str:
	if seconds is not None:
		seconds = int(seconds)
		d = seconds // (3600 * 24)
		h = seconds // 3600 % 24
		m = seconds % 3600 // 60
		s = seconds % 3600 % 60
		if d > 0:
			return '{:002d}D {:02d} hr {:02d}m {:02d}s'.format(d, h, m, s)
		elif h > 0:
			return '{:02d}H {:02d} min {:02d}s'.format(h, m, s)
		elif m > 0:
			return '{:02d}m {:02d} sec'.format(m, s)
		elif s > 0:
			return '{:02d} sec'.format(s)
		return '-'

  
def main():
	# Start time
	start_time=time.time()	

##########################################################################################################
	
	# Seed Everything
	RANDOM_SEED=42	
	random.seed(RANDOM_SEED)
	os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	torch.cuda.manual_seed(RANDOM_SEED)
	torch.backends.cudnn.benchmark=False
	torch.backends.cudnn.deterministic=True	

##########################################################################################################
	
	# Protein
	protein="trpcage" 
	PROTEIN="TRPCAGE"  
	
	# Build the VAE network with desired architecture #
	ENCODER_NEURONS = [190, 128, 64, 32, 16]
	ENCODER_ACTIVATION = ['Tanh', 'Tanh', 'Tanh', 'Tanh', 'Tanh']

	DECODER_NEURONS = [16, 32, 64, 128, 190]
	DECODER_ACTIVATION = ['Tanh', 'Tanh', 'Tanh', 'Tanh', 'Sigmoid']

	LATENT_DIM=2 # Latent dimension #

	# Training parameters #
	USE_GPU=True
	GPU_ID=5
	TRAIN_BATCH_SIZE=5000
	VAL_BATCH_SIZE=5000
	NUM_EPOCHS=1000
	OPTIMIZER=optim.SGD
	LEARNING_RATE=0.001
	ALPHA=10
	BETA=1.0
	REC_LOSS_TYPE="mse"
	
	# Annealing parameters
	USE_ANNEALING=True
	NUM_CYCLES=1
	SHAPE='cosine'
	CYCLICAL=True
	BASELINE=0
	FINAL_VALUE=0.025009
	FRACTION=1.0 
	
	# Model directory name and model filename
	# Model directory name and model filename
	FILENAME_INIT=f"{protein}_a_{ALPHA}_b_{BETA}"	
	MODEL_DIRECTORY=f"./model_params/"
	MODEL_FILENAME=f"{FILENAME_INIT}_vaeban_shape_{SHAPE}_cycles_{NUM_CYCLES}_baseline_{BASELINE}_fval_{FINAL_VALUE}_batch_{TRAIN_BATCH_SIZE}_epochs_{NUM_EPOCHS}"
	PTH=".pth"
	
	# Create the directory where the model is to be saved
	if os.path.exists(MODEL_DIRECTORY):
	    shutil.rmtree(MODEL_DIRECTORY)
	os.makedirs(MODEL_DIRECTORY)		
	
	# Pass all the arguments as a dictionary		
	args = {'encoder_neurons' : ENCODER_NEURONS,
		'encoder_activation' : ENCODER_ACTIVATION,
		'decoder_neurons' : DECODER_NEURONS,
		'decoder_activation' : DECODER_ACTIVATION,
		'latent_dim' : LATENT_DIM,			
		'cuda': USE_GPU,
		'gpu_id': GPU_ID,
		'train_batch_size': TRAIN_BATCH_SIZE,
		'val_batch_size': VAL_BATCH_SIZE,
		'num_epochs': NUM_EPOCHS,
		'optimizer' : OPTIMIZER,
		'learning_rate': LEARNING_RATE,
		'alpha': ALPHA,
		'beta': BETA,
		'rec_loss_type': REC_LOSS_TYPE,
		'model_name': MODEL_DIRECTORY+MODEL_FILENAME+PTH,
		'use_annealing': USE_ANNEALING,
		'num_cycles': NUM_CYCLES,
		'shape': SHAPE,
		'cyclical': CYCLICAL,
		'baseline': BASELINE,
		'final_value': FINAL_VALUE,
		'fraction': FRACTION}	    	
	
	
	# Save the model arguments    
	with open("args.pkl", "wb") as f:
		pickle.dump(args, f)	
		
##########################################################################################################

	#-----Path to training and testing data-----#
	path_to_file = f"/home/subinoy/STANDARD_TRAJECTORIES/DISTANCE_{PROTEIN}/"	
	
	#-----Load the training and testing data-----#
	train_data=np.load(path_to_file+f"X_train_{protein}_frac_0.9.npy")
	test_data=np.load(path_to_file+f"X_test_{protein}_frac_0.9.npy")
	
	#-----Scale the data-----#
	scale_data = True	
	scaler = Normalizer()

	#-----Scale the distances-----#
	if scale_data:	
		X_train_scaled=scaler.fit_transform(train_data)
		X_test_scaled=scaler.fit_transform(test_data)
	else:
	
		epsilon = 1.0
		#X_train_scaled=X_train.copy()
		#X_test_scaled=X_test.copy()		
		
		train_min_values = np.min(train_data, axis=1)		
		train_min_values = train_min_values.reshape(-1, 1)		
		train_max_values = np.max(train_data, axis=1)
		train_max_values = train_max_values.reshape(-1, 1)
		
		test_min_values = np.min(test_data, axis=1)		
		test_min_values = test_min_values.reshape(-1, 1)				
		test_max_values = np.max(test_data, axis=1)	
		test_max_values = test_max_values.reshape(-1, 1)	
		
		X_train_scaled=(train_data-train_min_values)/(train_max_values-train_min_values)
		X_test_scaled=(test_data-test_min_values)/(test_max_values-test_min_values)		
		
		X_train_scaled=2*epsilon*X_train_scaled - epsilon
		X_test_scaled=2*epsilon*X_test_scaled - epsilon
		
		X_train_scaled=Normalizer().fit_transform(X_train_scaled)
		X_test_scaled=Normalizer().fit_transform(X_test_scaled)	
    
	# Convert them to tensors
	x_train_tensor = torch.from_numpy(X_train_scaled)
	x_test_tensor = torch.from_numpy(X_test_scaled)
		
	#y_train_tensor = torch.from_numpy(y_train)
	#y_test_tensor = torch.from_numpy(y_test)

	#train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
	#val_dataset = TensorDataset(x_test_tensor, y_test_tensor)

	train_loader = DataLoader(x_train_tensor, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(x_test_tensor, batch_size=VAL_BATCH_SIZE, shuffle=False)		

##########################################################################################################
		

	# Build the model #			
	vae=dense_vae(args)
	
	# Train the model #
	loss_history = vae.train(train_loader, val_loader)
	
	# Save the loss #
	filename=f"loss_{MODEL_FILENAME}.pkl" 
	with open(filename, "wb") as f:
		pickle.dump(loss_history, f)
	print(f"\n Saved file :: {filename}")
	
	# Save model
	torch.save(vae.network.state_dict(), MODEL_DIRECTORY+MODEL_FILENAME+PTH)
	
	# Ending time
	end_time=time.time()	
	
	# Print Execution time
	print(f"\n Execution time = {time_format(end_time-start_time)}")	
	print("\n Done!\n")

if __name__=="__main__":
	main()


