# β Annealed VAE
To train the model with your desired parameters, you need to modify changes in the "main.py" file for both VAE and annealed VAE directories


  # Protein
	protein="trpcage" # this is the name of the system and will be used for naming the filenames later
	PROTEIN="TRPCAGE" # this is the name of the system (in uppercase) and will be used for naming the filenames later 
	
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
	
  # Annealing parameters (only for model where you want to anneal the β parameter)
	USE_ANNEALING=True
	NUM_CYCLES=1
	SHAPE='cosine' # change the shape to 'linear' or 'logistic' as per your choice
	CYCLICAL=True 
	BASELINE=0 # Initial value of beta
	FINAL_VALUE=1.0 # Final value of beta
	FRACTION=1.0 # This value controls the epoch at which the beta reaches to the final value and remains there till the cycle ends.

 # Annealing parameters (only for general VAE model)
	USE_ANNEALING=False
	NUM_CYCLES=-1
	SHAPE=None
	CYCLICAL=True
	BASELINE=0
	FINAL_VALUE=1.0
	FRACTION=1.0 

# Model directory name and model filename  (only for model where you want to anneal the β parameter)
	FILENAME_INIT=f"{protein}_a_{ALPHA}_b_{BETA}"	
	MODEL_DIRECTORY=f"./model_params/"
	MODEL_FILENAME=f"{FILENAME_INIT}_vaeban_shape_{SHAPE}_cycles_{NUM_CYCLES}_baseline_{BASELINE}_fval_{FINAL_VALUE}_batch_{TRAIN_BATCH_SIZE}_epochs_{NUM_EPOCHS}"
	PTH=".pth"
 
  # Model directory name and model filename (only for general VAE model)
	FILENAME_INIT=f"{protein}_a_{ALPHA}_b_{BETA}"	
	MODEL_DIRECTORY=f"./model_params/"
 	MODEL_FILENAME=f"{FILENAME_INIT}_vae_batch_{TRAIN_BATCH_SIZE}_epochs_{NUM_EPOCHS}"
	PTH=".pth"

# Path to training and testing data
	path_to_file = f"/path/to_training/and/validation/data/DISTANCE_{PROTEIN}/"	
	
	#-----Load the training and testing data-----#
	train_data=np.load(path_to_file+f"X_train_{protein}.npy")
	test_data=np.load(path_to_file+f"X_test_{protein}.npy")
# Train the model
1. Open a terminal and activate the conda environment that has pytorch
2. Type: ./main.py and press Enter

# Create the latent space
1. Open a jupyter notebook

Add these lines to a cell 


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
	import matplotlib.pyplot as plt
	from sklearn.manifold import TSNE
	from sklearn.cluster import KMeans
	from torch.nn import functional as F
	from sklearn.decomposition import PCA
	from humanfriendly import format_timespan
	from torch.utils.data import DataLoader, TensorDataset
	from torch.utils.data.sampler import SubsetRandomSampler
	from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
	from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score,\
	adjusted_rand_score, normalized_mutual_info_score
	
	%matplotlib inline


 	protein="trpcage"
	PROTEIN="TRPCAGE"
	
	#-----Path to training and testing data-----#
	path_to_file = f"/home/data/STANDARD_TRAJECTORIES/DISTANCE_{PROTEIN}/"	
	
	#-----Load the training and testing data-----#
	file_name = f"{protein}_distance.npy" # This contains the full dataset


	#-----Scale the data-----#
	scaler=Normalizer()
	#-----Load the distances-----#
	data=np.load(f"{path_to_file}{file_name}")
	#-----Scale the distances-----#
	data_scaled=scaler.fit_transform(data)


	full_data_tensor = torch.from_numpy(data_scaled.astype(np.float32))
	full_loader = DataLoader(full_data_tensor, batch_size=len(data_scaled), shuffle=False)	
	
	
	args=pickle.load(open("args.pkl", "rb"))
	
	#-----Create an instance of the dense_gmvae class-----#
	model = dense_vae(args)
	
	#-----Load the saved model state dict-----#
	model.network.load_state_dict(torch.load(args['model_name']))
	
	#-----Saving filename-----#
	saving_filename=f"latent_"+str(args['latent_dim'])+"_"+args['model_name'][15:-4]
	
	 
	#-----Create the latent data (mean vector)-----#
	full_features = model.compute_latent(full_loader)[0]
	np.save(f"latent_data_{saving_filename}.npy", full_features)

