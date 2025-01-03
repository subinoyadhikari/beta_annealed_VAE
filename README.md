# β Annealed VAE


#----------PROGRAM INPUTS----------#
To train the model, you need to modify changes in the main.py file


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
	FINAL_VALUE=1.0
	FRACTION=1.0 

  # Model directory name and model filename
	FILENAME_INIT=f"{protein}_a_{ALPHA}_b_{BETA}"	
	MODEL_DIRECTORY=f"./model_params/"
