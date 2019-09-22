from multi_label_CNN import train_model, predict_attributes

import os, sys

data_folder = sys.argv[ 1 ]

if os.path.isdir(data_folder):
	# train the model
	train_model(data_folder)

if os.path.isfile(data_folder):
	saved_model_path = sys.argv[ 2 ]
	if not os.path.isfile( "./models/cl_multi_label.h5" ) :
		print( "There is no model trained in models folder... run training module first." )
		sys.exit()
	print( "Model is saved in : %s", saved_model_path )
	# predict the image attributes
	predict_attributes(saved_model_path, data_folder)