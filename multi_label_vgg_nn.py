from preprocessing import get_attribute_dims
from classifier import get_pretrained_model, create_attributes_model, AttributeFCN

# Labels
labels_file = "./data/data/attributes.csv"
label_values_file = "./data/data/label_values.json"

# Trianing validation images
TRAIN_IMAGES_FOLDER = "./data/data/train"
VALID_IMAGES_FOLDER = "./data/data/test"
# TEST_IMAGES_FOLDER = ""

if __name__ == "__main__":
	target_dims = get_attribute_dims(label_values_file)
	print(target_dims)
	pretrained_conv_model, _, _ = get_pretrained_model( "vgg16", pop_last_pool_layer=True )
	
	attribute_models = create_attributes_model( AttributeFCN, 512, pretrained_conv_model,
	                                            target_dims,
	                                            "weights/vgg16-fcn-266-2/",
	                                            labels_file,
	                                            TRAIN_IMAGES_FOLDER,
	                                            VALID_IMAGES_FOLDER,
	                                            num_epochs=10,
	                                            is_train=True,
	                                            use_gpu=False )

