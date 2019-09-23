# predict_deep_fashion_attributes
Garments in fashion domain are characterised by attributes like pattern, neck types, sleeve types, size, material. Ecommerce websites use this data to help users navigate through their catalog via filters and effective categorization. To predict these visual features using Deep Learning. Please train a ​Single Neural Network to predict all three attributes for each garment.

#### Data Representation
* All images are stored in images folder. 
* Attributes for each image is stored in attributes.csv file with id as image filenames and "neck
", "sleeve_length" and "pattern" as their attributes
    * Neck attribute is distributed to 0-6 types
    * Sleeve length attribute is distributed to 0-4 types
    * Pattern attribute is distributed to 0-9 types
    
#### Convolutional Neural Network is trained :
* Download the trained model [here]() and
 paste them in "models" folder 
 * To train the model run : `python3 main_multi_label.py /path/to/data/folder`
 * To predict individual attributes for an image run : `python3 main_multi_label.py /path/example/image
   /path/to/saved_model/` 
    * The current accuracy `90.11%` on 10% validation set.

#### Fine-tuning pretrained vgg16 model and retraining last layers with our dataset
* Download the trained model [here]()
* To train teh model run : `python3 main_multi_label_pretrained.py "path/to/data/folder`
* To predict individual attributes for an image run : `python3 main_multi_label_pretrained.py "path/to
/example/image/to/predict  /path/to/saved/model`
    * The current accuracy `xx.xx%` on 10% validation set.
    * The approach is to share the pretrained layers between all attributes and defining last layers for
     each attributes to predict inidividually. 
    
### Need for improvement
* Taking care of #NA values, will take much more development. One way is implemented to treat them as a
 separate class. 
 * A new approach is to train a multi-task network which performs better in terms of model generalization. 
 