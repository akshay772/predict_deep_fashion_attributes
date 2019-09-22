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
 * To train the model run : `python3 main_CNN.py /path/to/data/folder`
 * To predict individual sentences run : `python3 main_CNN.py "example sentence to classify" /path
/saved_model/` 
    * The current accuracy `90.11%` on 10% validation set.
    
### Need for improvement
 