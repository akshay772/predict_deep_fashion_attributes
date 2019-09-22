import numpy as np
import pandas as pd

import os

from PIL import Image, ImageOps
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def preprocessing(train):
    # new expanded data frame
    new_columns = [ train.columns[ 0 ], "neck0", "neck1", "neck2", "neck3", "neck4", "neck5", "neck6",
        "neck7", "sleeve_length0", "sleeve_length1", "sleeve_length2", "sleeve_length3", "sleeve_length4",
        "sleeve_length5", "pattern0", "pattern1", "pattern2", "pattern3", "pattern4", "pattern5", "pattern6",
        "pattern7", "pattern8", "pattern9", "pattern10" ]
    new_train = pd.DataFrame( columns=new_columns )
    new_train.filename = train.filename
    new_train.loc[ train.neck == np.float64( 0.0 ), "neck0" ] = 1
    new_train.loc[ train.neck == np.float64( 1.0 ), "neck1" ] = 1
    new_train.loc[ train.neck == np.float64( 2.0 ), "neck2" ] = 1
    new_train.loc[ train.neck == np.float64( 3.0 ), "neck3" ] = 1
    new_train.loc[ train.neck == np.float64( 4.0 ), "neck4" ] = 1
    new_train.loc[ train.neck == np.float64( 5.0 ), "neck5" ] = 1
    new_train.loc[ train.neck == np.float64( 6.0 ), "neck6" ] = 1
    new_train.loc[ train.neck == np.float64( 7.0 ), "neck7" ] = 1
    new_train.loc[ train.sleeve_length == np.float64( 0.0 ), "sleeve_length0" ] = 1
    new_train.loc[ train.sleeve_length == np.float64( 1.0 ), "sleeve_length1" ] = 1
    new_train.loc[ train.sleeve_length == np.float64( 2.0 ), "sleeve_length2" ] = 1
    new_train.loc[ train.sleeve_length == np.float64( 3.0 ), "sleeve_length3" ] = 1
    new_train.loc[ train.sleeve_length == np.float64( 4.0 ), "sleeve_length4" ] = 1
    new_train.loc[ train.sleeve_length == np.float64( 5.0 ), "sleeve_length5" ] = 1
    new_train.loc[ train.pattern == np.float64( 0.0 ), "pattern0" ] = 1
    new_train.loc[ train.pattern == np.float64( 1.0 ), "pattern1" ] = 1
    new_train.loc[ train.pattern == np.float64( 2.0 ), "pattern2" ] = 1
    new_train.loc[ train.pattern == np.float64( 3.0 ), "pattern3" ] = 1
    new_train.loc[ train.pattern == np.float64( 4.0 ), "pattern4" ] = 1
    new_train.loc[ train.pattern == np.float64( 5.0 ), "pattern5" ] = 1
    new_train.loc[ train.pattern == np.float64( 6.0 ), "pattern6" ] = 1
    new_train.loc[ train.pattern == np.float64( 7.0 ), "pattern7" ] = 1
    new_train.loc[ train.pattern == np.float64( 8.0 ), "pattern8" ] = 1
    new_train.loc[ train.pattern == np.float64( 9.0 ), "pattern9" ] = 1
    new_train.loc[ train.pattern == np.float64( 10.0 ), "pattern10" ] = 1
    new_train.fillna( np.float64( 0 ), inplace=True )
    
    return new_train


def resize_img( im ) :
    print( "\nbefore : ", im.size )
    old_size = im.size
    desired_size = 350
    ratio = float( desired_size ) / max( old_size )
    new_size = tuple( [ int( x ) for x in old_size ] )
    im = im.resize( new_size, Image.ANTIALIAS )
    new_im = Image.new( "RGB", (desired_size, desired_size) )
    new_im.paste( im, ((desired_size - new_size[ 0 ]) // 2, (desired_size - new_size[ 1 ]) // 2) )
    
    delta_w = desired_size - new_size[ 0 ]
    delta_h = desired_size - new_size[ 1 ]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    print( padding )
    im = ImageOps.expand( im, padding, fill="white" )
    return im


def create_model() :
    model = Sequential()
    model.add( Conv2D( filters=16, kernel_size=(5, 5), activation="relu", input_shape=(350, 350, 3) ) )
    model.add( MaxPooling2D( pool_size=(2, 2) ) )
    model.add( Dropout( 0.25 ) )
    model.add( Conv2D( filters=32, kernel_size=(5, 5), activation='relu' ) )
    model.add( MaxPooling2D( pool_size=(2, 2) ) )
    model.add( Dropout( 0.25 ) )
    model.add( Conv2D( filters=64, kernel_size=(5, 5), activation="relu" ) )
    model.add( MaxPooling2D( pool_size=(2, 2) ) )
    model.add( Dropout( 0.25 ) )
    model.add( Conv2D( filters=64, kernel_size=(5, 5), activation='relu' ) )
    model.add( MaxPooling2D( pool_size=(2, 2) ) )
    model.add( Dropout( 0.25 ) )
    model.add( Flatten() )
    model.add( Dense( 128, activation='relu' ) )
    model.add( Dropout( 0.5 ) )
    model.add( Dense( 64, activation='relu' ) )
    model.add( Dropout( 0.5 ) )
    model.add( Dense( 25, activation='sigmoid' ) )
    print( model.summary() )
    model.compile( optimizer='adam', loss='binary_crossentropy', metrics=[ 'accuracy' ] )
    
    return model

def train_model(data_folder):
    attributes_path = os.path.join(data_folder, "attributes.csv")
    train = pd.read_csv(attributes_path)
    train.neck.fillna(np.float64(7), inplace=True)
    train.sleeve_length.fillna(np.float64(5), inplace=True)
    train.pattern.fillna(np.float64(10), inplace=True)
    new_train = preprocessing(train)
    train_image = []
    image_not_in_folder = []
    images_path = os.path.join(data_folder, "images")
    for i in tqdm(range(new_train.shape[0])):
        file_path = os.path.join(images_path,new_train.filename[i])
        if os.path.isfile(file_path):
            img = image.load_img(file_path)
            img = resize_img(img)
            print("after resize : ", img.size, "\n")
            img = image.img_to_array(img)
            img = img/255
            train_image.append(img)
        else:
            image_not_in_folder.append(i)
    
    X = np.array(train_image)
    new_train.drop(new_train.index[image_not_in_folder[0]], inplace=True)
    new_train.reset_index(drop=True, inplace=True)
    y = np.array(new_train.drop(["filename"], axis=1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    
    model = create_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)
    model.save("./models/cl_multi_label.h5")
    
    
def predict_attributes(saved_model_path, image_path):
    model = load_model(saved_model_path)
    img = image.load_img(image_path)
    img = resize_img( img )
    img = image.img_to_array( img )
    img = img / 255
    
    classes = np.asarray(["neck0", "neck1", "neck2", "neck3", "neck4", "neck5", "neck6",
        "neck7", "sleeve_length0", "sleeve_length1", "sleeve_length2", "sleeve_length3", "sleeve_length4",
        "sleeve_length5", "pattern0", "pattern1", "pattern2", "pattern3", "pattern4", "pattern5", "pattern6",
        "pattern7", "pattern8", "pattern9", "pattern10" ])

    proba = model.predict( img.reshape( 1, 350, 350, 3 ) )
    top_3 = np.argsort( proba[ 0 ] )[ :-4 :-1 ]
    for i in range( 3 ) :
        print( "{}".format( classes[ top_3[ i ] ] ) + " ({:.3})".format( proba[ 0 ][ top_3[ i ] ] ) )

