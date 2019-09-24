from torchvision import transforms
from PIL import Image
from glob import glob

import torch.utils.data as data
import pandas as pd
import numpy as np
import os


def load_label_values( label_values_file ) :
    import json
    
    with open( label_values_file, 'r' ) as f :
        label_values = json.load( f )
    
    # Covert String numbers to integers
    for key, values in label_values[ "idx_to_names" ].items() :
        label_values[ "idx_to_names" ][ key ] = { int( k ) : v for k, v in values.items() }
    
    # for key, values in label_values[ "values_to_idx" ].items() :
    # 	label_values[ "values_to_idx" ][ key ] = { k : int( v ) for k, v in values.items() }
    
    return label_values


def get_attribute_dims( label_values_file ) :
    label_values = load_label_values( label_values_file )
    return label_values[ "attribute_dims" ]


def get_transforms(is_train=False):
    if is_train:
        data_transforms = transforms.Compose([
            transforms.Scale(266),
            transforms.CenterCrop((400, 266)),
            # transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Scale(266),
            transforms.CenterCrop((400, 266)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return data_transforms


def default_loader(path):
    return Image.open(path).convert('RGB')


def get_labels(labels_file, target_column, images_folder):
    labels_df = pd.read_csv(labels_file)
    print("Before pre-processing... ", labels_df.shape)
    if target_column == "neck":
        labels_df[target_column].fillna(np.float64(7), inplace=True)
    if target_column == "sleeve_length":
        labels_df[ target_column ].fillna( np.float64( 4 ), inplace=True )
    if target_column == "pattern":
        labels_df[ target_column ].fillna( np.float64( 11 ), inplace=True )
    labels_df.set_index("filename", inplace=True)
    labels_df = labels_df.loc[ ~labels_df.index.duplicated( keep='first' ) ]
    
    # remove file row that does not exists in data folder
    for i in labels_df.index:
        if not os.path.isfile(os.path.join(images_folder, i)):
            labels_df.drop( i, inplace=True )
    print( "After pre-processing... ", labels_df.shape )
    return labels_df


class AttributeDataset( data.Dataset ) :
    # This is memory efficient because all the images are not stored in the memory at once but read as
    # required. Here data.Dataset is a class of torch.utils
    def __init__( self, images_folder, labels_df, target_column, transform=None, target_transform=None,
            loader=default_loader ) :
        
        super().__init__()
        
        self.images_folder = images_folder
        # Index should be the filename in the root folder
        self.labels_df = labels_df
        self.target_column = target_column
        # self.class_to_idx = { target_col: idx for target_col in self.target_columns }
        
        self.imgs = self._get_data()
        
        if len( self.imgs ) == 0 :
            raise (RuntimeError( "Found 0 images in subfolders of: " + images_folder + "\n"
                                                                                       "Supported image "
                                                                                       "extensions are: " +
                                 ",".join(
                IMG_EXTENSIONS ) ))
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    def _get_data( self ) :
        images = [ ]
        
        for file_location in glob( os.path.join( self.images_folder, "*.jpg" ) ) :
            filename = file_location.split( "/" )[ -1 ]
            target_value = self.labels_df.loc[ filename, self.target_column ]
            if not np.isnan( target_value ) :
                item = (file_location, int( target_value ))
                images.append( item )
        return images
    
    # customly writing these two functions to override base class functions
    def __getitem__( self, index ) :
        path, target = self.imgs[ index ]
        img = self.loader( path )
        if self.transform is not None :
            img = self.transform( img )
        if self.target_transform is not None :
            target = self.target_transform( target )
        
        return path, img, target
    
    def __len__( self ) :
        return len( self.imgs )


def make_dsets( IMAGES_FOLDER, LABELS_FILE, target_column, batch_size=32, num_workers=4, is_train=True,
        shuffle=True ) :
    # Data Augmentation and Normalization
    data_transforms = get_transforms( is_train )
    
    labels_df = get_labels( LABELS_FILE, target_column, IMAGES_FOLDER )
    dset = AttributeDataset( IMAGES_FOLDER, labels_df, target_column=target_column,
                             transform=data_transforms )
    
    dset_loader = data.DataLoader( dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers )
    return dset_loader

