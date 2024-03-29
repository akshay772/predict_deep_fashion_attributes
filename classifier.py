import os, time, copy
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from preprocessing import make_dsets


def get_pretrained_model(arch="resnet18", pop_last_pool_layer=False, use_gpu=False):
    if arch == "resnet18":
        resnet = torchvision.models.resnet18(pretrained=True)
        pretrained_features = nn.Sequential(*list(resnet.children())[:-1])
        pretrained_fc = resnet.fc
        fc_dim = 512
    elif arch == "vgg16":
        vgg = torchvision.models.vgg16(pretrained=True)
        pretrained_features = vgg.features
        pretrained_fc = vgg.classifier
        fc_dim = 4096
    elif arch == "alexnet":
        alexnet = torchvision.models.alexnet(pretrained=True)
        pretrained_features = alexnet.features
        pretrained_fc = alexnet.classifier
        fc_dim = 4096
    if pop_last_pool_layer:
        pretrained_features = nn.Sequential(*list(pretrained_features.children())[:-1])
    for param in pretrained_features.parameters():
        param.requires_grad = False
    if use_gpu:
        pretrained_features = pretrained_features.cuda()
    return pretrained_features, pretrained_fc, fc_dim


class AttributeFCN( nn.Module ) :
    def __init__( self, in_channels, out_dims, return_conv_layer=False ) :
        super().__init__()
        self.return_conv_layer = return_conv_layer
        model_steps = [ nn.BatchNorm2d( in_channels ),
	        nn.Conv2d( in_channels, 256, 3, padding=1 ),
            nn.ReLU(),
            nn.BatchNorm2d( 256 ),
	        nn.Conv2d( 256, 128, 3, padding=1 ),
	        nn.ReLU(),
	        nn.BatchNorm2d( 128 ),
            nn.Conv2d( 128, 64, 3, padding=1 ),
	        nn.ReLU(),
	        nn.BatchNorm2d( 64 ),
            nn.Conv2d( 64, out_dims, 1 ) ]
        model_steps_dummy = [ # nn.BatchNorm2d(in_channels),
            nn.Conv2d( in_channels, out_dims, 1 ) ]
        self.conv_model = nn.Sequential( *model_steps )
    
    def forward( self, x ) :
        # Get Conv Layer output. Output channels = number of classes
        classes_conv_out = self.conv_model( x )
        
        # Do Global Average Pooling on the Conv Layer with Number of Channels = Classes
        pool_size = (classes_conv_out.size( 2 ), classes_conv_out.size( 3 ))
        average_pool = F.avg_pool2d( classes_conv_out, kernel_size=classes_conv_out.size()[ 2 : ] )
        average_pool_flatten = average_pool.view( average_pool.size( 0 ), -1 )
        # print(average_pool_flatten)
        classes_softmax = F.log_softmax( average_pool_flatten )
        
        if self.return_conv_layer :
            return classes_conv_out, classes_softmax
        else :
            return classes_softmax


def optim_scheduler_ft(model, epoch, init_lr=0.01, lr_decay_epoch=7):
    lr = init_lr * (0.1**(epoch//lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer


def train_attribute_model( model, pretrained_model, train_dset_loader, valid_dset_loader=None,
        criterion=nn.CrossEntropyLoss(), optim_scheduler=optim_scheduler_ft, use_gpu=None, num_epochs=25,
        verbose=False, flatten_pretrained_out=False ) :
    since = time.time()
    best_model = model
    best_acc = 0.0
    
    if use_gpu is None :
        use_gpu = torch.cuda.is_available()
    
    # Define Phases and Get the dataset Sizes
    phases = [ "train" ]
    dset_sizes = {
        "train" : len( train_dset_loader.dataset )
        }
    if valid_dset_loader is not None :
        phases.append( "valid" )
        dset_sizes[ "valid" ] = len( valid_dset_loader.dataset )
    
    if use_gpu :
        pretrained_model.cuda()
        model.cuda()
    
    for epoch in range( num_epochs ) :
        if verbose :
            print( "Epoch {}/{}".format( epoch, num_epochs - 1 ) )
        
        # Each epoch has a train and validation Phase
        for phase in phases :
            if phase == "train" :
                optimizer = optim_scheduler( model, epoch )
                model.train()
            else :
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
            
            # Iterate over data
            dset_loader = valid_dset_loader if phase == "valid" else train_dset_loader
            for data in dset_loader :
                # Get the inputs
                batch_files, inputs, labels = data
                
                # Wrap them in Variable
                if use_gpu :
                    inputs, labels = inputs.cuda(), labels.cuda()
                if phase == "train" :
                    inputs = Variable( inputs )
                else :
                    inputs = Variable( inputs, volatile=True )
                labels = Variable( labels )
                
                # Get output from pre-trained model and re-shape to Flatten
                out_features = pretrained_model( inputs )
                if flatten_pretrained_out :
                    out_features = out_features.view( out_features.size( 0 ), -1 )
                
                # Forward
                outputs = model( out_features )
                # _, preds = torch.max(outputs.data, 1)
                # loss = criterion(outputs, labels)
                # print(outputs, type(outputs))
                loss = F.nll_loss( outputs, labels )
                preds = outputs.data.max( 1 )[ 1 ]
                
                # Backward + Optimize only in Training Phase
                if phase == "train" :
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum( preds == labels.data ).item()
            
            epoch_loss = running_loss / dset_sizes[ phase ]
            epoch_acc = running_corrects / dset_sizes[ phase ]
            if verbose :
                print( "{} Loss: {:.4f} Acc: {:.4f}".format( phase, epoch_loss, epoch_acc ) )
            elif epoch % 3 == 0 :
                print( "{} Epoch {}/{} Loss: {:.4f} Acc: {:.4f}".format( phase, epoch, num_epochs - 1,
                                                                         epoch_loss, epoch_acc ) )
            
            # Deep copy the model
            if phase == "valid" and epoch_acc > best_acc :
                best_acc = epoch_acc
                best_model = copy.deepcopy( model )
    
    time_elapsed = time.time() - since
    print( "Training completed in {:0f}m {:0f}s".format( time_elapsed // 60, time_elapsed % 60 ) )
    print( "Best val Acc: {:4f}".format( best_acc ) )
    return best_model


def save_model(model, weights_path):
    torch.save(model.state_dict(), weights_path)


def load_model(ModelClass, in_channels, output_dim, weights_path=None, return_conv_layer=False, use_gpu=None):
    model = ModelClass(in_channels, output_dim, return_conv_layer)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    if use_gpu:
        model = model.cuda()
    return model


def train_model(model, pretrained_features, target_column, labels_file, train_images_folder,
                valid_images_folder=None,
                batch_size=32, num_workers=4, num_epochs=10,
                use_gpu=None,
                flatten_pretrained_out=False):

    train_dset_loader = make_dsets(train_images_folder, labels_file, target_column,
                                   batch_size=batch_size, num_workers=num_workers, is_train=True)
    valid_dset_loader = None
    if valid_images_folder:
        print("Found validation dataset.... preparing\n")
        valid_dset_loader = make_dsets(valid_images_folder, labels_file, target_column,
                                batch_size=batch_size, num_workers=num_workers, is_train=False)

    # Sleeve Length Model
    model = train_attribute_model(model, pretrained_features,
                                train_dset_loader=train_dset_loader,
                                valid_dset_loader=valid_dset_loader,
                                num_epochs=num_epochs,
                                use_gpu=use_gpu,
                                flatten_pretrained_out=flatten_pretrained_out)
    return model


def create_attributes_model(ModelClass, in_dims, pretrained_features, target_columns, weights_root,
                            labels_file, train_images_folder, valid_images_folder=None, is_train=True,
                            batch_size=32, num_workers=4, num_epochs=30, use_gpu=None):
    models = {}
    for col_name, col_dim in target_columns.items():
        print("Processing Attribute: {}".format(col_name))
        weights_path = os.path.join(weights_root, col_name + ".pth")
        load_weights_path = None
        if os.path.exists(weights_path):
            load_weights_path = weights_path
        model = load_model(ModelClass, in_dims, col_dim + 1, weights_path=load_weights_path, use_gpu=use_gpu)
        if is_train:
            print("Start Training for: {}".format(col_name))
            model = train_model(model, pretrained_features, col_name, labels_file, train_images_folder, valid_images_folder,
                               batch_size, num_workers, num_epochs, use_gpu=use_gpu)
        save_model(model, weights_path)
        models[col_name] = model
    return models

