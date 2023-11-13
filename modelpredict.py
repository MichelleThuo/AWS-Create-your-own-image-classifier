import matplotlib.pyplot as plt
import seaborn as sb
import time
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import json
from predict import predict_args
from modeltrain import *

def load_checkpoint(checkpoint_path = 'my_checkpoint.pth', gpu = False):
    '''Load the saved model.'''
    saved_model = torch.load(checkpoint_path)
    model_arch = saved_model['arch']
    
    #use GPU if it's available
    device = torch.device("cuda" if gpu else "cpu")
    
    model = models.vgg16(pretrained = True)
    model.name = 'vgg16'
    #model = saved_model(model_arch)
    #Freeze parameters so we don't backdrop through them
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(4096, 102),
                                    nn.LogSoftmax(dim=1))
    
    
    model.load_state_dict(saved_model['state_dict'], strict=False)
    model.classifier = saved_model['classifier']
    epochs = saved_model['epochs']
    model.class_to_idx = saved_model['class_to_idx']
    optimizer = saved_model['optimizer']
    #optimizer.load_state_dict(saved_model['optimizer'])
    model = saved_model['arch']
    
    model.to(device);
    return model

#model = load_checkpoint()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    original_width, original_height = image.size
    if original_width < original_height:
        width = 256
        height = width * (original_height // original_width)
    else:
        height = 256
        width = height * (original_width // original_height)
    newsize = (width, height)
    image = image.resize(newsize)
    left = (width - 224)//2
    top = (height - 224)//2
    right = (width + 224)//2
    bottom = (height + 224)//2
    
    #crop the center of the image
    image = image.crop((left, top, right, bottom))
    
    image = np.array(image)
    image = image.astype('float64')
    image = image / ([255, 255, 255])
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = image.transpose((2, 0, 1))
   
    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    #undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #model = load_checkpoint(model)
    model = load_checkpoint()
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0).float()
    outputs = model.forward(image)
    #cat_to_name = None
    #if predict_args.category_names:
        #with open(predict_args.category_names, 'r') as f:
            #cat_to_name = json.load(f)
    #else:
        #with open('cat_to_name.json','r') as f:
            #cat_to_name = json.load(f)
    with open(predict_args.category_names, 'r') as f:
        cat_to_name = json.load(f)        
    probs, classes = torch.exp(outputs).topk(topk)
    probs = probs[0].tolist()
    classes = [cat_to_name[str(cl+1)] for cl in classes[0].tolist()]
    
    return probs, classes
