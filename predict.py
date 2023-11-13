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

parser = argparse.ArgumentParser()

#Image input
parser.add_argument('input', type = str, action = 'store', nargs = '*', default = '/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg')
#Checkpoint
parser.add_argument('checkpoint', type = str, action = 'store', nargs = '*', default = '/home/workspace/ImageClassifier/my_checkpoint.pth')
#Top-k
parser.add_argument('--top_k', action = 'store', dest = 'top_k', type = int, default = 5)
#Category names
parser.add_argument('--category_names', action = 'store', dest = 'category_names', default = 'cat_to_name.json')
#Choose processor
parser.add_argument('--processor', action = 'store', dest = 'processor', default = 'GPU')

predict_args = parser.parse_args()
print("Image Input: ", predict_args.input, "Checkpoint: ", predict_args.checkpoint, "TopK: ", predict_args.top_k, "Category names: ",
      predict_args.category_names, "Processor: ", predict_args.processor)
print(predict_args)

from modelpredict import *

def main():
    model = load_checkpoint(predict_args.checkpoint)
    
    image = predict_args.input
    #imshow(image)
    probs, classes = predict(image, model, predict_args.top_k)
    
    import json
    
    with open(predict_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
#     # TODO: Display an image along with the top 5 classes
    #imshow(img)   
    list(cat_to_name.items())[0][1]
    names = []
    for i in range (len(classes)):

        j = list(cat_to_name.items())[i][1]
        names.append(j)

    print(names)
    print(probs)

    result = list(zip(names,probs))
    print(result)    


    base_color = sb.color_palette()[0]
    #sb.barplot(y = names, x = probs, color = base_color)
    
if __name__ == "__main__":
    main()
