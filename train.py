import matplotlib.pyplot as plt
import seaborn as sb
import time
import numpy as np
import pandas as pd
import PIL
import PIL.Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse

#from workspace_utils import keep_awake
#with active_session():
    
parser = argparse.ArgumentParser()
#where data is stored
parser.add_argument("data_dir", type = str, action = "store", nargs = "*", default = "/home/workspace/aipnd-project/flowers/")
#set directory to save checkpoints
parser.add_argument("--save_dir", action = "store",dest = "save_dir", default = "/home/workspace/ImageClassifier/my_checkpoint.pth")
#choose architecture
parser.add_argument("--arch", action = "store", dest = "model", default = "vgg16")
#set the learning rate hyperparameter
parser.add_argument("--learning_rate", action = "store", dest = "learning_rate", default = 0.0009)
#set the epoch hyperparameter
parser.add_argument("--epochs", action = "store", dest = "epochs", type = int, default = 4)
#set the hidden units hyperparameter
parser.add_argument("--hidden_units", action = "store", dest = "hidden_units", type = int, default = 4096)
#use GPU for training
parser.add_argument("--processor", action = "store", dest = "processor", default = "GPU")

args = parser.parse_args()

print("Image Directory: ", args.data_dir, "Save Directory: ", args.save_dir, "Model: ", args.model, 
      "Learning Rate: ", args.learning_rate, "Epochs: ", args.epochs, "Hidden units: ", args.hidden_units, 
      "Processor: ", args.processor)

from modeltrain import trainer

def main():
    print(args.model)
    trainer(args.data_dir, args.save_dir, args.model, args.learning_rate, args.epochs, args.hidden_units, args.processor)
#use the get_input_args function to run the program
if __name__ == "__main__":
    main()
