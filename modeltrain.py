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

def trainer(model, data_dir, save_dir, learning_rate, num_epochs, hidden_units, processor):
    if model == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("No model specified")
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)
    
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=32, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088,4096)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.15)),
        #('fc1', nn.Linear(4096,196)),
        #('relu1', nn.ReLU()),
        #('dropout', nn.Dropout(p=0.15)),
        #('fc2', nn.Linear(196, 150)),
        #('relu2', nn.ReLU()),
        ('output', nn.Linear(4096,102)),
        ('dropout', nn.Dropout(p=0.15)),
        # LogSoftmax is needed by NLLLoss criterion
        ('softmax', nn.LogSoftmax(dim=1))
    ]))
    
# Replace classifier
    model.classifier = classifier
    
    use_gpu = torch.cuda.is_available()
    print("GPU available:", use_gpu)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.to(device)

    criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0009)
    
    if processor == "GPU":
        model.cuda()
    else:
        model.cpu()
        
    epochs = 4
    steps = 0
    running_loss = 0
    print_every = 40

    for epoch in range(epochs):
        for inputs, labels in trainloaders:
            steps += 1
        # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloaders):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloaders):.3f}")
                running_loss = 0
                model.train()
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {
              'state_dict': model.state_dict(),
              'classifier':model.classifier,
              'epochs': epochs,
              'class_to_idx':model.class_to_idx,
              'optimizer':optimizer.state_dict(),
              'arch':model}

    torch.save(checkpoint, 'my_checkpoint.pth')
    
    return model
