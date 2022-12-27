# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:31:24 2022

@author: Lenovo
"""
from __future__ import print_function, division
import os
import cv2
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
plt.ion()  
import seaborn as sns
import shutil
import warnings
from tqdm.notebook import trange as notebook_trange
from tqdm import tqdm, trange
from fastai.vision.all import *
from copy import copy, deepcopy

warnings.filterwarnings('ignore')
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")
else:
    print("Using CPU")

input_data_path = "C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/train_marble/"
fnames = get_image_files(input_data_path)
df = pd.DataFrame()
for fname in fnames:
    df = df.append({'fname':fname, 'label': fname.parent.name}, ignore_index=True)
df = df[~(df.label=="all")]
df = df.reset_index(drop="index")
new_df = pd.DataFrame()
new_df["path"] = df.fname[1:].values
new_df["label"] = df.label[1:].values
df = new_df
df.head()

#os.mkdir("all_data")
#os.mkdir("all_data/good")
#os.mkdir("all_data/not_good")

for i in tqdm(range(df.shape[0])):
    if df["label"].iloc[i] == "good":
        shutil.copy2((df.path.iloc[i]), "C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/marble/good")
    else:
        shutil.copy2((df.path.iloc[i]), "C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/marble/defect")
    
#"/".join(str(df.path.iloc[1]).split("/")[5:])

path= "C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/validation_marble/"
targets=os.listdir(path)
print(targets)

data_dir= "C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/validation/"
train_path=os.path.join(data_dir,'C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/train_marble/')
valid_path=os.path.join(data_dir,'C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/validation_marble/')
test_path=os.path.join(data_dir,'C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/test_marble/')

def make_dir():
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        os.mkdir(train_path)
        os.mkdir(valid_path)
        os.mkdir(test_path)
        for target in targets:
            os.mkdir(os.path.join(train_path,target))
            os.mkdir(os.path.join(valid_path,target))
            os.mkdir(os.path.join(test_path,target))
def check_dir():
    print(f'{data_dir}: {os.path.isdir(data_dir)}')
    print(f'{train_path}: {os.path.isdir(train_path)}')
    print(f'{valid_path}: {os.path.isdir(valid_path)}')
    print(f'{test_path}: {os.path.isdir(test_path)}')
    for target in targets:
        print(f'{os.path.join(train_path,target)}: {os.path.isdir(os.path.join(train_path,target))}')
        print(f'{os.path.join(valid_path,target)}: {os.path.isdir(os.path.join(valid_path,target))}')
        print(f'{os.path.join(test_path,target)}: {os.path.isdir(os.path.join(test_path,target))}')

make_dir()
check_dir()

# Properties of images;
import cv2
image = cv2.imread(str(df.path.iloc[1]))
print(f'Image size: {image.shape}')
print(f'Max pixel value: {np.max(image)}')

5.2
0.4
.4
df.shape[0]*0.15

def load_train_images(path=path,n=100):
    for target in targets:
        data_path=os.path.join(path,target)
        dest=os.path.join(train_path,target)
        image_set=random.choices(os.listdir(data_path))
        print(f'Loading the training images for {target}')
        for file in tqdm(image_set):
            file_path=os.path.join(data_path,file)
            shutil.copy(file_path,dest)
            
def load_valid_images(path=path,n=100):
    for target in targets:
        data_path=os.path.join(path,target)
        dest=os.path.join(valid_path,target)
        image_set=random.choices(os.listdir(data_path))
        print(f'Loading the validation images for {target}')
        for file in tqdm(image_set):
            file_path=os.path.join(data_path,file)
            shutil.copy(file_path,dest)
            
def load_test_images(path=path,n=223):
    for target in targets:
        data_path=os.path.join(path,target)
        dest=os.path.join(test_path,target)
        image_set=random.choices(os.listdir(data_path))
        print(f'Loading the testing images for {target}')
        for file in tqdm(image_set):
            file_path=os.path.join(data_path,file)
            shutil.copy(file_path,dest)
    
import random
load_train_images()
#load_valid_images()
load_test_images()

data_dir = 'C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/marble_dataset/'
TRAIN = 'C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/validation/train'
VAL = 'C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/validation/valid'
TEST = 'C:/Users/Lenovo/Dropbox/My PC (LAPTOP-KKR98K2S)/Desktop/validation/test'

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        #transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.RandomResizedCrop(240),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], # Gaussian Noise
                             [0.229, 0.224, 0.225])
    ]),
    VAL: transforms.Compose([
        transforms.ToTensor(),
        #transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.Normalize([0.485, 0.456, 0.406], # Gaussian Noise
                             [0.229, 0.224, 0.225])
    ]),
    TEST: transforms.Compose([
        transforms.ToTensor(),
        #transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.Normalize([0.485, 0.456, 0.406], # Gaussian Noise
                             [0.229, 0.224, 0.225])
   ])
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x), 
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    )
    for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
print("Classes: ")
class_names = image_datasets[TRAIN].classes
print(image_datasets[TRAIN].classes)

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

# Get a batch of training data
inputs, classes = next(iter(dataloaders[TRAIN]))
show_databatch(inputs, classes)

def visualize_model(vgg, num_images=6):
    was_training = vgg.training
    
    # Set model for evaluation
    vgg.train(False)
    vgg.eval() 
    
    images_so_far = 0

    for i, data in enumerate(dataloaders[TEST]):
        inputs, labels = data
        size = inputs.size()[0]
        
        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
        
        outputs = vgg(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        print("Ground truth:")
        show_databatch(inputs.data.cpu(), labels.data.cpu())
        print("Prediction:")
        show_databatch(inputs.data.cpu(), predicted_labels)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        
        images_so_far += size
        if images_so_far >= num_images:
            break
        
    vgg.train(mode=was_training) # Revert model back to original training state

def eval_model(vgg, criterion):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)
    
    for i, data in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()
        
    avg_loss = loss_test / dataset_sizes[TEST]
    avg_acc = acc_test / dataset_sizes[TEST]
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)
    
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg19(pretrained=True)

mod = list(vgg.classifier.children())
mod.pop()
mod.append(torch.nn.Linear(4096, 2))# Input Layer and output
new_classifier = torch.nn.Sequential(*mod)
vgg.classifier = new_classifier
vgg = vgg.to(device)
criterion = torch.nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 each epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.1)
vgg

print("Test before training")
eval_model(vgg,criterion)

def train_model(vgg, criterion, optimizer, scheduler, num_epochs=16):
    since = time.time()
    best_model_wts = deepcopy(vgg.state_dict())
    best_acc = 0.0
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        vgg.train(True)
        for i, data in enumerate(dataloaders[TRAIN]):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)   
            # Use half training dataset
            if i >= train_batches / 2:
                break  
            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = vgg(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.data
            acc_train += torch.sum(preds == labels.data)
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train * 2 / dataset_sizes[TRAIN]
        avg_acc = acc_train * 2 / dataset_sizes[TRAIN]
        vgg.train(False)
        vgg.eval()
        for i, data in enumerate(dataloaders[VAL]):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
            inputs, labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
            optimizer.zero_grad()
            outputs = vgg(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        avg_loss_val = loss_val / dataset_sizes[VAL]
        avg_acc_val = acc_val / dataset_sizes[VAL]
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = deepcopy(vgg.state_dict())
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    vgg.load_state_dict(best_model_wts)
    return vgg

vgg = train_model(vgg, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=16)

print("Test after training")
eval_model(vgg,criterion)

visualize_model(vgg, num_images=16) #test before training

