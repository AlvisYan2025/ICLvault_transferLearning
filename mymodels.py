import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import statistics
import os
import random
from tqdm import tqdm

#resnet oct model definition
class ResnetOCTModel(nn.Module):
    def __init__(self, new_layers=[], dropout=0.5, freeze = True):
        super(ResnetOCTModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.new_dims = new_layers
        self.dropout = dropout
        self.freeze = freeze
        pretrained_model = models.resnet50(pretrained=True)
        for param in pretrained_model.parameters():
            if freeze:
                param.requires_grad = False
        self.resnet = pretrained_model
        num_features = pretrained_model.fc.out_features
        layers = []
        for layer_size in new_layers:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(num_features, layer_size))
            layers.append(nn.ReLU())
            num_features = layer_size
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(num_features, 1, bias = True)) 
        self.new_layer = nn.Sequential(*layers)
    def forward(self, x):
        # Extract features using the pre-trained model
        x = self.resnet(x)
        # Pass through custom layers
        x = self.new_layer(x)
        return x
    def examine_parameter_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}")
                print(f"  Mean: {param.data.mean().item()}")
                print(f"  Std: {param.data.std().item()}")
                print(f"  Min: {param.data.min().item()}")
                print(f"  Max: {param.data.max().item()}")
                print()
    def plot(self, train_loss, validation_loss, train_std, val_std, num_epochs, title=''):
        #plot the training graph
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label='Train Loss', color='blue')
        if train_std:
            plt.fill_between(epochs, np.array(train_loss) - np.array(train_std), np.array(train_loss) + np.array(train_std), color='blue', alpha=0.2)
        plt.plot(epochs, validation_loss, label='Validation Loss', color='orange')
        if val_std:
            plt.fill_between(epochs, np.array(validation_loss) - np.array(val_std), np.array(validation_loss) + np.array(val_std), color='orange', alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title + " Training and Validation Loss Trends")
        plt.legend()
        last_epoch = num_epochs
        last_train_loss = train_loss[-1]
        last_val_loss = validation_loss[-1]
        plt.annotate(f'Train Loss: {last_train_loss:.2f}', 
                    xy=(last_epoch, last_train_loss), 
                    xytext=(last_epoch, last_train_loss + 0.05),
                    arrowprops=dict(facecolor='blue', shrink=0.05),
                    fontsize=10, color='blue')
        
        plt.annotate(f'Validation Loss: {last_val_loss:.2f}', 
                    xy=(last_epoch, last_val_loss), 
                    xytext=(last_epoch, last_val_loss + 0.05),
                    arrowprops=dict(facecolor='orange', shrink=0.05),
                    fontsize=10, color='orange')
        plt.show()
    def train_model(self, num_epochs, dataloader_train, dataloader_val, criterion = nn.L1Loss(), learning_rate=0.0025, early_stop=True, regl2=0, graph=False, unfreeze_epoch=5, unfreeze_num = 5):
        '''train model with specified datasets'''
        print('training start')
        print('----------------------')
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=regl2)
        train_loss_avg = [] #average loss over all epochs 
        validation_loss_avg =[] 
        train_loss_std = [] #std of losses over all epochs
        validation_loss_std = []
        lossV= float('inf')
        stop = False
        for epoch in range(num_epochs):
            #gradual unfreezing
            layer_names = ['fc', 'layer4', 'layer3', 'layer3', 'layer2', 'layer1']
            if epoch >= unfreeze_epoch and self.freeze:
                unfreeze_ind = (epoch//unfreeze_num)%len(layer_names)
                for name, param in self.resnet.named_parameters():
                    if layer_names[unfreeze_ind] in name:
                        param.requires_grad = True
            #training loop
            train_loss = [] #inividual loss for a single epoch
            validation_loss = []
            self.train() 
            running_loss = 0.0
            for inputs, labels in dataloader_train:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                optimizer.zero_grad()
                outputs = self(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                #self.mseloss(outputs, labels)
                all_params = torch.cat([x.contiguous().view(-1) for x in self.parameters()])
                l2_regularization = regl2 * torch.norm(all_params, 2)
                loss = loss + l2_regularization
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                running_loss += loss.item()
            lossT = running_loss/len(dataloader_train)
            train_loss_avg.append(lossT)
            #train_loss_std.append(statistics.stdev(train_loss))
            #validation
            self.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in dataloader_val:
                    inputs, labels = inputs.to(self.device), labels.to(self.device).float()
                    outputs = self(inputs).squeeze(1)
                    loss = criterion(outputs, labels)
                    validation_loss.append(loss.item())
                    running_val_loss += loss.item()
            if (running_val_loss / len(dataloader_val)>=lossV):
                stop=True
            lossV = running_val_loss / len(dataloader_val)
            validation_loss_avg.append(lossV)
            #validation_loss_std.append(statistics.stdev(validation_loss))
            print(f"Epoch {epoch + 1}/{num_epochs}, Traning Loss: {lossT}, Validation loss: {lossV}")
            if stop and early_stop:
                print('validation stopped converging')
                break 
        if graph:
            self.plot(train_loss_avg, validation_loss_avg, 0, 0, num_epochs)
        return (train_loss_avg, validation_loss_avg, train_loss_std, validation_loss_std)
    def predict(self, inputs):
        self.eval()
        #self.dropout = 0.0
        with torch.no_grad():
            outputs = self(inputs)
        return outputs
    def evaluate(self, dataloader_test, criterion=nn.L1Loss(), show_val=False):
        self.eval()  
        running_loss = 0.0
        with torch.no_grad(): 
            for inputs, labels in dataloader_test:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float()  
                outputs = self(inputs).squeeze()
                loss = criterion(outputs, labels)
                if show_val:
                    print(f'predicted {outputs}, expected {labels}, loss {loss}')
                running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader_test)
        print(f"Evaluation Loss: {avg_loss:.4f}")

#dataset definition
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=39.38, std=42.014)  
])

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, num_crops=1, transform=transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.num_crops = num_crops
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label




def filter_by_lens(data, size):
    return data[data["ICL Size"]==size]
def get_paths_and_labels(subjects, data, data_dir='AS-OCT data'):
    '''given a list of subject names, returns a list of corresponding file names and labels(vault sizes)'''
    def is_valid(file_path):
        for dir in subjects:
            if dir in file_path:
                return True
        return False
    paths = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == "desktop.ini":
                continue
            file_path = os.path.join(root, file)
            if is_valid(file_path):
                label = data.loc[data['Subject (Eye)']== int(root.split('/')[-1].split(' ')[-1])]['Post-Op Vault']
                if not label.empty:
                    paths.append(file_path)
                    labels.append(label.values[0])
    return (paths, labels)

def split_subjects(data):
    other, test = train_test_split(data, test_size=0.15, shuffle=True)
    subject_test = ["Subject "+str(num) for num in test['Subject (Eye)']]
    train, val = train_test_split(other, test_size=0.2)
    subject_train = ["Subject "+str(num) for num in train['Subject (Eye)']]
    subject_val = ["Subject "+str(num) for num in val['Subject (Eye)']]
    print('selected subjects:', len(subject_test), len(subject_train),len(subject_val))
    return subject_test, subject_train, subject_val 
def construct_dataset(data, subject_test, subject_train, subject_val, batch_size=4):
    train_path, train_labels = get_paths_and_labels(subject_train,data, data_dir='OCT_Data')
    test_path, test_labels = get_paths_and_labels(subject_test,data, data_dir='OCT_Data')
    val_path, val_labels = get_paths_and_labels(subject_val,data, data_dir='OCT_Data')
    dataset_train = ImageDataset(image_paths=train_path, labels=train_labels, transform=transform)
    dataset_test = ImageDataset(image_paths=test_path, labels=test_labels, transform=transform)
    dataset_val = ImageDataset(image_paths=val_path, labels=val_labels, transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    print(f'datasets constructed. test length: {len(dataset_test)}, train length {len(dataset_train)}, val length {len(dataset_val)}')
    return dataloader_test, dataloader_train, dataloader_val