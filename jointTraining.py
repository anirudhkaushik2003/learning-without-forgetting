from __future__ import print_function
from __future__ import division

import sys
sys.path.append("../..")
import numpy as np
from iirc.datasets_loader import get_lifelong_datasets
from iirc.definitions import PYTORCH, IIRC_SETUP
from iirc.utils.download_cifar import download_extract_cifar100


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
import torch.nn.functional as F
from torchmetrics.classification import MultilabelJaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from tqdm import tqdm

from metrics import jaccard_sim, modified_jaccard_sim, strict_accuracy, recall
from resnetcifar import ResNetCIFAR

import time
import os
import copy
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

from IIRC_CIFAR_HIERARCHY import classHierarchy
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

download_extract_cifar100("../../data")

essential_transforms_fn = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
])
augmentation_transforms_fn = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
])

dataset_splits, tasks, class_names_to_idx = \
    get_lifelong_datasets(dataset_name = "iirc_cifar100",
                          dataset_root = "../../data", # the imagenet folder (where the train and val folders reside, or the parent directory of cifar-100-python folder
                          setup = IIRC_SETUP,
                          framework = PYTORCH,
                          tasks_configuration_id = 0,
                          essential_transforms_fn = essential_transforms_fn,
                          augmentation_transforms_fn = augmentation_transforms_fn,
                          joint = True
                         )

# print(len(tasks))
n_classes_per_task = []
for task in tasks:
    n_classes_per_task.append(len(task))
n_classes_per_task = np.array(n_classes_per_task)

# lifelong_datasets['train'].choose_task(2)
# print(list(zip(*lifelong_datasets['train']))[1])
for i in dataset_splits:
    print(i)

    
# initialize a pretrained model (imageNet)
model_name = "resnet" #choosing alexnet since it is "relatively" easy to train
# model_name = "squeezenet" # changed to squeezeNet since it gets same acc as alex but smaller
num_classes = 9 # in cifar100

batch_size = 4

num_epochs = 14

feature_extract = False #set to false so we can finetune entire model

ngpu = 1

def train_model(model, trainloader, testloader, criterion, optimizer, num_classes, task_id, num_epochs=5, temperature=1 ):
    since = time.time() # including this just because
    
    if task_id == 0:
        num_epochs *= 2 # train for 2x num_epochs for the first task as compared to other tasks
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        
                
        running_loss = 0.0
        data_len = 0
        # iterate over data
        for images,label1,label2 in tqdm(trainloader):
            images = images.to(device)
            label1 = one_hot_encode_labels(label1, num_classes=num_classes, gpu=True)
#             label2 = label2.to(device)


            #empty the gradients
            optimizer.zero_grad()

            outputs, _ = model(images)
            
            offset_1, offset_2 = compute_offsets(task_id)
            outputs = outputs[:, :offset_2]
            predictions = outputs > 0.0
            
            loss = criterion(outputs/temperature, label1)
            loss.backward()
            
            running_loss += loss.item()
            optimizer.step()
            
            train_metrics['jaccard_sim'] += jaccard_sim(predictions.to(torch.int32), label1.to(torch.int32)) * images.shape[0]
            train_metrics['modified_jaccard_sim'] += modified_jaccard_sim(predictions.to(torch.int32), label1.to(torch.int32)) * images.shape[0]
            train_metrics['strict_acc'] += strict_accuracy(predictions.to(torch.int32), label1.to(torch.int32)) * images.shape[0]
            train_metrics['recall'] += recall(predictions.to(torch.int32), label1.to(torch.int64)) * images.shape[0]
            data_len += images.shape[0]
            
        train_metrics['jaccard_sim'] /= data_len  
        train_metrics['modified_jaccard_sim'] /= data_len  
        train_metrics['strict_acc'] /= data_len  
        train_metrics['recall'] /= data_len 
        
        
                
        epoch_loss = running_loss / len(trainloader.dataset)
        print("len dataset = ",len(trainloader.dataset))
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
        print('{} Loss: {:.4f}'.format('train', epoch_loss))
        
        print()
        
        print(f"===Training Scores===")
        print(f"JS: {train_metrics['jaccard_sim']} ")
        print(f"modified JS: {train_metrics['modified_jaccard_sim']} ")
        print(f"strict accuracy: {train_metrics['strict_acc']} ")
        print(f"Recall: {train_metrics['recall']} ")
        test_model(model, testloader, num_classes, task_id, mode=0)
        model = model.to(device)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    return model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def one_hot_encode_labels(label, num_classes, gpu=False):
    
    label = torch.from_numpy(np.array([class_names_to_idx[i] for i in label]))
    label = F.one_hot(label, num_classes=num_classes)
    label = label.to(torch.float32)
    if gpu:
        label = label.to(device)
                       
    return label
                       
    
def compute_offsets(task):
        offset1 = int(sum(n_classes_per_task[:task]))
        offset2 = int(sum(n_classes_per_task[:task + 1]))
        return offset1, offset2
    
def test_model(model,testloader, num_classes, task_id, mode=0):
    with torch.no_grad():
        print(f"Begining Testing on task {task_id}")
        data_len = 0 
        
        for i,data in enumerate(tqdm(testloader)):
            images, label1,label2 = data
            images = images.to(device)
            # since subclass labels are introduced after their corresponding superclass labels,
            # in case we encounter a subclass label, we can assume it's superclass label has already been introduced
            # or that it's superclass does not exist
            if not any([class_names_to_idx[j] for j in label1] < seen_classes):
                continue
            if label1 in classHierarchy or label1 in classHierarchy.values(): #if subclass has superclass or is superclass
                if label1 in classHierarchy.values(): # if label is superclass label
                    label1 = one_hot_encode_labels(label1, num_classes=num_classes)
                    label = label1
                else: # if label is subclass and has superclass
                    label2 = label1
                    label1 = classHierarchy[label1]
                    
                    label1 = one_hot_encode_labels(label1, num_classes=num_classes)

                    label2 = one_hot_encode_labels(label2, num_classes=num_classes)

                    label = label1 | label2
                    
            else: # subclass has no superclass
                label1 = one_hot_encode_labels(label1, num_classes=num_classes)
                label = label1

            label = label.to(torch.int32)
            outputs, _ = model(images) # sigmoidless 
            outputs = outputs.detach().cpu()
            offset_1, offset_2 = compute_offsets(task_id)
            outputs = outputs[:, :offset_2]
            predictions = outputs > 0.0
#             print(predicted)

            valid_metrics['jaccard_sim'] += jaccard_sim(predictions.to(torch.int32), label.to(torch.int32)) * images.shape[0]
            valid_metrics['modified_jaccard_sim'] += modified_jaccard_sim(predictions.to(torch.int32), label.to(torch.int32)) * images.shape[0]
            valid_metrics['strict_acc'] += strict_accuracy(predictions.to(torch.int32), label.to(torch.int32)) * images.shape[0]
            valid_metrics['recall'] += recall(predictions.to(torch.int32), label.to(torch.int64)) * images.shape[0]
            data_len += images.shape[0]
            
        valid_metrics['jaccard_sim'] /= data_len  
        valid_metrics['modified_jaccard_sim'] /= data_len  
        valid_metrics['strict_acc'] /= data_len  
        valid_metrics['recall'] /= data_len  
            
        
            
#             correct += (predicted == label).sum().item()
#             correct /= batch_size

#             print(preds,label)
        if mode == 0:
            print("===In-task validation===")
            print(f"JS: {valid_metrics['jaccard_sim']} ")
            print(f"modified JS: {valid_metrics['modified_jaccard_sim']} ")
            print(f"strict accuracy: {valid_metrics['strict_acc']} ")
            print(f"Recall: {valid_metrics['recall']} ")
            
            
        elif mode == 1:
            print("===Post-task validation===")
            print(f"JS: {valid_metrics['jaccard_sim']} ")
            print(f"modified JS: {valid_metrics['modified_jaccard_sim']} ")
            print(f"strict accuracy: {valid_metrics['strict_acc']} ")
            print(f"Recall: {valid_metrics['recall']} ")
            
        elif mode == 2:
            print("===Final Test Scores===")
            print(f"JS: {valid_metrics['jaccard_sim']} ")
            print(f"modified JS: {valid_metrics['modified_jaccard_sim']} ")
            print(f"strict accuracy: {valid_metrics['strict_acc']} ")
            print(f"Recall: {valid_metrics['recall']} ")
            
            

            
# Setup 
# BCE loss for multi-label classification
# sigmoid activation after FC layer 
# everything above 0.5 is a predicted label

criterion = nn.BCEWithLogitsLoss(reduction="mean") # as output is sigmoidless

# get dataset corresponding to each split
train_data = dataset_splits["train"]
intask_val_data = dataset_splits["intask_valid"]
posttask_val_data = dataset_splits["posttask_valid"]
test_data = dataset_splits["test"]

# pre-trained Model on imageNet 
# resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
seen_classes = 0
seen_classes_list = []
output_layer_size = int(sum(n_classes_per_task))

# additional network features
temperature = 1.0
weight_decay = 1e-5


resnet = ResNetCIFAR(num_classes=output_layer_size, num_layers=20 )
resnet = resnet.to(device)
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay)


# scheduler # TO DO

# resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


# initialize data to train on first task
for task_id, task in enumerate(tasks):
    train_metrics = {"jaccard_sim" : 0, "modified_jaccard_sim" : 0, "strict_acc" : 0, "recall" : 0}
    valid_metrics = {"jaccard_sim" : 0, "modified_jaccard_sim" : 0, "strict_acc" : 0, "recall" : 0}
    train_data.choose_task(task_id)
    intask_val_data.choose_task(task_id)
    posttask_val_data.choose_task(task_id)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    InTask_valloader = torch.utils.data.DataLoader(intask_val_data, batch_size=batch_size, shuffle=True, num_workers=2)
    PostTask_valloader = torch.utils.data.DataLoader(posttask_val_data, batch_size=batch_size, shuffle=True, num_workers=2)
    
    seen_classes += n_classes_per_task[task_id]
    seen_classes_list = list(set(seen_classes_list) | set(task))
            
    if (device.type == 'cuda') and (ngpu > 1):
        print(f"Training on multiple gps ({ngpu})")    
        resnet = nn.DataParallel(resnet,list(range(ngpu)))
            
    print(f"Begining Training on Task {task_id+1}")
    resnet = train_model(resnet, trainloader, InTask_valloader, criterion, optimizer, seen_classes,task_id, num_epochs)
    test_model(resnet, InTask_valloader,seen_classes, task_id, mode=1)

# resnet = train_model(resnet, dataloader_dict, criterion, optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))