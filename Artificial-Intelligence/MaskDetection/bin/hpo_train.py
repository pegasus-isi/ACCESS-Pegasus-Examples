#!/usr/bin/env python3

import optuna
import glob,os
import numpy as np 
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
import PIL
import random
import torch
import argparse
import torchvision
from collections import Counter
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches
import time
import joblib
import re


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def generate_box(obj):
    """
    returns the bounding box coordinates as a list
    """
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    """
    returns the class of the detected object
    """
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "without_mask":
        return 0
    return 2

def generate_target(image_id, file): 
    """
    returns a dictionary target consisting of boxes, labels and image ID of the image
    :input: image ID, annotation file
    """
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all('object')

        num_objs = len(objects)
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        
        return target

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    print("Getting the model defined - currently not pretrained")
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("Model is defined")
    return model

class MaskDataset(object):
    def __init__(self, prefix, transforms):
        self.transforms = transforms
        self.prefix = prefix
        if prefix == "train_":
            self.imgs = glob.glob(prefix+"*.png")
        else:
            self.imgs = glob.glob(prefix+"*.png")
        
        self.size = len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks

        file_image = self.imgs[idx]
        ind = re.findall(r'\d+', self.imgs[idx])[0]
        file_label = 'maksssksksss'+ ind + '.xml'
        img = Image.open(file_image).convert("RGB")
        #Generate Label
        target = generate_target(int(ind), file_label)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return self.size


def plot_image(img_tensor, pred, annotation):
    """
    :input: tensor of image, predicted output (pred) and ground truth (annotation) of the image
    :output: returns plot of bounding boxes on the image
    """
    
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data
    ax.imshow(img.permute(1, 2, 0))
    
    for pred_b in pred[0]['boxes']:
        xmin2, ymin2, xmax2, ymax2 = pred_b
        pred = patches.Rectangle((xmin2,ymin2),(xmax2-xmin2),(ymax2-ymin2),linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(pred)
   
    for true_b in annotation[0]["boxes"]:
        xmin1, ymin1, xmax1, ymax1 = true_b      
        gt = patches.Rectangle((xmin1,ymin1),(xmax1-xmin1),(ymax1-ymin1),linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(gt)
        
    plt.show()

    
def validate(val_loader, model, device):

    model.eval()
    model.to(device)
    running_test_loss = 0

    with torch.no_grad():
        for imgs, annotations in val_loader:
            model.eval()
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            #outputs = model(imgs) - Not needed since we aren't calculating accuracy
            
            # to get val loss. 
            model.train()
            loss_dict = model(imgs,annotations)
            losses = sum(loss for loss in loss_dict.values())
            running_test_loss+=losses.item()
         
    test_loss_final = running_test_loss/len(val_loader)
    print("Test/Val Loss: {}".format(test_loss_final))
    
    return model, test_loss_final


def train(train_loader, model, optimizer, epoch, device):

    model.train()
    model.to(device)
    running_loss = 0
    for imgs, annotations in train_loader:
        
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        optimizer.zero_grad()   
        
        loss_dict = model(imgs, annotations)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
    
    train_loss = running_loss/len(train_loader)
    print("Train Loss: {}".format(train_loss))
    
    return model, train_loss


    
def hpo_monitor(study, trial):
    joblib.dump(study,"hpo_study_mask_detection.pkl")
    



def objective(trial):
    
    print("Performing trial {}".format(trial.number))

    data_transform = transforms.Compose([transforms.ToTensor(),])
    
    # prefix as the first parameter and transformation as the second
    train_dataset = MaskDataset("train_",data_transform)
    test_dataset = MaskDataset("val_",data_transform)
    
    # create data-loaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=True, collate_fn=collate_fn)
    
    # initialise model
    model = get_model_instance_segmentation(3)
    model.to(device)
    print("Model is defined and put onto GPU")
   
    losses_dict= {'train': {}, 'test': {}, 'accuracy': {}}
    
    params = [p for p in model.parameters() if p.requires_grad]    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])   
    lr = trial.suggest_float("lr",1e-5, 1e-1, log=True)
    
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.005)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)
        
    total_loss = 0
    
    for e in range(EPOCHS):
        
        print("{} out of {}".format(e+1, EPOCHS))
        model, train_loss = train(train_dataloader, model, optimizer, EPOCHS, device)
        model, test_loss = validate(test_dataloader, model, device)
        current_metrics = [e,train_loss, test_loss]
        losses_dict["train"][e] = train_loss
        losses_dict["test"][e] = test_loss
        total_loss += test_loss
       
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    total_loss/=EPOCHS
    
    return total_loss


def get_best_params(best):

    parameters = {}

    try:
        parameters["trial_id"] = best.number
        parameters["value"] = best.value
        parameters["params"] = best.params
    except Exception as e:
        parameters["trial_id"] = 0
        parameters["value"] =  0.00
        parameters["params"] = {"optimizer" : "Adam", "lr": 0.001}      
    
    f = open("best_hpo_params.txt","w")
    f.write(str(parameters))
    f.close()

   
def load_study():
    
    try:
        STUDY = joblib.load("hpo_study_mask_detection.pkl")      
        rem_trials = TRIALS - len(STUDY.trials_dataframe())
        print("Successfully loaded the existing study!") 
        
        if rem_trials > 0:
            STUDY.optimize(objective, n_trials=rem_trials, callbacks=[hpo_monitor])
        else:
            print("All trials done!")
        
    except Exception as e:
        print("Creating a new study!")
        
        STUDY = optuna.create_study(study_name='mask_detection')
        STUDY.optimize(objective, n_trials=TRIALS, callbacks=[hpo_monitor])

    best_trial = STUDY.best_trial
    get_best_params(best_trial)

    return

def main():
    print("Device the model will be trained with:")
    print(device)
    
    global EPOCHS
    global TRIALS
    parser = argparse.ArgumentParser(description="Mask-detection Workflow")
    parser.add_argument('--epochs', default=2, type=int, help="Enter number of epochs to train the model")
    parser.add_argument('--trials', type=int, default=2, help="Enter number of trials to perform HPO")
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    TRIALS = args.trials
    load_study()
    
    return
    
    
if __name__ == "__main__":
    main()
