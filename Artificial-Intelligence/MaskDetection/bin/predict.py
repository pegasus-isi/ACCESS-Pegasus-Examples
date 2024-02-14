import glob,os,sys
import numpy as np 
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
import PIL
import random
import torch
import torchvision
from collections import Counter
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import re


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class_name = {
    0 : "without mask",
    1 : "with mask",
    2 : "wearing mask incorrectly"
}

def plot_image2(img_tensor, annotation, plotted_image_name):
    
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))
    
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        
    plt.savefig(plotted_image_name)
    plt.show()


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

def get_model_instance_segmentation(num_classes):
    """
    initialise model
    """
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


#-------------- DATASET CLASS ------------------------------------
class MaskDataset(object):
    def __init__(self, prefix, transforms):
        self.transforms = transforms
        self.prefix = prefix
        if prefix == "train_":
            self.imgs = glob.glob(prefix+"*.png")+glob.glob("val_*.png")
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


def collate_func(batch):
    return tuple(zip(*batch))


def predict(model_file, predicted_image, predictions_file):
    data_transform = transforms.Compose([transforms.ToTensor(),])
    
    # prefix as the first parameter and transformation as the second
    pred_dataset = MaskDataset("pred_",data_transform)
    
    # create data-loaders
    pred_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=1,shuffle=True, collate_fn=collate_func)

    for imgs, annotations in pred_dataloader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        break

    model2 = get_model_instance_segmentation(3)
    model2.load_state_dict(torch.load(model_file))
    model2.eval()
    model2.to(device)
    preds = model2(imgs)
    
    plot_image2(imgs[0], preds[0], predicted_image)
    labels_arr = preds[0]["labels"].data.cpu().numpy()
    scores_arr = preds[0]["scores"].data.cpu().numpy()
    
    with open(predictions_file, "w") as txt_file:
        txt_file.write("{:5}  {:^4}  {:^23}  {}".format("box","class","label","confidence") + "\n")
        for i in range(len(scores_arr)):
            txt_file.write("box{}  {:^4}  {:^25}  {:1f}".format(i,labels_arr[i],class_name[labels_arr[i]],scores_arr[i]*100) + "\n")


if __name__ == "__main__":
    model_file = sys.argv[1]
    predicted_image = sys.argv[2]
    predictions_file = sys.argv[3]        
    predict(model_file, predicted_image, predictions_file)