#!/usr/bin/env python3

import glob, os, sys
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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_confusion_matrix(y_true, y_pred, cm_file):
    """creates and plots a confusion matrix given two list (targets and predictions)
    :param list y_true: list of all targets (in this case integers bc. they are indices)
    :param list y_pred: list of all predictions (in this case one-hot encoded)
    """

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(cm_file)
    plt.show()


def generate_label(obj):
    """
    returns the class of the detected object
    """
    if obj.find("name").text == "with_mask":
        return 1
    elif obj.find("name").text == "without_mask":
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
        objects = soup.find_all("object")

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
    xmin = int(obj.find("xmin").text)
    ymin = int(obj.find("ymin").text)
    xmax = int(obj.find("xmax").text)
    ymax = int(obj.find("ymax").text)

    return [xmin, ymin, xmax, ymax]


# -------------- DATASET CLASS ------------------------------------
class MaskDataset(object):
    def __init__(self, prefix, transforms):
        self.transforms = transforms
        self.prefix = prefix
        if prefix == "train_":
            self.imgs = glob.glob(prefix + "*.png") + glob.glob("val_*.png")
        else:
            self.imgs = glob.glob(prefix + "*.png")
        self.size = len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks

        file_image = self.imgs[idx]
        ind = re.findall(r"\d+", self.imgs[idx])[0]
        file_label = "maksssksksss" + ind + ".xml"

        img = Image.open(file_image).convert("RGB")
        # Generate Label
        target = generate_target(int(ind), file_label)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return self.size


def collate_func(batch):
    return tuple(zip(*batch))


def evaluate(model_file, evaluate_file, cm_file):

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # prefix as the first parameter and transformation as the second
    test_dataset = MaskDataset("test_", data_transform)

    # create data-loaders
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, collate_fn=collate_func
    )

    model2 = get_model_instance_segmentation(3)
    model2.load_state_dict(torch.load(model_file))
    model2.eval()
    model2.to(device)

    running_loss = 0.0
    loss_value = 0.0

    y_test = []
    predictions = []
    for images, targets in test_dataloader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        y_test.append(targets)
        with torch.no_grad():
            loss_dict = model2(images)

            # this returned object from the model:
            # len is 4 (so index here), which is probably because of the size of the batch
            # loss_dict[index]['boxes']
            # loss_dict[index]['labels']
            # loss_dict[index]['scores']
            for x in range(len(loss_dict)):
                loss_value += sum(loss for loss in loss_dict[x]["scores"])
                predictions.append(loss_dict[x]["scores"])

        running_loss += loss_value

    with open(evaluate_file, "w") as txt_file:
        txt_file.write(
            "Model Evaluation : running loss --->".format(float(running_loss)) + "\n"
        )

    predictions, targets = [], []
    for image, labels in test_dataloader:
        preds = model2(image)
        label = labels[0]
        y_test = label["labels"].detach().cpu().numpy()
        pred = preds[0]
        y_pred = pred["labels"].detach().cpu().numpy()
        for i in range(len(y_test)):
            predictions.append(y_pred[i])
            targets.append(y_test[i])

    create_confusion_matrix(targets, predictions, cm_file)


if __name__ == "__main__":
    model_file = sys.argv[1]
    evaluate_file = sys.argv[2]
    cm_image = sys.argv[3]
    evaluate(model_file, evaluate_file, cm_image)
