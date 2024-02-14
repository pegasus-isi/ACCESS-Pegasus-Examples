#!/usr/bin/env python3

import os,glob
from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt



def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "without_mask":
        return 0
    return 2

def get_class_count(labels):
    classes = []
    for i in range(len(labels)):
        f = open(labels[i])
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all('object')
        for ele in objects:
            classes.append(generate_label(ele))
    
    return Counter(classes)

def plot_class_distribution(class_points):
    fig = plt.figure()
 
    X = ["Without Mask", "With Mask", "Incorrectly Worn Mask"]
    Y = [class_points[0], class_points[1], class_points[2]]
    plt.bar(X,Y)
    plt.xlabel("Classes")
    plt.ylabel("Data points")
    plt.suptitle("Class Distribution")
    #save as a png
    plt.savefig("class_distribution.png")

if __name__ == "__main__":

    labels = glob.glob("*.xml")
    class_count = get_class_count(labels)
    plot_class_distribution(class_count)