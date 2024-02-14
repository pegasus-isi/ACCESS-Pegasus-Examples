#!/usr/bin/env python3

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import string

def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]

def plot_image(boxes, image, counter):
    fig, ax = plt.subplots()
    img = plt.imread(image)

    # Display the image
    plt.imshow(img)

    for box in boxes:
        xmin, ymin, xmax, ymax = box

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig("annotated_images_" +str(counter) +".png")
    #plt.show()
    
    
imagesList = {}
imagesList = glob.glob('*.png')
imagesList.sort()

counter=0
for item in imagesList:
    annon_file = "maksssksksss" +str(counter) +".xml"
    with open(annon_file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        boxes = []
        for i in objects:
            boxes.append(generate_box(i))
       
    plot_image(boxes, item, counter)
    counter+=1
    
    # debugging
    #print (item)
    #print (annon_file)
    












    

