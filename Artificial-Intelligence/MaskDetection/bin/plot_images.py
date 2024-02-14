#!/usr/bin/python

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]

def plot_image(boxes):
    fig, ax = plt.subplots()
    img = plt.imread("/home/scitech/shared-data/face-recognition-wf/Images2/maksssksksss0.png")

    # Display the image
    plt.imshow(img)

    for box in boxes:
        xmin, ymin, xmax, ymax = box

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig('savedImage.png')
    plt.show()

### Main Routine ###
with open("/home/scitech/shared-data/face-recognition-wf/annotations2/maksssksksss0.xml") as f:
    data = f.read()
    soup = BeautifulSoup(data, 'xml')
    objects = soup.find_all('object')

    boxes = []
    for i in objects:
        boxes.append(generate_box(i))

plot_image(boxes)
