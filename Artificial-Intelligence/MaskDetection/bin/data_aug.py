#!/usr/bin/env python3

"""
MACHINE LEARNING WORKFLOWS - STEP 2 - DATA PREPROCESSING (might be a multistep process of itself)

Data  Augmentation. 

Here we augment the images.


"""
import numpy as np
import glob, os
import cv2


DATASET_DIR = ""

class DataAugmentation():

    def load_img(self,img_path):
        return cv2.imread(img_path)


    def rotate(self, image, angle=20, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv2.warpAffine(image,M,(w,h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image


    def add_gaussian_noise(self, img):
        gaussian = np.random.normal(0, 0.20, (img.shape) )
        return img + gaussian

    
    
    def image_augment(self, path): 
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        ''' 
        img = self.load_img(path)
        img = img/255
        img_gaussian = self.add_gaussian_noise(img)
        return [img, img_gaussian]

def main():
    augmentation = DataAugmentation()

    for file in glob.glob( DATASET_DIR + "*.png"):
        augmented_imgs = augmentation.image_augment(file)
        label = file.split(".")[0]
        name = "train_" + label + "_aug_"
        i = 0
        for img in augmented_imgs:
            cv2.imwrite(DATASET_DIR + name + str(i) + '.png',img*255)# if you want to see the images do img*255
            i +=1



if __name__ == "__main__":
    main()
