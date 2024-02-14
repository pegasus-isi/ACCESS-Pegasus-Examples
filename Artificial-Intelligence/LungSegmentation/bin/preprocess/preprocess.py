#!/usr/bin/env python3
import sys
import argparse
import os
import cv2
import numpy as np
from cv2 import imread
import random
from pathlib import Path

CUTOFF = {"train": [0, 0.7], "val": [0.7, 0.9], "test": [0.9, 1]}
IGNORE_IMAGES = ['CHNCXR_0025_0.png', 'CHNCXR_0036_0.png', 'CHNCXR_0037_0.png', 'CHNCXR_0038_0.png', 'CHNCXR_0039_0.png', 'CHNCXR_0040_0.png', 'CHNCXR_0065_0.png', 'CHNCXR_0181_0.png', 'CHNCXR_0182_0.png', 'CHNCXR_0183_0.png', 'CHNCXR_0184_0.png', 'CHNCXR_0185_0.png', 'CHNCXR_0186_0.png', 'CHNCXR_0187_0.png', 'CHNCXR_0188_0.png', 'CHNCXR_0189_0.png', 'CHNCXR_0190_0.png', 'CHNCXR_0191_0.png', 'CHNCXR_0192_0.png', 'CHNCXR_0193_0.png', 'CHNCXR_0194_0.png', 'CHNCXR_0195_0.png', 'CHNCXR_0196_0.png', 'CHNCXR_0197_0.png', 'CHNCXR_0198_0.png', 'CHNCXR_0199_0.png', 'CHNCXR_0200_0.png', 'CHNCXR_0201_0.png', 'CHNCXR_0202_0.png', 'CHNCXR_0203_0.png', 'CHNCXR_0204_0.png', 'CHNCXR_0205_0.png', 'CHNCXR_0206_0.png', 'CHNCXR_0207_0.png', 'CHNCXR_0208_0.png', 'CHNCXR_0209_0.png', 'CHNCXR_0210_0.png', 'CHNCXR_0211_0.png', 'CHNCXR_0212_0.png', 'CHNCXR_0213_0.png', 'CHNCXR_0214_0.png', 'CHNCXR_0215_0.png', 'CHNCXR_0216_0.png', 'CHNCXR_0217_0.png', 'CHNCXR_0218_0.png', 'CHNCXR_0219_0.png', 'CHNCXR_0220_0.png', 'CHNCXR_0336_1.png', 'CHNCXR_0341_1.png', 'CHNCXR_0342_1.png', 'CHNCXR_0343_1.png', 'CHNCXR_0344_1.png', 'CHNCXR_0345_1.png', 'CHNCXR_0346_1.png', 'CHNCXR_0347_1.png', 'CHNCXR_0348_1.png', 'CHNCXR_0349_1.png', 'CHNCXR_0350_1.png', 'CHNCXR_0351_1.png', 'CHNCXR_0352_1.png', 'CHNCXR_0353_1.png', 'CHNCXR_0354_1.png', 'CHNCXR_0355_1.png', 'CHNCXR_0356_1.png', 'CHNCXR_0357_1.png', 'CHNCXR_0358_1.png', 'CHNCXR_0359_1.png', 'CHNCXR_0360_1.png', 'CHNCXR_0481_1.png', 'CHNCXR_0482_1.png', 'CHNCXR_0483_1.png', 'CHNCXR_0484_1.png', 'CHNCXR_0485_1.png', 'CHNCXR_0486_1.png', 'CHNCXR_0487_1.png', 'CHNCXR_0488_1.png', 'CHNCXR_0489_1.png', 'CHNCXR_0490_1.png', 'CHNCXR_0491_1.png', 'CHNCXR_0492_1.png', 'CHNCXR_0493_1.png', 'CHNCXR_0494_1.png', 'CHNCXR_0495_1.png', 'CHNCXR_0496_1.png', 'CHNCXR_0497_1.png', 'CHNCXR_0498_1.png', 'CHNCXR_0499_1.png', 'CHNCXR_0500_1.png', 'CHNCXR_0502_1.png', 'CHNCXR_0505_1.png', 'CHNCXR_0560_1.png', 'CHNCXR_0561_1.png', 'CHNCXR_0562_1.png', 'CHNCXR_0563_1.png', 'CHNCXR_0564_1.png', 'CHNCXR_0565_1.png']

np.random.seed(4)

class DataPreprocessing:  

    def readForTransformations(self, input_image, pth, is_mask):  
        img = cv2.imread(os.path.join(pth, input_image))
        if not is_mask: return cv2.resize(img,(X_shape,X_shape))[:,:,0] 
        return img 

    def flipImage(self, input_image, pth, is_mask=True):
        norm_img = np.zeros((800, 800))
        img = self.readForTransformations(input_image, pth, is_mask)
        if not is_mask: 
            flipped = cv2.flip(img, 1)
            return cv2.normalize(flipped,  norm_img, 0, 255, cv2.NORM_MINMAX)
        else:
            flipped = cv2.flip(img, 1)
            return flipped

    def rotateImage(self, input_image, pth, deg, is_mask = True):
        img = self.readForTransformations(input_image, pth, is_mask)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        if not is_mask: 
            norm_img = np.zeros((800, 800))
            cv2.normalize(rotated,  norm_img, 0, 255, cv2.NORM_MINMAX)
        return rotated
    
    @staticmethod
    def normalize(input_image):
        """
        This function takes images as input and normalize it i.e remove noise from the data. By normalizing images, we bring
        the image into a range of intensity values which is normal to our senses. In a normalized image, the mean is 0 and the 
        variance is 1.        
        :param input_image: image_file[.png]
        :return:  image_file[.png]
        """
        norm_img = np.zeros((800, 800))
        im = cv2.resize(cv2.imread(os.path.join(DIR, input_image)),(X_shape,X_shape))[:,:,0]        
        return cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
    
def parse_args(args):
    """
    This function takes the command line arguments.        
    :param args: commandline arguments
    :return:  parsed commands
    """
    parser = argparse.ArgumentParser(description="Preprocess Job - Normalizes the images for faster computation")
    parser.add_argument(
                "-i",
                "--input_dir",
                default=".",
                help="directory where input files will be read from"
            )

    parser.add_argument(
            "-m",
            "--mask_dir",
            default=".",
            help="directory where input masks will be read from"
        )

    parser.add_argument(
                "-o",
                "--output_dir",
                default=".",
                help="directory where output files will be written to"
            )
    parser.add_argument(
                "-t",
                "--type",
                default="train",
                help="the type of input set stating if it is test/train/val"
            )

    parser.add_argument(
                "-c",
                "--cutoff",
                default="f",
                help="If you wish to use cutoff"
            )

    return parser.parse_args(args)

if __name__=="__main__":
    args = parse_args(sys.argv[1:])
    DIR = args.input_dir
    
    X_shape = 256 #converting the input images to 256x256 size
    norm_img = np.zeros((800,800))
    all_masks = []
    images = []
    all_files = os.listdir(DIR)
    all_files = [i for i in all_files if i not in IGNORE_IMAGES]
    all_files_length = len(all_files)
    random.shuffle(all_files)
    lower_limit, upper_limit = CUTOFF[args.type][0]*all_files_length, CUTOFF[args.type][1]*all_files_length

    i = 0
    for f in all_files:
        if f.endswith(".png"):
            if "mask" in f:
                all_masks.append(f)
            else:
                # running without pegasus
                if args.cutoff == "t":
                    if i > upper_limit:
                        break

                    if i < lower_limit:
                        continue  

                    images.append(f)
                    i += 1
                elif f.startswith(args.type):
                    images.append(f)
                    

    if args.cutoff=='t': 
        all_masks = os.listdir(args.mask_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dp = DataPreprocessing()

    print('Created ', len(images))
    # Calling the normalize function to nomalize the images and saving them to the output directory
    for i in images: 
        img = i
        normalized_image1 = dp.normalize(i)
        cv2.imwrite(os.path.join(args.output_dir, i[0:-4]+"_norm.png"), normalized_image1)
        #creating transformations for the train images
        if args.type == "train": 
            if img.startswith("train_"):
                img = img[6:-4] # train_CHNCXR_0003_0.png -> CHNCXR_0003_0
            else:
                img = img[:-4] # CHNCXR_0003_0.png -> CHNCXR_0003_0

            mask_img = None 
            for m in all_masks:
                mname = m[:-9] # CHNCXR_0149_0_mask.png -> CHNCXR_0149_0
                if img == mname:
                    mask_img = m
                    break
            if mask_img is None: 
                print('Not found ', i, fname); continue
            mask_path = args.mask_dir if args.mask_dir else args.output_dir 
            flipped = dp.flipImage(i, DIR, False)
            cv2.imwrite(os.path.join(args.output_dir, args.type+'_'+ img +"_0_norm.png"), flipped)
            flipped = dp.flipImage(mask_img, mask_path)
            # print('Writing to ', os.path.join(args.output_dir, args.type+'_'+i[0:-4]+"_norm.png"))
            cv2.imwrite(os.path.join(args.output_dir, img + "_0_mask.png"), flipped)

            for ind, deg in enumerate([8]):
                rotated = dp.rotateImage(i, DIR, deg, False)
                cv2.imwrite(os.path.join(args.output_dir, args.type+'_'+ img +"_"+str(ind+1)+"_norm.png"), rotated)
                # print('Writing to ',i[0:-4]+"_"+str(ind+1)+"_mask.png")
                rotated = dp.rotateImage(mask_img, mask_path, deg)
                cv2.imwrite(os.path.join(args.output_dir, img +"_"+str(ind+1)+"_mask.png"), rotated)
    
    # in the case of clustering preprocess jobs, we want to cleanup input files ourself
    print("cleanup")
    for f in images:
        print("removing {}".format(f))
        Path(f).unlink()
        assert Path(f).exists() == False
