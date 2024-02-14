#!/usr/bin/env python3
import sys
import os
import cv2
import pickle
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from unet import UNet


def parse_args(args):
    """
        This function takes the command line arguments.        
        :param args: commandline arguments
        :return:  parsed commands
    """
    parser = argparse.ArgumentParser(description="Lung Image Segmentation Using UNet Architecture")
    parser.add_argument(
                "-i",
                "--input_dir",
                default=os.getcwd(),
                help="directory where input files will be read from"
            )

    parser.add_argument(
                "-o",
                "--output_dir",
                default=os.getcwd(),
                help="directory where output files will be written to"
            )
    
    parser.add_argument('-epochs',  metavar='num_epochs', type=int, default = 20, help = "Number of training epochs")
    parser.add_argument('--batch_size',  metavar='batch_size', type=int, default = 32, help = "Batch Size")
    parser.add_argument('--fig_sizex',  metavar='fig_sizex', type=int, default = 8.5, help = "Analysis graph's size x")
    parser.add_argument('--fig_sizey',  metavar='fig_sizey', type=int, default = 11, help = "Analysis graph's size y")
    parser.add_argument('--subplotx',  metavar='subplotx', type=int, default = 3, help = "Analysis graph's subplot no of rows")
    parser.add_argument('--subploty',  metavar='subploty', type=int, default = 1, help = "Analysis graph's subplot no of columns")
    return parser.parse_args(args)  

if __name__=="__main__":
    unet = UNet(parse_args(sys.argv[1:]))
    CURR_PATH = unet.args.output_dir
    
    files = os.listdir(CURR_PATH)

    test_data = [i for i in files if ".png" in i and i.startswith("test_")]
	
    dim = 256
    actual_images = [i for i in test_data]
    # actual_images = actual_images[1:2]
    X_test = [cv2.imread(os.path.join(CURR_PATH,i), 1) for i in actual_images]  

    X_test = np.array(X_test).reshape((len(X_test),dim,dim,3)).astype(np.float32)
    model = load_model(os.path.join(CURR_PATH,"model.h5"), compile=False)
    preds = model.predict(X_test)

    for i in range(len(preds)):
        im = preds[i]
        im[im>0.5] = 1
        im[im<0.5] = 0
        width,height=im.shape[1],im.shape[0]
        img=cv2.imread(os.path.join(CURR_PATH,actual_images[i]),0)
        img=cv2.resize(img,(256,256)).reshape(256,256, 1)
        masked_img=np.squeeze(img*im.reshape(256,256, 1))
        masked_img[masked_img>0.5] = 255
        masked_img[masked_img<0.5] = 0
        masked_img=cv2.resize(masked_img,(width,height)).astype(np.float32)
        cv2.imwrite(os.path.join(CURR_PATH,'pred_'+ str(test_data[i].split('.png')[0][5:]+'_mask.png')), masked_img)

