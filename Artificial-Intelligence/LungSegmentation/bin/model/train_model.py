#!/usr/bin/env python3
import os, sys
import pandas as pd
from unet import UNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import segmentation_models as sm
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score
from utils import GeneratePDF
from tensorflow.keras.callbacks import EarlyStopping
import argparse

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
    parser.add_argument(
                "-p",
                "--params_file",
                default="study_results.txt",
                help="Training params"
            )
    
    parser.add_argument('--epochs',  metavar='num_epochs', type=int, default = 40, help = "Number of training epochs")
    parser.add_argument('--batch_size',  metavar='batch_size', type=int, default = 32, help = "Batch Size")
    parser.add_argument('--fig_sizex',  metavar='fig_sizex', type=int, default = 8.5, help = "Analysis graph's size x")
    parser.add_argument('--fig_sizey',  metavar='fig_sizey', type=int, default = 11, help = "Analysis graph's size y")
    parser.add_argument('--subplotx',  metavar='subplotx', type=int, default = 3, help = "Analysis graph's subplot no of rows")
    parser.add_argument('--subploty',  metavar='subploty', type=int, default = 1, help = "Analysis graph's subplot no of columns")
    return parser.parse_args(args) 


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    unet = UNet(args)
    config = {}
    with open(os.path.join(args.input_dir, args.params_file), 'r') as f:
    config = eval(f.read())['params']

    BACKBONE = 'seresnet34'
    preprocess_input = get_preprocessing(BACKBONE)
    model = sm.Unet(BACKBONE, input_shape=(256,256,3), 
    encoder_weights='imagenet', decoder_block_type='transpose')

    w_path = os.path.join(unet.args.output_dir, "model_tmp.h5")
    path = os.path.join(unet.args.output_dir, "model.h5")

    checkpoint_callback = ModelCheckpoint(w_path, monitor='loss', mode="min", save_best_only=True)
    early_stopping = EarlyStopping( monitor='loss', min_delta=0, patience=4)
    callbacks = [checkpoint_callback, early_stopping] 

    # Compile the U-Net model
    model.compile(optimizer=Adam(lr=config['lr']), loss=unet.dice_coef_loss, metrics = [iou_score, 'accuracy'])

    # Call DataLoader function to get train and validation dataset
    train_vol, train_seg, valid_vol, valid_seg = unet.DataLoader()

    # Train the U-Net model
    history = model.fit(
              x = train_vol, 
              y = train_seg, 
              batch_size = unet.args.batch_size, 
              epochs = unet.args.epochs,
              callbacks=callbacks,
              validation_data =(valid_vol, valid_seg))

    # model.save(w_path)

    pdf_path = os.path.join(args.output_dir, 'Analysis.pdf')

    #generate analysis results
    pdf = GeneratePDF()
    pdf.create(unet, pdf_path, history)   
    os.replace(w_path, path)

