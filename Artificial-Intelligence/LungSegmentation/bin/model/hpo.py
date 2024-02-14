#!/usr/bin/env python3
#Time = 29035
import cv2
import os, sys
import argparse
from unet import UNet
import json
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
import optuna
from segmentation_models.metrics import iou_score
import joblib


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
                "-s",
                "--storage_path",
                type=str,
                default="",
                help="Storage location for parallel hpo jobs. Ex: mysql://username:pass@hostname/db"
            )
    parser.add_argument(
                "-o",
                "--output_dir",
                default=os.getcwd(),
                help="directory where output files will be written to"
            )
    parser.add_argument(
                "-r",
                "--results_file",
                default="study_results.txt",
                help="Study results output file"
            )
    parser.add_argument('--epochs',  metavar='num_epochs', type=int, default = 30, help = "Number of training epochs")
    parser.add_argument('--batch_size',  metavar='batch_size', type=int, default = 32, help = "Batch Size")
    parser.add_argument('--fig_sizex',  metavar='fig_sizex', type=int, default = 8.5, help = "Analysis graph's size x")
    parser.add_argument('--fig_sizey',  metavar='fig_sizey', type=int, default = 11, help = "Analysis graph's size y")
    parser.add_argument('--subplotx',  metavar='subplotx', type=int, default = 3, help = "Analysis graph's subplot no of rows")
    parser.add_argument('--subploty',  metavar='subploty', type=int, default = 1, help = "Analysis graph's subplot no of columns")
    return parser.parse_args(args)  

        

def get_best_params(best, filename, root):
    """
    Saves best parameters of Optuna Study.
    """
    
    parameters = {}
    parameters["trial_id"] = best.number
    parameters["value"] = best.value
    parameters["params"] = best.params
    f = open(os.path.join(root, filename),"w")
    f.write(str(parameters))
    f.close()

def tune_unet(trial, direction="minimize"):
    """
    This function is used to train the model and call the Ray Tune callback function after every epoch for 
    hyperparameter optimization.
    :parameter config: hyperarameters list                
    """

    lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-5, 3e-5, 1e-7, 2e-3])
    model = sm.Unet('resnet34', input_shape=(256,256,3), encoder_weights='imagenet')

    early_stopping = EarlyStopping( monitor='loss', min_delta=0, patience=4)
    callbacks = [early_stopping]

    # Compile the U-Net model
    model.compile(optimizer=Adam(lr=lr), loss=[unet.dice_coef_loss], metrics = [iou_score, 'accuracy'])

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
    loss = history.history["loss"]

    return abs(sum(loss)/len(loss))


def hpo_monitor(study, trial):
    """
    Save optuna hpo study
    """
    joblib.dump(study, os.path.join(unet.args.output_dir, "study_checkpoint_tmp.pkl"))
    os.rename(
        os.path.join(unet.args.output_dir, "study_checkpoint_tmp.pkl"), 
        os.path.join(unet.args.output_dir, "study_checkpoint.pkl")
    )



def create_study(abs_folder_path, write_path, result_path, storage_path):
    """
    This function creates study object which contains data from each epoch, with different hyperparameters, of the training.
    :parameter checkpoint_file: File
                            Checkpoint file contains previous study object (if any) or an empty file where study object 
                            is dumped
    """           

    try:
        STUDY = joblib.load(abs_folder_path)
        todo_trials = unet.N_TRIALS - len(STUDY.trials_dataframe())
        if todo_trials > 0 :
            print("There are {} trials to do out of {}".format(todo_trials, unet.N_TRIALS))
            STUDY.optimize(tune_unet, n_trials=todo_trials,  callbacks=[hpo_monitor])
            best_trial = STUDY.best_trial
            get_best_params(best_trial, result_path, unet.args.output_dir)
        else:
            print("This study is finished. Nothing to do.")

    except Exception as e:
        if storage_path=='':
            STUDY = optuna.create_study(direction='minimize', study_name='Lung Segmentation')
        else:
            STUDY = optuna.create_study(direction='minimize', storage=storage_path, load_if_exists=True, study_name='Lung Segmentation')
        
        STUDY.optimize(tune_unet, n_trials= unet.N_TRIALS,  callbacks=[hpo_monitor])
        best_trial = STUDY.best_trial
        get_best_params(best_trial, result_path, unet.args.output_dir)


if __name__=="__main__":
    global unet
    args = parse_args(sys.argv[1:])
    unet = UNet(args)
    unet.DataLoader()
    hpo_checkpoint = os.path.join(unet.args.output_dir, "study_checkpoint.pkl")
    hpo_checkpoint_tmp = os.path.join(unet.args.output_dir, "study_checkpoint_tmp.pkl")
    hpo_results=args.results_file
    storage=args.storage_path
    create_study(hpo_checkpoint, hpo_checkpoint_tmp, hpo_results, storage)

