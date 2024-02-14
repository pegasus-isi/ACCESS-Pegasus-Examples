import random
from Pegasus.api import *
import numpy as np
import re

def split_data_filenames(filenames):

    random.shuffle(filenames)
    train, validate, test = np.split(filenames, [int(len(filenames)*0.7), int(len(filenames)*0.8)])
    files_split_dict = {}
    files_split_dict["train"] = train
    files_split_dict["test"] = test
    files_split_dict["validate"] = validate
    
    train_filenames = [file for file in train] 
    val_filenames = [file for file in validate] 
    test_filenames =  [file for file in test] 
    return train_filenames,val_filenames,test_filenames, files_split_dict



def create_ann_list(filenames):
    files_ids = []
    imgs = []

    for filename in filenames:
        stringList = filename.split(".")
        tempName = stringList[0]
        digit = re.findall(r'\d+', tempName)[0]
        files_ids.append(digit)
        fname = filename.split("/")[-1]
        imgs.append(File(fname))
    ann = []
    for ids in files_ids:
        name = 'maksssksksss{}.xml'.format(ids)
        ann.append(File(name))
    return imgs, ann


def create_augmented_filelist(filenames, i):
    files_ids = []
    for filename in filenames:
        stringList = filename.split(".")
        tempName = stringList[0]
        digit = re.findall(r'\d+', tempName)[0]
        files_ids.append(digit)
    resized_files = []
    for ids in files_ids:
    	for j in range(i):
    		name = 'train_maksssksksss{}_aug_{}.png'.format(ids,j)
    		resized_files.append(File(name))
    return resized_files