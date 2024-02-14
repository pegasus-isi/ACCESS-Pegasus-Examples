#!/usr/bin/env python3
import os
from unet import UNet
import numpy as np
import cv2
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import inch, cm
import argparse
import sys
import re

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


def row(canvas, x, y, y_inc, title, arr):
  canvas.drawString(x-inch*0.1, inch*5, "Raw Images")
  for i in arr:
    canvas.drawImage(os.path.join(unet.args.output_dir,i), x, y, inch, inch)
    y = y+y_inc

def draw_table(canvas, orig_images, actual_mask_images, masks_pred):
  y_inc = inch*1.25
  x_inc = inch*2.5
  x = inch*0.8
  y = inch*.25
  orig_y = y

  row(canvas, x, y, y_inc, "Raw Images", orig_images)
  x, y = x+x_inc, orig_y
  row(canvas, x, y, y_inc, "Actual Masks", actual_mask_images)
  x, y = x+x_inc, orig_y
  row(canvas, x, y, y_inc, "Predicted Masks", masks_pred)
  canvas.save()

def get_images(all_files):
  predicted_mask_images = []
  all_masks = []
  actual_mask_images = []

  orig_images = []
  masks_pred = []
  c = 3
  for i in all_files:
    if "mask" in i:
      if i.startswith("pred_"):
        predicted_mask_images.append(i)
      else:
        all_masks.append(i)
    elif c > 0 and "test" in i:
      orig_images.append(i)
      c = c-1
  for oig in orig_images:
    orig_name = oig[5:-9]
    for img in predicted_mask_images:
      fname = img[5:-14]
      found = False
      if fname == orig_name:
        masks_pred.append(img)
        for m in all_masks:
          mname = m[0:-9]
          if fname == mname:
            found = True
            actual_mask_images.append(m)
            break
      if found: break

  return orig_images, actual_mask_images, masks_pred

if __name__=="__main__":
  dim = 256
  num_classes = 1
  global unet

  unet = UNet(parse_args(sys.argv[1:]))
  path = unet.args.output_dir

    
  # handle the case where jobs are clustered and we don't want all other files picked up
  # aside from what is needed for this job
  p = re.compile(r"test_([A-Z]{6}_\d{4}_\d)_norm.png")
  all_files = list()
  for f in os.listdir(path):
    # if predicted mask, add to all_files
    if f.startswith("pred_"):
      all_files.append(f)
      print("added {}".format(f))
    # else make sure file matches the pattern of a test file and
    # also add the corresponding mask (which should exist here)
    else:
      matched = p.match(f)
      if matched:
        # add test lung image file
        all_files.append(f)
        print("added {}".format(f))
        m = "{}_mask.png".format(matched.group(1))
        all_files.append(m)
        print("added {}".format(m))


  orig_images, actual_mask_images, masks_pred = get_images(all_files)

  print(' Images', orig_images, actual_mask_images, masks_pred)
  predicted_masks = [cv2.imread(os.path.join(path,i), 1) for i in masks_pred]
  predicted_masks = np.array(predicted_masks).reshape((len(predicted_masks),dim,dim, 3)).astype(np.float32)
  actual_masks = [cv2.resize(cv2.imread(os.path.join(path,i), 1),(dim, dim)) for i in actual_mask_images]
  actual_masks = np.array(actual_masks).reshape((len(actual_masks),dim,dim,3)).astype(np.float32)

  num_classes = predicted_masks.shape[-1]
  predicted_masks = np.array([ np.argmax(predicted_masks, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)

  axes = (1,2) # W,H axes of each image
  intersection = np.sum(np.abs(predicted_masks * actual_masks), axis=axes)
  mask_sum = np.sum(np.abs(actual_masks), axis=axes) + np.sum(np.abs(predicted_masks), axis=axes)
  #union = mask_sum  - intersection

  smooth = 0.001
  dice = 2 * (intersection + smooth)/(mask_sum + smooth)
  dice = np.mean(dice)


  canvas = Canvas(os.path.join(unet.args.output_dir, "EvaluationAnalysis.pdf"))
  canvas.setFont("Times-Roman", 20)
  canvas.drawString(inch*3,inch*11, "Evaluation Analysis")
  canvas.setFont("Times-Roman", 12)
  canvas.drawString(inch*3,inch*10, "Dice coefficient ="+ str(dice))
  draw_table(canvas, orig_images, actual_mask_images, masks_pred)
