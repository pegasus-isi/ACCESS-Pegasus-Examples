import os
import cv2
import numpy as np 
from keras import backend as keras

IGNORE_IMAGES = ['CHNCXR_0025_0.png', 'CHNCXR_0036_0.png', 'CHNCXR_0037_0.png', 'CHNCXR_0038_0.png', 'CHNCXR_0039_0.png', 'CHNCXR_0040_0.png', 'CHNCXR_0065_0.png', 'CHNCXR_0181_0.png', 'CHNCXR_0182_0.png', 'CHNCXR_0183_0.png', 'CHNCXR_0184_0.png', 'CHNCXR_0185_0.png', 'CHNCXR_0186_0.png', 'CHNCXR_0187_0.png', 'CHNCXR_0188_0.png', 'CHNCXR_0189_0.png', 'CHNCXR_0190_0.png', 'CHNCXR_0191_0.png', 'CHNCXR_0192_0.png', 'CHNCXR_0193_0.png', 'CHNCXR_0194_0.png', 'CHNCXR_0195_0.png', 'CHNCXR_0196_0.png', 'CHNCXR_0197_0.png', 'CHNCXR_0198_0.png', 'CHNCXR_0199_0.png', 'CHNCXR_0200_0.png', 'CHNCXR_0201_0.png', 'CHNCXR_0202_0.png', 'CHNCXR_0203_0.png', 'CHNCXR_0204_0.png', 'CHNCXR_0205_0.png', 'CHNCXR_0206_0.png', 'CHNCXR_0207_0.png', 'CHNCXR_0208_0.png', 'CHNCXR_0209_0.png', 'CHNCXR_0210_0.png', 'CHNCXR_0211_0.png', 'CHNCXR_0212_0.png', 'CHNCXR_0213_0.png', 'CHNCXR_0214_0.png', 'CHNCXR_0215_0.png', 'CHNCXR_0216_0.png', 'CHNCXR_0217_0.png', 'CHNCXR_0218_0.png', 'CHNCXR_0219_0.png', 'CHNCXR_0220_0.png', 'CHNCXR_0336_1.png', 'CHNCXR_0341_1.png', 'CHNCXR_0342_1.png', 'CHNCXR_0343_1.png', 'CHNCXR_0344_1.png', 'CHNCXR_0345_1.png', 'CHNCXR_0346_1.png', 'CHNCXR_0347_1.png', 'CHNCXR_0348_1.png', 'CHNCXR_0349_1.png', 'CHNCXR_0350_1.png', 'CHNCXR_0351_1.png', 'CHNCXR_0352_1.png', 'CHNCXR_0353_1.png', 'CHNCXR_0354_1.png', 'CHNCXR_0355_1.png', 'CHNCXR_0356_1.png', 'CHNCXR_0357_1.png', 'CHNCXR_0358_1.png', 'CHNCXR_0359_1.png', 'CHNCXR_0360_1.png', 'CHNCXR_0481_1.png', 'CHNCXR_0482_1.png', 'CHNCXR_0483_1.png', 'CHNCXR_0484_1.png', 'CHNCXR_0485_1.png', 'CHNCXR_0486_1.png', 'CHNCXR_0487_1.png', 'CHNCXR_0488_1.png', 'CHNCXR_0489_1.png', 'CHNCXR_0490_1.png', 'CHNCXR_0491_1.png', 'CHNCXR_0492_1.png', 'CHNCXR_0493_1.png', 'CHNCXR_0494_1.png', 'CHNCXR_0495_1.png', 'CHNCXR_0496_1.png', 'CHNCXR_0497_1.png', 'CHNCXR_0498_1.png', 'CHNCXR_0499_1.png', 'CHNCXR_0500_1.png', 'CHNCXR_0502_1.png', 'CHNCXR_0505_1.png', 'CHNCXR_0560_1.png', 'CHNCXR_0561_1.png', 'CHNCXR_0562_1.png', 'CHNCXR_0563_1.png', 'CHNCXR_0564_1.png', 'CHNCXR_0565_1.png']

class UNet:
    def __init__(self, parsed_args):
        self.args = parsed_args
        self.curr = self.args.input_dir#os.getcwd()
        OUTPUT_FOLDER=self.args.input_dir
        self.N_TRIALS = 4

    def DataLoader(self):
        """
        This function takes all the training images and masks from the current working directory and converts
        the images into a numpy array.        
        :return:  
            X_train: ndarray
                      2D array containing train images            
            y_train: ndarray
                      2D array containing train masks  
            X_valid: ndarray
                      2D array containing validation images  
            y_valid: ndarray
                      2D array containing validation masks  
        """
        path = self.curr   
        train_masks = []
        val_masks = []
	
        all_files = [i for i in os.listdir(path) if ".png" in i and i not in IGNORE_IMAGES]
        #masks = os.listdir(masks_path)

        train_data = [i for i in all_files if "train_" in i]
        val_data = [i for i in all_files if "val_" in i]
        masks_data = [i for i in all_files if "mask" in i]
        
        for img in train_data:
          fname = img[6:-9]
          found= False
          for m in masks_data:
            mname = m[0:-9]
            if fname == mname:
              found = True
              train_masks.append(m)
              break
          if not found: print('Not found ', fname)
                
        for img in val_data:
          fname = img[4:-9]
          for m in masks_data:
            mname = m[0:-9]
            if fname == mname:
              val_masks.append(m)
              break  
        
        print('Images --- ', len(train_data))
        print('Val-- ',len(val_data))
        print('t Masks--', len(train_masks))
        print('v Masks--', len(val_masks))   

        dim = 256
        X_train = [cv2.imread(os.path.join(path,i), 1) for i in train_data]
        y_train = [cv2.resize(
                    cv2.imread(os.path.join(path,i), 0)
                ,(dim, dim)) for i in train_masks]

        X_valid = [cv2.imread(os.path.join(path,i), 1) for i in val_data]
        y_valid = [cv2.resize(
                    cv2.imread(os.path.join(path,i), 0)
                ,(dim, dim)) for i in val_masks]

        X_train = np.array(X_train).reshape((len(X_train),dim,dim,3))
        y_train = np.array(y_train).reshape((len(y_train),dim,dim,1))
        X_valid = np.array(X_valid).reshape((len(X_valid),dim,dim,3))
        y_valid = np.array(y_valid).reshape((len(y_valid),dim,dim,1))
        assert len(X_train) == len(y_train)
        assert len(X_valid) == len(y_valid)

        return X_train.astype(np.float32), y_train.astype(np.float32), X_valid.astype(np.float32), y_valid.astype(np.float32)

    def dice_coef(self, y_true, y_pred):
        """
        This function is used to gauge the similarity of two samples. It is also called F1-score.
        :parameter y_true: actual mask of the image
        :parameter y_pred: predicted mask of the image
        :return: dice_coefficient value        
        """
        y_true_f = keras.flatten(y_true)
        y_pred_f = keras.flatten(y_pred)
        intersection = keras.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)
    
    def dice_coef_loss(self, y_true, y_pred):
        """
        This function is used to gauge the similarity of two samples. It is also called F1-score.
        :parameter y_true: actual mask of the image
        :parameter y_pred: predicted mask of the image
        :return: dice_coefficient value        
        """
        return -self.dice_coef(y_true, y_pred)


