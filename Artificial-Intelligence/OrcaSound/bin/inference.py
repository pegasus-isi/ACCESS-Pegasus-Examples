#!/usr/bin/env python3

import os, sys, json, glob
import torch
import numpy as np
import pandas as pd
import params 
import argparse

from model import get_model_or_checkpoint
from scipy.io import wavfile
from collections import defaultdict
from dataloader import AudioFileWindower 
from pathlib import Path
from tqdm import tqdm


"""
Input: wav folder
Output: prediction

Steps:
    * Dataloader splits things into windows 
    * Iterate and accumulate predictions 
    - Call aggregation function 
    - Write out a JSON file 

"""

class OrcaDetectionModel():
    def __init__(self, model_path, threshold=0.7, min_num_positive_calls_threshold=3, hop_s=2.45, rolling_avg=False, use_cuda=False):
        #i initialize model
        self.model, _ = get_model_or_checkpoint(params.MODEL_NAME, model_path, use_cuda=use_cuda)
        self.model.eval()
        #self.mean = os.path.join(model_path, params.MEAN_FILE)
        #self.invstd = os.path.join(model_path, params.INVSTD_FILE)
        self.mean = None
        self.invstd = None
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold
        self.hop_s = hop_s
        self.rolling_avg = rolling_avg

    def split_and_predict(self, wav_file_path):
        """
        Args contains:
            - wavfile_path
            - model_path 
        """

        # initialize parameters
        wavfile_path = wav_file_path
        chunk_duration=params.INFERENCE_CHUNK_S

        audio_file_windower = AudioFileWindower(
                [wavfile_path], mean=self.mean, invstd=self.invstd, hop_s=self.hop_s
            )
        window_s = audio_file_windower.window_s

        # initialize output JSON
        result_json = {
            "local_predictions":[],
            "local_confidences":[]
            }

        # iterate through dataloader and add accumulate predictions
        for i in tqdm(range(len(audio_file_windower))):
            # get a mel spec for the window 
            audio_file_windower.get_mode = 'mel_spec'
            mel_spec_window, _ = audio_file_windower[i]
            # run inference on window
            input_data = torch.from_numpy(mel_spec_window).float().unsqueeze(0).unsqueeze(0)
            pred, _ = self.model(input_data)
            posterior = np.exp(pred.detach().cpu().numpy())

            pred_id = 0
            if posterior[0,1] > self.threshold:
                pred_id = 1
            confidence = round(float(posterior[0,1]),3)

            result_json["local_predictions"].append(pred_id)
            result_json["local_confidences"].append(confidence)
        
        submission = pd.DataFrame(dict(
            wav_filename=Path(wav_file_path).name,
            start_time_s=[i*self.hop_s for i in range(len(audio_file_windower))],
            duration_s=self.hop_s,
            confidence=result_json['local_confidences']
        ))

        if self.rolling_avg:
            rolling_scores = submission['confidence'].rolling(2).mean()
            rolling_scores[0] = submission['confidence'][0]
            submission['confidence'] = rolling_scores
            result_json["local_confidences"] = submission['confidence'].tolist()
        result_json['submission'] = submission

        return result_json


    def aggregate_predictions(self, result_json):
        """
        Given N local window predictions Pi, aggregate into a global one.
        Currently we try to reduce false positives so have strict thresholds
        """

        # calculate nth percentile of result_json["local_confidences"], this is global confidence
        local_confidences = result_json["local_confidences"]
        local_predictions = result_json["local_predictions"]

        pred_array = np.array(local_predictions)
        conf_array = np.array(local_confidences)
        total_num_positive_predictions = sum(pred_array)

        global_prediction = 0
        if total_num_positive_predictions >= self.min_num_positive_calls_threshold:
            global_prediction = 1
        result_json["global_prediction"] = global_prediction
        
        positive_predictions_conf = conf_array[pred_array == 1]
        global_confidence = 0
        if positive_predictions_conf.size > 0:
            global_confidence = np.average(positive_predictions_conf)
        result_json["global_confidence"] = global_confidence*100

        return result_json

    def predict(self, wav_file_path):
        result_json = self.split_and_predict(wav_file_path)
        result_json = self.aggregate_predictions(result_json)
        return result_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identifies wav files with Orca sounds."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default=".",
        help="Path to the input directory with `.wav` files. Default is `.`",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="predictions.json",
        help="Path to the predictions file. Default is `predictions.json`.",
    )
    parser.add_argument(
        "-s",
        "--sensor",
        default="_",
        help="Sensor name to be saved in predictions file.",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        default="_",
        help="Timestamp to be saved in predictions file.",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        action="store_true",
        help="Enable CUDA.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="model.pkl",
        help="Path to the model that will be used for inference. Default is `model.pkl`.",
    )

    args = parser.parse_args()
    
    orca_model = OrcaDetectionModel(args.model, use_cuda=args.cuda)
    results = {}
    for input_wav in sorted(glob.glob(os.path.join(args.input_dir, "*.wav"))):
        result_json = orca_model.predict(input_wav)
        results[Path(input_wav).name] = {"local_predictions": result_json["local_predictions"], 
        "local_confidences": result_json["local_confidences"],
        "global_prediction": result_json["global_prediction"], 
        "global_confidence": result_json["global_confidence"]}

    final_json = {args.sensor: {args.timestamp: [results]}}

    with open(args.output, 'w') as f:
        json.dump(final_json, f)

