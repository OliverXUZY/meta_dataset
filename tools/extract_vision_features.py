import os
import random
import h5py
from PIL import Image
import json
#import cv2
import numpy as np
import time
import torch
import sys
sys.path.insert(0, '/home/zhuoyan/vision/meta_dataset/')
import clip

import datasets


def extract_features_per_task(dataset, model, task):
    task_features = {}
    source = task['source']
    task_features['source'] = source
    print(f"Processing {source}...")
    features = []
    for class_id, idx in task['indices']:
        img = dataset.get_next(source, class_id, idx)
        x = img.unsqueeze(0).cuda()  # Add batch dimension and send to GPU
        with torch.no_grad():
            feature = model(x)
        features.append(feature.view(-1).cpu().data.numpy().tolist())
    task_features['features'] = features

    return task_features

def main(split = "train", model_name = "clip"):
    if split == "train":
        trainSet = datasets.datasets['meta-dataset']()
        set_name = "trainSet"
    else:
        raise NotImplementedError
        
    
    indices_json_path = '/datadrive2/datasets/meta_dataset_taskEmb/100_per_domain'
    # Read from a JSON file
    with open(os.path.join(indices_json_path,'finetuning_tasks_indices.json'), 'r') as json_file:
        indices = json.load(json_file)
    
    print("check load indices")
    print("type of indices: ", type(indices))
    print("len of indices: ", len(indices))

    if model_name == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14")
    elif model_name == "clip":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.cuda().eval()  # Send model to GPU and set it to evaluation mode

    # Assuming testSet and subset are defined here or loaded
    if model_name == "clip":
        func = model.encode_image
    else:
        func = model

    start_time = time.time()

    for task_id, task in enumerate(indices[202:], start= 202):
        task_features = extract_features_per_task(trainSet, func, task)

        with open(os.path.join(indices_json_path,f'{model_name}/task_{task_id}.json'), 'w') as json_file:
            json.dump(task_features, json_file, indent = 4)

        print("done {}-th task, time elapsed {} min, total time: {} min".format(
            task_id,
            (time.time() - start_time)/60,
            (time.time() - start_time)/60/(task_id+1 - 202)*len(indices[202:]),  
            ))

    print("check features")
    print(task_features.keys())
    print(task_features['source'])
    val = task_features['features']
    fea = torch.tensor(val)
    print(fea.shape)

if __name__ == "__main__":
   main()