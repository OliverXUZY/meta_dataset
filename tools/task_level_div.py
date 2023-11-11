import sys
sys.path.insert(0, '/home/zhuoyan/vision/meta_dataset')
import time
import json
import torch
import os
import numpy as np

from distance_tools import load_test_features, simfunc, compute_mahalanobis_distance, MeanEmbeddingSimilarity,load_train_task_features


def main():
    # target = 'traffic_sign'
    trsource = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'quickdraw', 'fungi', 'vgg_flower']
    all = ['traffic_sign', 'mscoco', 'ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'quickdraw', 'fungi', 'vgg_flower']
        
    root = "/datadrive2/datasets/meta_dataset_taskEmb/100_per_domain"
    num_tasks = 700

    test_features = []
    for i in range(10):
        test_features.append(load_test_features(irun = i))
    
    for target in all:
        start_time = time.time()
        print(f"==== processing {target} ... ======")
        with open(os.path.join(root, f'{target}_sim_sorted_index.json'), 'r') as file:
            sorted_index = json.load(file)
        
        div_by_tasks_cached_file = os.path.join(root, f'{target}_div_by_tasks.npy')
        if os.path.exists(div_by_tasks_cached_file):
            print("load from : ", div_by_tasks_cached_file)
            final = np.load(div_by_tasks_cached_file)
            print(f"target: {target} | div shape: {final.shape}")
        else:
            final = []
            start_time = time.time()
            for i in range(10):
                features = test_features[i]

                current_set = []
                mh_distances = []
                count = 1
                for task_id in sorted_index:
                    task_features = load_train_task_features(os.path.join(root, "clip"), task_id)
                    current_set.append(task_features)
                    train_set_features = torch.concat(current_set)
                    distance = compute_mahalanobis_distance(features[target], train_set_features).item()
                    mh_distances.append(distance)
                    count += 1
                    if count % 20 == 1:
                        print(f"target: {target} | {i}-th test features | done {count}-th task")
                final.append(mh_distances)

                print("done {}-th task, time elapsed {} min, total time: {} min".format(
                count,
                (time.time() - start_time)/60,
                (time.time() - start_time)/60/(i+1)*10,  
                ))

            final = np.array(final)
            print("save to : ", div_by_tasks_cached_file)
            
            np.save(div_by_tasks_cached_file, final)
            print(f"target: {target} | div shape: {final.shape}")

            
if __name__ == "__main__":
    main()


