import sys
sys.path.insert(0, '/home/zhuoyan/vision/meta_dataset')
import time
import json
import torch
import os

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
        final = []
        for i in range(10):
            features = test_features[i]
            similarity_tmp = []
            for task_id in range(num_tasks):
                task_features = load_train_task_features(os.path.join(root, "clip"), task_id)
                
                similarity_tmp.append(MeanEmbeddingSimilarity(features[target], task_features))
            final.append(similarity_tmp)
        
        sim = torch.tensor(final)
        print("target: {}| check similarity shape: {}".format(target, sim.shape)) # [10,700]

        sim = sim.mean(0)

        sorted_index = sorted(range(len(sim)), key=lambda k: sim[k],reverse=True)

        ## check whether need to filtered out
        with open(os.path.join(root,'finetuning_tasks_indices.json'), 'r') as json_file:
            indices = json.load(json_file)

        filtered_index = []
        if target in trsource:
            for i in sorted_index:
                if indices[i]['source'] != target:
                    filtered_index.append(i)
        
        print("target: {}| length of filtered_index: {}".format(target, len(filtered_index)))

        if filtered_index:
            print("target: {}| save filtered_index with length: {}".format(target, len(filtered_index)))
            with open(os.path.join(root, f'{target}_sim_sorted_index.json'), 'w') as file:
                json.dump(filtered_index, file)
        else:
            print("target: {}| save sorted_index with length: {}".format(target, len(sorted_index)))
            with open(os.path.join(root, f'{target}_sim_sorted_index.json'), 'w') as file:
                json.dump(sorted_index, file)
            
        ## check domain
        # indices_json_path = '/datadrive2/datasets/meta_dataset_taskEmb/100_per_domain'
        # # Read from a JSON file
        # with open(os.path.join(indices_json_path,'finetuning_tasks_indices.json'), 'r') as json_file:
        #     indices = json.load(json_file)
        
        # for i in sorted_index:
        #     print(indices[i]['source'])

if __name__ == "__main__":
   
    main()




