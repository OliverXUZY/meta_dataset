import subprocess
import argparse
# List of configurations
configs = [
    "configs/clip/finetune_select.yaml",
    "configs/dinov2/finetune_select.yaml",
    "configs/mocov3/finetune_select.yaml",
]

settings = [
    "selected_tasks",
    # "all_tasks",
    # "bad_tasks",
]

load_model_paths = [
    "./save/meta_dataset_select/clip/{}/{}/meta-dataset-select_clip_ViT-B32_fs-centroid_15y2s18q_300m_100M",
    "./save/meta_dataset_select/dinov2/{}/{}/meta-dataset-select_dinov2_vitb14_fs-centroid_15y2s18q_300m_100M",
    "./save/meta_dataset_select/mocov3/{}/{}/meta-dataset-select_mocov3_vit_fs-centroid_15y2s18q_300m_100M",
    ]
    
# Base command
# base_cmd = "python finetune_select.py"
base_cmd = "python finetune.py"
# val_domains = 'traffic_sign'
from_scratch = True


def parse_args():
    parser = argparse.ArgumentParser(description="This is my training script.")
    parser.add_argument('-d', '--domain', type=str, help='specify domain')
    args = parser.parse_args()
    return args

def run_domain(val_domains):
    # Loop over configurations
    for idx, config in enumerate(configs):
        for id, setting in enumerate(settings):
            # Extract the model name from the config path for the save path
            model_name = config.split("/")[1]
            print("finetuning model {}".format(model_name))    
            # Construct the save path
            save_path = f"./save/first50/{model_name}/{val_domains}/{setting}" 

            ## all domains train together
            # save_path = f"/datadrive2/save/meta_dataset/{model_name}/{save_name}"    
            
            
            # Construct the full command
            cmd = f"{base_cmd} --config {config}  --val_domains {val_domains} --target_in_trainDataset {val_domains} --output_path {save_path} --setting {setting}"

            if not from_scratch:
                load_model_path = load_model_paths[idx].format(val_domains, setting)
                cmd += " --path {}".format(load_model_path)

            print(cmd)
            
            # Execute the command
            
            subprocess.run(cmd, shell=True)

def main():
    args = parse_args()
    if args.domain:
        run_domain(args.domain)
        break
    print("No specify args domain")
    all = ['omniglot', 'aircraft', 'cu_birds', 'quickdraw', 'fungi', 'vgg_flower', 'ilsvrc_2012', 'mscoco', 'traffic_sign']
    # all = ['vgg_flower']

    for domain in all:
        run_domain(domain)

            
if __name__ == "__main__":
    main()
