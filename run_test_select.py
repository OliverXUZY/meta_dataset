import subprocess
import argparse
# List of configurations
configs = [
    "configs/clip/test.yaml",
    "configs/dinov2/test.yaml",
    "configs/mocov3/test.yaml"
]

# List of domains
settings = [
    'selected_tasks',
    # 'bad_tasks',
    # 'all_tasks',
    ]

log_paths = [
    # "/datadrive2/save/meta_dataset/clip/{}",
    # "/datadrive2/save/meta_dataset/dinov2/{}",
    # "/datadrive2/save/meta_dataset/mocov3/{}",
    ]

# Save paths with placeholders
# test_domain = "traffic_sign"
save_paths = [
    "./save/first50/clip/{}/{}/meta-dataset-select_clip_ViT-B32_fs-centroid_15y2s18q_300m_100M",
    "./save/first50/dinov2/{}/{}/meta-dataset-select_dinov2_vitb14_fs-centroid_15y2s18q_300m_100M",
    "./save/first50/mocov3/{}/{}/meta-dataset-select_mocov3_vit_fs-centroid_15y2s18q_300m_100M",
    ]



def parse_args():
    parser = argparse.ArgumentParser(description="This is my training script.")
    parser.add_argument('-d', '--domain', type=str, help='specify domain')
    args = parser.parse_args()
    return args

# Base command
base_cmd = "python test.py"

zero_shot = False

def run_domain(test_domain):
    # Loop over configurations
    for idx, config in enumerate(configs):
        # Extract the model name from the config path for the save path
        model_name = config.split("/")[1]
        print("testing model {}".format(model_name))
        
        # Loop over select
        for id, setting in enumerate(settings):
            if zero_shot:
                log_path = log_paths[idx].format(setting)
                cmd = f"{base_cmd} --config {config} --domains {test_domain} --log_path {log_path}"

            else:
                # Construct the save path by filling the placeholder
                save_path = save_paths[idx].format(test_domain,setting)
                # Construct the full command
                cmd = f"{base_cmd} --config {config} --domains {test_domain} --save_path {save_path}"

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
