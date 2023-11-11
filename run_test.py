import subprocess

# List of configurations
configs = [
    "configs/clip/test.yaml",
    # "configs/dinov2/test.yaml",
    # "configs/mocov3/test.yaml"
]

# List of domains
domains = [
    # 'quickdraw', 
    # 'omniglot', 
    # 'ilsvrc_2012', 
    # 'aircraft', 
    # 'fungi', 
    # 'cu_birds', 
    # 'vgg_flower', 
    # 'traffic_sign',
    # 'mscoco',
    # "omniglot",
    # "iq",
    # "iqofv",
    # "iqofva",
    # "iqofvac",
    # "qo","qov","qovf","qovfc","qovfca"
    # 'ilsvrc_2012', 'ifvaoq'
    # 'oi','oiv','oivf','oivfc','oivfca'
    # "fungi","fi","fc",'fci','fcio','fcioq','fcioqa'
    'selected_tasks'
    ]

log_paths = [
    # "/datadrive2/save/meta_dataset/clip/{}",
    # "/datadrive2/save/meta_dataset/dinov2/{}",
    # "/datadrive2/save/meta_dataset/mocov3/{}",
    ]

# Save paths with placeholders
test_domain = "traffic_sign"
save_paths = [
    # "/datadrive2/save/meta_dataset/clip/{}/{}/meta-dataset_clip_ViT-B32_fs-centroid_15y2s18q_300m_100M",
    # "/datadrive2/save/meta_dataset/dinov2/{}/{}/meta-dataset_dinov2_vitb14_fs-centroid_15y2s18q_300m_100M",
    # "/datadrive2/save/meta_dataset/mocov3/{}/{}/meta-dataset_mocov3_vit_fs-centroid_15y2s18q_300m_100M",
    ]





# Base command
base_cmd = "python test.py"

zero_shot = False


# Loop over configurations
for idx, config in enumerate(configs):
    # Extract the model name from the config path for the save path
    model_name = config.split("/")[1]
    print("testing model {}".format(model_name))
    
    # Loop over domains
    for id, domain in enumerate(domains):
        if zero_shot:
            log_path = log_paths[idx].format(domain)
            cmd = f"{base_cmd} --config {config} --domains {test_domain} --log_path {log_path}"

        else:
            # Construct the save path by filling the placeholder
            save_path = save_paths[idx].format(test_domain,domain)
            # Construct the full command
            cmd = f"{base_cmd} --config {config} --domains {test_domain} --save_path {save_path}"

        print(cmd)
        
        # Execute the command
        subprocess.run(cmd, shell=True)

# python test.py --config "configs/mocov3/mini-imagenet/test.yaml" --save_path "/datadrive/select_data/save/mocov3/domain-net/select/rpq/select-domain-net_mocov3_vit_fs-centroid_15y2s18q_300m_100M"