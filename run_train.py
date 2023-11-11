import subprocess

# List of configurations
configs = [
    "configs/clip/finetune.yaml",
    "configs/dinov2/finetune.yaml",
    "configs/mocov3/finetune.yaml",
]


# List of domains
domains = [
    # "quickdraw",
    # "omniglot",
    # "quickdraw,ilsvrc_2012",
    # "quickdraw,ilsvrc_2012,aircraft",
    # "quickdraw,ilsvrc_2012,fungi,aircraft",
    # "quickdraw,ilsvrc_2012,omniglot,fungi,vgg_flower",
    # "quickdraw,ilsvrc_2012,omniglot,fungi,vgg_flower,aircraft",
    # "quickdraw,omniglot",
    # "quickdraw,omniglot,vgg_flower",
    # "quickdraw,omniglot,vgg_flower,fungi",
    # "quickdraw,omniglot,vgg_flower,fungi,cu_birds",
    # "quickdraw,omniglot,vgg_flower,fungi,cu_birds,aircraft",
    # "quickdraw,ilsvrc_2012,omniglot,fungi,vgg_flower,aircraft,cu_birds",
    # "ilsvrc_2012",
    # "ilsvrc_2012,cu_birds",
    # "omniglot,ilsvrc_2012",
    # "omniglot,ilsvrc_2012,vgg_flower",
    # "omniglot,ilsvrc_2012,vgg_flower,fungi",
    # "omniglot,ilsvrc_2012,vgg_flower,fungi,cu_birds",
    # "omniglot,ilsvrc_2012,vgg_flower,fungi,cu_birds,aircraft",
    "fungi",
    'fungi,ilsvrc_2012',
    'fungi,cu_birds',
    'fungi,cu_birds,ilsvrc_2012',
    'fungi,cu_birds,ilsvrc_2012,omniglot',
    'fungi,cu_birds,ilsvrc_2012,omniglot,quickdraw',
    "quickdraw,ilsvrc_2012,omniglot,cu_birds,fungi,aircraft",
   ]
save_names = [
    # "quickdraw",
    # "omniglot",
    # "iq",
    # "iqofv",
    # "iqofva",
    # "iqofvac",
    # "ilsvrc_2012",
    # "all"
    # "qo","qov","qovf","qovfc","qovfca"
    # 'ilsvrc_2012', 'ifvaoq'
    # 'oi','oiv','oivf','oivfc','oivfca',
    "fungi","fi","fc",'fci','fcio','fcioq','fcioqa'
    
]
    
load_model_paths = [
    "/datadrive2/save/meta_dataset/clip/mscoco/{}/meta-dataset_clip_ViT-B32_fs-centroid_15y2s18q_300m_100M",
    "/datadrive2/save/meta_dataset/dinov2/mscoco/{}/meta-dataset_dinov2_vitb14_fs-centroid_15y2s18q_300m_100M",
    "/datadrive2/save/meta_dataset/mocov3/mscoco/{}/meta-dataset_mocov3_vit_fs-centroid_15y2s18q_300m_100M",
    ]
# Base command
# base_cmd = "python finetune_select.py"
base_cmd = "python finetune.py"
val_domains = 'vgg_flower'
from_scratch = True


# Loop over configurations
for idx, config in enumerate(configs):
    # Extract the model name from the config path for the save path
    model_name = config.split("/")[1]
    print("finetuning model {}".format(model_name))
    
    # Loop over domains
    for domain, save_name in zip(domains, save_names):
        # Construct the save path
        save_path = f"/datadrive2/save/meta_dataset/{model_name}/{val_domains}/{save_name}" 

        ## all domains train together
        # save_path = f"/datadrive2/save/meta_dataset/{model_name}/{save_name}"    
        
        
        # Construct the full command
        cmd = f"{base_cmd} --config {config} --domains {domain} --val_domains {val_domains} --output_path {save_path}"

        ## all domains train together
        # cmd = f"{base_cmd} --config {config} --domains {domain} --val_domains {val_domains} --output_path {save_path} --n_batch_train 200"

        if not from_scratch:
            load_model_path = load_model_paths[idx].format(save_name)
            cmd += " --path {}".format(load_model_path)
        print(cmd)
        
        # Execute the command
        subprocess.run(cmd, shell=True)
