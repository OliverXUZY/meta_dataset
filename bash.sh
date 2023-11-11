#############################     zero_shot
# python test.py \
#     --config "configs/clip/test.yaml" \
#     --log_path "/datadrive2/save/meta_dataset/clip" \
#     --domains "quickdraw"

# python finetune.py \
#     --config "configs/clip/finetune.yaml" \
#     --output_path "/datadrive2/save/meta_dataset/clip/mscoco/ilsvrc_2012" \
#     --domains "ilsvrc_2012" --val_domains mscoco \

python test.py \
    --config "configs/mocov3/test.yaml" \
    --save_path "/datadrive2/save/meta_dataset/clip/mscoco/ilsvrc_2012/meta-dataset_clip_ViT-B32_fs-centroid_15y2s18q_300m_100M" \
    --domains "mscoco"



# python finetune.py \
#         --config configs/clip/finetune.yaml \
#         --domains quickdraw,omniglot,ilsvrc_2012    \
#         --val_domains mscoco \
#         --output_path /datadrive2/save/meta_dataset/clip/mscoco/qoif \
#         --path /datadrive2/save/meta_dataset/clip/mscoco/qoif/meta-dataset_clip_ViT-B32_fs-centroid_15y2s18q_300m_100M


