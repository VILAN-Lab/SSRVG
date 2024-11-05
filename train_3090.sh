##!/bin/bash
#source ~/.bashrc
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate tmm_msvlt
#nohup ./train_3090.sh > ./log/refcocog5.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=1
# ReferItGame
#python -u -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --batch_size 16 --epochs 90 --aug_crop --aug_scale --aug_translate --dataset referit --max_query_len 30 --output_dir outputs/referit_kca7_64
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25985 --use_env train.py --batch_size 16 --epochs 10 --aug_scale --aug_translate --dataset referit --max_query_len 40 --resume /home/user/D/32T/zsy/tmm_msvlt/outputs/five_total_train/checkpoint.pth --output_dir outputs/five_total_train_finetune_referit --lr 1e-6 --lr_bert 1e-8 --lr_visu_cnn 1e-8
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25985 --use_env train.py --batch_size 16 --epochs 40 --aug_scale --aug_translate --dataset referit --max_query_len 40 --output_dir outputs/finetune_referit_120

#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25985 --use_env eval.py --batch_size 40 --dataset referit --max_query_len 40 --eval_model /home/user/D/32T/zsy/tmm_msvlt/outputs/finetune_referit_120/best_checkpoint.pth --eval_set test --output_dir outputs/finetune_referit_120_ca6

# # RefCOCO
#python -u -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --batch_size 16 --epochs 90 --aug_crop --aug_scale --aug_translate --dataset unc --max_query_len 30 --output_dir outputs/refcoco_kca7_64
#python -u train.py --batch_size 32 --epochs 90 --aug_crop --aug_scale --aug_translate --dataset unc --max_query_len 30 --output_dir outputs/refcoco_kca7 --resume outputs/refcoco_kca7/checkpoint.pth
#python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --batch_size 64 --lr 1e-4 --aug_crop --aug_scale --aug_translate --dataset unc --max_query_len 30 --output_dir outputs/refcoco_pe_negative_nobert --no_amp --epochs 20 --optimizer rmsprop --lr_scheduler step --lr_drop 15
#python -u -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --resume /home/user/D/32T/zsy/tmm_msvlt/outputs/swin_small_scoremodule_twolevel_refcocog_chuanlian_two_loss_thc/checkpoint.pth --batch_size 16 --epochs 120 --aug_scale --aug_translate --dataset unc --max_query_len 30 --output_dir outputs/swin_small_scoremodule_twolevel_refcocog_chuanlian_two_loss_thc
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25985 --use_env train.py --batch_size 16 --epochs 10 --aug_scale --aug_translate --dataset unc --max_query_len 40 --resume /home/user/D/32T/zsy/tmm_msvlt/outputs/five_total_train/checkpoint.pth --output_dir outputs/five_total_train_finetune_coco --lr 1e-6 --lr_bert 1e-8 --lr_visu_cnn 1e-8



# # RefCOCO+
#python -u -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --batch_size 16 --epochs 120 --erase 0.1 --aug_crop --aug_scale --aug_translate --dataset unc+ --max_query_len 30 --output_dir outputs/refcocop_kca7_64 --resume outputs/refcocop_kca7_64/checkpoint.pth
#python -u -m torch.distributed.launch --nproc_per_node=8 --master_port 25985 --use_env train.py --batch_size 16 --epochs 40 --aug_scale --aug_translate --dataset unc+ --max_query_len 40 --resume /home/user/D/32T/zsy/tmm_msvlt/outputs/total_train/checkpoint0086.pth --resume_scheduler 1 --lr 1e-5 --output_dir outputs/total_train_unc+
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25985 --use_env train.py --batch_size 16 --epochs 10 --aug_scale --aug_translate --dataset unc+ --max_query_len 40 --resume /home/user/D/32T/zsy/tmm_msvlt/outputs/five_total_train/checkpoint.pth --output_dir outputs/five_total_train_finetune_cocop --lr 1e-6 --lr_bert 1e-8 --lr_visu_cnn 1e-8
# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --aug_scale --aug_translate --aug_crop --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50
#python -u -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 16  --epochs 90 --aug_crop --aug_scale --aug_translate --dataset gref --max_query_len 40 --output_dir outputs/refcocogg_kca7_64
# # RefCOCOg umd-split
#python train.py --batch_size 64 --aug_scale --aug_translate --aug_crop --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_usplit_r101_a100_nowpa --no_aux_loss --no_amp --no_wpa_loss
#python -u -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --resume /home/user/D/zpz/tmm_msvlt/outputs/refcocog_5_12/checkpoint.pth --batch_size 6 --epochs 120 --aug_scale --aug_translate --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_5_12
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25985 --use_env train.py --batch_size 16 --epochs 10 --aug_scale --aug_translate --dataset gref_umd --max_query_len 40 --resume /home/user/D/32T/zsy/tmm_msvlt/outputs/five_total_train/checkpoint.pth  --output_dir outputs/five_total_train_finetune_cocog --lr 1e-6 --lr_bert 1e-8 --lr_visu_cnn 1e-8

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25985 --use_env train.py --batch_size 16 --epochs 40 --aug_scale --aug_translate --dataset gref_umd --max_query_len 40 --dec_layers 1 --output_dir outputs/cocog_decoder1

#python -u -m torch.distributed.launch --nproc_per_node=8 --master_port 25985 --use_env train.py --batch_size 16 --epochs 40 --aug_scale --aug_translate --dataset gref_umd --max_query_len 40 --resume /home/user/D/32T/zsy/tmm_msvlt/outputs/total_train/checkpoint0086.pth --resume_scheduler 1 --lr 1e-5 --output_dir outputs/total_train_cocog

# # Flickr
#python -u -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --batch_size 32 --epochs 90 --aug_crop --aug_scale --aug_translate --dataset flickr --max_query_len 15 --output_dir outputs/flickr_kca 7 --resume outputs/flickr_kca7/checkpoint.pth
#python -u -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 16 --epochs 90 --aug_crop --aug_scale --aug_translate --dataset flickr --max_query_len 15 --resume outputs/swin_small_scoremodule_twolevel_flickr_chuanlian_two_loss_thc/best_checkpoints.pth  --output_dir outputs/swin_bas
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 16 --epochs 120 --aug_crop --aug_scale --aug_translate --dataset flickr --max_query_len 15 --output_dir outputs/flickr_chuanlian_two_loss_thc
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25985 --use_env train.py --batch_size 16 --epochs 90 --aug_scale --aug_translate --dataset flickr --max_query_len 40 --output_dir outputs/finetune_flickr

# eval
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -u -m torch.distributed.launch --nproc_per_node=6 --master_port 25985 --use_env eval.py --batch_size 16 --epochs 90 --dataset unc+ --max_query_len 40 --output_dir outputs/total_train --eval_set val --eval_model /home/user/D/32T/zsy/tmm_msvlt/outputs/total_train/best_checkpoint.pth
#!/bin/bash

# total test
# 导出 eval_model 环境变量
#export eval_model=/home/user/D/32T/zsy/tmm_msvlt/outputs/five_total_train/best_checkpoint.pth
#export outputs=outputs/five_total_train
#CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25982 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset unc --eval_set val &
#CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25983 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset unc --eval_set testA &
#CUDA_VISIBLE_DEVICES=2 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25984 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset unc --eval_set testB &
#CUDA_VISIBLE_DEVICES=3 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25985 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset unc+ --eval_set val &
#CUDA_VISIBLE_DEVICES=4 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25986 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset unc+ --eval_set testA &
#CUDA_VISIBLE_DEVICES=5 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25987 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset unc+ --eval_set testB &
#CUDA_VISIBLE_DEVICES=6 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25988 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset gref_umd --eval_set val &
#CUDA_VISIBLE_DEVICES=7 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25989 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset gref_umd --eval_set test &

#CUDA_VISIBLE_DEVICES=7 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25951 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset flickr --eval_set test &
#CUDA_VISIBLE_DEVICES=5 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 25952 --use_env eval.py --batch_size 128 --max_query_len 40 --eval_model $eval_model --output_dir outputs --dataset referit --eval_set test &

