#!/bin/bash
mkdir -p checkpoints
python -u train.py --name ps3d-chairs --stage chairs --validation chairs --gpus 0 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision 
python -u train.py --name ps3d-things --stage things --validation sintel --restore_ckpt checkpoints/ps3d-chairs.pth --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 400 720 --wdecay 0.0001 --mixed_precision
python -u train.py --name ps3d-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/ps3d-things.pth --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --mixed_precision
python -u train.py --name ps3d-kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/ps3d-sintel.pth --gpus 0 --num_steps 50000 --batch_size 5 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --mixed_precision
