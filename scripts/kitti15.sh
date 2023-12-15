#!/usr/bin/env bash
set -x
DATAPATH="/dssg/home/scs2010810793/data/benchmark/benchmark/kitti/2015/"

python main_kitti15.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-gc --logdir ./checkpoints/kitti15/HDA/piancha_Horizon_def3_best_s1/ \
    --loadckpt ./checkpoints/sceneflow/HDA/piancha_def_Horizon3/truetrue8/checkpoint_000015.ckpt \
    --test_batch_size 1 \
    --out_add 'true' --key_query_same 'true' --deformable_groups 8

python main_kitti15.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-gc --logdir ./checkpoints/kitti15/HDA/piancha_Horizon_def3_best_s2/ \
    --loadckpt ./checkpoints/sceneflow/HDA/piancha_def_Horizon3/truetrue8/checkpoint_000015.ckpt \
    --test_batch_size 1 \
    --out_add 'true' --key_query_same 'true' --deformable_groups 8

python main_kitti15.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --epochs 300 --lrepochs "200:10" \
    --model gwcnet-gc --logdir ./checkpoints/kitti15/HDA/piancha_Horizon_def3_best_s3/ \
    --loadckpt ./checkpoints/sceneflow/HDA/piancha_def_Horizon3/truetrue8/checkpoint_000015.ckpt \
    --test_batch_size 1 \
    --out_add 'true' --key_query_same 'true' --deformable_groups 8

