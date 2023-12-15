#!/usr/bin/env bash
set -x
DATAPATH="/dssg/home/scs2010810793/data/benchmark/benchmark/UnrealStereo4K/"
python main_UnrealStereo_test.py --datapath $DATAPATH --testlist ./filenames/UnrealStereo_test.txt --model gwcnet-gc \
 --loadckpt ./checkpoints/UnrealStereo/HDA/piancha_Horizon_def3_best_s1/truetrue8/checkpoint_000277_best.ckpt \
 --logdir ./checkpoints/UnrealStereo/HDA --dataset UnrealStereo \
 --out_add 'true' --key_query_same 'true' --deformable_groups 8 \
 --epochs 1 --lrepochs "10,12,14,16:2" \
 