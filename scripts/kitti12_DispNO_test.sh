#!/usr/bin/env bash
set -x
DATAPATH="/dssg/home/scs2010810793/data/benchmark/benchmark/kitti/2012/"
python main_kitti_DispNO_test.py --dataset kittiDispNO --datapath $DATAPATH \
    --testlist ./filenames/kitti12_train.txt --model gwcnet-gc-dispNO \
    --loadckpt ./checkpoints/kitti12/DispNO/piancha_Horizon_def3_best_s2/truetrue8/checkpoint_000267_best.ckpt \
    --out_add 'true' --key_query_same 'true' --deformable_groups 8 \
    --output_representation 'bimodal' --sampling 'dda' \
    --scale_min 5 --scale_max 5 \

