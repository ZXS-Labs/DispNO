#!/usr/bin/env bash
set -x
DATAPATH="/dssg/home/scs2010810793/data/benchmark/benchmark/UnrealStereo4K/"
python main_UnrealStereo_DispNO_range_test.py --datapath $DATAPATH --testlist ./filenames/UnrealStereo_test.txt --model gwcnet-gc-dispNO-range \
 --loadckpt ./checkpoints/UnrealStereo/DispNO/piancha_Horizon_def3_best_s2/truetrue8/checkpoint_000275_best.ckpt \
 --logdir ./checkpoints/UnrealStereo/DispNO --dataset UnrealStereoDispNO \
 --out_add 'true' --key_query_same 'true' --deformable_groups 8 \
 --epochs 1 --lrepochs "10,12,14,16:2" \
 --output_representation 'bimodal' --sampling 'dda' \
 --test_scale_min 4 --test_scale_max 4 \
 --start_disp 0 --end_disp 512 \
 