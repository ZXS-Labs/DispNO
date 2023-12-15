#!/usr/bin/env bash
set -x
DATAPATH="/dssg/home/scs2010810793/data/benchmark/benchmark/middlebury-offi/trainingH/"
python main_middlebury_DispNO_range_test.py --datapath $DATAPATH --testlist ./filenames/middlebury_test.txt --model gwcnet-gc-dispNO-range \
 --loadckpt ./checkpoints/middlebury/DispNO/piancha_Horizon_def3_best_s3/truetrue8/checkpoint_000397_best.ckpt \
 --logdir ./checkpoints/middlebury/DispNO --dataset middleburyDispNO \
 --out_add 'true' --key_query_same 'true' --deformable_groups 8 \
 --output_representation 'bimodal' --sampling 'dda' --epochs 1 --lrepochs "10,12,14,16:2" \
 --scale_min 5 --scale_max 5 \
 --start_disp 15 --end_disp 303 \
