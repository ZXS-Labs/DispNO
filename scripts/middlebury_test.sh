#!/usr/bin/env bash
set -x
DATAPATH="/dssg/home/scs2010810793/data/benchmark/benchmark/middlebury-offi/trainingH/"
python main_middlebury_test.py --datapath $DATAPATH --testlist ./filenames/middlebury_test.txt --model gwcnet-gc \
 --loadckpt ./checkpoints/middlebury/HDA/piancha_Horizon_def3_best_s3/truetrue8/checkpoint_000351_best.ckpt \
 --out_add 'true' --key_query_same 'true' --deformable_groups 8
