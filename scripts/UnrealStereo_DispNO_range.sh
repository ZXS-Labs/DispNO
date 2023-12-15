#!/usr/bin/env bash
set -x
DATAPATH="/dssg/home/scs2010810793/data/benchmark/benchmark/UnrealStereo4K/"
python main_UnrealStereo_DispNO_range.py --dataset UnrealStereoDispNO \
    --datapath $DATAPATH --trainlist ./filenames/UnrealStereo_train.txt --testlist ./filenames/UnrealStereo_val.txt \
    --epochs 300 --lrepochs "200, 600:10" \
    --model gwcnet-gc-dispNO-range --logdir ./checkpoints/UnrealStereo/DispNO/piancha_Horizon_def3_best_s1/\
    --loadckpt ./checkpoints/sceneflow/DispNO/truetrue8/checkpoint_000015.ckpt \
    --test_batch_size 1 \
    --out_add 'true' --key_query_same 'true' --deformable_groups 8 \
    --output_representation 'bimodal' --sampling 'dda' \
    --train_scale_min 4 --train_scale_max 7 --test_scale_min 4 --test_scale_max 4 \
    --start_disp 0 --end_disp 512 \

python main_UnrealStereo_DispNO_range.py --dataset UnrealStereoDispNO \
    --datapath $DATAPATH --trainlist ./filenames/UnrealStereo_train.txt --testlist ./filenames/UnrealStereo_val.txt \
    --epochs 300 --lrepochs "200, 600:10" \
    --model gwcnet-gc-dispNO-range --logdir ./checkpoints/UnrealStereo/DispNO/piancha_Horizon_def3_best_s2/ \
    --loadckpt ./checkpoints/sceneflow/DispNO/truetrue8/checkpoint_000015.ckpt \
    --test_batch_size 1 \
    --out_add 'true' --key_query_same 'true' --deformable_groups 8 \
    --output_representation 'bimodal' --sampling 'dda' \
    --train_scale_min 4 --train_scale_max 7 --test_scale_min 4 --test_scale_max 4 \
    --start_disp 0 --end_disp 512 \

python main_UnrealStereo_DispNO_range.py --dataset UnrealStereoDispNO \
    --datapath $DATAPATH --trainlist ./filenames/UnrealStereo_train.txt --testlist ./filenames/UnrealStereo_val.txt \
    --epochs 300 --lrepochs "200, 600:10" \
    --model gwcnet-gc-dispNO-range --logdir ./checkpoints/UnrealStereo/DispNO/piancha_Horizon_def3_best_s3/ \
    --loadckpt ./checkpoints/sceneflow/DispNO/truetrue8/checkpoint_000015.ckpt \
    --test_batch_size 1 \
    --out_add 'true' --key_query_same 'true' --deformable_groups 8 \
    --output_representation 'bimodal' --sampling 'dda' \
    --train_scale_min 4 --train_scale_max 7 --test_scale_min 4 --test_scale_max 4 \
    --start_disp 0 --end_disp 512 \

