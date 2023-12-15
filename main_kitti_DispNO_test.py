from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
from skimage import io
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
parser.add_argument('--deformable_groups', type=int, required=True) 
parser.add_argument('--out_add', type=str)
parser.add_argument('--key_query_same', type=str)
parser.add_argument('--output_representation', type=str, required=True, help='regressing disparity')
parser.add_argument('--sampling', type=str, default='dda', required=True)
parser.add_argument('--scale_min', type=int, default=1)
parser.add_argument('--scale_max', type=int, default=1)

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False, args.sampling, scale_min=args.scale_min, scale_max=args.scale_max)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](d=args.maxdisp, out_add= True if args.out_add == 'true' else False, key_query_same=True if args.key_query_same == 'true' else False, deformable_groups=args.deformable_groups, output_representation=args.output_representation)

model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

def test():
    os.makedirs('./predictions_kitti15_DispNO_scale5', exist_ok=True)
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        scalar_outputs, image_outputs, disp_est_np = test_sample(sample)
        disp_est_np = tensor2numpy(disp_est_np)
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        
        print("scalar_outputs:", scalar_outputs)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            fn = os.path.join("predictions_kitti15_DispNO_scale5", fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)

    avg_test_scalars = avg_test_scalars.mean()
    print("avg_test_scalars:", avg_test_scalars)


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()

    imgL, imgR, disp_gt, points, scale, o_shape = sample['left'], sample['right'], sample['disparity'], sample['points'], sample['scale'], sample['o_shape']
    imgL = imgL.cuda() # torch.Size([1, 3, 512, 960])
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda() # torch.Size([1, 512, 960])
    points = points.cuda() # torch.Size([1, 491520, 2])
    scale = scale.cuda() # torch.Size([16])
    o_shape = o_shape.cuda() # torch.Size([1, 2])

    res = model(imgL, imgR, points, scale, o_shape)
    disp_ests = [res[0]]
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    scalar_outputs = {}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(scalar_outputs), image_outputs, res[0]


if __name__ == '__main__':
    test()
