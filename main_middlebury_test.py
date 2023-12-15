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
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
from skimage import io
import sys

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='middlebury', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
parser.add_argument('--deformable_groups', type=int, required=True) 
parser.add_argument('--out_add', type=str)
parser.add_argument('--key_query_same', type=str)
# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](d=args.maxdisp, out_add= True if args.out_add == 'true' else False, key_query_same=True if args.key_query_same == 'true' else False, deformable_groups=args.deformable_groups)

model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def test():
    os.makedirs('./predictions_middlebury_HDA_scale4', exist_ok=True)
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs, disp_est_np= test_sample(sample)
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
            disp_vis = disp_est[top_pad:, :-right_pad].astype("float32")
            save_path = "HDA_output/trainingH/" + fn.split('/')[-2]
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            fileName = "HDA_output/trainingH/" + fn.split('/')[-2] + "/disp0HDA.pfm"
            writePFM(fileName, disp_vis)
            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            fn = os.path.join("predictions_middlebury_HDA_scale4", fn.split('/')[-2] + '.png')
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)

    avg_test_scalars = avg_test_scalars.mean()
    print("avg_test_scalars:", avg_test_scalars)

# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()

    imgL, imgR, disp_gt, s, o_shape = sample['left'], sample['right'], sample['disparity'], sample['s'], sample['o_shape']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    s = s.cuda()
    o_shape = o_shape.cuda()

    ret = model(imgL, imgR, s.item(), o_shape)
    disp_ests = [ret[0]]
    # disparity range is defined based on actual disparity for Middlebury
    mask = (disp_gt < 400) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, ret[0]


if __name__ == '__main__':
    test()
