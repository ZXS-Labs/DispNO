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
from models import __models__, unimodal_loss, bimodal_loss
from utils import *
from torch.utils.data import DataLoader
import gc
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', type=str)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--out_add', type=str)
parser.add_argument('--key_query_same', type=str)
parser.add_argument('--deformable_groups', type=int, required=True)
parser.add_argument('--output_representation', type=str, required=True, help='regressing disparity')
parser.add_argument('--sampling', type=str, default='dda', required=True)
parser.add_argument('--scale_min', type=int, default=1)
parser.add_argument('--scale_max', type=int, default=1)

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
args.logdir = args.logdir + str(args.out_add) + str(args.key_query_same) + str(args.deformable_groups)
print(args.logdir)
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True, args.sampling, scale_min=args.scale_min, scale_max=args.scale_max)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)

# model, optimizer
model = __models__[args.model](d=args.maxdisp, out_add= True if args.out_add == 'true' else False, 
                               key_query_same=True if args.key_query_same == 'true' else False, 
                               deformable_groups=args.deformable_groups, output_representation=args.output_representation)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume == 'true':
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))

def train():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            del scalar_outputs

            print('Epoch {}/{}, Iter {}/{}, disp loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                      batch_idx,
                                                                                      len(TrainImgLoader), loss,
                                                                                      time.time() - start_time))
       # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, points, labels, scale = sample['left'], sample['right'], sample['samples'], sample["labels"], sample['scale']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    points = points.cuda()
    labels = labels.cuda()
    scale = scale.cuda() # torch.Size([16])
    optimizer.zero_grad()
    disp_ests = model(imgL, imgR, points, scale)
    labels = labels / args.maxdisp
    mask = torch.mul(labels > 0, labels <= 1.)
    if args.output_representation == "bimodal":
        loss = bimodal_loss(disp_ests['mu0'][mask], disp_ests['mu1'][mask], disp_ests['sigma0'][mask], disp_ests['sigma1'][mask],
                            disp_ests['pi0'][mask], disp_ests['pi1'][mask], labels[mask], dist="laplacian").mean()
    elif args.output_representation == "unimodal":
        loss = unimodal_loss(disp_ests['disp'][mask], disp_ests['var'][mask], labels[mask]).mean()
    else:
        loss = torch.abs(disp_ests['disp'][mask] - labels[mask]).mean()

    scalar_outputs = {"loss": loss}
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)

if __name__ == '__main__':
    train()

