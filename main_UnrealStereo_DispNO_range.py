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
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--start_disp', type=int, default=0, help='maximum disparity')
parser.add_argument('--end_disp', type=int, default=512, help='maximum disparity')

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
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=50, help='the frequency of saving checkpoint')
parser.add_argument('--out_add', type=str)
parser.add_argument('--key_query_same', type=str)
parser.add_argument('--deformable_groups', type=int, required=True)
parser.add_argument('--output_representation', type=str, required=True, help='regressing disparity')
parser.add_argument('--sampling', type=str, default='dda', required=True)
parser.add_argument('--train_scale_min', type=int, default=1)
parser.add_argument('--train_scale_max', type=int, default=1)
parser.add_argument('--test_scale_min', type=int, default=1)
parser.add_argument('--test_scale_max', type=int, default=1)

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
args.logdir = args.logdir + str(args.out_add) + str(args.key_query_same) + str(args.deformable_groups)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True, args.sampling, scale_min=args.train_scale_min, scale_max=args.train_scale_max)
test_dataset = StereoDataset(args.datapath, args.testlist, False, args.sampling, scale_min=args.test_scale_min, scale_max=args.test_scale_max)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](d=args.maxdisp, start_disp=args.start_disp, end_disp=args.end_disp, out_add= True if args.out_add == 'true' else False, key_query_same=True if args.key_query_same == 'true' else False, deformable_groups=args.deformable_groups, output_representation=args.output_representation)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
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
    Thres3 = 1
    EPE = 1
    D1 = 1
    Thres1 = 1
    Thres2 = 1
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
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if epoch_idx % 1 == 0:
            avg_test_scalars = AverageMeterDict()
            for batch_idx, sample in enumerate(TestImgLoader):
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
                if do_summary:
                    save_scalars(logger, 'test', scalar_outputs, global_step)
                    save_images(logger, 'test', image_outputs, global_step)
                avg_test_scalars.update(scalar_outputs)
                print('Epoch {}/{}, Iter {}/{}, test epe = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                        batch_idx,
                                                                                        len(TestImgLoader), scalar_outputs["EPE"][0],

                                                                                        time.time() - start_time))
                del scalar_outputs, image_outputs

            avg_test_scalars = avg_test_scalars.mean()
            if epoch_idx > 250 and avg_test_scalars['Thres3'][0] <= Thres3:
                Thres3 = avg_test_scalars['Thres3'][0]
                Thres1 = avg_test_scalars['Thres1'][0]
                Thres2 = avg_test_scalars['Thres2'][0]
                D1 = avg_test_scalars['D1'][0]
                EPE = avg_test_scalars['EPE'][0]
                checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/checkpoint_{:0>6}_best.ckpt".format(args.logdir, epoch_idx))
            save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
            print("avg_test_scalars", avg_test_scalars , "best_Thres3", Thres3)
            if epoch_idx == args.epochs - 1:
                All = 'EPE:' + str(EPE) + '\nD1:' + str(D1) + '\nThres1:' + str(Thres1) + '\nThres2:' + str(Thres2) + '\nThres3:' + str(Thres3)
                fh = open(args.logdir + '/result.txt', 'a')
                fh.write(All)
                fh.close()
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
    labels = (labels - args.start_disp) / (args.end_disp - args.start_disp)
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
# test one sample
@make_nograd_func

def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt, points, scale, o_shape = sample['left'], sample['right'], sample['disparity'], sample['points'], sample['scale'], sample['o_shape']
    imgL = imgL.cuda() 
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda() 
    points = points.cuda() 
    scale = scale.cuda() 
    o_shape = o_shape.cuda() 

    res = model(imgL, imgR, points, scale, o_shape)
    disp_ests = [res[0]]
    mask = (disp_gt < (args.end_disp)) & (disp_gt > (args.start_disp))

    scalar_outputs = {}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func().forward(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(scalar_outputs), image_outputs
if __name__ == '__main__':
    train()
