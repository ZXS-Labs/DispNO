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
import os
import skimage
from skimage import io

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--out_add', type=str)
parser.add_argument('--key_query_same', type=str)
parser.add_argument('--deformable_groups', type=int, required=True)

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)
# create summary logger
print("creating new summary file")
args.logdir = args.logdir + "/" + str(args.out_add) + str(args.key_query_same) + str(args.deformable_groups)
logger = SummaryWriter(args.logdir)
args.loadckpt = './checkpoints/sceneflow/HDA/piancha_def_Horizon3/truetrue8/checkpoint_000015.ckpt'

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](d=args.maxdisp, out_add= True if args.out_add == 'true' else False, key_query_same=True if args.key_query_same == 'true' else False, deformable_groups=args.deformable_groups)
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

def test():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        # test
        os.makedirs('./predictions_sceneflow_hda_scale5', exist_ok=True)
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            do_summary = True
            loss, scalar_outputs, image_outputs, disp_est_np = test_sample(sample, compute_metrics=do_summary)

            disp_est_np = tensor2numpy(disp_est_np)
            top_pad_np = tensor2numpy(sample["top_pad"])
            right_pad_np = tensor2numpy(sample["right_pad"])
            left_filenames = sample["left_filename"]
            save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
            if batch_idx % 10 == 0:
                for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
                    assert len(disp_est.shape) == 2
                    disp_est = np.array(disp_est, dtype=np.float32)
                    fn_arr = fn.split('/')
                    fn = os.path.join("predictions_sceneflow_hda_scale5", fn_arr[2] + '-' + fn_arr[3] + '-' + fn_arr[5])
                    print("saving to", fn, disp_est.shape)
                    disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
                    skimage.io.imsave(fn, disp_est_uint)

        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TestImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        EPE = str(avg_test_scalars['EPE'][0])
        Thres1 = str(avg_test_scalars['Thres1'][0])
        Thres2 = str(avg_test_scalars['Thres2'][0])
        Thres3 = str(avg_test_scalars['Thres3'][0])
        f = open(args.logdir + '/result.txt', 'a')
        f.write(EPE + "\n" + Thres1 + "\n" + Thres2  + "\n" + Thres3 + "\n")
        gc.collect()

# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt, s, o_shape = sample['left'], sample['right'], sample['disparity'], sample['s'], sample['o_shape']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    s = s.cuda()
    o_shape = o_shape.cuda()
     
    ret = model(imgL, imgR, s.item(), o_shape)
    disp_ests = [ret[0]]
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, ret[0]


if __name__ == '__main__':
    test()

