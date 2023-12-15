import numpy as np
import re
import torchvision.transforms as transforms
import torch
import cv2


def get_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def get_boundaries(disp, th=1., dilation=10):
    edges_y = np.logical_or(np.pad(np.abs(disp[1:, :] - disp[:-1, :]) > th, ((1, 0), (0, 0)), mode='constant'),
                            np.pad(np.abs(disp[:-1, :] - disp[1:, :]) > th, ((0, 1), (0, 0)), mode='constant'))
    edges_x = np.logical_or(np.pad(np.abs(disp[:, 1:] - disp[:, :-1]) > th, ((0, 0), (1, 0)), mode='constant'),
                            np.pad(np.abs(disp[:, :-1] - disp[:,1:]) > th, ((0, 0), (0, 1)), mode='constant'))
    edges = np.logical_or(edges_y,  edges_x).astype(np.float32)

    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges

def scale_coords(points, max_length):
    return torch.clamp(2 * points/(max_length-1.)- 1., -1., 1.)

def npy_imread(path):
    gt=None
    try:
        gt = np.load(path, mmap_mode='c')
    except:
        print('Cannot open groundtruth depth: '+ path)

    # Remove invalid values
    gt[np.isinf(gt)] = 0

    return gt

