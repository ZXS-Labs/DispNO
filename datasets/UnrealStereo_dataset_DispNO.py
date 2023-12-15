import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, get_boundaries, scale_coords, npy_imread

class UnrealStereoDatsetDispNO(Dataset):
    def __init__(self, datapath, list_filename, training, sampling, scale_min=1, scale_max=1):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.sampler = sampling
        self.num_sample_inout = 50000
        self.dilation_factor = 10
        self.scale_min = scale_min
        self.scale_max = scale_max
        if self.training:
            self.img_height = 256
            self.img_width = 512
            self.lr_height = 64
            self.lr_width = 128

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        return npy_imread(filename)

    def __len__(self):
        return len(self.left_filenames)
    
    def __init_grid(self):
        nu = np.linspace(0, self.crop_width - 1, self.crop_width)
        nv = np.linspace(0, self.crop_height - 1, self.crop_height)
        u, v = np.meshgrid(nu, nv)

        self.u = u.flatten()
        self.v = v.flatten()
    
    def __get_coords(self, gt):
        #  Subpixel coordinates
        u = self.u + np.random.random_sample(self.u.size)
        v = self.v + np.random.random_sample(self.v.size)

        # Nearest neighbor
        d = gt[np.clip(np.rint(v).astype(np.uint16), 0, self.crop_height-1),
                 np.clip(np.rint(u).astype(np.uint16), 0, self.crop_width-1)]

        # Remove invalid disparitiy values
        u = u[np.nonzero(d)]
        v = v[np.nonzero(d)]
        d = d[np.nonzero(d)]

        return np.stack((u, v, d), axis=-1)
    
    def sampling(self, disparity):
        gt = disparity.data.numpy().squeeze()

        if self.sampler == "random":
            random_points = self.__get_coords(gt)
            idx = np.random.choice(random_points.shape[0], self.num_sample_inout) # (50000)
            points = random_points[idx, :] # (50000, 3)

        elif self.sampler == "dda":
            edges = get_boundaries(gt, dilation=self.dilation_factor)
            random_points = self.__get_coords(gt * (1. - edges))
            edge_points = self.__get_coords(gt * edges)

            # if edge points exist
            if edge_points.shape[0]>0:

                # Check tot num of edge points
                cond = edges.sum()//2 -  self.num_sample_inout//2 < 0
                if cond:
                    tot= (self.num_sample_inout - int(edges.sum())//2, int(edges.sum())//2)
                elif random_points.shape[0] < self.num_sample_inout//2:
                    tot = (random_points.shape[0], self.num_sample_inout - random_points.shape[0])
                else:
                    tot = (self.num_sample_inout//2, self.num_sample_inout//2)

                if random_points.shape[0] == 0:
                    idx_edges = np.random.choice(edge_points.shape[0], self.num_sample_inout)
                    points = edge_points[idx_edges, :]
                else:
                    idx = np.random.choice(random_points.shape[0], tot[0])
                    idx_edges = np.random.choice(edge_points.shape[0], tot[1])
                    points = np.concatenate([edge_points[idx_edges, :], random_points[idx, :]], 0)
            # use uniform sample otherwise
            else:
                random_points = self.__get_coords(gt)
                idx = np.random.choice(random_points.shape[0], self.num_sample_inout)
                points = random_points[idx, :]

        return np.array(points.T, dtype=np.float32), np.array(points[:,2:3].T, dtype=np.float32)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        s = random.uniform(self.scale_min, self.scale_max)

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = round(self.lr_width * s), round(self.lr_height * s)
            self.crop_width = crop_w
            self.crop_height = crop_h

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            left_img = left_img.resize((self.img_width, self.img_height), Image.BILINEAR)
            right_img = right_img.resize((self.img_width, self.img_height), Image.BILINEAR)

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            disparity = np.expand_dims(disparity, 0)
            disparity = torch.from_numpy(disparity.copy()).float()
            disparity = disparity.permute(1, 2, 0)
            
            o_shape = torch.from_numpy(np.asarray((crop_h, crop_w)).copy())

            self.__init_grid()
            samples, labels = self.sampling(disparity)
            samples = torch.from_numpy(samples)

            # Coordinated between [-1, 1]
            u = scale_coords(samples[0:1, :], self.crop_width)
            v = scale_coords(samples[1:2, :], self.crop_height)
            points = torch.cat([u,v],0)

            return {"left": left_img,
                    "right": right_img,
                    "o_shape": o_shape, # torch.Size([2])
                    "samples": points, # (3, 50000)
                    "labels": labels, # (1, 50000)
                    "scale": s,
                    }
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 540

            o_shape = torch.from_numpy(np.asarray((crop_h, crop_w)).copy())

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            disparity = torch.from_numpy(disparity.copy()).float()

            img_width = int(crop_w // s * 4)
            img_height = int(crop_h // s * 4)
            left_img = left_img.resize((img_width, img_height), Image.BILINEAR)
            right_img = right_img.resize((img_width, img_height), Image.BILINEAR)

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            nx = np.linspace(0, crop_w - 1, crop_w)
            ny = np.linspace(0, crop_h - 1, crop_h)
            u, v = np.meshgrid(nx, ny)

            coords = np.stack((u.flatten(), v.flatten()), axis=-1)
            coords = torch.Tensor(coords).float()

            # Coordinated between [-1, 1]
            u = scale_coords(coords[:, 0:1], crop_w)
            v = scale_coords(coords[:, 1:2], crop_h)
            coords = torch.cat([u,v],1)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "points": coords,
                    "o_shape": o_shape,
                    "scale": s,
                    "top_pad": 0,
                    "right_pad": 0,
                    "left_filename": self.left_filenames[index],
                    "right_filename": self.right_filenames[index]}