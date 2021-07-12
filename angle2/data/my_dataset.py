import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.image_folder import make_dataset
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform


class MyDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))  # image 绝对路径

        self.input_nc=opt.input_nc
        self.output_nc = opt.output_nc
       # assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):


        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path)  #  修改成删�?.convert('RGB')
        norma = transforms.Normalize((0.5,), (0.5,))
        w, h = AB.size

        w2 = int(w / 3)
        A = AB.crop((0, 0, w2, h))
        A1 = AB.crop((w2, 0, 2*w2, h))
        A2 = AB.crop((2*w2, 0, 3*w2, h))



        # A1 = transforms.ToTensor()(A1)
        # A2 = transforms.ToTensor()(A2)
        # A3 = transforms.ToTensor()(A3)
        #
        # B = transforms.ToTensor()(B)


        #



     #   transform_params = get_params(self.opt, A.size)
     #   A_transform = get_transform(self.opt, transform_params)
#

        A =torch.unsqueeze(torch.from_numpy(np.asarray(A,dtype="float32")),0)/65535
        A1 = torch.unsqueeze(torch.from_numpy(np.asarray(A1, dtype="float32")),0)/65535
        A2 = torch.unsqueeze(torch.from_numpy(np.asarray(A2, dtype="float32")),0)/65535

        
 
        B = torch.cat((A1, A2), 0)

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)

        # if input_nc == 1:  # RGB to gray
        #     tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #     A = tmp.unsqueeze(0)
        #
        # if output_nc == 1:  # RGB to gray
        #     tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #     B = tmp.unsqueeze(0)

        return {'A': A,'A1': A1, 'A2': A2,'B': B, 'A_paths': AB_path}                 #

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'MyDataset'
