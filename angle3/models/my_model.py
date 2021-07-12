import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pandas as pd
#from re_adam import RAdam
from . import re_adam
from math import exp,log
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=2):
    _1D_window = gaussian(window_size, 8).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 2, window_size, window_size).contiguous()
    return window


def gaussian_conv(img,window_size, sigma):
    filter_=create_window(window_size).cuda()
    return F.conv2d(img,filter_,padding=[window_size//2,window_size//2])
    
def gauss_weighted_l1(img1,img2,window_size, sigma):
    diff=torch.abs(img1-img2)  #
    l1=gaussian_conv(diff,window_size,sigma)
    return l1
    
class MyModel(BaseModel):
    def name(self):
        return 'MyModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0, no_lsgan=False, norm='batch')
        # parser.set_defaults(dataset_mode='aligned')
        # parser.set_defaults(netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.gan_loss=[]
        self.fft_loss=[]
        self.l1_loss=[]
        self.ssim_loss=[]
        self.g_loss=[]
        self.discri_loss=[]
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = [ 'D_real', 'D_fake','G_GAN', 'G_L1','G_ssim','G_fft']   #'G_ssim'
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B'] #'real_A1', 'real_A2',
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
          #  ssim  
        #self.ms_ssim_module = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=False, channel=2)
        # load/define networks
        self.ssim_module = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=False, channel=2)               # ssim_module
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
      #  print(self.netG)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
      #  print(self.netD)
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
       
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        # AtoB = self.opt.direction == 'AtoB'
        #  real_a   [1,3,256,256]          #  real_b
        self.real_A = input['A' ].to(self.device)     # [1,1,1024,1024]          # if AtoB else 'B'
       # self.real_A1 = input['A1'].to(self.device)
        #self.real_A2 = input['A2'].to(self.device)
        self.real_B = input['B' ].to(self.device)            # if AtoB else 'A'
        self.image_paths = input['A_paths']             #  if AtoB else 'B_paths'

    def forward(self):
       
        self.fake_B = self.netG(self.real_A)
      
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B

        #  fake_AB [1,4,256,256]     pred_fake [1,1,30,30]
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        self.real_AB = torch.cat((self.real_A, self.real_B), 1)   
        pred_real = self.netD(self.real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.discri_loss.append(self.loss_D.item())
        
        plt.figure()
        plt.plot(self.discri_loss)
        plt.savefig('./loss fig/'+'discri_loss.png',dpi=600)
        plt.close()
        discri_list=pd.DataFrame(data={'discri_loss':self.discri_loss})
        discri_list.to_csv('./loss/discri_loss.csv',index=False)
        
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        fft_fake=torch.rfft(self.fake_B,2)
        
        fft_real=torch.rfft(self.real_B,2)
        self.loss_G_fft = self.criterionL1(fft_fake, fft_real) 
        # Second, G(A) = B
      #  wf_fake=torch.sum(fake_AB,1)
     #   wf_real=torch.sum(self.real_AB,1)
     #   self.loss_G_MSE = self.criterionMSE(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 50
      #  self.loss_G_WF_L1 = self.criterionL1(wf_fake, wf_real)
      #  self.loss_msssim= (1- self.ssim_module(self.fake_B, self.real_B).mean())*50
        
        self.loss_G_ssim= (1-self.ssim_module(self.fake_B, self.real_B).mean())* 10                          #   this line ##
   #     self.loss_G_L1_matrix= gauss_weighted_l1(self.fake_B, self.real_B,11,1.5)
      #   self.loss_msssim_matrix= 1- self.ms_ssim_module(self.fake_B, self.real_B) 
      #  self.loss_ssim_matrix= 1- self.ssim_module(self.fake_B, self.real_B) 
        
     #   self.loss_G_L1=self.loss_G_L1_matrix.mean()
    #    self.loss_ssim=self.loss_ssim_matrix.mean()
        
   #     self.loss_G = self.loss_G_GAN + (self.loss_G_L1_matrix * 0.16 + self.loss_ssim_matrix * 0.84).mean()
     
        self.loss_G =  self.loss_G_GAN+ self.loss_G_L1 + self.loss_G_ssim + self.loss_G_fft
        self.gan_loss.append(self.loss_G_GAN.item())
        self.l1_loss.append(self.loss_G_L1.item())
        self.fft_loss.append(self.loss_G_fft.item())
        self.ssim_loss.append(self.loss_G_ssim.item())
        self.g_loss.append(self.loss_G.item())
        plt.figure()
        plt.plot(self.gan_loss)
        plt.savefig('./loss fig/'+'gan_loss.png',dpi=600)
        plt.close()
        plt.figure()
        plt.plot(self.l1_loss)
        plt.savefig('./loss fig/'+'l1_loss.png',dpi=600)
        plt.close()
        plt.figure()
        plt.plot(self.fft_loss)
        plt.savefig('./loss fig/'+'fft_loss.png',dpi=600)
        plt.close()
        plt.figure()
        plt.plot(self.ssim_loss)
        plt.savefig('./loss fig/'+'ssim_loss.png',dpi=600)
        plt.close()
        plt.figure()
        plt.plot(self.g_loss)
        plt.savefig('./loss fig/'+'g_loss.png',dpi=600)
        plt.close()
        gan_list=pd.DataFrame(data={'gan_loss':self.gan_loss})
        gan_list.to_csv('./loss/gan_loss.csv',index=False)
        l1_list=pd.DataFrame(data={'l1_loss':self.l1_loss})
        l1_list.to_csv('./loss/l1_loss.csv',index=False)
        fft_list=pd.DataFrame(data={'fft_loss':self.fft_loss})
        fft_list.to_csv('./loss/fft_loss.csv',index=False)
        ssim_list=pd.DataFrame(data={'ssim_loss':self.ssim_loss})
        ssim_list.to_csv('./loss/ssim_loss.csv',index=False)
        g_list=pd.DataFrame(data={'g_loss':self.ssim_loss})
        g_list.to_csv('./loss/g_loss.csv',index=False)
       # self.loss_G = self.loss_G_GAN + self.loss_G_L1 +self.loss_msssim
      #  self.loss_G = self.loss_G_GAN + self.loss_G_L1 +   self.loss_ssim
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
