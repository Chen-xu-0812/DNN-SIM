import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
from math import pi,cos,sin
import numpy as np

def add_stripes(image):
    w=Variable(torch.Tensor(image.shape[1]), requires_grad=True)
    wo=w/2
    k2=75.23
    ModFac = 0.8
    aA = 0.5 * ModFac;
    aB = 0.5 * ModFac;
    aC = 0.5 * ModFac;
    mA = 0.5;
    mB = 0.5;
    mC = 0.5;

    x = np.arange(w)
    y = np.arange(w)

    [X,Y] = np.meshgrid(x,y)
    X = Variable(torch.from_numpy(X), requires_grad=True)
    Y = Variable(torch.from_numpy(Y), requires_grad=True)

    #  illunination phase shifts along the three directions
    p0Ao = Variable(torch.Tensor(0 * pi / 3), requires_grad=True)
    p0Ap = Variable(torch.Tensor(2 * pi / 3), requires_grad=True)
    p0Am = Variable(torch.Tensor(4 * pi / 3), requires_grad=True)
    p0Bo = Variable(torch.Tensor(0 * pi / 3), requires_grad=True)
    p0Bp = Variable(torch.Tensor(2 * pi / 3), requires_grad=True)
    p0Bm = Variable(torch.Tensor(4 * pi / 3), requires_grad=True)
    p0Co = Variable(torch.Tensor(0 * pi / 3), requires_grad=True)
    p0Cp = Variable(torch.Tensor(2 * pi / 3), requires_grad=True)
    p0Cm = Variable(torch.Tensor(4 * pi / 3), requires_grad=True)

    # Illuminating patterns
    thetaA = Variable(torch.Tensor(0 * pi / 3), requires_grad=True)
    thetaB = Variable(torch.Tensor(1 * pi / 3), requires_grad=True)
    thetaC = Variable(torch.Tensor(2 * pi / 3), requires_grad=True)

    k2a = Variable(torch.Tensor((k2 / w) * cos(thetaA), (k2 / w) * sin(thetaA)), requires_grad=True)
    k2b = Variable(torch.Tensor((k2 / w) * cos(thetaB), (k2 / w) * sin(thetaB)), requires_grad=True)
    k2c = Variable(torch.Tensor((k2 / w) * cos(thetaC), (k2 / w) * sin(thetaC)), requires_grad=True)

    # random phase shift errors
    t = torch.rand(9,1)
    NN = Variable(torch.Tensor(1 * (0.5 - t) * pi / 18))

    # illunination phase shifts with random errors
    psAo = p0Ao + NN(1, 1)
    psAp = p0Ap + NN(2, 1)
    psAm = p0Am + NN(3, 1)
    psBo = p0Bo + NN(4, 1)
    psBp = p0Bp + NN(5, 1)
    psBm = p0Bm + NN(6, 1)
    psCo = p0Co + NN(7, 1)
    psCp = p0Cp + NN(8, 1)
    psCm = p0Cm + NN(9, 1)

    #  illunination patterns
    sAo = mA + aA*cos(2*pi*(k2a(1,1)*(X-wo)+k2a(1,2)*(Y-wo))+psAo)
    sAp = mA + aA * cos(2 * pi * (k2a(1, 1) * (X - wo) + k2a(1, 2) * (Y - wo)) + psAp)
    sAm = mA + aA * cos(2 * pi * (k2a(1, 1)* (X - wo) + k2a(1, 2) * (Y - wo)) + psAm)
    sBo = mB + aB * cos(2 * pi * (k2b(1, 1) * (X - wo) + k2b(1, 2) * (Y - wo)) + psBo)

    sBp = mB + aB * cos(2 * pi * (k2b(1, 1) * (X - wo) + k2b(1, 2) * (Y - wo)) + psBp)
    sBm = mB + aB * cos(2 * pi * (k2b(1, 1) * (X - wo) + k2b(1, 2) * (Y - wo)) + psBm)
    sCo = mC + aC * cos(2 * pi * (k2c(1, 1) * (X - wo) + k2c(1, 2) * (Y - wo)) + psCo)

    sCp = mC + aC * cos(2 * pi * (k2c(1, 1) * (X - wo) + k2c(1, 2) * (Y - wo)) + psCp)
    sCm = mC + aC * cos(2 * pi * (k2c(1, 1) * (X - wo) + k2c(1, 2) * (Y - wo)) + psCm)


    # superposed Objects
    s1a = torch.Tensor.mul(image,sAo)
    s2a = torch.Tensor.mul(image, sAp)
    s3a = torch.Tensor.mul(image, sAm)
    s1b = torch.Tensor.mul(image, sBo)
    s2b = torch.Tensor.mul(image, sBp)
    s3b = torch.Tensor.mul(image,sBm)
    s1c = torch.Tensor.mul(image, sCo)
    s2c = torch.Tensor.mul(image, sCp)
    s3c = torch.Tensor.mul(image, sCm)






if __name__ == '__main__':
    add_stripes('F:\何超坚果云\chenxu\SIM\datasets\sim_9\train\1.tif')