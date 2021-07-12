import torch
from torch.autograd import Variable
from math import pi,cos,sin
import numpy as np

def stripe(image_tensor):
    
    w = image_tensor.shape[2]
    wo = torch.Tensor(np.array(w / 2))
    k2 = 75.23
    ModFac = 0.8
    aA = torch.Tensor(np.array(0.5 * ModFac))
    aB = torch.Tensor(np.array(0.5 * ModFac))
    aC = torch.Tensor(np.array(0.5 * ModFac))
    mA = torch.Tensor(np.array(0.5))
    mB = torch.Tensor(np.array(0.5))
    mC = torch.Tensor(np.array(0.5))
    
    x = np.arange(w)
    y = np.arange(w)
    
    [X, Y] = np.meshgrid(x, y)
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    
    # #  illunination phase shifts along the three directions
    p0Ao = torch.Tensor(np.array(0 * pi / 3))
    p0Ap = torch.Tensor(np.array(2 * pi / 3))
    p0Am = torch.Tensor(np.array(4 * pi / 3))
    p0Bo = torch.Tensor(np.array(0 * pi / 3))
    p0Bp = torch.Tensor(np.array(2 * pi / 3))
    p0Bm = torch.Tensor(np.array(4 * pi / 3))
    p0Co = torch.Tensor(np.array(0 * pi / 3))
    p0Cp = torch.Tensor(np.array(2 * pi / 3))
    p0Cm = torch.Tensor(np.array(4 * pi / 3))
    
    # Illuminating patterns
    thetaA = torch.Tensor(np.array(0 * pi / 3))
    thetaB = (1 * pi / 3)
    thetaC = (2 * pi / 3)
    
    # k2a = torch.FloatTensor(np.array([(k2 / w) * cos(thetaA),(k2 / w) * sin(thetaA)]))
    k2a = torch.FloatTensor(np.array([(k2 / w) * cos(thetaA), (k2 / w) * sin(thetaA)]))
    k2b = torch.FloatTensor(np.array([(k2 / w) * cos(thetaB), (k2 / w) * sin(thetaB)]))
    k2c = torch.FloatTensor(np.array([(k2 / w) * cos(thetaC), (k2 / w) * sin(thetaC)]))
    
    #  illunination phase shifts along the three directions
    
    # random phase shift errors
    t = torch.rand(9, 1)
    NN = torch.FloatTensor(1 * (0.5 - t) * pi / 18)
    
    # illunination phase shifts with random errors
    psAo = p0Ao + NN[0]
    psAp = p0Ap + NN[1]
    psAm = p0Am + NN[2]
    psBo = p0Bo + NN[3]
    psBp = p0Bp + NN[4]
    psBm = p0Bm + NN[5]
    psCo = p0Co + NN[6]
    psCp = p0Cp + NN[7]
    psCm = p0Cm + NN[8]
    # r= torch.cos(2 * pi * (k2a[0] * (X - wo) + k2a[1] * (Y - wo)))
    
    #  illunination patterns
    sAo = mA + aA * torch.cos(2 * pi * (k2a[0] * (X - wo) + k2a[1] * (Y - wo)) + psAo)
    sAp = mA + aA * torch.cos(2 * pi * (k2a[0] * (X - wo) + k2a[1] * (Y - wo)) + psAp)
    sAm = mA + aA * torch.cos(2 * pi * (k2a[0] * (X - wo) + k2a[1] * (Y - wo)) + psAm)
    
    sBo = mB + aB * torch.cos(2 * pi * (k2b[0] * (X - wo) + k2b[1] * (Y - wo)) + psBo)
    sBp = mB + aB * torch.cos(2 * pi * (k2b[0] * (X - wo) + k2b[1] * (Y - wo)) + psBp)
    sBm = mB + aB * torch.cos(2 * pi * (k2b[0] * (X - wo) + k2b[1] * (Y - wo)) + psBm)
    
    sCo = mC + aC * torch.cos(2 * pi * (k2c[0] * (X - wo) + k2c[1] * (Y - wo)) + psCo)
    sCp = mC + aC * torch.cos(2 * pi * (k2c[0] * (X - wo) + k2c[1] * (Y - wo)) + psCp)
    sCm = mC + aC * torch.cos(2 * pi * (k2c[0] * (X - wo) + k2c[1] * (Y - wo)) + psCm)
    
    # superposed Objects
    s1a = torch.Tensor.mul(torch.squeeze(image_tensor), sAo.cuda())
    s2a = torch.Tensor.mul(torch.squeeze(image_tensor), sAp.cuda())
    s3a = torch.Tensor.mul(torch.squeeze(image_tensor), sAm.cuda())
    s1b = torch.Tensor.mul(torch.squeeze(image_tensor), sBo.cuda())
    s2b = torch.Tensor.mul(torch.squeeze(image_tensor), sBp.cuda())
    s3b = torch.Tensor.mul(torch.squeeze(image_tensor), sBm.cuda())
    s1c = torch.Tensor.mul(torch.squeeze(image_tensor), sCo.cuda())
    s2c = torch.Tensor.mul(torch.squeeze(image_tensor), sCp.cuda())
    s3c = torch.Tensor.mul(torch.squeeze(image_tensor), sCm.cuda())

    return (s1a , s2a ,s3a ,s1b ,s2b ,s2c ,s3a ,s3b ,s3c)