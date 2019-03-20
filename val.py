import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import model
from dataloader import _pil_loader
import os
from PIL import Image


if __name__ == '__main__':
    mean = [0.429, 0.431, 0.397]
    std  = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testPath = "UCF101_results/ucf101_interp_new"
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    # device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("checkpoints/SuperSloMo26.ckpt", map_location=device)
    flowComp = model.UNet(6, 4)
    flowComp.load_state_dict(state_dict["state_dictFC"])
    flowComp.eval()
    flowComp.to(device)
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.load_state_dict(state_dict["state_dictAT"])
    ArbTimeFlowIntrp.eval()
    ArbTimeFlowIntrp.to(device)
    validationFlowBackWarp = model.backWarp(256, 256, device)
    validationFlowBackWarp = validationFlowBackWarp.to(device)
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    TP = transforms.Compose([revNormalize, transforms.ToPILImage()])
    validationFrameIndex = torch.IntTensor([3])
    for folder in os.listdir(testPath):
        frame0 = _pil_loader(os.path.join(testPath, folder, "frame_00.png"))
        frame0 = transform(frame0).unsqueeze(0)
        frame1 = _pil_loader(os.path.join(testPath, folder, "frame_02.png"))
        frame1 = transform(frame1).unsqueeze(0)
        I0 = frame0.to(device)
        I1 = frame1.to(device)
        flowOut = flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:,:2,:,:]
        F_1_0 = flowOut[:,2:,:,:]

        fCoeff = model.getFlowCoeff(validationFrameIndex, device)

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)
        
        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
            
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0
            
        g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
        
        wCoeff = model.getWarpCoeff(validationFrameIndex, device)    
        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
        image = TP(Ft_p[0])
        image.save(os.path.join(testPath, folder, "frame_01_new.png"), "PNG")
        os.remove(os.path.join(testPath, folder, "frame_01_ours.png"))


