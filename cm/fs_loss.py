import torch
import torch.nn as nn
import torch.nn.functional as F
# Parts of these codes are from: https://github.com/Linfeng-Tang/SeAFusion
from PIL import Image
class AverageFilter(nn.Module):
    def __init__(self):
        super(AverageFilter, self).__init__()
        kernel = [[1/9, 1/9, 1/9],
                  [1/9, 1/9, 1/9],
                  [1/9, 1/9, 1/9]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                filtered = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weight, padding=1)
                tensor_list.append(filtered)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)

# input = torch.randn(1,1,3,3).cuda()
# print(input)
# AverageFilter = AverageFilter()
# output = AverageFilter(input)
# print(output)

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        # self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        # self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)


# class Fusionloss(nn.Module):
#     def __init__(self):
#         super(Fusionloss, self).__init__()
#         self.sobelconv = Sobelxy()
#         self.mse_criterion = torch.nn.MSELoss()
#
#     def forward(self, image_vis, image_ir, generate_img):
#         image_y = image_vis
#         B, C, W, H = image_vis.shape
#         image_ir = image_ir.expand(B, C, W, H)
#         x_in_max = torch.max(image_y, image_ir)
#         loss_in = F.l1_loss(generate_img, x_in_max)
#         # Gradient
#         y_grad = self.sobelconv(image_y)
#         ir_grad = self.sobelconv(image_ir)
#         B, C, K, W, H = y_grad.shape
#         ir_grad = ir_grad.expand(B, C, K, W, H)
#         generate_img_grad = self.sobelconv(generate_img)
#         x_grad_joint = torch.maximum(y_grad, ir_grad)
#         loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)
#
#         return loss_in, loss_grad


def get_self_information_size(tensor):


    sobelconv = Sobelxy()
    avgconv=AverageFilter()

    output = sobelconv(tensor).squeeze(2)
    # output = avgconv(tensor).squeeze(2)
    # output = tensor.squeeze(2)
    output = torch.square(output)
    output = torch.mean(output)


    return output


class Fusion_loss(nn.Module):
    def __init__(self):

        super(Fusion_loss,self).__init__()
        self.get_self_information_size = get_self_information_size
        self.mse_loss = torch.nn.MSELoss()
        self.sobelconv = Sobelxy()


    def forward(self, image_vis, image_ir, image_X, image_Y):

        image_X_Imsize = self.get_self_information_size(image_X)
        image_Y_Imsize = self.get_self_information_size(image_Y)

        loss_in = image_Y_Imsize / image_X_Imsize

        # Gradient
        y_grad = self.sobelconv(image_vis)
        ir_grad = self.sobelconv(image_ir)
        B, C, K, W, H = y_grad.shape
        ir_grad = ir_grad.expand(B, C, K, W, H)
        generate_img_grad = self.sobelconv(image_X)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)
        return loss_in ,loss_grad