import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import random
import numpy as np

<<<<<<< HEAD

=======
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model1 = models.resnet18(pretrained=True)
        self.pretrained_model2 = models.resnet18(pretrained=True)
        self.pretrained_model3 = models.resnet18(pretrained=True)
<<<<<<< HEAD
        self.conv1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1)
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1)
        self.conv3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=1)
        self.classifier = nn.Linear(1000, 1)
        self.soft = nn.Softmax(2)

=======
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.classifier = nn.Linear(1000, 1)
        self.soft = nn.Softmax(2)


>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
<<<<<<< HEAD
        order_index = torch.LongTensor(np.concatenate(
            [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
=======
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        if torch.cuda.is_available():
            a = a.cuda()
            order_index = order_index.cuda()
        return torch.index_select(a, dim, order_index)

<<<<<<< HEAD
    def forward(self, x, x1, x2):
=======
    def forward(self, x,x1,x2):
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        x = torch.squeeze(x, dim=0)
        x = self.pretrained_model1.conv1(x)
        x = self.pretrained_model1.bn1(x)
        x = self.pretrained_model1.maxpool(x)
        x = self.pretrained_model1.layer1(x)
        x = self.pretrained_model1.layer2(x)
        x = self.pretrained_model1.layer3(x)
        x = self.pretrained_model1.layer4(x)
        attention = self.conv1(x)
<<<<<<< HEAD
        attention = self.soft(attention.view(
            *attention.size()[:2], -1)).view_as(attention)
=======
        attention =  self.soft(attention.view(*attention.size()[:2], -1)).view_as(attention)
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        attention0 = attention
        maximum = torch.max(attention.flatten(2), 2).values
        maximum = Net.tile(maximum, 1, attention.shape[2]*attention.shape[3])
        attention_norm = attention.flatten(2).flatten(1) / maximum
<<<<<<< HEAD
        attention_norm = torch.reshape(
            attention_norm, (attention.shape[0], attention.shape[1], attention.shape[2], attention.shape[3]))
=======
        attention_norm= torch.reshape(attention_norm, (attention.shape[0],attention.shape[1],attention.shape[2],attention.shape[3]))
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        o1 = x*attention_norm

        x1 = torch.squeeze(x1, dim=0)
        x1 = self.pretrained_model2.conv1(x1)
        x1 = self.pretrained_model2.bn1(x1)
        x1 = self.pretrained_model2.maxpool(x1)
        x1 = self.pretrained_model2.layer1(x1)
        x1 = self.pretrained_model2.layer2(x1)
        x1 = self.pretrained_model2.layer3(x1)
        x1 = self.pretrained_model2.layer4(x1)
        attention = self.conv1(x1)
<<<<<<< HEAD
        attention = self.soft(attention.view(
            *attention.size()[:2], -1)).view_as(attention)
=======
        attention =  self.soft(attention.view(*attention.size()[:2], -1)).view_as(attention)
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        attention1 = attention
        maximum = torch.max(attention.flatten(2), 2).values
        maximum = Net.tile(maximum, 1, attention.shape[2]*attention.shape[3])
        attention_norm = attention.flatten(2).flatten(1) / maximum
<<<<<<< HEAD
        attention_norm = torch.reshape(
            attention_norm, (attention.shape[0], attention.shape[1], attention.shape[2], attention.shape[3]))
        o2 = x1*attention_norm

=======
        attention_norm= torch.reshape(attention_norm, (attention.shape[0],attention.shape[1],attention.shape[2],attention.shape[3]))
        o2 = x1*attention_norm


>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        x2 = torch.squeeze(x2, dim=0)
        x2 = self.pretrained_model3.conv1(x2)
        x2 = self.pretrained_model3.bn1(x2)
        x2 = self.pretrained_model3.maxpool(x2)
        x2 = self.pretrained_model3.layer1(x2)
        x2 = self.pretrained_model3.layer2(x2)
        x2 = self.pretrained_model3.layer3(x2)
        x2 = self.pretrained_model3.layer4(x2)
        attention = self.conv1(x2)
<<<<<<< HEAD
        attention = self.soft(attention.view(
            *attention.size()[:2], -1)).view_as(attention)
=======
        attention =  self.soft(attention.view(*attention.size()[:2], -1)).view_as(attention)
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        attention2 = attention
        maximum = torch.max(attention.flatten(2), 2).values
        maximum = Net.tile(maximum, 1, attention.shape[2]*attention.shape[3])
        attention_norm = attention.flatten(2).flatten(1) / maximum
<<<<<<< HEAD
        attention_norm = torch.reshape(
            attention_norm, (attention.shape[0], attention.shape[1], attention.shape[2], attention.shape[3]))
        o3 = x2*attention_norm

        output = torch.cat((o1, o2, o3), dim=0)
        out = self.pretrained_model1.avgpool(output)
=======
        attention_norm= torch.reshape(attention_norm, (attention.shape[0],attention.shape[1],attention.shape[2],attention.shape[3]))
        o3 = x2*attention_norm

        output = torch.cat((o1,o2,o3), dim=0)
        out= self.pretrained_model1.avgpool(output)
>>>>>>> 465269e44a99a1697fb91d4c4815629a581887cf
        out = self.pretrained_model1.fc(out.squeeze(3).squeeze(2))
        out = torch.max(out, 0, keepdim=True)[0]
        final = self.classifier(out)

        return final, attention0, attention1, attention2
