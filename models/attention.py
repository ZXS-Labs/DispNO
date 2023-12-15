import torch.nn as nn
import torch
import numpy as np
import torch.autograd as autograd
from skimage import morphology
from torch.nn import functional as F

class SpatialAM_Module(nn.Module):
    def __init__(self, out_add, key_query_same, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SpatialAM_Module, self).__init__()
        self.out_add = out_add
        self.key_query_same = key_query_same
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels // scale
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
        )
        if self.key_query_same:
            self.f_query = self.f_key
        else:
            self.f_query = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.key_channels),
            )

        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        # coefficient of context in add mechanism
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fusion = nn.Conv2d(self.out_channels * 2, self.out_channels, 1, 1, 0, bias=True)


        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):

        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()  # N C H W
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        if self.out_add:
            context = self.gamma * context + x  # add
        else:
            context = torch.cat((context, x), 1)
            context = self.fusion(context)
        return context

class SpatialAM3D_Module(nn.Module):
    def __init__(self, out_add, key_query_same, in_channels, key_channels, value_channels, out_channels=None):
        super(SpatialAM3D_Module, self).__init__()
        self.out_add = out_add
        self.key_query_same = key_query_same
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.key_channels),
        )
        if self.key_query_same:
            self.f_query = self.f_key
        else:
            self.f_query = nn.Sequential(
                nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(self.key_channels),
            )

        self.f_value = nn.Conv3d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv3d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        # coefficient of context in add mechanism
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fusion = nn.Conv3d(self.out_channels * 2, self.out_channels, 1, 1, 0, bias=True)


        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):

        batch_size, d, h, w = x.size(0), x.size(2), x.size(3), x.size(4)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()  # N C H W
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        if self.out_add:
            context = self.gamma * context + x  # add
        else:
            context = torch.cat((context, x), 1)
        return context

class Horizon_attention_module(nn.Module):
    def __init__(self, out_add, key_query_same, channels, key_query_channel):
        super(Horizon_attention_module, self).__init__()
        self.out_add = out_add
        self.key_query_same = key_query_same
        self.in_channels = channels
        self.key_channels = key_query_channel
        self.value_channels = self.in_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels)
        )

        if self.key_query_same:
            self.f_query = self.f_key
        else:
            self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels)
            )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.fusion = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(-1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)


    def forward(self, x):
        b, c, h, w = x.shape
        Query = self.f_query(x).permute(0, 2, 3, 1)
        Key = self.f_key(x).permute(0, 2, 1, 3)
        score1 = torch.bmm(Query.contiguous().view(-1, w, self.key_channels),
                          Key.contiguous().view(-1, self.key_channels, w))
        attention_map = self.softmax(score1)
        buffer = self.f_value(x).permute(0, 2, 3, 1).contiguous().view(-1, w, c)
        buffer = torch.bmm(attention_map, buffer).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)
        if self.out_add:
            out = self.gamma * buffer + x  # add
        else:
            out = torch.cat((x, buffer), 1)
        return out

class Vertical_attention_module(nn.Module):
    def __init__(self, out_add, key_query_same, channels, key_query_channel):
        super(Vertical_attention_module, self).__init__()
        self.out_add = out_add
        self.key_query_same = key_query_same
        self.in_channels = channels
        self.key_channels = key_query_channel
        self.value_channels = self.in_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels)
        )

        if self.key_query_same:
            self.f_query = self.f_key
        else:
            self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels)
            )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        self.fusion = nn.Conv2d(channels * 2, channels, 1, 1, 0, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(-1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)


    def forward(self, x):
        b, c, h, w = x.shape

        Query = self.f_query(x).permute(0, 3, 2, 1)
        Key = self.f_key(x).permute(0, 3, 1, 2)
        score1 = torch.bmm(Query.contiguous().view(-1, h, self.key_channels),
                          Key.contiguous().view(-1, self.key_channels, h))
        attention_map = self.softmax(score1)
        buffer = self.f_value(x).permute(0, 3, 2, 1).contiguous().view(-1, h, c)
        buffer = torch.bmm(attention_map, buffer).contiguous().view(b, w, h, c).permute(0, 3, 2, 1)
        if self.out_add:
            out = self.gamma * buffer + x  # add
        else:
            out = torch.cat((x, buffer), 1)
        return out
class Parallax_attention_module(nn.Module):
     def __init__(self, key_query_same, in_channels, key_channels, value_channels, out_channels=None):
         super(Parallax_attention_module, self).__init__()
         self.in_channels = in_channels
         self.key_channels = key_channels
         self.value_channels = value_channels
         self.key_query_same = key_query_same

         self.f_key = nn.Sequential(
             nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                       kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(self.key_channels),
         )

         if self.key_query_same:
             self.f_query = self.f_key
         else:
             self.f_query = nn.Sequential(
                 nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                           kernel_size=1, stride=1, padding=0, bias=False),
                 nn.BatchNorm2d(self.key_channels),
             )

         self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                  kernel_size=1, stride=1, padding=0)

         self.softmax = nn.Softmax(-1)
         self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0, bias=True)

     def forward(self, x_left, x_right):
         b, c, h, w = x_left.shape
         Q = self.f_query(x_left).permute(0, 2, 3, 1)
         S = self.f_key(x_right).permute(0, 2, 1, 3)

         score = torch.bmm(Q.contiguous().view(-1, w, c),
                           S.contiguous().view(-1, c, w))

         M_right_to_left = self.softmax(score)

         Q = self.f_query(x_right).permute(0, 2, 3, 1)
         S = self.f_key(x_left).permute(0, 2, 1, 3)

         score = torch.bmm(Q.contiguous().view(-1, w, c),
                           S.contiguous().view(-1, c, w))

         M_left_to_right = self.softmax(score)

         V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
         V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
         V_left_to_right = morphologic_process(V_left_to_right)

         V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
         V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
         V_right_to_left = morphologic_process(V_right_to_left)

class EA_Module(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, query_channels, out_channels=None):
        super(EA_Module, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.query_channels = query_channels
        self.out_channels = out_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1,
                               stride=1, padding=0)

        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.query_channels, kernel_size=1,
                               stride=1, padding=0)

        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1,
                               stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1,
                           stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        B, H, W = x.size(0), x.size(2), x.size(3)
        key = self.f_key(x).view(B, self.key_channels, -1)
        value = self.f_value(x).view(B, self.value_channels, -1)
        value_tran = value.permute(0, 2, 1)
        global_context_vectors = torch.matmul(key, value_tran)
        global_context_vectors = (self.key_channels ** -.5) * global_context_vectors
        global_context_vectors = F.softmax(global_context_vectors, dim=-1)

        query = self.f_query(x).view(B, self.query_channels, -1)
        query_tran = query.permute(0, 2, 1)
        context = torch.matmul(query_tran, global_context_vectors)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(B, self.value_channels, *x.size()[2:])
        context = self.W(context)

        context = self.gamma * context + x  # add
        return context
class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = self.gamma * reprojected_value + input_

        return attention

class ParallaxAM_Module(nn.Module):
    def __init__(self, out_add, key_query_same, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(ParallaxAM_Module, self).__init__()
        self.out_add = out_add
        self.key_query_same = key_query_same
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels // scale
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.key_channels),
        )
        if self.key_query_same:
            self.f_query = self.f_key
        else:
            self.f_query = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.key_channels),
            )

        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        # coefficient of context in add mechanism
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fusion = nn.Conv2d(self.out_channels * 2, self.out_channels, 1, 1, 0, bias=True)


        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
		
    def forward(self, x, y):

        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(y).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(y).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()  # N C H W
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        if self.out_add:
            context = self.gamma * context + x  # add
        else:
            context = torch.cat((context, x), 1)
            context = self.fusion(context)
        return context
