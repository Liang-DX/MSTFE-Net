from braindecode.models.eegnet import Conv2dWithConstraint
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x)
        y=y.squeeze(-1).permute(0,2,1)
        y=self.conv(y)
        y=self.sigmoid(y)
        y=y.permute(0,2,1).unsqueeze(-1)
        return x*y.expand_as(x)
class ChannelFusion(nn.Module):
    def __init__(self, in_channels):
        super(ChannelFusion, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
    def forward(self, x):
        avg_out = self.avgPool(x)
        max_out = self.max_pool(x)
        out = avg_out + max_out
        out = self.elu(out)
        out = self.conv(out)
        return self.sigmoid(out)


class LocalTemporalFusion(nn.Module):
    def __init__(self, kernel_size=7):
        super(LocalTemporalFusion, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(2, 1, 9, padding=4, bias=False)
        self.conv3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(2, 1, 5, padding=2, bias=False)
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        split_tensors = []
        for i in range(4):
            start_idx = i * x.size(3) // 4
            end_idx = start_idx + x.size(3) // 4
            split_tensors.append(x[:, :, :, start_idx:end_idx])

        outputs = []
        for i, split_tensor in enumerate(split_tensors):
            if i == 0:
                output = self.conv1(split_tensor)
            elif i == 1:
                output = self.conv2(split_tensor)
            elif i == 2:
                output = self.conv3(split_tensor)
            elif i == 3:
                output = self.conv4(split_tensor)
            outputs.append(output)
        x = torch.cat(outputs, dim=3)
        x = self.conv(x)
        return self.sigmoid(x)


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionConv, self).__init__()

        self.conv_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3)
        self.local_fusion = LocalTemporalFusion()
        self.channel_fusion = ChannelFusion(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)


    def forward(self, x1, x2):

        x_fused = torch.cat([x1, x2], dim=1)
        x_fused_c = x_fused * self.channel_fusion(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.local_fusion(x_fused_s)
        x_out = self.conv(x_fused_s + x_fused_c)
        return x_out


class MSFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFF, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2 ):
        x_fused = self.fusion_conv(x1, x2)
        x3 = torch.cat([x1, x2], dim=1)
        cut = self.shortcut(x3)
        return x_fused + cut
class Branch_Right(nn.Module):
    def __init__(self, in_channels):
        super(Branch_Right, self).__init__()
        out_channels = in_channels

        self.project1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),)

        self.project2 = nn.Sequential(
            nn.Conv2d(out_channels, 2*out_channels, 1, bias=False),
            nn.BatchNorm2d(2*out_channels),)

    def forward(self, x0,x1,x2,x3):
        B,T,N,C = x0.shape
        x_ = torch.cat([x0,x1,x2,x3],0)
        x__ = x_.reshape(-1,T,N*B,C)
        x__ = self.project1(x__.permute(0,3,2,1)).permute(0,3,2,1)
        weight = F.soft_max(x__, dim=0)
        x_ = x_.reshape(-1,T,N*B,C)
        out = (weight * x_).sum(0)
        out = out.reshape(B,T,N,C)
        return self.project2(out.permute(0,3,2,1)).permute(0,3,2,1)
class gatedFusion(nn.Module):

    def __init__(self, dim):
        super(gatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=True)
        self.fc2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x1, x2):
        x11 = self.fc1(x1)
        x22 = self.fc2(x2)

        z = torch.sigmoid(x11+x22)

        out = z*x1 + (1-z)*x2
        return out
class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []

        for i in range(out_shape):
            index = i*self.stride
            input = x[:, :, index:index+self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)

        out = torch.cat(out, dim=-1)

        return out
def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim**.5
    attn = F.soft_max(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, n_head*self.d_k)
        self.w_k = nn.Linear(d_model, n_head*self.d_k)
        self.w_v = nn.Linear(d_model, n_head*self.d_v)
        self.w_o = nn.Linear(n_head*self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    # [batch_size, n_channel, d_model]
    def forward(self, query, key, value):

        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        
        out, _ = attention(q, k, v)
        
        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.dropout(self.w_o(out))

        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, fc_ratio, attn_dropout=0.5, fc_dropout=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(dim, heads, attn_dropout)
        self.feed_forward = FeedForward(dim, dim*fc_ratio, fc_dropout)
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
    
    def forward(self, data):
        out = data + self.multihead_attention(data,data, data)
        res = self.layernorm1(out)
        res = out + self.feed_forward(res)
        output = self.layernorm2(res)
        return output

class MyModel(nn.Module):
    def __init__(self, classes=4, channels=22, dim=32, pool_size=50,
    pool_stride=15, heads=8, fc=4, depth=4, attn_dropout=0.5, fc_dropout=0.5,F2 = 8,cnn_dropout = 0.25):
        super().__init__()

        self.temp_Conv1 = nn.Sequential(
            Conv2dWithConstraint(1,  dim//4, kernel_size=[1, 51], padding=(0,25), max_norm=2.),
            nn.BatchNorm2d(dim // 4),
        )
        self.temp_Conv2 = nn.Sequential(
            Conv2dWithConstraint(1, dim // 4, kernel_size=[1, 25], padding=(0, 12), max_norm=2.),
            nn.BatchNorm2d(dim // 4),
        )
        self.temp_Conv3 = nn.Sequential(
            Conv2dWithConstraint(1, dim // 4, kernel_size=[1,15], padding=(0, 7), max_norm=2.),
            nn.BatchNorm2d(dim // 4),
        )
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, (channels, 1), padding=0, groups=F2, bias=False, max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.Dropout(cnn_dropout),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
        )
        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, kernel_size=[channels, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.Dropout(cnn_dropout),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
        )
        self.spatial_3 = nn.Sequential(
            Conv2dWithConstraint(F2, F2, kernel_size=[channels, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.Dropout(cnn_dropout),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid',
                                 max_norm=2.),
            nn.BatchNorm2d(F2),
        )
        self.elu = nn.ELU()
        self.varPool = VarPoold(pool_size, pool_stride)
        self.avgPool = nn.AvgPool1d(pool_size, pool_stride)
        self.dropout = nn.Dropout()
        self.transformer_encoders = nn.ModuleList(
            [TransformerEncoder(24, heads, fc, attn_dropout, fc_dropout) for _ in range(depth)]
        )
        self.msff = MSFF(in_channels=2 * 32, out_channels=32)
        self.branch_left = nn.Sequential(
            nn.Conv2d(32, 32, (2, 1)),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        self.branch_right = Branch_Right(in_channels=6)
        self.eca = ECAAttention(kernel_size=3)
        self.gt = gatedFusion(dim=1)
        self.classify = nn.Linear(24*32, classes)
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x1 = self.temp_Conv1(x)
        x1 = self.spatial_1(x1)
        x2 = self.temp_Conv2(x)
        x2 = self.spatial_2(x2)
        x3 = self.temp_Conv3(x)
        x3 = self.spatial_3(x3)
        x = torch.cat((x1,x2,x3), dim=1)
        x = self.elu(x)
        x = x.squeeze(dim=2)
        x1 = self.avgPool(x)
        x2 = self.varPool(x)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')
        for encoder in self.transformer_encoders:
            x1 = encoder(x1)
            x2 = encoder(x2)
        x1 = x1.unsqueeze(dim=2)
        x1 = x1.reshape(x1.size(0),-1,2,x1.size(3))
        x2 = x2.unsqueeze(dim=2)
        x2 = x2.reshape(x1.size(0),-1,2,x1.size(3))
        x = self.msff(x1,x2)
        chunks = torch.chunk(x, 4,dim=3)
        x = self.branch_left(x)
        k = self.branch_right(chunks[0],chunks[1],chunks[2],chunks[3])
        k = k.reshape(k.size(0),k.size(1),1, -1).permute(0,1,3,2)
        x = x.permute(0,1,3,2)
        z = self.gt(x,k)
        z = self.eca(z)
        z = z.reshape(x.size(0), -1)
        out = self.classify(z)

        return out
