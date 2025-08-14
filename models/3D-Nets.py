import torch
import torch.nn as nn
import torchvision.ops as ops
from recurrent_mechanisms import CGRU_cell, CLSTM_cell
from typing import Union


class Feature_Convs_3D(nn.Module):
    # Convolutional block purely for spatial extraction.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if not out_channels:
            out_channels = in_channels
    
        layers = [
            nn.Conv3d(in_channels=in_channels, out_channels=(in_channels + out_channels) // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=(in_channels + out_channels) // 2),
            nn.PReLU(),
      
            nn.Conv3d(in_channels=(in_channels + out_channels) // 2, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.PReLU()
        ]
        self.feature_convs = nn.Sequential(*layers)
  
    def forward(self, x):
        return self.feature_convs(x)


class Bichannel_Fusion_3D(nn.Module):
    def __init__(self, fire_channels, aux_channels, out_channels):
        super().__init__()
    
        layers_1 = [
            nn.Conv3d(in_channels=fire_channels, out_channels=fire_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=fire_channels),
            nn.PReLU()
        ]
        self.channel_1 = nn.Sequential(*layers_1)
    
        layers_2 = [
            nn.Conv3d(in_channels=aux_channels, out_channels=aux_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=aux_channels),
            nn.PReLU()
        ]
        self.channel_2 = nn.Sequential(*layers_2)
    
        layers_3 = [
              nn.Conv3d(in_channels=fire_channels+aux_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
              nn.BatchNorm3d(num_features=out_channels),
              nn.PReLU(),
      
              nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
              nn.BatchNorm3d(num_features=out_channels),
              nn.PReLU()
        ]
        self.depth_expand = nn.Sequential(*layers_3)
  
    def forward(self, x_fire, x_aux):
        batch_len, time_steps, _, _, _ = x_fire.shape
    
        x_fire = self.channel_1(x_fire)
        x_aux = self.channel_2(x_aux)
        x = torch.cat([x_fire, x_aux], dim=1)
        x = self.depth_expand(x)
        return x


class Time_Skip_3D(nn.Module):
    def __init__(self, input_dim, hidden_dim, input_shape, static_dim, rtype):
        super().__init__()
    
        self.rtype = rtype
        self.conv_REC = rtype(shape=(input_shape, input_shape), input_channels=input_dim, filter_size=3, num_features=hidden_dim)
    
        layers = [
            nn.Conv3d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=input_dim),
            nn.PReLU()
        ]
        self.time_skip = nn.Sequential(*layers)
    
        layers_1 = [
            nn.Conv3d(in_channels=static_dim, out_channels=static_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=static_dim),
            nn.PReLU()
        ]
        self.static_modal = nn.Sequential(*layers_1)
    
        layers_2 = [
            nn.Conv3d(in_channels=input_dim+static_dim, out_channels=input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=input_dim),
            nn.PReLU(),
    
            nn.Conv3d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=input_dim),
            nn.PReLU()
        ]
        self.output_conv = nn.Sequential(*layers_2)
  
    def forward(self, x, x_static):
        batch_len, _, time_steps, _, _ = x.shape
    
        x = x.permute(2, 0, 1, 3, 4)
        _, x_output = self.conv_REC(x, seq_len=time_steps)
        x_output = torch.stack(x_output).permute(1, 2, 0, 3, 4)
        x = self.time_skip(x_output)
        x_static = self.static_modal(x_static)
        x = torch.cat([x, x_static], dim=1)
        x = self.output_conv(x)
        return x


class DownSpace_Conv_3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, ver:int=1):
        super().__init__()
    
        layers = [
            nn.MaxPool3d(kernel_size=stride),
    
            Feature_Convs_3D(in_channels=in_channels, out_channels=out_channels),
    
            nn.Dropout3d(p=0.1)
        ]
        self.downspace_conv = nn.Sequential(*layers)
  
    def forward(self, x):
        return self.downspace_conv(x)


class Bottleneck_Net_3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4, stride=2):
        super().__init__()
    
        layers_1 = [
            nn.MaxPool3d(kernel_size=stride),
    
            nn.Flatten(start_dim=2, end_dim=3),
    
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
            nn.BatchNorm2d(num_features=in_channels),
            nn.PReLU(),
    
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.PReLU(),
    
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
            nn.BatchNorm2d(num_features=in_channels),
            nn.PReLU(),
    
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
    
            nn.Dropout2d(p=0.1)
        ]
        self.downsample = nn.Sequential(*layers_1)
    
        layers_2 = [
            nn.Conv2d(in_channels=out_channels, out_channels=factor * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=factor * out_channels),
            nn.PReLU(),
      
            nn.Conv2d(in_channels=factor * out_channels, out_channels=factor * out_channels, kernel_size=3, padding=1, bias=False, groups=factor * out_channels),
            nn.BatchNorm2d(num_features=factor * out_channels),
            nn.PReLU(),
      
            nn.Conv2d(in_channels=factor * out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        ]
        self.bottleneck = nn.Sequential(*layers_2)
    
        layers_3 = [
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False, groups=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
            nn.PReLU(),
    
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.PReLU(),
    
            nn.Unflatten(dim=2, unflattened_size=(1, 64)),
    
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=stride, stride=stride, bias=False),
            nn.BatchNorm3d(num_features=in_channels),
            nn.PReLU(),
        ]
        self.upsample = nn.Sequential(*layers_3)

    def forward(self, x):
        x = self.downsample(x)
        x_first = x
        x = self.bottleneck(x)
        return self.upsample(x + x_first)


class UpSpace_Conv_3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=2, last_layer=False, ver:int=1):
        super().__init__()
    
        if not out_channels:
            out_channels=in_channels
        layers = [
            Feature_Convs_3D(in_channels=in_channels, out_channels=out_channels),
    
            nn.ConvTranspose3d(in_channels=out_channels, out_channels=out_channels, kernel_size=stride, stride=stride, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.PReLU()
        ]
    
        if last_layer:
          layers = [
              Feature_Convs_3D(in_channels=in_channels, out_channels=out_channels),
    
              nn.Flatten(start_dim=1, end_dim=2),
    
              nn.Conv2d(in_channels=4*out_channels, out_channels=1, kernel_size=1, bias=False),
              nn.Sigmoid()
          ]
        self.upspace_conv = nn.Sequential(*layers)
  
    def forward(self, x_enc, x_dec):
        return self.upspact_conv(x_enc + x_dec)


### GRU/LSTM Networks Using 3-D Autoencoder
class RMC_3D_Nets(nn.Module):
    # These networks take in a 5D input (B, T, C, H, W) and forecast a 4D output (B, C, H, W)
    # B = batch_size
    # T = time_steps
    # C = channels
    # H = height
    # W = width

    def __init__(self, rtype: Union[CGRU_cell, CLSTM_cell]):
        super().__init__()
        self.rytpe = rtype
    
        self.bichannel = Bichannel_Fusion_3D(fire_channels=2, aux_channels=13, out_channels=32) # (N-1) x 32 x 256 x 256
        self.skip = Time_Skip_3D(input_dim=32, hidden_dim=64, input_shape=256, static_dim=4, rtype=rtype) # 4 x 32 x 256 x 256
        self.down_1 = DownSpace_Conv_3D(in_channels=32, out_channels=64, stride=2) # 2 x 64 x 128 x 128
        self.bottleneck = Bottleneck_Net_3D(in_channels=64, out_channels=128, factor=6, stride=2) # 1 x (128 -> 768) x 32 x 32
        self.up_1 = UpSpace_Conv_3D(in_channels=64, out_channels=32, stride=2) # 2 x 64 x 128 x 128
        self.up_0 = UpSpace_Conv_3D(in_channels=32, out_channels=16, last_layer=True) # (4 -> 1) x (32 -> 1) x 256 x 256

    def forward(self, x_fire, x_aux):
        x_fire = x_fire.to("cuda")
    
        # Isolate channels with static inputs as tensor.
        x_aux_static = x_aux[:, :, [12, 13, 14, 15], :, :].to('cuda')
        keep_indices = torch.ones(x_aux.shape[2], dtype=torch.bool, device='cpu')
        keep_indices[[12, 13, 14, 15]] = False
        x_aux = x_aux[:, :, keep_indices, :, :].to('cuda')
    
        x_fire = torch.transpose(x_fire, 1, 2)
        x_aux = torch.transpose(x_aux, 1, 2)
        x_aux_static = torch.transpose(x_aux_static, 1, 2)
    
        x = self.bichannel(x_fire=x_fire, x_aux=x_aux)
        x_skip = self.skip(x=x, x_static=x_aux_static)
        x_down_1 = self.down_1(x_skip)
        x_bottleneck = self.bottleneck(x_down_1)
        x_up_1 = self.up_1(x_down_1, x_bottleneck)
        x_up_0 = self.up_0(x_skip, x_up_1)
        return x_up_0
  
    def init_weights(self):
        # Initialize differently based on Network Convolutions versus Recurrent Convolutions.
        for m in self.modules():
            if isinstance(m, self.rtype):
                m.init_weights()
    
            elif isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.ConvTranspose3d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", a=0.25)
