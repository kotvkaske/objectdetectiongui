import os
import torch
from torch import nn
import torch.nn.functional as F


class SegNet(nn.Module):
    """Архитектура SegNet для бинарной сегментации (0 - фон, 1 - человек)"""

    def __init__(self):
        super().__init__()
        self.encoder_conv_00 = nn.Sequential(*[
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])
        self.encoder_conv_01 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])
        self.encoder_conv_10 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        self.encoder_conv_11 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        self.encoder_conv_20 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.encoder_conv_21 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.encoder_conv_22 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.encoder_conv_30 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.encoder_conv_31 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.encoder_conv_32 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])

        #### DECODER
        self.decoder_convtr_32 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_31 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_30 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_22 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_21 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_20 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128)
        ])
        self.decoder_convtr_11 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(128)
        ])
        self.decoder_convtr_10 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64)
        ])
        self.decoder_convtr_01 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1),
            nn.BatchNorm2d(64)
        ])
        self.decoder_convtr_00 = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=1,
                               kernel_size=3,
                               padding=1)
        ])

    def forward(self, x):
        # encoder
        dim_0 = x.size()
        x_00 = F.relu(self.encoder_conv_00(x))
        x_01 = F.relu(self.encoder_conv_01(x_00))
        x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        x_22 = F.relu(self.encoder_conv_22(x_21))
        x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        x_32 = F.relu(self.encoder_conv_32(x_31))
        x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)

        # Decoder Stage - 4
        x_3d = F.max_unpool2d(x_3, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d = F.relu(self.decoder_convtr_32(x_3d))
        x_31d = F.relu(self.decoder_convtr_31(x_32d))
        x_30d = F.relu(self.decoder_convtr_30(x_31d))
        dim_3d = x_30d.size()

        # Decoder Stage - 3
        x_2d = F.max_unpool2d(x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d = F.relu(self.decoder_convtr_22(x_2d))
        x_21d = F.relu(self.decoder_convtr_21(x_22d))
        x_20d = F.relu(self.decoder_convtr_20(x_21d))
        dim_2d = x_20d.size()

        # Decoder Stage - 2
        x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = F.relu(self.decoder_convtr_10(x_11d))
        dim_1d = x_10d.size()

        # Decoder Stage - 1
        x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)
        dim_0d = x_00d.size()

        # x_softmax = F.softmax(x_00d, dim=1)

        return x_00d


class convblock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel=3, padding=0, stride=1):
        super(convblock, self).__init__()
        self.first_conv = nn.Sequential(
            *[nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=padding, stride=stride),
              nn.BatchNorm2d(mid_channels),
              nn.BatchNorm2d(mid_channels),
              nn.ReLU(),
              nn.Conv2d(mid_channels, out_channels, kernel_size=kernel),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()])

    def forward(self, x):
        x = self.first_conv(x)
        return x


def copy_crop_concat(x1, x2):  ##
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    x = torch.cat([x2, x1], dim=1)
    return x


class UNET_custom(nn.Module):
    def __init__(self):
        super(UNET_custom, self).__init__()
        self.first_block = convblock(3, 64, 64, padding=2)
        self.first_max_pool = nn.MaxPool2d(2, ceil_mode=True)

        self.second_block = convblock(64, 128, 128, padding=2)
        self.second_block_pool = nn.MaxPool2d(2, ceil_mode=True)

        self.third_block = convblock(128, 256, 256, padding=2)
        self.third_block_pool = nn.MaxPool2d(2, ceil_mode=True)

        self.forth_block = convblock(256, 512, 512, padding=2)
        #        self.forth_block_pool = nn.MaxPool2d(2, ceil_mode=True)

        self.forth_deconvblock = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.forth_convblock = convblock(512, 256, 256, padding=2)

        self.third_deconvblock = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.third_convblock = convblock(256, 128, 128, padding=2)

        self.second_deconvblock = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.second_convblock = convblock(128, 64, 64, padding=2)

        self.outp_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.first_block(x)
        x1 = x.clone()
        x = self.first_max_pool(x)
        #        print(x.shape, 'спуск 1')
        x = self.second_block(x)
        x2 = x.clone()
        x = self.second_block_pool(x)
        #        print(x.shape, 'спуск 2')
        x = self.third_block(x)
        x3 = x.clone()
        x = self.third_block_pool(x)
        #        print(x.shape, 'спуск 3')

        x = self.forth_block(x)
        #        print(x.shape, 'спуск 4')

        x = self.forth_deconvblock(x)
        x = copy_crop_concat(x, x3)
        x = self.forth_convblock(x)

        # # print(x.shape, 'подъем 2')
        x = self.third_deconvblock(x)
        x = copy_crop_concat(x, x2)
        x = self.third_convblock(x)
        # # print(x.shape, 'подъем 3')

        x = self.second_deconvblock(x)
        x = copy_crop_concat(x, x1)
        x = self.second_convblock(x)
        x = self.outp_layer(x)
        return x
