import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=5, stride=1, padding=1),
            nn.Conv2d(features_d, features_d, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 3, 2, 1),
            self._block(features_d * 2, features_d * 4, 3, 2, 1),
            self._block(features_d * 4, features_d * 8, 3, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=3, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(Generator, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels,  features * 1, 3, 2, 1)  # Input: [1, 64, 64], Output: [features, 32, 32]
        self.enc2 = self._conv_block(features * 1, features * 2, 3, 2, 1)  # Input: [features, 32, 32], Output: [features*2, 16, 16]
        self.enc3 = self._conv_block(features * 2, features * 4, 3, 2, 1)  # Input: [features*2, 16, 16], Output: [features*4, 8, 8]
        self.enc4 = self._conv_block(features * 4, features * 8, 3, 2, 1)  # Input: [features*4, 8, 8], Output: [features*8, 4, 4]

        # Decoder
        self.dec0 = self._tconv_block(features * 8, features * 4, 4, 2, 1)  # Input: [features*8, 4, 4], Output: [features*4, 8, 8]
        self.dec1 = self._tconv_block(features * 8, features * 4, 4, 2, 1)  # Input: [features*8, 8, 8], Output: [features*4, 16, 16]
        self.dec2 = self._tconv_block(features * 6, features * 2, 4, 2, 1)  # Input: [features*6, 16, 16], Output: [features*2, 32, 32]
        self.dec3 = self._tconv_block(features * 3, features * 1, 4, 2, 1)  # Input: [features*3, 16, 16], Output: [features, 64, 64]
        self.dec_out = nn.Sequential(
            nn.Conv2d(
                features+1,
                out_channels,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.Tanh()
        )  # Input: [features, 64, 64], Output: [out_channels, 64, 64]

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _tconv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)  # Output shape: [features, 32, 32]
        enc2 = self.enc2(enc1)  # Output shape: [features*2, 16, 16]
        enc3 = self.enc3(enc2)  # Output shape: [features*4, 8, 8]
        enc4 = self.enc4(enc3)  # Output shape: [features*8, 4, 4]
        # Decoding path
        dec0 = self.dec0(enc4)  # Output shape: [features*4, 8, 8]
        dec0 = torch.cat((dec0, enc3), dim=1)  # Output shape: [features*8, 8, 8]

        dec1 = self.dec1(dec0)  # Output shape: [features*2, 16, 16]
        dec1 = torch.cat((dec1, enc2), dim=1)  # Output shape: [features*4, 16, 16]

        dec2 = self.dec2(dec1)  # Output shape: [features, 32, 32]
        dec2 = torch.cat((dec2, enc1), dim=1)  # Output shape: [features*2, 32, 32]

        dec3 = self.dec3(dec2)  # Output shape: [features, 64, 64]
        dec3 = torch.cat((dec3, x), dim=1)
        out = self.dec_out(dec3)
        return out


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()