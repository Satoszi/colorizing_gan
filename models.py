import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, filters):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self._block(3         , 2*filters, 4, 2, 0, bias=False), # out 32x32 #Shouldn't be batchnorm
            self._block(2*filters, 4*filters , 4, 2, 1, bias=False), # out 16x16
            self._block(4*filters , 8*filters , 4, 2, 1, bias=False), # out 8x8
            self._block(8*filters , 16*filters , 4, 2, 1, bias=False), # out 4x4
            self._block(16*filters , 1   , 4, 2, 1, bias=False, out=True), # out 1x1 #Shouldn't be batchnorm
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias, out=False):
        return nn.Sequential(            
            nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, 
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, 
                               bias=bias),
            
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid() if out else nn.LeakyReLU(0.2)
            )
        
    def forward(self, x):
        return self.disc( x)


class Generator(nn.Module):
    def __init__(self,nz, filters):
        super(Generator, self).__init__()
        self.gan = nn.Sequential(
            self._block(nz        , 16*filters, 4, 1, 0, bias=False), # out 4x4
            self._block(16*filters, 8*filters , 4, 2, 1, bias=False), # out 8x8
            self._block(8*filters , 4*filters , 4, 2, 1, bias=False), # out 16x16
            self._block(4*filters , 2*filters , 4, 2, 1, bias=False), # out 32x32
            self._block(2*filters , 3         , 4, 2, 1, bias=False, out=True), # out 64x64
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bias, out=False):
        return nn.Sequential(            
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels, 
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, 
                               bias=bias),
            
            nn.BatchNorm2d(out_channels),
            nn.Tanh() if out else nn.ReLU()
            )
        
    def forward(self, x):
        return self.gan( x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# def test():
#     x = torch.randn((5,3,64,64))
#     disc = Discriminator(16)
#     print(disc.forward(x).shape)
    
#     x = torch.randn((5,100,1,1))
#     gen = Generator(100,16)
#     print(gen.forward(x).shape)