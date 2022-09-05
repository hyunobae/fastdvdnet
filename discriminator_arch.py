import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, pgd):
        super(Discriminator, self).__init__()
        self.pgd = pgd
        self.fsize = 64
        if self.pgd == 0:
            self.fsize = 16

        elif self.pgd == 1:
            self.fsize = 32

        elif self.pgd == 2:
            self.fsize = 64

        self.net = nn.Sequential(
            nn.Conv2d(3, self.fsize, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.fsize, self.fsize, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.fsize, self.fsize*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.fsize*2, self.fsize*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.fsize*2, self.fsize*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.fsize*4, self.fsize*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.fsize*4, self.fsize*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(self.fsize*8, self.fsize*8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.fsize*8, self.fsize*16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.fsize*16, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))