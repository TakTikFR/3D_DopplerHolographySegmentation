import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Briques de base ─────────────────────────────────────────────────────────

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, upsample='deconv'):
        super().__init__()
        if upsample == 'bilinear':
            # trilinear en 3D
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        elif upsample == 'deconv':
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            raise ValueError("Upsample method doit être 'bilinear' ou 'deconv' (pixelshuffle n'a pas d'équivalent 3D natif).")

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Padding sur les 3 dimensions spatiales : D, H, W
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
            diffZ // 2, diffZ - diffZ // 2,
        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ─── UNet 3D principal ───────────────────────────────────────────────────────

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, out_channels=32, upsample='deconv'):
        """
        n_channels : canaux d'entrée (ex: 1 pour un volume mono-canal)
        n_classes  : canaux de sortie (segmentation)
        out_channels : base de channels (réduit à 32 vs 64 en 2D
                       car la mémoire GPU explose en 3D)
        upsample   : 'deconv' ou 'bilinear' ('pixelshuffle' non supporté en 3D)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        factor = 2 if upsample == 'bilinear' else 1
        C = out_channels

        # Encodeur
        self.inc   = DoubleConv3D(n_channels, C)
        self.down1 = Down3D(C,      C * 2)
        self.down2 = Down3D(C * 2,  C * 4)
        self.down3 = Down3D(C * 4,  C * 8)
        self.down4 = Down3D(C * 8,  C * 16 // factor)  # bottleneck

        # Décodeur
        self.up1 = Up3D(C * 16, C * 8  // factor, upsample)
        self.up2 = Up3D(C * 8,  C * 4  // factor, upsample)
        self.up3 = Up3D(C * 4,  C * 2  // factor, upsample)
        self.up4 = Up3D(C * 2,  C,                upsample)

        # Couche de sortie
        self.outc = nn.Conv3d(C, n_classes, kernel_size=1)

    def forward(self, pixel_values, labels=None, **kwargs):
        x = pixel_values
        # Encodeur
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Décodeur avec skip connections
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)

