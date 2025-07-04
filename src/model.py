import matplotlib.pyplot as plt
from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def weight_init_normal (m) :
    classname = m.__class__.__name__
    if classname.find ('Conv') != -1 :
        nn.init.normal_ (m.weight.data, 0.0, 0.02)
        if m.bias is not None :
            nn.init.constant_ (m.bias.data, 0.0)
    elif classname.find ('BatchNorm2d') != -1 :
        nn.init.normal_ (m.weight.data, 1.0, 0.02)
        nn.init.constant_ (m.bias.data, 0.0)

class Tumor3DCNN (nn.Module) :
    def __init__ (self, features=32) :
        super ().__init__ ()
        self.features = features
        self.conv1 = nn.Conv3d (1, features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d (features)
        self.resconv1 = ResBlock (features)
        self.conv2 = nn.Conv3d (features, features * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d (features * 2)
        self.resconv2 = ResBlock (features * 2)
        self.conv3 = nn.Conv3d (features * 2, features * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d (features * 4)
        self.resconv3 = ResBlock (features * 4)
        self.pool = nn.MaxPool3d (2, 2)
        self.fc1 = nn.Linear (features * 4 * 8 * 8 * 8, features * 4)
        self.dropout = nn.Dropout (0.5)
        self.fc2 = nn.Linear (features * 4, 1)
        self.init_weights ()

    def forward (self, x) :
        x = self.pool (F.relu (self.bn1 (self.conv1 (x))))
        x = self.bn1 (self.resconv1 (x))
        x = self.pool (F.relu (self.bn2 (self.conv2 (x))))
        x = self.bn2 (self.resconv2 (x))
        x = self.pool (F.relu (self.bn3 (self.conv3 (x))))
        x = self.bn3 (self.resconv3 (x))
        #print (x.shape)
        #print (x.size (0))
        x = x.view (x.size (0), 8 * 8 * 8 * self.features * 4)
        x = F.relu (self.fc1 (x))
        x = self.dropout (x)
        x = self.fc2 (x)
        #print (x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

class Block (nn.Module) :
    def __init__ (self, in_channels, out_channels, use_dropout=False, down=True) :
        super ().__init__ ()
        self.conv = nn.Sequential (
            nn.Conv2d (in_channels, out_channels, 4, 2, 1, bias=False)
            if down
            else nn.ConvTranspose2d (in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d (out_channels),
            nn.LeakyReLU (0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout (0.5)
    
    def forward (self, x) :
        x = self.conv (x)
        if self.use_dropout :
            return self.dropout (x)
        return x

class Unet (nn.Module) :
    def __init__ (self, use_dropout=False, features=64) :
        super ().__init__ ()
        self.initial_down = nn.Sequential (
            nn.Conv2d (1, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU (0.2),
        )
        self.down1 = Block (features, features * 2, use_dropout=False)
        self.down2 = Block (features * 2, features * 4, use_dropout=False)
        self.down3 = Block (features * 4, features * 8, use_dropout=False)
        self.down4 = Block (features * 8, features * 8, use_dropout=False)
        self.down5 = Block (features * 8, features * 8, use_dropout=False)
        self.down6 = Block (features * 8, features * 8, use_dropout=False)
        self.bottleneck = nn.Sequential (
            nn.Conv2d (features * 8, features * 8, 4, 2, 1),
            nn.ReLU (),
        )
        self.up1 = Block (features * 8, features * 8, use_dropout=True, down=False)
        self.up2 = Block (features * 16, features * 8, use_dropout=True, down=False)
        self.up3 = Block (features * 16, features * 8, use_dropout=True, down=False)
        self.up4 = Block (features * 16, features * 8, use_dropout=True, down=False)
        self.up5 = Block (features * 16, features * 4, use_dropout=True, down=False)
        self.up6 = Block (features * 8, features * 2, use_dropout=True, down=False)
        self.up7 = Block (features * 4, features, use_dropout=True, down=False)
        self.final_up = nn.Sequential (
            nn.ConvTranspose2d (features * 2, 1, 4, 2, 1),
            #nn.Tanh ()
        )

    def forward (self, x) :
        d1 = self.initial_down (x)
        #print ("d1.shape = ", d1.shape)
        d2 = self.down1 (d1)
        d3 = self.down2 (d2)
        d4 = self.down3 (d3)
        #d5 = self.down4 (d4)
        #d6 = self.down5 (d5)
        #d7 = self.down6 (d6)
        #print ("d7.shape = ", d7.shape)
        bottleneck = self.bottleneck (d4)
        u1 = self.up1 (bottleneck)
        #u2 = self.up2 (torch.cat ([u1, d7], dim=1))
        #u3 = self.up3 (torch.cat ([u2, d6], dim=1))
        #u4 = self.up4 (torch.cat ([u1, d5], dim=1))
        u5 = self.up5 (torch.cat ([u1, d4], dim=1))
        u6 = self.up6 (torch.cat ([u5, d3], dim=1))
        u7 = self.up7 (torch.cat ([u6, d2], dim=1))
        final = self.final_up (torch.cat ([u7, d1], dim=1))
        return final

class ResBlock (nn.Module) :
    def __init__ (self, features) :
        super ().__init__ ()
        self.conv1 = nn.Conv3d (features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d (features, features, kernel_size=3, padding=1)
        self.alpha = nn.Parameter (torch.tensor (0.5))

    def forward (self, x) :
        x = F.relu (self.conv1 (x))
        y = self.conv2 (x)
        a = torch.sigmoid (self.alpha)
        return F.relu (a * x + (1 - a) * y)

