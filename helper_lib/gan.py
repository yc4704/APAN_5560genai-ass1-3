from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()

        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)

        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)
        return self.tanh(x)