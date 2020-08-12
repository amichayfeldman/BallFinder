import torch


class BallDetector(torch.nn.Module):
    def __init__(self):
        super(BallDetector, self).__init__()
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2),
                                          torch.nn.BatchNorm2d(num_features=8),
                                          torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
                                          torch.nn.BatchNorm2d(num_features=8),
                                          torch.nn.MaxPool2d(kernel_size=2))

        self.block2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
                                          torch.nn.BatchNorm2d(num_features=16),
                                          torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
                                          torch.nn.BatchNorm2d(num_features=16),
                                          torch.nn.MaxPool2d(kernel_size=2))

        self.upsample2 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2)

        self.block3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
                                          torch.nn.BatchNorm2d(num_features=32),
                                          torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
                                          torch.nn.BatchNorm2d(num_features=32),
                                          torch.nn.MaxPool2d(kernel_size=2))

        self.upsample3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4)

        self.block4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3),
                                          torch.nn.BatchNorm2d(num_features=56),
                                          torch.nn.Conv2d(in_channels=56, out_channels=2, kernel_size=3))

    def forward(self, x):
        out_1 = self.block1(x)
        out_2 = self.block2(out_1)
        out_3 = self.block3(out_2)

        upsampled_out2 = self.upsample2(out_2)
        upsampled_out3 = self.upsample3(out_3)

        concatenated = torch.cat((out_1, upsampled_out2, upsampled_out3), dim=1)
        out_4 = self.block4(concatenated)
        return out_4



