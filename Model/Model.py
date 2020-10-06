import torch
from Utils.help_funcs import divide_input_to_patches

def find_closest_num(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

    if abs(n - n1) < abs(n - n2):
        return n1
    return n2


def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BallDetector(torch.nn.Module):
    def __init__(self, config):
        super(BallDetector, self).__init__()
        self.config = config
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
        out_tensor = torch.zeros(size=torch.Size([x.shape[0], x.shape[2], x.shape[3]]))
        for start_row_idx, end_row_idx, start_col_idx, end_col_idx in divide_input_to_patches(x_shape=list(x.shape),
                                                                                              config=self.config):
            patch_out = self.feed_forward(input=x[:, :, start_row_idx:end_row_idx, start_col_idx:end_col_idx])
            # TODO: think to add logic to sync patches outputs
            out_tensor[:, :, start_row_idx:end_row_idx, start_col_idx:end_col_idx] = patch_out
        return out_tensor

    def feed_forward(self, input):
        out_1 = self.block1(input)
        out_2 = self.block2(out_1)
        out_3 = self.block3(out_2)

        upsampled_out2 = self.upsample2(out_2)
        upsampled_out3 = self.upsample3(out_3)

        concatenated = torch.cat((out_1, upsampled_out2, upsampled_out3), dim=1)
        out_4 = self.block4(concatenated)
        return out_4


