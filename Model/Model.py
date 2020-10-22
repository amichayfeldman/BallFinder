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
        block1_c = 8
        self.block1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=block1_c, kernel_size=5, stride=2),
                                          torch.nn.BatchNorm2d(num_features=block1_c),
                                          torch.nn.Conv2d(in_channels=block1_c, out_channels=block1_c, kernel_size=3),
                                          torch.nn.BatchNorm2d(num_features=block1_c),
                                          torch.nn.MaxPool2d(kernel_size=2))

        block2_c = 16
        self.block2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=block1_c, out_channels=block2_c, kernel_size=3,
                                                          padding=1),
                                          torch.nn.BatchNorm2d(num_features=block2_c),
                                          torch.nn.Conv2d(in_channels=block2_c, out_channels=block2_c, kernel_size=3,
                                                          padding=1),
                                          torch.nn.BatchNorm2d(num_features=block2_c),
                                          torch.nn.MaxPool2d(kernel_size=2))

        # self.upsample2 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2)
        # self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='bicubic')

        block3_c = 32
        self.block3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=block2_c, out_channels=block3_c, kernel_size=3,
                                                          padding=1),
                                          torch.nn.BatchNorm2d(num_features=block3_c),
                                          torch.nn.Conv2d(in_channels=block3_c, out_channels=block3_c, kernel_size=3,
                                                          padding=1),
                                          torch.nn.BatchNorm2d(num_features=block3_c),
                                          torch.nn.MaxPool2d(kernel_size=2))

        # self.upsample3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4)
        # self.upsample3 = torch.nn.Upsample(scale_factor=2, mode='bicubic')

        block4_c = block1_c + block2_c + block3_c
        self.block4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=block4_c, out_channels=block4_c, kernel_size=3,
                                                          padding=1),
                                          torch.nn.BatchNorm2d(num_features=block4_c),
                                          torch.nn.Conv2d(in_channels=block4_c, out_channels=2, kernel_size=3,
                                                          padding=1))

        self.out1_1_shape, self.out1_2_shape, self.out2_shape, self.out3_shape = [None] * 4

    def forward(self, x):
        # compute output tensor shape according to patch sizes and
        patch_w, patch_h = self.config.getint('Params', 'patch_w'), self.config.getint('Params', 'patch_h')
        in_w, in_h = x.shape[3], x.shape[2]
        self.out1_1_shape = (int((in_h - 5) / 2) + 1, int((in_w - 5) / 2) + 1)
        self.out1_2_shape = (int((self.out1_1_shape[0] - 3 + 1) / 2), int((self.out1_1_shape[1] - 3 + 1) / 2))

        # TODO: change this method to multi-threading
        out_tensor = torch.zeros(size=torch.Size([x.shape[0], 2, self.out1_2_shape[0], self.out1_2_shape[1]]))
        out_tensor.to(device="cuda:0")

        i_h, i_w = 0, 0
        for start_row_idx, end_row_idx, start_col_idx, end_col_idx in divide_input_to_patches(x_shape=list(x.shape),
                                                                                              config=self.config):
            patch_out = self.feed_forward(X=x[:, :, start_row_idx:end_row_idx, start_col_idx:end_col_idx])
            if int(patch_out.shape[2] * (1 + (i_h / 2))) <= out_tensor.shape[2]:
                p_h_start, p_h_end = i_h * int(patch_out.shape[2] / 2), int(patch_out.shape[2] * (1 + (i_h / 2)))
            else:
                p_h_start, p_h_end = -patch_out.shape[2], out_tensor.shape[2]
            if int(patch_out.shape[3] * (1 + (i_w / 2))) <= out_tensor.shape[3]:
                p_w_start, p_w_end = i_w * int(patch_out.shape[3] / 2), int(patch_out.shape[3] * (1 + (i_w / 2)))
            else:
                p_w_start, p_w_end = -patch_out.shape[3], out_tensor.shape[3]
                i_h += 1
                i_w = -1
            i_w += 1
            out_tensor[:, :, p_h_start:p_h_end, p_w_start:p_w_end] = patch_out

        # softmax = torch.nn.Softmax(dim=1)
        # output = softmax(out_tensor)
        return out_tensor

    def feed_forward(self, X):
        out_1 = self.block1(X)
        out_2 = self.block2(out_1)
        out_3 = self.block3(out_2)

        upsampling_layer = torch.nn.Upsample(size=(out_1.shape[2], out_1.shape[3]), mode='bicubic')

        upsampled_out2 = upsampling_layer(out_2)
        upsampled_out3 = upsampling_layer(out_3)

        concatenated = torch.cat((out_1, upsampled_out2, upsampled_out3), dim=1)
        out_4 = self.block4(concatenated)
        return out_4


