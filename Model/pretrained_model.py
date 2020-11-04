import torch
import torchvision
from Utils.help_funcs import divide_input_to_patches
import copy
from.Model import init_weights

class PreTrainedVggModel(torch.nn.Module):
    def __init__(self, config, device):
        super(PreTrainedVggModel, self).__init__()
        self.config = config
        self.device = device
        pretrained_backbone = torchvision.models.vgg19_bn(pretrained=True)
        self.vgg_block1 = copy.deepcopy(pretrained_backbone.features[:7])  # out channels = 64
        self.vgg_block2 = copy.deepcopy(pretrained_backbone.features[7:14])  # out channels = 128
        self.vgg_block3 = copy.deepcopy(pretrained_backbone.features[14:27])  # out channels = 256

        del pretrained_backbone
        # self.vgg_block4 = pretrained_backbone.features[27:40]  # out channels = 512

        self.block_5 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=448, out_channels=32, kernel_size=3),
                                           torch.nn.BatchNorm2d(num_features=32),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),
                                           torch.nn.BatchNorm2d(num_features=16),
                                           torch.nn.ReLU(inplace=True),
                                           torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1),
                                           )

    def feed_forward(self, x):
        out_1 = self.vgg_block1(x)
        out_2 = self.vgg_block2(out_1)
        out_3 = self.vgg_block3(out_2)
        # out_4 = self.vgg_block4(out_3)

        upsampling_layer = torch.nn.Upsample(size=(out_1.shape[2], out_1.shape[3]), mode='bicubic', align_corners=True)

        upsampled_out2 = upsampling_layer(out_2)
        upsampled_out3 = upsampling_layer(out_3)
        # upsampled_out4 = upsampling_layer(out_4)
        concatenated = torch.cat((out_1, upsampled_out2, upsampled_out3), dim=1)
        out_5 = self.block_5(concatenated)
        return out_5

    def forward(self, X):
        # compute output tensor shape according to patch sizes and
        in_w, in_h = X.shape[3], X.shape[2]
        out1_1_shape = (int(in_h - 3 + 2) + 1, int(in_w - 3 + 2) + 1)
        out1_2_shape = (out1_1_shape[0] - 3 + 1, out1_1_shape[1] - 3 + 1)

        # TODO: change this method to multi-threading
        out_tensor = torch.zeros(size=torch.Size([X.shape[0], 2, out1_2_shape[0], out1_2_shape[1]]))
        out_tensor.to(device="cuda:0")

        i_h, i_w = 0, 0
        for start_row_idx, end_row_idx, start_col_idx, end_col_idx in divide_input_to_patches(x_shape=list(X.shape),
                                                                                              config=self.config):
            patch_out = self.feed_forward(x=X[:, :, start_row_idx:end_row_idx, start_col_idx:end_col_idx].to(self.device))
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

    def freeze_backbone(self):
        for block in [self.vgg_block1, self.vgg_block2, self.vgg_block3]:
            for child in block.children():
                if isinstance(child, torch.nn.Conv2d) or isinstance(child, torch.nn.BatchNorm2d):
                    child.weight.requires_grad = False
                    if child.bias is not None:
                        child.bias.requires_grad = False

    def init_model_out_block(self):
        self.block_5.apply(init_weights)

