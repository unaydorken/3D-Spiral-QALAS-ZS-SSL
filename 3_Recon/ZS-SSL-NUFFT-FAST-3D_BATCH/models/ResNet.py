import torch
import torch.nn as nn

def activation_func(activation, is_inplace=False):
    return nn.ModuleDict([
        ['ReLU', nn.ReLU(inplace=is_inplace)],
        ['None', nn.Identity()],
    ])[activation]


def BatchNorm(is_batch_norm, features, ndim=2):

    if not is_batch_norm:
        return nn.Identity()

    if ndim == 2:
        return nn.BatchNorm2d(features)

    elif ndim == 3:
        return nn.BatchNorm3d(features)

    else:
        raise ValueError(
            f"Unsupported ndim: {ndim}"
        )


def scaling_layer(is_scaling=True):

    scale_factor = torch.tensor(
        [0.1],
        dtype=torch.float32,
    )

    if is_scaling:
        return nn.Mul(scale_factor)
    else:
        return nn.Identity()


def conv_layer(
    filter_size,
    padding=1,
    is_batch_norm=False,
    activation_type='ReLU',
    ndim=2,
    *args,
    **kwargs,
):

    kernel_size, in_c, out_c = filter_size

    if ndim == 2:
        Conv = nn.Conv2d

    elif ndim == 3:
        Conv = nn.Conv3d

    else:
        raise ValueError(
            f"Unsupported ndim: {ndim}"
        )

    return nn.Sequential(
        Conv(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        ),
        BatchNorm(
            is_batch_norm,
            out_c,
            ndim=ndim,
        ),
        activation_func(
            activation_type
        ),
    )


def ResNetBlock(
    filter_size,
    ndim=2,
):

    return nn.Sequential(
        conv_layer(
            filter_size,
            activation_type='ReLU',
            ndim=ndim,
        ),
        conv_layer(
            filter_size,
            activation_type='None',
            ndim=ndim,
        ),
    )


class ResNetBlocksModule(nn.Module):

    def __init__(
        self,
        device,
        filter_size,
        num_blocks,
        ndim=2,
        checkpoint_group_size=4,
    ):

        super().__init__()

        self.device = device
        self.ndim = ndim

        # Number of residual blocks recomputed together
        # during backward for 3D training.
        self.checkpoint_group_size = checkpoint_group_size

        self.layers = nn.ModuleList([
            ResNetBlock(
                filter_size=filter_size,
                ndim=ndim,
            )
            for _ in range(num_blocks)
        ])

        self.trace = []

    def forward(self, x):

        scale_factor = torch.tensor(
            [0.1],
            dtype=torch.float32,
            device=x.device,
        )

        for layer in self.layers:

            x = (
                x
                + layer(x)
                * scale_factor
            )

        return x


class ResNet(nn.Module):

    def __init__(
        self,
        device,
        in_ch=2,
        num_of_resblocks=5,
        ndim=2,
        features=8,
    ):

        super(ResNet, self).__init__()

        if ndim not in (2, 3):
            raise ValueError(
                f"Unsupported ndim: {ndim}"
            )

        self.in_ch = in_ch
        self.num_of_resblocks = num_of_resblocks
        self.device = device
        self.ndim = ndim

        kernel_size = 3
        padding = 1

        # Map input to feature maps
        filter1 = [
            kernel_size,
            in_ch,
            features,
        ]

        # ResNet blocks
        filter2 = [
            kernel_size,
            features,
            features,
        ]

        # Map feature maps back to input channels
        filter3 = [
            kernel_size,
            features,
            in_ch,
        ]

        self.layer1 = conv_layer(
            filter_size=filter1,
            activation_type='None',
            ndim=ndim,
        )

        self.layer2 = ResNetBlocksModule(
            device=self.device,
            filter_size=filter2,
            num_blocks=num_of_resblocks,
            ndim=ndim,
        )

        self.layer3 = conv_layer(
            filter_size=filter2,
            activation_type='None',
            ndim=ndim,
        )

        self.layer4 = conv_layer(
            filter_size=filter3,
            activation_type='None',
            ndim=ndim,
        )

    def forward(self, input_x):

        l1_out = self.layer1(
            input_x
        )

        l2_out = self.layer2(
            l1_out
        )

        l3_out = self.layer3(
            l2_out
        )

        temp = (
            l3_out
            + l1_out
        )

        nw_out = self.layer4(
            temp
        )

        return nw_out