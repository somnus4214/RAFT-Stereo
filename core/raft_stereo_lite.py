import torch
import torch.nn as nn
import torch.nn.functional as F

from core.corr import (
    AlternateCorrBlock,
    CorrBlock1D,
    CorrBlockFast1D,
    PytorchAlternateCorrBlock1D,
)
from core.extractor import ResidualBlock
from core.update import BasicMotionEncoder, ConvGRU, FlowHead, interp, pool2x
from core.utils.utils import coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class LiteResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(LiteResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False
        )
        self.conv2 = DepthwiseSeparableConv(planes, planes)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8
        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1 or in_planes != planes:
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                self.norm3,
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class LiteBasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0, downsample=3):
        super(LiteBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=7, stride=1 + (downsample > 2), padding=3
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(96, stride=1 + (downsample > 0))

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

    def _make_layer(self, dim, stride=1):
        layer1 = LiteResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = LiteResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x, dual_inp=False):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = x.split(split_size=batch_dim, dim=0)

        return x


class LiteMultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], norm_fn="batch", dropout=0.0, downsample=3):
        super(LiteMultiBasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=7, stride=1 + (downsample > 2), padding=3
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(96, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(96, stride=2)
        self.layer5 = self._make_layer(96, stride=2)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                LiteResidualBlock(96, 96, self.norm_fn, stride=1),
                nn.Conv2d(96, dim[2], 3, padding=1),
            )
            output_list.append(conv_out)
        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                LiteResidualBlock(96, 96, self.norm_fn, stride=1),
                nn.Conv2d(96, dim[1], 3, padding=1),
            )
            output_list.append(conv_out)
        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(96, dim[0], 3, padding=1)
            output_list.append(conv_out)
        self.outputs32 = nn.ModuleList(output_list)

    def _make_layer(self, dim, stride=1):
        layer1 = LiteResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = LiteResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x, dual_inp=False, num_layers=3):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if dual_inp:
            v = x
            x = x[: (x.shape[0] // 2)]

        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        x = self.layer4(x)
        outputs16 = [f(x) for f in self.outputs16]
        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        x = self.layer5(x)
        outputs32 = [f(x) for f in self.outputs32]
        return (
            (outputs08, outputs16, outputs32, v)
            if dual_inp
            else (outputs08, outputs16, outputs32)
        )


class LiteMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.gru08 = ConvGRU(
            hidden_dims[2],
            encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1),
        )
        if args.n_gru_layers >= 2:
            self.gru16 = ConvGRU(
                hidden_dims[1],
                hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2],
            )
        if args.n_gru_layers == 3:
            self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])

        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=128, output_dim=2)
        factor = 2**self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor**2) * 9, 1, padding=0),
        )

    def forward(
        self,
        net,
        inp,
        corr=None,
        flow=None,
        iter08=True,
        iter16=True,
        iter32=True,
        update=True,
    ):
        if iter32 and self.args.n_gru_layers == 3:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16 and self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru16(
                    net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1])
                )
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru08(
                    net[0], *(inp[0]), motion_features, interp(net[1], net[0])
                )
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])
        mask = 0.25 * self.mask(net[0])
        return net, mask, delta_flow


class LiteRAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Determine actual hidden dims based on lite settings
        # Defaults to args.hidden_dims if they are customized, else defaults to lighter 96
        base_hidden = getattr(args, "hidden_dim", 96)

        if getattr(args, "model_type", "base") == "lite" and base_hidden != 128:
            self.hidden_dims = [base_hidden] * args.n_gru_layers
            context_dims = self.hidden_dims
        else:
            self.hidden_dims = args.hidden_dims
            context_dims = args.hidden_dims

        self.cnet = LiteMultiBasicEncoder(
            output_dim=[self.hidden_dims, context_dims],
            norm_fn=args.context_norm,
            downsample=args.n_downsample,
        )
        self.update_block = LiteMultiUpdateBlock(
            self.args, hidden_dims=self.hidden_dims
        )

        self.context_zqr_convs = nn.ModuleList(
            [
                nn.Conv2d(context_dims[i], self.hidden_dims[i] * 3, 3, padding=3 // 2)
                for i in range(self.args.n_gru_layers)
            ]
        )

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                LiteResidualBlock(96, 96, "instance", stride=1),
                nn.Conv2d(96, 128, 3, padding=1),
            )
        else:
            self.fnet = LiteBasicEncoder(
                output_dim=128, norm_fn="instance", downsample=args.n_downsample
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        N, D, H, W = flow.shape
        factor = 2**self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        image1_raw = image1
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(
                    torch.cat((image1, image2), dim=0),
                    dual_inp=True,
                    num_layers=self.args.n_gru_layers,
                )
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [
                list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                for i, conv in zip(inp_list, self.context_zqr_convs)
            ]

        if self.args.corr_implementation == "reg":
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt":
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda":
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda":
            corr_block = AlternateCorrBlock

        corr_fn = corr_block(
            fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels
        )

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and getattr(
                    self.args, "slow_fast_gru", False
                ):
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=True,
                        iter16=False,
                        iter08=False,
                        update=False,
                    )
                if self.args.n_gru_layers >= 2 and getattr(
                    self.args, "slow_fast_gru", False
                ):
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=self.args.n_gru_layers == 3,
                        iter16=True,
                        iter08=False,
                        update=False,
                    )
                net_list, up_mask, delta_flow = self.update_block(
                    net_list,
                    inp_list,
                    corr,
                    flow,
                    iter32=self.args.n_gru_layers == 3,
                    iter16=self.args.n_gru_layers >= 2,
                )

            delta_flow[:, 1] = 0.0
            coords1 = coords1 + delta_flow

            if test_mode and itr < iters - 1:
                continue

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:, :1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_predictions[-1]

        return flow_predictions
