"""
1D ResNet模型实现
基于Ronin源代码，适配用于IMU速度估计

原始代码来自: https://github.com/Sachini/ronin
论文: RoNIN: Robust Neural Inertial Navigation in the Wild (ICRA 2020)
"""

import torch
import torch.nn as nn


def conv3(in_planes, out_planes, kernel_size, stride=1, dilation=1):
    """1D卷积层，kernel_size为3"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=False,
    )


class BasicBlock1D(nn.Module):
    """1D ResNet基础块"""

    expansion = 1

    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None
    ):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3(in_planes, out_planes, kernel_size, stride, dilation)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_planes, out_planes, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    """1D ResNet瓶颈块"""

    expansion = 4

    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None
    ):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.conv2 = conv3(out_planes, out_planes, kernel_size, stride, dilation)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.conv3 = nn.Conv1d(
            out_planes, out_planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FCOutputModule(nn.Module):
    """
    全连接输出模块
    """

    def __init__(self, in_planes, num_outputs, **kwargs):
        """
        构造全连接输出层

        Args:
            in_planes: 输入通道数
            num_outputs: 输出维度
            fc_dim: 全连接层维度
            dropout: dropout概率
            in_dim: 输入时间维度
            trans_planes: 可选的过渡卷积层通道数
        """
        super(FCOutputModule, self).__init__()
        fc_dim = kwargs.get("fc_dim", 1024)
        dropout = kwargs.get("dropout", 0.5)
        in_dim = kwargs.get("in_dim", 7)
        trans_planes = kwargs.get("trans_planes", None)

        if trans_planes is not None:
            self.transition = nn.Sequential(
                nn.Conv1d(in_planes, trans_planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(trans_planes),
            )
            in_planes = trans_planes
        else:
            self.transition = None

        self.fc = nn.Sequential(
            nn.Linear(in_planes * in_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_outputs),
        )

    def get_dropout(self):
        return [m for m in self.fc if isinstance(m, torch.nn.Dropout)]

    def forward(self, x):
        if self.transition is not None:
            x = self.transition(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


class GlobAvgOutputModule(nn.Module):
    """
    全局平均池化输出模块
    """

    def __init__(self, in_planes, num_outputs):
        super(GlobAvgOutputModule, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_planes, num_outputs)

    def get_dropout(self):
        return []

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNet1D(nn.Module):
    """
    1D ResNet网络
    用于时间序列IMU数据处理
    """

    def __init__(
        self,
        num_inputs,
        num_outputs,
        block_type,
        group_sizes,
        base_plane=64,
        output_block=None,
        zero_init_residual=False,
        **kwargs,
    ):
        super(ResNet1D, self).__init__()
        self.base_plane = base_plane
        self.inplanes = self.base_plane

        # 输入模块
        self.input_block = nn.Sequential(
            nn.Conv1d(
                num_inputs,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # 残差组
        self.planes = [self.base_plane * (2**i) for i in range(len(group_sizes))]
        kernel_size = kwargs.get("kernel_size", 3)
        strides = [1] + [2] * (len(group_sizes) - 1)
        dilations = [1] * len(group_sizes)
        groups = [
            self._make_residual_group1d(
                block_type,
                self.planes[i],
                kernel_size,
                group_sizes[i],
                strides[i],
                dilations[i],
            )
            for i in range(len(group_sizes))
        ]
        self.residual_groups = nn.Sequential(*groups)

        # 输出模块
        if output_block is None:
            self.output_block = GlobAvgOutputModule(
                self.planes[-1] * block_type.expansion, num_outputs
            )
        else:
            self.output_block = output_block(
                self.planes[-1] * block_type.expansion, num_outputs, **kwargs
            )

        self._initialize(zero_init_residual)

    def _make_residual_group1d(
        self, block_type, planes, kernel_size, blocks, stride=1, dilation=1
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block_type.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block_type.expansion),
            )

        layers = []
        layers.append(
            block_type(
                self.inplanes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block_type.expansion

        for _ in range(1, blocks):
            layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        x = self.output_block(x)
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_ronin_resnet(arch="resnet18", num_inputs=6, num_outputs=2, window_size=200):
    """
    创建RoNIN ResNet模型

    Args:
        arch: 架构类型 ('resnet18', 'resnet50', 'resnet101')
        num_inputs: 输入通道数 (默认6: IMU 6轴)
        num_outputs: 输出维度 (默认2: vx, vy)
        window_size: 窗口大小 (默认200)

    Returns:
        ResNet1D模型
    """
    fc_config = {
        "fc_dim": 512,
        "in_dim": window_size // 32 + 1,  # 根据网络降采样计算
        "dropout": 0.5,
        "trans_planes": 128,
    }

    if arch == "resnet18":
        network = ResNet1D(
            num_inputs,
            num_outputs,
            BasicBlock1D,
            [2, 2, 2, 2],
            base_plane=64,
            output_block=FCOutputModule,
            kernel_size=3,
            **fc_config,
        )
    elif arch == "resnet50":
        fc_config["fc_dim"] = 1024
        network = ResNet1D(
            num_inputs,
            num_outputs,
            BasicBlock1D,
            [3, 4, 6, 3],
            base_plane=64,
            output_block=FCOutputModule,
            kernel_size=3,
            **fc_config,
        )
    elif arch == "resnet101":
        fc_config["fc_dim"] = 1024
        network = ResNet1D(
            num_inputs,
            num_outputs,
            BasicBlock1D,
            [3, 4, 23, 3],
            base_plane=64,
            output_block=FCOutputModule,
            kernel_size=3,
            **fc_config,
        )
    else:
        raise ValueError(f"Invalid architecture: {arch}")

    return network


if __name__ == "__main__":
    # 测试模型
    model = get_ronin_resnet("resnet18", num_inputs=6, num_outputs=2, window_size=200)
    print(f"Total parameters: {model.get_num_params():,}")

    # 测试前向传播
    batch_size = 4
    num_inputs = 6
    seq_len = 200
    x = torch.randn(batch_size, num_inputs, seq_len)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
