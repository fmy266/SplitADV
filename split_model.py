import torch, sys

sys.path.append("..")
import utils.general_utils as utils
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ViewModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def split(model_name, split_idx):
    """
    model_name: resnet
    """
    model = utils.load_model("resnet", "cifar10", size=18) if model_name == "resnet" else utils.load_model(model_name,
                                                                                                           "cifar10")
    if model_name == "resnet":
        if split_idx == 1:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1)
            intermediate_layer = torch.nn.Sequential(model.layer1, model.layer2, model.layer3, model.layer4,
                                                     torch.nn.AvgPool2d(4), ViewModel())
            late_layer = torch.nn.Linear(512, 10)
        elif split_idx == 2:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1, model.layer1)
            intermediate_layer = torch.nn.Sequential(model.layer2, model.layer3, model.layer4,
                                                     torch.nn.AvgPool2d(4), ViewModel())
            late_layer = torch.nn.Linear(512, 10)
        elif split_idx == 3:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1, model.layer1, model.layer2)
            intermediate_layer = torch.nn.Sequential(model.layer3, model.layer4, torch.nn.AvgPool2d(4), ViewModel())
            late_layer = torch.nn.Linear(512, 10)
        elif split_idx == 4:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1, model.layer1, model.layer2, model.layer3)
            intermediate_layer = torch.nn.Sequential(model.layer4, torch.nn.AvgPool2d(4), ViewModel())
            late_layer = torch.nn.Linear(512, 10)
    elif model_name == "mobilenetv2":
        if split_idx == 1:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1)
            intermediate_layer = torch.nn.Sequential(model.layers, model.conv2, model.bn2, torch.nn.AvgPool2d(4),
                                                     ViewModel())
            late_layer = torch.nn.Linear(1280, 10)
        elif split_idx == 2:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1, model.layers[:4])
            intermediate_layer = torch.nn.Sequential(model.layers[4:], model.conv2, model.bn2, torch.nn.AvgPool2d(4),
                                                     ViewModel())
            late_layer = torch.nn.Linear(1280, 10)
        elif split_idx == 3:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1, model.layers[:8])
            intermediate_layer = torch.nn.Sequential(model.layers[8:], model.conv2, model.bn2, torch.nn.AvgPool2d(4),
                                                     ViewModel())
            late_layer = torch.nn.Linear(1280, 10)
        elif split_idx == 4:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1, model.layers[:12])
            intermediate_layer = torch.nn.Sequential(model.layers[12:], model.conv2, model.bn2, torch.nn.AvgPool2d(4),
                                                     ViewModel())
            late_layer = torch.nn.Linear(1280, 10)
    elif model_name == "densenet":
        if split_idx == 1:
            early_layer = torch.nn.Sequential(model.conv1)
            intermediate_layer = torch.nn.Sequential(model.dense1, model.trans1, model.dense2, model.trans2,
                                                     model.dense3,
                                                     model.trans3, model.dense4, torch.nn.AvgPool2d(4), ViewModel())
            late_layer = torch.nn.Linear(384, 10)
        elif split_idx == 2:
            early_layer = torch.nn.Sequential(model.conv1, model.dense1, model.trans1)
            intermediate_layer = torch.nn.Sequential(model.dense2, model.trans2, model.dense3,
                                                     model.trans3, model.dense4, torch.nn.AvgPool2d(4), ViewModel())
            late_layer = torch.nn.Linear(384, 10)
        elif split_idx == 3:
            early_layer = torch.nn.Sequential(model.conv1, model.dense1, model.trans1, model.dense2, model.trans2)
            intermediate_layer = torch.nn.Sequential(model.dense3, model.trans3, model.dense4,
                                                     torch.nn.AvgPool2d(4), ViewModel())
            late_layer = torch.nn.Linear(384, 10)
        elif split_idx == 4:
            early_layer = torch.nn.Sequential(model.conv1, model.dense1, model.trans1, model.dense2, model.trans2,
                                              model.dense3, model.trans3)
            intermediate_layer = torch.nn.Sequential(model.dense4, torch.nn.AvgPool2d(4), ViewModel())
            late_layer = torch.nn.Linear(384, 10)
    elif model_name == "efficientnet":
        if split_idx == 1:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1)
            intermediate_layer = torch.nn.Sequential(model.layers, torch.nn.AdaptiveAvgPool2d(1), ViewModel())
            late_layer = torch.nn.Linear(320, 10)
        elif split_idx == 2:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1, model.layers[:3])
            intermediate_layer = torch.nn.Sequential(model.layers[3:], torch.nn.AdaptiveAvgPool2d(1), ViewModel())
            late_layer = torch.nn.Linear(320, 10)
        elif split_idx == 3:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1, model.layers[:6])
            intermediate_layer = torch.nn.Sequential(model.layers[6:], torch.nn.AdaptiveAvgPool2d(1), ViewModel())
            late_layer = torch.nn.Linear(320, 10)
        elif split_idx == 4:
            early_layer = torch.nn.Sequential(model.conv1, model.bn1, model.layers[:9])
            intermediate_layer = torch.nn.Sequential(model.layers[9:], torch.nn.AdaptiveAvgPool2d(1), ViewModel())
            late_layer = torch.nn.Linear(320, 10)

    return early_layer, intermediate_layer, late_layer


def get_shadow_model(model, strategy, split_idx):
    if model == "resnet":
        if strategy == "single":
            if split_idx == 1:
                dummy_early_layer = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            elif split_idx == 2:
                dummy_early_layer = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            elif split_idx == 3:
                dummy_early_layer = torch.nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1, bias=False)
            elif split_idx == 4:
                dummy_early_layer = torch.nn.Conv2d(3, 256, kernel_size=9, stride=3, bias=False)
        elif strategy == "multi":
            if split_idx == 1:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 64, 1))
            elif split_idx == 2:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 32, 1), BasicBlock(32, 64, 1))
            elif split_idx == 3:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 32, 1), BasicBlock(32, 64, 1),
                                                        BasicBlock(64, 128, 2))
            elif split_idx == 4:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 32, 1), BasicBlock(32, 64, 1),
                                                        BasicBlock(64, 128, 2), BasicBlock(128, 256, 2))
    if model == "mobilenetv2":
        if strategy == "single":
            if split_idx == 1:
                dummy_early_layer = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            elif split_idx == 2:
                dummy_early_layer = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            elif split_idx == 3:
                dummy_early_layer = torch.nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=1, bias=False)
            elif split_idx == 4:
                dummy_early_layer = torch.nn.Conv2d(3, 96, kernel_size=9, stride=3, bias=False)
        elif strategy == "multi":
            if split_idx == 1:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 32, 1))
            elif split_idx == 2:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 16, 1), BasicBlock(16, 32, 2))
            elif split_idx == 3:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 16, 1), BasicBlock(16, 32, 2),
                                                        BasicBlock(32, 64, 2))
            elif split_idx == 4:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 16, 1), BasicBlock(16, 32, 1),
                                                        BasicBlock(32, 64, 2), BasicBlock(64, 96, 2))
    if model == "densenet":
        if strategy == "single":
            if split_idx == 1:
                dummy_early_layer = torch.nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
            elif split_idx == 2:
                dummy_early_layer = torch.nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False)
            elif split_idx == 3:
                dummy_early_layer = torch.nn.Conv2d(3, 96, kernel_size=9, stride=3, bias=False)
            elif split_idx == 4:
                dummy_early_layer = torch.nn.Conv2d(3, 192, kernel_size=9, stride=6, bias=False)
        elif strategy == "multi":
            if split_idx == 1:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 24, 1))
            elif split_idx == 2:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 24, 1), BasicBlock(24, 48, 2))
            elif split_idx == 3:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 24, 1), BasicBlock(24, 48, 2),
                                                        BasicBlock(48, 96, 2))
            elif split_idx == 4:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 24, 1), BasicBlock(24, 48, 2),
                                                        BasicBlock(48, 96, 2), BasicBlock(96, 192, 2))
    if model == "efficientnet":
        if strategy == "single":
            if split_idx == 1:
                dummy_early_layer = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            elif split_idx == 2:
                dummy_early_layer = torch.nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
            elif split_idx == 3:
                dummy_early_layer = torch.nn.Conv2d(3, 80, kernel_size=9, stride=6, bias=False)
            elif split_idx == 4:
                dummy_early_layer = torch.nn.Conv2d(3, 112, kernel_size=9, stride=6, bias=False)
        elif strategy == "multi":
            if split_idx == 1:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 32, 1))
            elif split_idx == 2:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 32, 1), BasicBlock(32, 24, 2))
            elif split_idx == 3:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 32, 2), BasicBlock(32, 24, 2),
                                                        BasicBlock(24, 80, 2))
            elif split_idx == 4:
                dummy_early_layer = torch.nn.Sequential(BasicBlock(3, 32, 1), BasicBlock(32, 24, 2),
                                                        BasicBlock(24, 80, 2), BasicBlock(80, 112, 2))
    return dummy_early_layer
