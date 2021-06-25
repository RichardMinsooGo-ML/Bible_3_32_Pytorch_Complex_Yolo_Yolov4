import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from utils.utils import to_cpu

sys.path.append('../')
from models.yolo_layer import YoloLayer
from utils.train_utils import parse_model_config
# from utils.utils import build_targets, to_cpu, parse_model_config

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def create_network(blocks):
    """
    Constructs module list of layer blocks from module configuration in blocks
    """
    hyperparams = blocks.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, block in enumerate(blocks):
        modules = nn.Sequential()

        if block["type"] == "convolutional":
            batch_normalize = int(block["batch_normalize"])
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(block["stride"]),
                    padding=pad,
                    bias=not batch_normalize,
                ),
            )
            if batch_normalize:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if block["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif block["type"] == "maxpool":
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            if stride == 1 and kernel_size == 2:
                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif block["type"] == "upsample":
            upsample = Upsample(scale_factor=int(block["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif block["type"] == "route":
            layers = [int(x) for x in block["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyModule())

        elif block["type"] == "shortcut":
            filters = output_filters[1:][int(block["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyModule())

        elif block["type"] == "yolo":
            anchor_masks = [int(i) for i in block["mask"].split(",")]
            # Extract anchors
            anchors = [float(i) for i in block["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1], math.sin(anchors[i + 2]), math.cos(anchors[i + 2])) for i in
                       range(0, len(anchors), 3)]
            
            
            anchors = [anchors[i] for i in anchor_masks]
            num_classes = int(block["classes"])
            # self.num_classes = num_classes
            img_size = int(hyperparams["height"])
            # Define detection layer

            yolo_layer = YoloLayer( anchors=anchors, num_classes=num_classes,stride=1,
                                   ignore_thresh=0.5)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        # Register module list and number of output filters
        output_filters.append(filters)
        module_list.append(modules)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyModule(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyModule, self).__init__()

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfgfile, img_size=416):
        super(Darknet, self).__init__()
        self.blocks = parse_model_config(cfgfile)
        self.hyperparams, self.module_list = create_network(self.blocks)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)):
            if block["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif block["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in block["layers"].split(",")], 1)
            elif block["type"] == "shortcut":
                layer_i = int(block["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif block["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

        """Parses and loads the weights stored in 'weights_path'"""
    def load_darknet_weights(self, weights_path):

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75
        elif "yolov3-tiny.conv.15" in weights_path:
            cutoff = 15

        ptr = 0
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)):
            if i == cutoff:
                break
            if block["type"] == "convolutional":
                conv_layer = module[0]
                if block["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w