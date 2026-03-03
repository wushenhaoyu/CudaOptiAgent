import os
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

kernel_path = "kernel.cu"

if not os.path.exists(kernel_path):
    raise FileNotFoundError(f"CUDA kernel file not found at: {kernel_path}")

with open(kernel_path, "rb") as f:
    content_hash = hashlib.md5(f.read()).hexdigest()[:8]

cuda_extension = load(
    name=f"DenseNet121_{content_hash}",
    sources=[kernel_path],
    verbose=False,
)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)  # Concatenate along channel axis
        return x


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        return self.transition(x)


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        """
        :param growth_rate: The growth rate of the DenseNet (new features added per layer)
        :param num_classes: The number of output classes for classification
        """
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block is followed by a transition layer, except the last one
        num_features = 64
        block_layers = [6, 12, 24, 16]  # Corresponding layers in DenseNet121

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, 3, height, width)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        params = []

        for module in self.features:
            if isinstance(module, nn.Conv2d):
                params.append(module.weight)
                if module.bias is not None:
                    params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                params.append(module.weight)
                params.append(module.bias)
                params.append(module.running_mean)
                params.append(module.running_var)

        for block in self.dense_blocks:
            for layer in block.layers:
                for submodule in layer:
                    if isinstance(submodule, nn.BatchNorm2d):
                        params.append(submodule.weight)
                        params.append(submodule.bias)
                        params.append(submodule.running_mean)
                        params.append(submodule.running_var)
                    elif isinstance(submodule, nn.Conv2d):
                        params.append(submodule.weight)
                        if submodule.bias is not None:
                            params.append(submodule.bias)

        for transition in self.transition_layers:
            for submodule in transition.transition:
                if isinstance(submodule, nn.BatchNorm2d):
                    params.append(submodule.weight)
                    params.append(submodule.bias)
                    params.append(submodule.running_mean)
                    params.append(submodule.running_var)
                elif isinstance(submodule, nn.Conv2d):
                    params.append(submodule.weight)
                    if submodule.bias is not None:
                        params.append(submodule.bias)

        params.append(self.final_bn.weight)
        params.append(self.final_bn.bias)
        params.append(self.final_bn.running_mean)
        params.append(self.final_bn.running_var)

        params.append(self.classifier.weight)
        params.append(self.classifier.bias)

        return cuda_extension.DenseNet121(x, *params)


# Testing the DenseNet121 model
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet


def get_inputs():
    return [torch.rand(batch_size, 3, height, width)]


def get_init_inputs():
    return [32, num_classes]