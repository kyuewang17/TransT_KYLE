from __future__ import absolute_import, print_function

import logging
from collections import OrderedDict
import torch
import torchsummary
import torch.nn as nn
import torch.nn.functional as F


# CUDA Device Configuration
__CUDA_DEVICE__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_logger(logging_level=logging.INFO, log_name="root", logging_filepath=None):
    # Define Logger
    if log_name == "root":
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name=log_name)

    # Set Logger Display Level
    logger.setLevel(level=logging_level)

    # Set Formatter
    formatter = logging.Formatter("[%(levelname)s] | %(asctime)s : %(message)s")
    # formatter = "[%(levelname)s] | %(asctime)s : %(message)s"

    # Set Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # stream_handler.setFormatter(CustomFormatter(formatter))
    logger.addHandler(stream_handler)

    # Set File Handler
    if logging_filepath is not None:
        file_handler = logging.FileHandler(logging_filepath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Overlap Base Model Class
class BASE_MODEL(nn.Module):
    def __init__(self, **kwargs):
        super(BASE_MODEL, self).__init__()

        # Unpack KWARGS
        self.overlap_criterion = kwargs.get("overlap_criterion")

        # Define OrderedDict of Layer Names
        self.layers_dict = OrderedDict()

    def _init_params(self, dist):
        # Return if Length of Ordered Dictionary if 0
        if len(self.layers_dict) == 0:
            return

        # Assertion
        assert dist in ["normal", "uniform"]

        # Iterate Layers Dictionary, initialize weights
        for layer_names in self.layers_dict.values():
            # Search for Activation Layer
            weight_init_method = None
            for layer_name in layer_names:
                if layer_name.upper().__contains__("RELU"):
                    weight_init_method = "he"
                    break
                elif layer_name.upper().__contains__("TANH"):
                    weight_init_method = "xavier"
                    break

            # Iterate for Layers,
            for layer_name in layer_names:
                # Get Layer
                layer = getattr(self, layer_name)

                # Initialize
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    if weight_init_method == "he":
                        if dist == "uniform":
                            nn.init.kaiming_uniform_(layer.weight)
                        else:
                            nn.init.kaiming_normal_(layer.weight)

                    elif weight_init_method == "xavier":
                        if dist == "uniform":
                            nn.init.xavier_uniform_(layer.weight)
                        else:
                            nn.init.xavier_normal_(layer.weight)

                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def get_feat_dims(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


# Overlap Classifier Model Class
class OVERLAP_CLASSIFIER(BASE_MODEL):
    def __init__(self, **kwargs):
        super(OVERLAP_CLASSIFIER, self).__init__(**kwargs)

        # === Unpack KWARGS === #
        # Get Classification Loss
        self.cls_loss = kwargs.get("cls_loss")
        self.__last_softmax_on_training = False

        dimensions = kwargs.get("dimensions")
        assert isinstance(dimensions, list) and len(dimensions) > 1
        layers = kwargs.get("layers")
        assert isinstance(layers, list) and len(layers) + 1 == len(dimensions)
        hidden_activations = kwargs.get("hidden_activations")
        assert isinstance(hidden_activations, list) and len(hidden_activations) + 1 == len(layers)
        batchnorm_layers = kwargs.get("batchnorm_layers", [False] * len(hidden_activations))
        assert isinstance(batchnorm_layers, list) and len(batchnorm_layers) + 1 == len(layers)
        dropout_probs = kwargs.get("dropout_probs", [0] * len(hidden_activations))
        assert isinstance(dropout_probs, list) and len(dropout_probs) + 1 == len(layers)
        final_activation = kwargs.get("final_activation")

        init_dist = kwargs.get("init_dist")
        assert init_dist in ["normal", "uniform"]

        # Define Layers via OrderedDict
        for idx in range(len(dimensions)-1):
            in_dims, out_dims = dimensions[idx], dimensions[idx+1]

            # Append Index to Key
            self.layers_dict[idx] = []

            # === Define Weighted Layers === #
            if layers[idx] == "fc":
                layer_name = "fc{}".format(idx+1)
                self.layers_dict[idx].append(layer_name)
                _layer = nn.Linear(in_features=in_dims, out_features=out_dims)
                setattr(self, layer_name, _layer)
            else:
                raise NotImplementedError()

            # === Define Mid-Layers (non-weighted) === #
            if idx < len(dimensions) - 2:
                # == Batch Normalization Layer == #
                if batchnorm_layers[idx]:
                    layer_name = "bn{}".format(idx + 1)
                    self.layers_dict[idx].append(layer_name)
                    if self.layers_dict[idx][-2].__contains__("fc"):
                        # todo: Later, sophisticate this code for "conv" layer
                        _layer = nn.BatchNorm1d(out_dims)
                    else:
                        raise NotImplementedError()
                    setattr(self, layer_name, _layer)

                # == Hidden Activation Layer == #
                if hidden_activations[idx] == "ReLU":
                    layer_name = "relu{}".format(idx + 1)
                    self.layers_dict[idx].append(layer_name)
                    _layer = nn.ReLU()
                    setattr(self, layer_name, _layer)
                elif hidden_activations[idx] == "LeakyReLU":
                    layer_name = "leakyrelu{}".format(idx + 1)
                    self.layers_dict[idx].append(layer_name)
                    _layer = nn.LeakyReLU()
                    setattr(self, layer_name, _layer)
                else:
                    raise NotImplementedError()

                # == Dropout Layer == #
                if dropout_probs[idx] > 0:
                    layer_name = "dropout{}".format(idx + 1)
                    self.layers_dict[idx].append(layer_name)
                    _layer = nn.Dropout(dropout_probs[idx])
                    setattr(self, layer_name, _layer)

            # === Final Activation Layer === #
            else:
                if final_activation == "sigmoid":
                    layer_name = "sigmoid"
                    self.layers_dict[idx].append(layer_name)
                    _layer = nn.Sigmoid()
                    setattr(self, layer_name, _layer)
                elif final_activation == "tanh":
                    layer_name = "tanh"
                    self.layers_dict[idx].append(layer_name)
                    _layer = nn.Tanh()
                    setattr(self, layer_name, _layer)
                elif final_activation == "softmax":
                    if self.cls_loss not in ["CE", "WCE"]:
                        layer_name = "softmax"
                        self.layers_dict[idx].append(layer_name)
                        _layer = nn.Softmax()
                        setattr(self, layer_name, _layer)
                    else:
                        self.__last_softmax_on_training = True
                else:
                    raise NotImplementedError()

        # Initialize Parameters
        self._init_params(dist=init_dist)

    def get_feat_dims(self, position):
        assert position in ["input", "output"]

        if len(self.layers_dict) == 0:
            return -1

        # Get Layer Block
        if position == "input":
            raise NotImplementedError()
        else:
            block_layers = self.layers_dict[next(reversed(self.layers_dict))]
            block_layers = reversed(block_layers)

        # Iterate Block, Detect FC or Conv Layers
        # --> note that for "output", block layers are reversed.
        return_channels = -1
        for layer_name in block_layers:
            if layer_name.upper().__contains__("FC") or layer_name.upper().__contains__("CONV"):
                # Get Layer
                layer = getattr(self, layer_name)

                # Get Input Features (Channels)
                if isinstance(layer, nn.Linear):
                    if position == "input":
                        raise NotImplementedError()
                    else:
                        return_channels = layer.out_features
                elif isinstance(layer, nn.Conv2d):
                    if position == "input":
                        raise NotImplementedError()
                    else:
                        return_channels = layer.out_channels
                else:
                    raise NotImplementedError()

                # Break
                break

        return return_channels

    def get_layer_weights(self, layer_name):
        raise NotImplementedError()

    def forward(self, f):
        # Apply Global Statistical Poolings
        f1 = F.avg_pool2d(f, f.shape[2:]).view(f.shape[0], -1)
        f2 = F.max_pool2d(f, f.shape[2:]).view(f.shape[0], -1)
        f3 = f.view(f.shape[0], f.shape[1], -1).std(2)

        # Concatenate
        f = torch.cat((f1, f2, f3), dim=1)

        # Forward
        for idx, layer_names in self.layers_dict.items():
            for layer_name in layer_names:
                layer = getattr(self, layer_name)
                f = layer(f)

        return f


if __name__ == "__main__":
    # Set Batch Size, Spatial Dimension, and Channel Dimension for Debugging
    B, S1, S2, C = 64, 32, 32, 256

    # Logger
    _logger = set_logger()

    # Generate Random Samples for Debugging
    rand_samples = torch.rand(size=(B, C, S1, S2)).to(
        device=__CUDA_DEVICE__, dtype=torch.float32
    )

    # Init Classifier Model
    classifier = OVERLAP_CLASSIFIER(
        cls_loss="CE",

        # dimensions=[256, 32, 10, 2],
        dimensions=[768, 100, 32, 2],
        layers=["fc", "fc", "fc"],
        hidden_activations=["ReLU", "ReLU"],
        batchnorm_layers=[True, True],
        dropout_probs=[0.2, 0.2],
        final_activation="softmax",

        init_dist="normal"
    )
    classifier.to(device=__CUDA_DEVICE__)

    # Print Test
    torchsummary.summary(classifier, input_size=(C, S1, S2))

    # Test Forward
    out = classifier(rand_samples)

    print(123)
    pass
