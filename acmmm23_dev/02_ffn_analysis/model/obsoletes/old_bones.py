from __future__ import absolute_import, print_function

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF


# CUDA Device Configuration
__CUDA_DEVICE__ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    def get_io_channels(self, type):
        assert type in ["input", "output"]

        if len(self.layers_dict) == 0:
            return -1

        # Get Layer Block
        if type == "input":
            block_layers = self.layers_dict[next(iter(self.layers_dict))]
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
                    if type == "input":
                        return_channels = layer.in_features
                    else:
                        return_channels = layer.out_features
                elif isinstance(layer, nn.Conv2d):
                    if type == "input":
                        return_channels = layer.in_channels
                    else:
                        return_channels = layer.out_channels
                else:
                    raise NotImplementedError()

                # Break
                break

        return return_channels

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class OVERLAP_CLASSIFIER(BASE_MODEL):
    def __init__(self, **kwargs):
        super(OVERLAP_CLASSIFIER, self).__init__(**kwargs)

        # Unpack KWARGS
        self.loss_type = kwargs.get("loss_type")
        dimensions = kwargs.get("dimensions")
        assert isinstance(dimensions, list) and len(dimensions) > 1
        layers = kwargs.get("layers")
        assert isinstance(layers, list) and len(layers) + 1 == len(dimensions)
        hidden_activations = kwargs.get("hidden_activations")
        assert isinstance(hidden_activations, list) and len(hidden_activations) == len(layers)
        batchnorm_layers = kwargs.get("batchnorm_layers", [False] * len(layers))
        assert isinstance(batchnorm_layers, list) and len(batchnorm_layers) == len(layers)
        dropout_probs = kwargs.get("dropout_probs", [0] * len(layers))
        assert isinstance(dropout_probs, list) and len(dropout_probs) == len(layers)
        final_activation = kwargs.get("final_activation")

        # Last Softmax on Non-Training
        self.__last_softmax = False

        # Define Layers via OrderedDict
        for idx in range(len(dimensions)-1):
            in_dims, out_dims = dimensions[idx], dimensions[idx+1]

            # Append Index to Key
            self.layers_dict[idx] = []

            # ==== Define Weighted Layers ==== #
            if layers[idx] == "fc":
                layer_name = "fc{}".format(idx+1)
                self.layers_dict[idx].append(layer_name)
                _layer = nn.Linear(in_features=in_dims, out_features=out_dims)
                setattr(self, layer_name, _layer)
            else:
                raise NotImplementedError()

            # ==== Define BatchNorm Layers ==== #
            if batchnorm_layers[idx]:
                layer_name = "bn{}".format(idx+1)
                self.layers_dict[idx].append(layer_name)
                # _layer =

            if idx < len(dimensions) - 2:
                if batchnorm_layers[idx]:
                    layer_name = "bn{}"



            raise NotImplementedError()






if __name__ == "__main__":

    # Set Batch Size, Spatial Dimension, and Channel Dimension for Debugging
    B, S, C = 64, 1024, 256

    # Generate Random Samples for Debugging
    rand_samples = torch.rand(size=(B, S, C)).to(
        device=__CUDA_DEVICE__, dtype=torch.float64
    )


    print(123)
    pass
