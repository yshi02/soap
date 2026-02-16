# Fuse the batch norm params into the trained convolution weights for specified modules
# Currently this only targets Conv2d and BatchNorm2d

import argparse
import importlib
import pathlib
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
)
parser.add_argument(
    "--original_module",
    help="Python module implementing the Original NN",
    type=str,
    default="LISTA_model_16x64"
)
parser.add_argument(
    "--fused_module",
    help="Python module implementing the NN where conv and BN layers are fused",
    type=str,
    default="LISTA_model_16x64_fused_batchnorm",
)
parser.add_argument(
    "--nn_name",
    help="Name of the NN in the Python module",
    type=str,
    default="LISTA_Net",
)
parser.add_argument(
    "--nn_module",
    nargs="+",
    help="Name of the NN module for which the fusing should apply to",
    type=str,
    default=["conv_y_layers"]
)
parser.add_argument(
    "--input",
    help="Path to the saved torch object for the NN",
    type=pathlib.Path,
    required=True,
)
parser.add_argument(
    "--output",
    help="Path where the converted parameters should be saved as a torch object",
    type=pathlib.Path,
    required=True,
)
args = parser.parse_args()

original_module = importlib.import_module(args.original_module)
original_nn = getattr(original_module, args.nn_name)()

input_object = torch.load(args.input, map_location="cpu", weights_only=True)
input_state_dict = (
    input_object["model_state_dict"]
    if "model_state_dict" in input_object.keys()
    else input_object
)

# Make sure the saved model matches the NN
try:
    original_nn.load_state_dict(input_state_dict, strict=True)
except RuntimeError as err:
    print(f"Saved model does not match with the NN: {err}")
    sys.exit(1)

def fuse_conv_bn(conv, bn, use_float64=False):
    """
    Fuse a Conv2d and BatchNorm2d into a single Conv2d with bias.
    ref: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
    """
    fused = torch.nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )

    # W_bn = diag(gamma / sqrt(var + eps))
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.running_var + bn.eps)))

    # b_fused = W_bn @ b_conv + (beta - gamma * mean / sqrt(var + eps))
    b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))

    with torch.no_grad():
        fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))
        fused.bias.copy_(torch.mm(w_bn, b_conv.unsqueeze(1)).squeeze(1) + b_bn)

    return fused


# Load the fused model architecture (no BatchNorm layers)
fused_module = importlib.import_module(args.fused_module)
fused_nn = getattr(fused_module, args.nn_name)()

# Put original model in eval mode (so BN uses running stats, not batch stats)
original_nn.eval()

# For each target module, fuse conv+bn pairs and collect the fused layers
for module_name in args.nn_module:
    original_mlist = getattr(original_nn, module_name)

    # Each element in the ModuleList is a Sequential
    for seq_idx, seq in enumerate(original_mlist):
        layers = list(seq.children())
        new_layers = []
        i = 0
        while i < len(layers):
            if isinstance(layers[i], torch.nn.Conv2d):
                if i + 1 < len(layers) and isinstance(layers[i + 1], torch.nn.BatchNorm2d):
                    if args.verbose:
                        print(f"Fusing {module_name}.{seq_idx}[{i}] Conv2d + [{i+1}] BatchNorm2d")
                    fused = fuse_conv_bn(layers[i], layers[i + 1])
                    new_layers.append(fused)
                    i += 2  # skip the BN layer
                    continue
            new_layers.append(layers[i])
            i += 1

        # Replace the Sequential in the original model with fused layers
        original_mlist[seq_idx] = torch.nn.Sequential(*new_layers)

# Now the original_nn has fused weights and no BN layers in target modules.
# Its structure should match fused_nn. Extract and load the state dict.
fused_state_dict = original_nn.state_dict()

try:
    fused_nn.load_state_dict(fused_state_dict, strict=True)
except RuntimeError as err:
    print(f"Fused state dict does not match fused NN architecture: {err}")
    print("Check that --fused_module matches the original module but without BatchNorm layers.")
    sys.exit(1)

# Save the new state dict
torch.save(dict(model_state_dict=fused_nn.state_dict()), args.output)

print(f"Output saved to {args.output}")
