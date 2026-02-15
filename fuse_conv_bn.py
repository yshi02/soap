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
while True:
    try:
        original_nn.load_state_dict(input_state_dict, strict=True)
        break
    except RuntimeError as err:
        print(f"Saved model does not match with the NN: {err}")
        sys.exit(1)

# We are assuming that the target top-level modules all only have one sequential list
# assert(all(len(module) == 1 for module in target_modules.values()))

found_target_module = False

# Iterate over each module
# For target modules, find conv and bn layers that can be fused
# For non-target modules, add them to the new NN
new_nn = torch.nn.Module()
for name, module in original_nn.named_modules():
    if name not in args.nn_module:
        setattr(new_nn, name, module)
        continue

    found_target_module = True
    if args.verbose:
        printf(f"Found target module {name}")

    mlist = target_modules[name][0]

    # Use a simple pointer to iterate over the layers and skip fusable batchnorms
    i = 0
    new_mlist = []
    while i < len(mlist):
        new_mlist.append(mlist[i])
        if isinstance(mlist[i], torch.nn.Conv2d):
            # If the next layer is batchnorm, we can fuse it
            if (i < len(mlist) - 1) and (isinstance(mlist[i+1], torch.nn.BatchNorm2d)):
                if args.verbose:
                    print(f"Found fusable conv & BN layers at {name}[0][{i}(+1)]")
                # TODO: implement fusing here

                # Don't add the BN layer to the new module list since it's now fused
                i += 1
        i += 1
    print(new_mlist)
    new_module = nn.Module()
    setattr(new_module, name, )
    n = torch.nn.ModuleList(new_mlist)
    print(n)
    torch.save(n.state_dict(), "n.pt")

# Convert *only* the weights and biases
new_state_dict = {}
for key, val in input_state_dict.items():
    pass

# Save the new state dict
# Wrapped in a new dict to effectively remove all other things like epoch or optimizer_state_dict
# torch.save(dict(model_state_dict=new_state_dict), args.output)

print(f"Output saved to {args.output}")
