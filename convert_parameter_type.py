# Alternate floating-point data formats in PTX:
# https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats
#
# torch.dtype types:
# https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch-dtype

import argparse
import importlib
import pathlib
import sys
import torch

TYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "e5m2": torch.float8_e5m2,
    "e4m3": torch.float8_e4m3fn,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--module", help="Python module implementing the NN", type=str, required=True
)
parser.add_argument(
    "--nn_name",
    help="Name of the NN in the Python module",
    type=str,
    default="LISTA_Net",
)
parser.add_argument(
    "--input",
    help="Path to the saved torch object for the NN",
    type=pathlib.Path,
    required=True,
)
parser.add_argument(
    "--input_type",
    help="Parameter type of the input",
    type=str,
    choices=["fp32"],
    default="fp32",
)
parser.add_argument(
    "--output",
    help="Path where the converted parameters should be saved as a torch object",
    type=pathlib.Path,
    required=True,
)
parser.add_argument(
    "--type",
    help="Type of the parameters to be converted to",
    type=str,
    choices=["bf16", "e5m2", "e4m3"],
    required=True,
)
args = parser.parse_args()

module = importlib.import_module(args.module)
nn = getattr(module, args.nn_name)()

input_object = torch.load(args.input, map_location="cpu", weights_only=True)
input_state_dict = input_object["model_state_dict"]

input_type = TYPES[args.input_type]
new_type = TYPES[args.type]

# Make sure the saved model matches with the NN
while True:
    try:
        nn.load_state_dict(input_state_dict, strict=True)
        break
    except RuntimeError as err:
        print(f"Saved model does not match with the NN: {err}")
        sys.exit(1)

# Convert *only* the weights and biases
new_state_dict = {}
for key, val in input_state_dict.items():
    if isinstance(val, torch.Tensor):
        if val.dtype != input_type:
            print(f"dtype for tensor {key} != {args.input_type} (expected), aborting")
            sys.exit(1)
        new_state_dict[key] = val.to(dtype=new_type)
    else:
        new_state_dict[key] = val

# Save the new state dict
# Wrapped in a new dict to effectively remove all other things like epoch or optimizer_state_dict
torch.save(dict(model_state_dict=new_state_dict), args.output)

print(f"Output saved to {args.output}")
