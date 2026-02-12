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
    # Integer types
    "int64" : torch.int64,
    "uint64": torch.uint64,
    "int32" : torch.int32,
    "uint32": torch.uint32,
    "int16" : torch.int16,
    "uint16": torch.uint16,
    "int8"  : torch.int8,
    "uint8" : torch.uint8,

    # Real fp types
    "fp64": torch.float64,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "e5m2": torch.float8_e5m2,
    "e4m3": torch.float8_e4m3fn,

    # Complex fp types
    "complex128" : torch.complex128,
    "complex64"  : torch.complex64,
    "complex32"  : torch.complex32,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
)
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
    "--target_type",
    help="Target parameter type; only parameters of this type will be converted",
    type=str,
    choices=TYPES.keys(),
    required=True,
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
    choices=TYPES.keys(),
    required=True,
)
args = parser.parse_args()

module = importlib.import_module(args.module)
nn = getattr(module, args.nn_name)()

input_object = torch.load(args.input, map_location="cpu", weights_only=True)
input_state_dict = (
    input_object["model_state_dict"]
    if "model_state_dict" in input_object.keys()
    else input_object
)

target_type = TYPES[args.target_type]
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
        if val.dtype != target_type:
            if args.verbose:
                print(f"{key} has type {val.dtype} (!= {target_type}), skipping conversion")
            new_state_dict[key] = val
        else:
            if args.verbose:
                print(f"Converting {key} to {new_type}")
            new_state_dict[key] = val.to(dtype=new_type)
    else:
        new_state_dict[key] = val

# Save the new state dict
# Wrapped in a new dict to effectively remove all other things like epoch or optimizer_state_dict
torch.save(dict(model_state_dict=new_state_dict), args.output)

print(f"Output saved to {args.output}")
