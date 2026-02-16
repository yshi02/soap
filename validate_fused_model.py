# Validate that a fused (no BatchNorm) model produces the same outputs as the original model
# by comparing layer-by-layer intermediate activations and final output.

import argparse
import importlib
import pathlib
import sys
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--original_module",
    help="Python module implementing the original NN (with BatchNorm)",
    type=str,
    default="LISTA_model_16x64",
)
parser.add_argument(
    "--fused_module",
    help="Python module implementing the fused NN (no BatchNorm)",
    type=str,
    default="LISTA_model_16x64_fused_batchnorm",
)
parser.add_argument(
    "--nn_name",
    help="Name of the NN class in both modules",
    type=str,
    default="LISTA_Net",
)
parser.add_argument(
    "--nn_module",
    nargs="+",
    help="Target module names that were fused (for layer-by-layer comparison)",
    type=str,
    default=["conv_y_layers"],
)
parser.add_argument(
    "--original_weights",
    help="Path to the original model weights",
    type=pathlib.Path,
    required=True,
)
parser.add_argument(
    "--fused_weights",
    help="Path to the fused model weights",
    type=pathlib.Path,
    required=True,
)
parser.add_argument(
    "--input_shape",
    nargs=4,
    help="Input tensor shape: batch channels height width",
    type=int,
    default=[1, 512, 4, 32],
)
parser.add_argument(
    "--seed",
    help="Random seed for reproducible input",
    type=int,
    default=42,
)
parser.add_argument(
    "--atol",
    help="Absolute tolerance for comparison",
    type=float,
    default=5e-1,
)
parser.add_argument(
    "--rtol",
    help="Relative tolerance for comparison",
    type=float,
    default=1e-1,
)
args = parser.parse_args()


def load_model(module_name, nn_name, weights_path):
    mod = importlib.import_module(module_name)
    nn = getattr(mod, nn_name)()
    obj = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = obj["model_state_dict"] if "model_state_dict" in obj else obj
    nn.load_state_dict(state_dict, strict=True)
    nn.eval()
    return nn


# Load both models
original_nn = load_model(args.original_module, args.nn_name, args.original_weights)
fused_nn = load_model(args.fused_module, args.nn_name, args.fused_weights)

# Generate deterministic random input
torch.manual_seed(args.seed)
y_input = torch.randn(*args.input_shape)
# phi and theta are single-channel tensors with same spatial dims
phi = torch.randn(args.input_shape[0], 1, args.input_shape[2], args.input_shape[3])
theta = torch.randn(args.input_shape[0], 1, args.input_shape[2], args.input_shape[3])

print(f"Input shape: {list(y_input.shape)}")
print(f"Tolerances: atol={args.atol}, rtol={args.rtol}")
print()

all_passed = True


def compare(name, a, b):
    global all_passed
    max_abs = (a - b).abs().max().item()
    max_rel = ((a - b).abs() / (b.abs() + 1e-12)).max().item()
    close = torch.allclose(a, b, atol=args.atol, rtol=args.rtol)
    status = "PASS" if close else "FAIL"
    if not close:
        all_passed = False
    print(f"  [{status}] {name:40s}  max_abs_err={max_abs:.2e}  max_rel_err={max_rel:.2e}")
    return close


print("=" * 80)
print("Layer-by-layer comparison for target modules")
print("=" * 80)

for module_name in args.nn_module:
    orig_mlist = getattr(original_nn, module_name)
    fused_mlist = getattr(fused_nn, module_name)

    for seq_idx in range(len(orig_mlist)):
        orig_seq = list(orig_mlist[seq_idx].children())
        fused_seq = list(fused_mlist[seq_idx].children())

        print(f"\n--- {module_name}[{seq_idx}] ---")
        print(f"  Original layers: {len(orig_seq)},  Fused layers: {len(fused_seq)}")

        # Run through both sequences layer by layer, tracking corresponding positions
        # Original: Conv, BN, ReLU, Conv, BN, ReLU, ...
        # Fused:    Conv, ReLU, Conv, ReLU, ...
        # We compare after each "logical block" (after ReLU or after the last layer)
        with torch.no_grad():
            orig_out = y_input.clone()
            fused_out = y_input.clone()
            oi = 0  # original layer index
            fi = 0  # fused layer index
            block_idx = 0

            while oi < len(orig_seq) and fi < len(fused_seq):
                # Advance original through one logical block
                orig_block_layers = []
                orig_out = orig_seq[oi](orig_out)
                orig_block_layers.append(type(orig_seq[oi]).__name__)
                oi += 1
                # If this was a Conv followed by BN, consume the BN too
                if oi < len(orig_seq) and isinstance(orig_seq[oi], torch.nn.BatchNorm2d):
                    orig_out = orig_seq[oi](orig_out)
                    orig_block_layers.append(type(orig_seq[oi]).__name__)
                    oi += 1
                # If next is ReLU, consume it
                if oi < len(orig_seq) and isinstance(orig_seq[oi], torch.nn.ReLU):
                    orig_out = orig_seq[oi](orig_out)
                    orig_block_layers.append(type(orig_seq[oi]).__name__)
                    oi += 1

                # Advance fused through one logical block
                fused_block_layers = []
                fused_out = fused_seq[fi](fused_out)
                fused_block_layers.append(type(fused_seq[fi]).__name__)
                fi += 1
                # If next is ReLU, consume it
                if fi < len(fused_seq) and isinstance(fused_seq[fi], torch.nn.ReLU):
                    fused_out = fused_seq[fi](fused_out)
                    fused_block_layers.append(type(fused_seq[fi]).__name__)
                    fi += 1

                orig_desc = " -> ".join(orig_block_layers)
                fused_desc = " -> ".join(fused_block_layers)
                compare(
                    f"block {block_idx} ({orig_desc} vs {fused_desc})",
                    orig_out,
                    fused_out,
                )
                block_idx += 1

# ---- Full forward pass comparison ----
print()
print("=" * 80)
print("Full forward pass comparison")
print("=" * 80)

with torch.no_grad():
    orig_final = original_nn(y_input, phi, theta)
    fused_final = fused_nn(y_input, phi, theta)

print()
compare("Final output", orig_final, fused_final)

print()
print("=" * 80)
if all_passed:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED — see above for details")
    sys.exit(1)
