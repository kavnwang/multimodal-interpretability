import torch
import nnsight
from PIL import Image
import argparse
import os
import json
from IPython.display import clear_output
from nnsight.modeling.vllm import VLLM
from transformers import AutoProcessor
import bitsandbytes
import einops
from vllm.inputs import TextPrompt
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def overlay_heatmap(image, heatmap, cmap="jet", alpha=0.5):
    """
    Given a PIL image and a 2D numpy heatmap (values in [0,1]),
    overlay the heatmap onto the image using matplotlib.
    """
    plt.imshow(image)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis('off')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_json", type=str, required=True,

                        help="Path to the JSON file containing prompts/images.")
    args = parser.parse_args()

    # Load the JSON data (list of dicts)
    with open(args.prompt_json, "r") as f:
        data_list = json.load(f)

    # Initialize model and processor just once
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    model = VLLM(
        model_name,
        device="cuda",
        dispatch=True,
        # dtype="bfloat16",
        # quantization="bitsandbytes",
        # load_format="bitsandbytes",
    )

    # Get model stats for patching
    NUM_LAYERS = len(model.language_model.model.layers)
    NUM_HEADS = int(model.language_model.model.config.num_attention_heads)
    HEAD_SIZE = None  # We'll set it once we see the attn shape.

    # We'll keep a single RUNNING_SUM_FILE for all data, if you want:
    RUNNING_SUM_FILE = "./data/running_sums.txt"
    running_sums = {}

    # Initialize or load running sums from file
    if os.path.exists(RUNNING_SUM_FILE):
        with open(RUNNING_SUM_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key_str, val_str = line.split(":")
                key_str = key_str.strip()
                val_str = val_str.strip()
                try:
                    running_sums[key_str] = float(val_str)
                except ValueError:
                    pass
    else:
        # If it doesn't exist, create placeholders for each LxHy
        with open(RUNNING_SUM_FILE, "w") as f:
            for layer in range(NUM_LAYERS):
                for head in range(NUM_HEADS):
                    key_str = f"L{layer}H{head}"
                    f.write(f"{key_str}: 0.0000\n")
        # Then read it back for usage
        with open(RUNNING_SUM_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key_str, val_str = line.split(":")
                key_str = key_str.strip()
                val_str = val_str.strip()
                running_sums[key_str] = float(val_str)

    # Now iterate over each entry in the JSON
    for item in data_list:
        index = item["index"]
        clean_image_path = item["clean_image_path"]
        corrupted_image_path = item["corrupted_image_path"]
        prompt = item["prompt"]
        clean_word = item["clean_word"]
        corrupted_word = item["corrupted_word"]

        # Load images
        clean_image = Image.open(clean_image_path)
        corrupted_image = Image.open(corrupted_image_path)

        # Figure out the token indices for "clean_word" and "corrupted_word"
        clean_token_ids = model.tokenizer(clean_word).input_ids
        corrupted_token_ids = model.tokenizer(corrupted_word).input_ids

        # Typically, the first ID is the BOS or other special token,
        # so we often pick index=1. But you might need to confirm:
        # e.g. the second element of the token_ids
        # (In your code, you used `.max().item()` which is somewhat unusual,
        # but let's replicate your logic.)
        clean_index = torch.tensor(clean_token_ids)[1].item()
        corrupted_index = torch.tensor(corrupted_token_ids)[1].item()

        # Capture the attention & logits on the "clean" version
        with model.trace(temperature=0.0, top_p=1.0) as tracer:
            with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": clean_image})) as result:
                clean_attn = [
                    model.language_model.model.layers[layer].self_attn.attn.output.save()
                    for layer in range(NUM_LAYERS)
                ]
                clean_logits = model.logits.output.save()

        # Capture the attention & logits on the "corrupted" version
        with model.trace(temperature=0.0, top_p=1.0) as tracer:
            with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": corrupted_image})) as result:
                corrupted_attn = [
                    model.language_model.model.layers[layer].self_attn.attn.output.save()
                    for layer in range(NUM_LAYERS)
                ]
                corrupted_logits = model.logits.output.save()

        # Convert model output to strings
        clean_output = model.tokenizer.decode(clean_logits.argmax(dim=-1))
        corrupted_output = model.tokenizer.decode(corrupted_logits.argmax(dim=-1))

        # Stack attention across layers for shaping
        clean_attn = torch.stack(clean_attn, dim=0)
        corrupted_attn = torch.stack(corrupted_attn, dim=0)

        if HEAD_SIZE is None:
            # HEAD_SIZE can be determined from your attention dimension
            # The last dimension might be hidden_size, so we do:
            HEAD_SIZE = clean_attn.size(2) // NUM_HEADS

        # These diffs are used for your patching measurement
        clean_diff = clean_logits[0][clean_index].item() - clean_logits[0][corrupted_index].item()
        corrupted_diff = corrupted_logits[0][clean_index].item() - corrupted_logits[0][corrupted_index].item()

        print(f"\n----- Processing index={index} -----")
        print(f"Clean Output = {clean_output}")
        print(f"Corrupted Output = {corrupted_output}")

        # Reshape so we have (num_layers, seq, num_heads, HEAD_SIZE)
        clean_attn_heads = clean_attn.reshape(
            clean_attn.size(0), clean_attn.size(1), NUM_HEADS, HEAD_SIZE
        )
        corrupted_attn_heads = corrupted_attn.reshape(
            corrupted_attn.size(0), corrupted_attn.size(1), NUM_HEADS, HEAD_SIZE
        )

        # Prepare to store patching results for this item
        save_file = f"./data/patching_results-{index}.txt"
        total_patches = NUM_LAYERS * NUM_HEADS

        # Figure out if we're resuming
        if os.path.exists(save_file):
            with open(save_file, "r") as f:
                try:
                    patching_results = [int(line.strip()) for line in f if line.strip()]
                    start_count = len(patching_results)
                except ValueError:
                    print("Save file is empty or corrupted. Starting fresh.")
                    patching_results = []
                    start_count = 0
        else:
            patching_results = []
            with open(save_file, "w") as f:
                pass
            print("No existing save file found. Starting from scratch.")
            start_count = 0

        # We'll keep an accumulator for patch diffs to write them in chunks
        accumulated_results = []
        print(f"Resuming from iteration {start_count} out of {total_patches}.")
        i = 0
        # Perform the actual patching across all layers/heads
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                if i < start_count:
                    i += 1
                    continue

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                with model.trace(TextPrompt(prompt=prompt, multi_modal_data={"image": corrupted_image})) as tracer:
                    patched_attn = corrupted_attn_heads[layer].clone()
                    # The code below tries to handle seq-length differences
                    if patched_attn.size(0) < clean_attn_heads.size(1):
                        patched_attn[:, head] = clean_attn_heads[layer, :-1, head].clone()
                    else:
                        patched_attn[:, head] = clean_attn_heads[layer, :, head].clone()

                    # Reshape back
                    patched_attn = patched_attn.reshape(
                        int(patched_attn.size(0)),
                        int(patched_attn.size(1) * patched_attn.size(2))
                    )

                    # Replace attention for that layer
                    model.language_model.model.layers[layer].self_attn.attn.output = patched_attn

                    # Grab the final logits after patching
                    patched_output = model.logits.output.save()

                # Compute the difference relative to corrupted_diff
                patch_diff = (patched_output[0][clean_index].item() -
                              patched_output[0][corrupted_index].item()) - corrupted_diff

                accumulated_results.append(patch_diff)

                # Write in chunks to reduce I/O
                if (i + 1) % 10 == 0:
                    with open(save_file, "a") as f:
                        for val in accumulated_results:
                            f.write(f"{val}\n")
                    accumulated_results.clear()

                i += 1

        # If anything remains in the accumulator, write it out
        if accumulated_results:
            with open(save_file, "a") as f:
                for val in accumulated_results:
                    f.write(f"{val}\n")
        # Optionally clean up
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        clean_attn_per_token = clean_attn_heads.abs().sum(dim=(0, 2, 3))
        corrupted_attn_per_token = corrupted_attn_heads.abs().sum(dim=(0, 2, 3))
        total_attn_per_token = clean_attn_per_token + corrupted_attn_per_token

        sorted_values, sorted_indices = torch.sort(total_attn_per_token, descending=True)
        output_file = f"./data/patching_token-{index}.csv"
        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                pass

        with open(output_file, "w") as f:
            f.write("index,sum\n")
            for idx, val in zip(sorted_indices, sorted_values):
                f.write(f"{idx.item()},{val.item()}\n")
        # Read back patching results for final summary info
        with open(save_file, "r") as f:
            lines = f.read().strip().split("\n")

        results = [float(line) for line in lines]
        t = torch.tensor(results)

        # Identify largest & smallest 20 diffs
        large_indices = t.topk(20).indices.clone().detach()
        small_indices = t.topk(20, largest=False).indices.clone().detach()

        large_layers = large_indices // NUM_HEADS
        large_heads = large_indices % NUM_HEADS

        small_layers = small_indices // NUM_HEADS
        small_heads = small_indices % NUM_HEADS

        large_pairs = [
            f"L{layer.item()}H{head.item()}: {value.item():.4f}"
            for layer, head, value in zip(large_layers, large_heads, t.topk(20, largest=True).values)
        ]

        small_pairs = [
            f"L{layer.item()}H{head.item()}: {value.item():.4f}"
            for layer, head, value in zip(small_layers, small_heads, t.topk(20, largest=False).values)
        ]

        num_visual_tokens = 576   # change if needed (e.g. if using a different patch grid)
        grid_size = int(np.sqrt(num_visual_tokens))  # e.g. 14

        # Loop over each top head and generate side-by-side visualizations
        for layer_tensor, head_tensor in zip(large_layers, large_heads):
            layer = layer_tensor.item()
            head = head_tensor.item()
            
            # --- For the clean prompt ---
            # Extract the (seq_length, HEAD_SIZE) tensor for this head in this layer.
            attn_clean = clean_attn_heads[layer, :, head]  # shape: (seq_length, HEAD_SIZE)
            # Collapse the HEAD_SIZE dimension (here we take the mean of the absolute value)
            attn_clean_scalar = attn_clean.abs().mean(dim=-1)  # shape: (seq_length,)
            # Assume the visual tokens are the last num_visual_tokens tokens:
            attn_clean_visual = attn_clean_scalar[:num_visual_tokens].float()
            # Reshape into a square grid:
            try:
                attn_clean_grid = attn_clean_visual.view(grid_size, grid_size).cpu().numpy()
            except Exception as e:
                print(f"Could not reshape clean attention for index {index}, layer {layer} head {head}: {e}")
                continue
            # Normalize to [0,1]:
            attn_clean_grid = (attn_clean_grid - attn_clean_grid.min()) / (attn_clean_grid.max() - attn_clean_grid.min() + 1e-8)
            
            # --- For the corrupted prompt ---
            attn_corr = corrupted_attn_heads[layer, :, head]  # shape: (seq_length, HEAD_SIZE)
            attn_corr_scalar = attn_corr.abs().mean(dim=-1)
            attn_corr_visual = attn_corr_scalar[:num_visual_tokens].float()
            try:
                attn_corr_grid = attn_corr_visual.view(grid_size, grid_size).cpu().numpy()
            except Exception as e:
                print(f"Could not reshape corrupted attention for index {index}, layer {layer} head {head}: {e}")
                continue
            attn_corr_grid = (attn_corr_grid - attn_corr_grid.min()) / (attn_corr_grid.max() - attn_corr_grid.min() + 1e-8)
            
            # --- Upsample the heatmaps to the image size ---
            # (Here we use PIL's resize; you could also use cv2 or torch.nn.functional.interpolate.)
            # Convert the heatmaps to 8-bit images for resizing:
            attn_clean_img = Image.fromarray(np.uint8(attn_clean_grid * 255)).resize(clean_image.size, resample=Image.BILINEAR)
            attn_corr_img = Image.fromarray(np.uint8(attn_corr_grid * 255)).resize(corrupted_image.size, resample=Image.BILINEAR)
            # Back to numpy arrays normalized to [0,1]:
            attn_clean_np = np.array(attn_clean_img) / 255.0
            attn_corr_np = np.array(attn_corr_img) / 255.0
            
            # --- Plot side-by-side ---
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # Clean image with attention overlay
            ax[0].imshow(clean_image)
            ax[0].imshow(attn_clean_np, cmap="jet", alpha=0.5)
            ax[0].set_title(f"Index {index} - Layer {layer} Head {head} (Clean)")
            ax[0].axis("off")
            
            # Corrupted image with attention overlay
            ax[1].imshow(corrupted_image)
            ax[1].imshow(attn_corr_np, cmap="jet", alpha=0.5)
            ax[1].set_title(f"Index {index} - Layer {layer} Head {head} (Corrupted)")
            ax[1].axis("off")
            
            plt.tight_layout()
            
            # Save the visualization; you can also display it with plt.show()
            vis_path = f"./data/attention_vis_index{index}_L{layer}_H{head}.png"
            plt.savefig(vis_path)
            plt.close(fig)

        # Update global running sums
        for i, diff_val in enumerate(results):
            layer = i // NUM_HEADS
            head = i % NUM_HEADS
            key_str = f"L{layer}H{head}"
            running_sums[key_str] = running_sums.get(key_str, 0.0) + diff_val

        # Write the updated running sums file
        with open(RUNNING_SUM_FILE, "w") as f:
            for key_str, val in running_sums.items():
                f.write(f"{key_str}: {val:.4f}\n")

        # Prepare final summary data for this index
        output_data = {
            "prompt": prompt,
            "clean_output": clean_output,
            "corrupted_output": corrupted_output,
            "clean_diff": clean_diff,
            "corrupted_diff": corrupted_diff,
            "largest_20_differences": large_pairs,
            "smallest_20_differences": small_pairs
        }

        summary_file = f"./data/patching_summary-{index}.json"
        if not os.path.exists(summary_file):
            with open(summary_file, "w") as f:
                pass

        with open(summary_file, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"Finished index {index}\n")
        # --- Finally, sort and display running_sums by their numeric value ---
    if os.path.exists(RUNNING_SUM_FILE):
        with open(RUNNING_SUM_FILE, "r") as f:
            lines = f.read().strip().split("\n")

        # Parse lines of the form "L0H1: 3.1415"
        running_sum_pairs = []
        for line in lines:
            if not line.strip():
                continue
            key_str, val_str = line.split(":")
            key_str = key_str.strip()
            val_str = val_str.strip()
            val = float(val_str)
            running_sum_pairs.append((key_str, val))

        running_sum_pairs.sort(key=lambda x: x[1], reverse=True)
        with open(RUNNING_SUM_FILE, "w") as f:
            for key, val in running_sum_pairs:
                f.write(f"{key}: {val:.4f}\n")
if __name__ == "__main__":
    main()

