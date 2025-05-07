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
from tqdm import tqdm

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
    parser.add_argument("--output_dir", type=str, default="./data/token_patching",
                        help="Directory to save the patching results.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

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

    # Define the specific attention heads to analyze
    SPECIFIC_HEADS = [
        (19, 9),   # L19H9: 11.4375
        (16, 0),   # L16H0: 11.1875
        (20, 17),  # L20H17: 9.6875
        (20, 21),  # L20H21: 9.5312
        (19, 8),   # L19H8: 9.4062
        (28, 18),  # L28H18: 9.3750
        (18, 12),  # L18H12: 8.7500
        (17, 25),  # L17H25: 7.3750
        (31, 22),  # L31H22: 6.9688
    ]

    # We'll keep a single RUNNING_SUM_FILE for all data
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
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(RUNNING_SUM_FILE), exist_ok=True)
        
        # If it doesn't exist, create placeholders for each specific head
        with open(RUNNING_SUM_FILE, "w") as f:
            for layer, head in SPECIFIC_HEADS:
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
        
        # Calculate the number of tokens in the instructions part
        # First, extract the instruction part (after <image> tag)
        instruction_tokens = []
        if "<image>" in prompt:
            image_pos = prompt.find("<image>")
            instruction_start = image_pos + len("<image>")
            instruction_end = prompt.find("[/INST]")
            
            if instruction_end > instruction_start:
                instruction_text = prompt[instruction_start:instruction_end]
                instruction_tokens = model.tokenizer(instruction_text, add_special_tokens=False).input_ids
                print(f"Found {len(instruction_tokens)} tokens in instruction: {instruction_text}")

        # Load images
        clean_image = Image.open(clean_image_path)
        corrupted_image = Image.open(corrupted_image_path)

        # Figure out the token indices for "clean_word" and "corrupted_word"
        clean_token_ids = model.tokenizer(clean_word).input_ids
        corrupted_token_ids = model.tokenizer(corrupted_word).input_ids

        # Typically, the first ID is the BOS or other special token,
        # so we often pick index=1. But you might need to confirm:
        # e.g. the second element of the token_ids
        clean_index = torch.tensor(clean_token_ids)[1].item()
        corrupted_index = torch.tensor(corrupted_token_ids)[1].item()

        # Capture the attention & logits on the "clean" version
        with model.trace(temperature=0.0, top_p=1.0) as tracer:
            with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": clean_image})) as result:
                clean_attn = [
                    model.language_model.model.layers[layer].self_attn.attn.output.save()
                    for layer, _ in SPECIFIC_HEADS
                ]
                clean_logits = model.logits.output.save()

        # Capture the attention & logits on the "corrupted" version
        with model.trace(temperature=0.0, top_p=1.0) as tracer:
            with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": corrupted_image})) as result:
                corrupted_attn = [
                    model.language_model.model.layers[layer].self_attn.attn.output.save()
                    for layer, _ in SPECIFIC_HEADS
                ]
                corrupted_logits = model.logits.output.save()

        # Convert model output to strings
        clean_output = model.tokenizer.decode(clean_logits.argmax(dim=-1))
        corrupted_output = model.tokenizer.decode(corrupted_logits.argmax(dim=-1))

        print(f"\n----- Processing index={index} -----")
        print(f"Clean Output = {clean_output}")
        print(f"Corrupted Output = {corrupted_output}")

        # These diffs are used for your patching measurement
        clean_diff = clean_logits[0][clean_index].item() - clean_logits[0][corrupted_index].item()
        corrupted_diff = corrupted_logits[0][clean_index].item() - corrupted_logits[0][corrupted_index].item()

        # For token-level analysis, we'll use the specific heads
        num_visual_tokens = 576   # change if needed (e.g. if using a different patch grid)
        grid_size = int(np.sqrt(num_visual_tokens))  # e.g. 24
        
        # Calculate the first token index after image and instruction tokens
        first_content_token = num_visual_tokens + len(instruction_tokens)
        print(f"First content token index: {first_content_token} (after {num_visual_tokens} image tokens + {len(instruction_tokens)} instruction tokens)")

        # For each head, perform token-level patching
        for layer, head in SPECIFIC_HEADS:
            print(f"\nProcessing Layer {layer} Head {head}")
            
            # Results for this head
            token_patch_results = []
            
            # For each visual token position
            for token_idx in tqdm(range(num_visual_tokens), desc=f"L{layer}H{head} tokens"):
                torch.cuda.empty_cache()
                
                # Perform patching for this specific token position
                with model.trace() as tracer:
                    # Get the corrupted attention and clone it
                    patched_attn = corrupted_attn[layer].clone()
                    
                    # Get the clean attention for this token
                    clean_token_attn = clean_attn[layer][token_idx, head].clone()
                    
                    # Replace just this token's attention for this head
                    patched_attn[token_idx, head] = clean_token_attn
                    
                    # Replace attention for that layer
                    model.language_model.model.layers[layer].self_attn.attn.output = patched_attn
                    
                    # Invoke the model with the corrupted image
                    generation_config = {"generation_config": {"temperature": 0.0, "top_p": 1.0}}
                    with tracer.invoke(prompt=prompt, multi_modal_data={"image": corrupted_image}, **generation_config) as result:
                        # Grab the final logits after patching
                        patched_output = model.logits.output.save()
                
                # Compute the difference relative to corrupted_diff
                patch_diff = (patched_output[0][clean_index].item() - 
                              patched_output[0][corrupted_index].item()) - corrupted_diff
                
                # Store the result
                token_patch_results.append((token_idx, patch_diff))
            
            # Sort results by patch difference (largest to smallest)
            token_patch_results.sort(key=lambda x: x[1], reverse=True)
            
            # Save results to file
            output_file = os.path.join(args.output_dir, f"token_patching_index{index}_L{layer}H{head}.csv")
            with open(output_file, "w") as f:
                f.write("token_idx,patch_diff\n")
                for token_idx, diff in token_patch_results:
                    f.write(f"{token_idx},{diff:.6f}\n")
            
            # Create visualization for this head
            # Reshape token patch differences into a grid
            patch_diff_grid = np.zeros((grid_size, grid_size))
            
            # Start position reference for visualization (the image tokens start at 0)
            # This value is used for reference/logging only
            vis_start_idx = 0
            
            for token_idx, diff in token_patch_results:
                # Adjust row and column based on token_idx
                row = token_idx // grid_size
                col = token_idx % grid_size
                patch_diff_grid[row, col] = diff
            
            # Normalize for visualization
            patch_diff_grid = (patch_diff_grid - patch_diff_grid.min()) / (patch_diff_grid.max() - patch_diff_grid.min() + 1e-8)
            
            # Resize to image dimensions
            patch_diff_img = Image.fromarray(np.uint8(patch_diff_grid * 255)).resize(clean_image.size)
            patch_diff_np = np.array(patch_diff_img) / 255.0
            
            # Create visualization
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            
            # Clean image
            ax[0].imshow(clean_image)
            ax[0].set_title(f"Clean Image")
            ax[0].axis("off")
            
            # Corrupted image
            ax[1].imshow(corrupted_image)
            ax[1].set_title(f"Corrupted Image")
            ax[1].axis("off")
            
            # Patching heatmap
            ax[2].imshow(corrupted_image)
            ax[2].imshow(patch_diff_np, cmap="jet", alpha=0.5)
            ax[2].set_title(f"Token Patching Heatmap L{layer}H{head}\nInstruction tokens: {len(instruction_tokens)}")
            ax[2].axis("off")
            
            plt.tight_layout()
            vis_path = os.path.join(args.output_dir, f"token_patching_vis_index{index}_L{layer}H{head}.png")
            plt.savefig(vis_path)
            plt.close(fig)
        
        # Create a summary file for this index
        summary_data = {
            "prompt": prompt,
            "clean_word": clean_word,
            "corrupted_word": corrupted_word,
            "clean_output": clean_output,
            "corrupted_output": corrupted_output,
            "clean_diff": clean_diff,
            "corrupted_diff": corrupted_diff,
            "instruction_token_count": len(instruction_tokens),
            "first_content_token": first_content_token, 
            "heads_analyzed": [f"L{layer}H{head}" for layer, head in SPECIFIC_HEADS]
        }
        
        summary_file = os.path.join(args.output_dir, f"token_patching_summary_index{index}.json")
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)
        
        print(f"Finished processing index {index}")

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
