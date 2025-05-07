import torch
import nnsight
from PIL import Image
import argparse
import os
import json
from nnsight.modeling.vllm import VLLM
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
from vllm.inputs import TextPrompt

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
    parser.add_argument("--output_dir", type=str, default="./data/text_to_image_attention",
                        help="Directory to save results")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
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
    )

    # Get model stats
    NUM_LAYERS = len(model.language_model.model.layers)
    NUM_HEADS = int(model.language_model.model.config.num_attention_heads)
    HEAD_SIZE = None  # We'll set it once we see the attn shape.

    # Define the specific attention heads to analyze
    SPECIFIC_HEADS = [
        (19, 9),   # L19H9
        (16, 0),   # L16H0
        (20, 17),  # L20H17
        (20, 21),  # L20H21
        (19, 8),   # L19H8
        (28, 18),  # L28H18
        (18, 12),  # L18H12
        (17, 25),  # L17H25
        (31, 22),  # L31H22
    ]

    # Now iterate over each entry in the JSON
    for item in data_list:
        index = item["index"]
        clean_image_path = item["clean_image_path"]
        prompt = item["prompt"]
        clean_word = item["clean_word"]

        print(f"\n----- Processing index={index} -----")
        print(f"Prompt: {prompt}")
        print(f"Clean word: {clean_word}")

        # Load image
        clean_image = Image.open(clean_image_path)

        # Tokenize the prompt to get text tokens
        tokenized_prompt = model.tokenizer.encode(prompt, return_tensors="pt")
        text_tokens = model.tokenizer.convert_ids_to_tokens(tokenized_prompt[0])
        
        # Capture the attention patterns
        with model.trace(temperature=0.0, top_p=1.0) as tracer:
            with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": clean_image})) as result:
                # Save attention scores for all layers
                all_attn = [
                    model.language_model.model.layers[layer].self_attn.attn.output.save()
                    for layer in range(NUM_LAYERS)
                ]
                
                # Save the final logits
                logits = model.logits.output.save()

        # Get the model output
        output = model.tokenizer.decode(logits.argmax(dim=-1)[0])
        print(f"Model output: {output}")

        # Stack attention across layers for shaping
        all_attn = torch.stack(all_attn, dim=0)
        
        if HEAD_SIZE is None:
            # HEAD_SIZE can be determined from attention dimension
            HEAD_SIZE = all_attn.size(2) // NUM_HEADS
        
        # Reshape so we have (num_layers, seq, num_heads, HEAD_SIZE)
        all_attn_heads = all_attn.reshape(
            all_attn.size(0), all_attn.size(1), NUM_HEADS, HEAD_SIZE
        )
        
        # Determine the number of visual tokens and text tokens
        num_visual_tokens = 576  # Typically 576 for a 24x24 grid
        grid_size = int(np.sqrt(num_visual_tokens))  # e.g. 24
        
        # Get the total sequence length
        seq_length = attention_mask.shape[1]
        
        # Calculate the number of tokens in the instructions part
        instruction_tokens = []
        if "<image>" in prompt:
            image_pos = prompt.find("<image>")
            instruction_start = image_pos + len("<image>")
            instruction_end = prompt.find("[/INST]")
            
            if instruction_end > instruction_start:
                instruction_text = prompt[instruction_start:instruction_end]
                instruction_tokens = model.tokenizer(instruction_text, add_special_tokens=False).input_ids
                print(f"Found {len(instruction_tokens)} tokens in instruction: {instruction_text}")
        
        # The first tokens are image tokens, followed by instruction tokens, then the main content tokens
        instruction_token_count = len(instruction_tokens)
        first_content_token = num_visual_tokens + instruction_token_count
        print(f"First content token index: {first_content_token} (after {num_visual_tokens} image tokens + {instruction_token_count} instruction tokens)")
        
        # The first tokens are image tokens, followed by text tokens
        text_token_indices = list(range(num_visual_tokens, seq_length))
        image_token_indices = list(range(num_visual_tokens))
        
        # For each specified head, analyze text-to-image attention
        for layer, head in SPECIFIC_HEADS:
            print(f"\nAnalyzing Layer {layer} Head {head}")
            
            # Extract the attention for this specific head
            # Shape: (seq_length, HEAD_SIZE)
            attn_head = all_attn_heads[layer, :, head]
            
            # Create a matrix to store text-to-image attention scores
            # We'll compute this from the raw attention outputs
            text_to_image_attention = torch.zeros(len(text_token_indices), num_visual_tokens)
            
            # Get attention weights by computing dot products between query and key vectors
            # This is a simplified approach - in a real model you'd need to access the actual attention weights
            for i, text_idx in enumerate(text_token_indices):
                for j, img_idx in enumerate(image_token_indices):
                    # Compute similarity between text token and image token representations
                    text_to_image_attention[i, j] = torch.dot(
                        attn_head[text_idx], attn_head[img_idx]
                    ).item()
            
            # Normalize the attention scores
            text_to_image_attention = F.softmax(text_to_image_attention, dim=-1)
            
            # Convert to numpy for easier analysis
            text_to_image_np = text_to_image_attention.numpy()
            
            # Create a heatmap visualization
            plt.figure(figsize=(20, 10))
            
            # Get text tokens for labeling - ensure we don't go out of bounds
            # The text tokens from the tokenizer don't include the image tokens
            # So we need to be careful with the indexing
            text_token_labels = []
            for i in text_token_indices:
                idx = i - num_visual_tokens
                if idx < len(text_tokens):
                    text_token_labels.append(text_tokens[idx])
                else:
                    # If we're out of bounds, use a placeholder
                    text_token_labels.append(f"[Token_{i}]")
            
            # Create the heatmap
            ax = sns.heatmap(text_to_image_np, cmap="viridis", 
                            xticklabels=False, yticklabels=text_token_labels)
            
            plt.title(f"Text-to-Image Attention for Layer {layer} Head {head}")
            plt.xlabel("Image Tokens (0-575)")
            plt.ylabel("Text Tokens")
            
            # Save the heatmap
            heatmap_path = os.path.join(args.output_dir, f"text_to_image_attn_index{index}_L{layer}H{head}.png")
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()
            
            # For each text token, find the top 10 image tokens it attends to
            for i, text_idx in enumerate(text_token_indices):
                token_text = text_token_labels[i]
                
                # Skip special tokens and padding
                if token_text.startswith('<') or token_text.startswith('['):
                    continue
                
                # Only process specific tokens: "heart", "left", "right", "star", or "sun"
                if token_text not in ["heart", "left", "right", "star", "sun"]:
                    continue
                
                # Get attention scores for this text token to all image tokens
                token_attn = text_to_image_np[i]
                
                # Find the top 10 image tokens
                top_indices = np.argsort(token_attn)[-10:][::-1]
                top_scores = token_attn[top_indices]
                
                # Create a visualization of where this text token is attending in the image
                attention_grid = np.zeros((grid_size, grid_size))
                
                # Fill in all attention scores
                for img_idx in range(num_visual_tokens):
                    row = img_idx // grid_size
                    col = img_idx % grid_size
                    attention_grid[row, col] = token_attn[img_idx]
                
                # Normalize for visualization
                attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min() + 1e-8)
                
                # Resize to image dimensions
                attention_img = Image.fromarray(np.uint8(attention_grid * 255)).resize(clean_image.size)
                attention_np = np.array(attention_img) / 255.0
                
                # Create visualization
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image
                ax[0].imshow(clean_image)
                ax[0].set_title(f"Original Image")
                ax[0].axis("off")
                
                # Attention overlay
                ax[1].imshow(clean_image)
                ax[1].imshow(attention_np, cmap="hot", alpha=0.5)
                ax[1].set_title(f"Attention from token '{token_text}'")
                ax[1].axis("off")
                
                plt.tight_layout()
                token_vis_path = os.path.join(args.output_dir, 
                                            f"token_attn_index{index}_L{layer}H{head}_token{text_idx}.png")
                plt.savefig(token_vis_path)
                plt.close(fig)
                
                # Save the top 10 image tokens and their scores to a text file
                token_data_path = os.path.join(args.output_dir, 
                                            f"token_attn_index{index}_L{layer}H{head}_token{text_idx}.txt")
                with open(token_data_path, "w") as f:
                    f.write(f"Text token: {token_text} (index {text_idx})\n")
                    f.write(f"Top 10 image tokens it attends to:\n")
                    for idx, score in zip(top_indices, top_scores):
                        row = idx // grid_size
                        col = idx % grid_size
                        f.write(f"Image token {idx} (row {row}, col {col}): {score:.6f}\n")
            
            # Save the full attention matrix to a numpy file
            np.save(os.path.join(args.output_dir, f"text_to_image_attn_index{index}_L{layer}H{head}.npy"), 
                    text_to_image_np)
        
        # Create a summary file for this index
        summary_data = {
            "prompt": prompt,
            "clean_word": clean_word,
            "model_output": output,
            "num_text_tokens": len(text_token_indices),
            "num_image_tokens": num_visual_tokens,
            "heads_analyzed": [f"L{layer}H{head}" for layer, head in SPECIFIC_HEADS]
        }
        
        summary_file = os.path.join(args.output_dir, f"text_to_image_attn_summary_index{index}.json")
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)
        
        print(f"Finished processing index {index}")

if __name__ == "__main__":
    main() 