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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_json", type=str, required=True,
                        help="Path to the JSON file containing prompts/images.")
    parser.add_argument("--output_dir", type=str, default="./data/image_to_text_attention",
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
        
        # Set up generation config
        generation_config = {"generation_config": {"temperature": 0.0, "top_p": 1.0}}

        # Capture the attention patterns
        with model.trace() as tracer:
            with tracer.invoke(prompt=prompt, multi_modal_data={"image": clean_image}, **generation_config) as result:
                # Save attention scores for specific heads
                # We need to save the attention scores (not just the outputs)
                attention_scores = {}
                for layer, head in SPECIFIC_HEADS:
                    attention_scores[(layer, head)] = model.language_model.model.layers[layer].self_attn.attn.attn_scores.output.save()
                
                # Also save the attention mask to identify text vs. image tokens
                attention_mask = model.language_model.model.attention_mask.output.save()
                
                # Save the final logits
                logits = model.logits.output.save()

        # Get the model output
        output = model.tokenizer.decode(logits.argmax(dim=-1)[0])
        print(f"Model output: {output}")

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
        
        # For each head, analyze image-to-text attention
        for layer, head in SPECIFIC_HEADS:
            print(f"\nAnalyzing Layer {layer} Head {head}")
            
            # Get the attention scores for this head
            # Shape: [batch_size, num_heads, seq_length, seq_length]
            attn_scores = attention_scores[(layer, head)]
            
            # Extract scores for this specific head
            # Shape: [batch_size, seq_length, seq_length]
            head_scores = attn_scores[0, head]
            
            # Create a matrix to store image-to-text attention scores
            # Rows: image tokens, Columns: text tokens
            image_to_text_attention = torch.zeros(num_visual_tokens, len(text_token_indices))
            
            # Fill the matrix with attention scores
            for i, img_idx in enumerate(image_token_indices):
                for j, text_idx in enumerate(text_token_indices):
                    image_to_text_attention[i, j] = head_scores[img_idx, text_idx].item()
            
            # Convert to numpy for easier analysis
            image_to_text_np = image_to_text_attention.numpy()
            
            # Create a heatmap visualization - but this would be too large
            # Instead, we'll compute the average attention from each image token to all text tokens
            avg_attention_to_text = image_to_text_np.mean(axis=1)
            
            # Reshape into a grid
            attention_grid = np.zeros((grid_size, grid_size))
            for img_idx in range(num_visual_tokens):
                row = img_idx // grid_size
                col = img_idx % grid_size
                attention_grid[row, col] = avg_attention_to_text[img_idx]
            
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
            ax[1].set_title(f"Average Attention to Text\nLayer {layer} Head {head}")
            ax[1].axis("off")
            
            plt.tight_layout()
            avg_vis_path = os.path.join(args.output_dir, f"avg_image_to_text_attn_index{index}_L{layer}H{head}.png")
            plt.savefig(avg_vis_path)
            plt.close(fig)
            
            # Now, for each text token, create a heatmap showing which image regions attend to it
            for j, text_idx in enumerate(text_token_indices):
                token_text = text_tokens[j]
                
                # Skip special tokens and padding
                if token_text.startswith('<') or token_text.startswith('['):
                    continue
                
                # Get attention scores from all image tokens to this text token
                token_attn = image_to_text_np[:, j]
                
                # Reshape into a grid
                token_grid = np.zeros((grid_size, grid_size))
                for img_idx in range(num_visual_tokens):
                    row = img_idx // grid_size
                    col = img_idx % grid_size
                    token_grid[row, col] = token_attn[img_idx]
                
                # Normalize for visualization
                token_grid = (token_grid - token_grid.min()) / (token_grid.max() - token_grid.min() + 1e-8)
                
                # Resize to image dimensions
                token_img = Image.fromarray(np.uint8(token_grid * 255)).resize(clean_image.size)
                token_np = np.array(token_img) / 255.0
                
                # Create visualization
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image
                ax[0].imshow(clean_image)
                ax[0].set_title(f"Original Image")
                ax[0].axis("off")
                
                # Attention overlay
                ax[1].imshow(clean_image)
                ax[1].imshow(token_np, cmap="hot", alpha=0.5)
                ax[1].set_title(f"Image Regions Attending to '{token_text}'")
                ax[1].axis("off")
                
                plt.tight_layout()
                token_vis_path = os.path.join(args.output_dir, 
                                            f"image_to_token_attn_index{index}_L{layer}H{head}_token{text_idx}.png")
                plt.savefig(token_vis_path)
                plt.close(fig)
                
                # Find the top 10 image tokens that attend to this text token
                top_indices = np.argsort(token_attn)[-10:][::-1]
                top_scores = token_attn[top_indices]
                
                # Save the top 10 image tokens and their scores to a text file
                token_data_path = os.path.join(args.output_dir, 
                                            f"image_to_token_attn_index{index}_L{layer}H{head}_token{text_idx}.txt")
                with open(token_data_path, "w") as f:
                    f.write(f"Text token: {token_text} (index {text_idx})\n")
                    f.write(f"Top 10 image tokens attending to it:\n")
                    for idx, score in zip(top_indices, top_scores):
                        row = idx // grid_size
                        col = idx % grid_size
                        f.write(f"Image token {idx} (row {row}, col {col}): {score:.6f}\n")
            
            # Save the full attention matrix to a numpy file
            np.save(os.path.join(args.output_dir, f"image_to_text_attn_index{index}_L{layer}H{head}.npy"), 
                    image_to_text_np)
        
        # Create a summary file for this index
        summary_data = {
            "prompt": prompt,
            "clean_word": clean_word,
            "model_output": output,
            "num_text_tokens": len(text_token_indices),
            "num_image_tokens": num_visual_tokens,
            "heads_analyzed": [f"L{layer}H{head}" for layer, head in SPECIFIC_HEADS]
        }
        
        summary_file = os.path.join(args.output_dir, f"image_to_text_attn_summary_index{index}.json")
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)
        
        print(f"Finished processing index {index}")

if __name__ == "__main__":
    main() 