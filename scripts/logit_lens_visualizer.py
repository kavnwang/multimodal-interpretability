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
from typing import List, Tuple, Dict, Optional, Any, Union
from vllm.inputs import TextPrompt
import matplotlib.colors as mcolors
import re

def analyze_and_visualize_image_tokens(
    model: VLLM,
    images: Union[Image.Image, List[Image.Image]],
    prompt: str,
    processor: Any,
    save_dir: str = "./logit_lens_results",
    target_words: List[str] = ["heart", "sun"]
) -> Dict:
    """
    Perform logit lens analysis on image tokens and generate heatmaps for target words.
    Uses a 24x24 grid and calculates log probability increases per patch.
    
    Args:
        model: The VLLM model
        images: A single image or list of images
        prompt: The prompt containing <image> token(s) to indicate image positions
        processor: The model processor
        save_dir: Directory to save results
        target_words: Words to analyze in the logit lens
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure images is a list
    if not isinstance(images, list):
        images = [images]
    
    # Properly format prompt following the template: "USER: xxx\nASSISTANT:"
    if not prompt.startswith("USER:"):
        prompt = f"USER: {prompt}\nASSISTANT:"
    
    # Calculate the number of instruction tokens before the first image token
    # First tokenize the text with spaces replacing <image> tokens
    prompt_without_images = re.sub(r'<image>', ' [IMG] ', prompt)
    tokens_before_images = []
    
    # Tokenize the prompt without images
    tokenized_prompt = processor.tokenizer.encode(prompt_without_images)
    
    # Find the position of special [IMG] tokens in the tokenized prompt
    img_token_positions = []
    for i, token_id in enumerate(tokenized_prompt):
        token = processor.tokenizer.decode([token_id])
        if '[IMG]' in token:
            img_token_positions.append(i)
    
    # Direct approach to find <image> token position in the original prompt
    # Find the first index of <image> in the text
    image_tag_index = prompt.find("<image>")
    
    if image_tag_index == -1:
        print("Warning: No <image> tag found in the prompt. Calculating USER: prefix tokens.")
        # Calculate how many tokens "USER: " takes
        user_prefix_tokens = len(processor.tokenizer.encode("USER: "))  # Subtract 2 for special tokens
        instruction_tokens_count = user_prefix_tokens
    else:
        # Tokenize the text up to the first <image> tag
        text_before_image = prompt[:image_tag_index]
        instruction_tokens_count = len(processor.tokenizer.encode(text_before_image))  # Subtract 2 for special tokens
        print(f"Detected {instruction_tokens_count} instruction tokens before first <image> token")
    
    # Use a fixed 24x24 grid (576 patches) per image
    grid_size = 24
    num_patches = grid_size * grid_size
    
    # For multiple images, we'll have num_patches * len(images) total image token positions
    total_image_patches = num_patches * len(images)
    
    # Calculate the actual image token positions, offsetting by instruction tokens
    image_token_positions = torch.arange(total_image_patches) + instruction_tokens_count
    
    results = {}
    hidden_states = []
    for layer in range(len(model.language_model.model.layers)):
        with model.trace(temperature=0.0, top_p=1.0) as tracer:
            with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": images})) as invoke:
                layer_output = model.language_model.model.layers[layer].output.save()
                # Get final logits for comparison
                final_logits = model.logits.output.save()
            
        hidden_states.append(layer_output[0])

    # Stack hidden states [num_layers, seq_len, hidden_dim]
    hidden_states = torch.stack(hidden_states)
    
    # Get the unembedding matrix (transpose of embedding matrix)
    unembed = model.language_model.model.embed_tokens.weight
    
    # Project hidden states to vocabulary space to get logits per layer
    logits_per_layer = torch.matmul(hidden_states, unembed.T)
    
    # Find token IDs for target words
    target_token_ids = {}
    for word in target_words:
        token_id = processor.tokenizer.encode(word, add_special_tokens=False)
        if len(token_id) == 1:  # Only use single token words for simplicity
            target_token_ids[word] = token_id[0]
    
    # Create heatmaps for each layer and target word
    num_layers = len(model.language_model.model.layers)
    seq_length = logits_per_layer.size(1)
    
    # Create a dictionary to store heatmap data
    heatmap_data = {}
    for word in target_words:
        if word in target_token_ids:
            # For each word, store data for all images
            heatmap_data[word] = {
                f"image_{i}": torch.zeros((num_layers, num_patches)) 
                for i in range(len(images))
            }
    
    # Extract logits for target words at image token positions
    for layer_idx in range(num_layers):
        for i, pos in enumerate(image_token_positions):
            # Determine which image this patch belongs to
            image_idx = i // num_patches
            patch_idx = i % num_patches
            
            # Check if position is within sequence length
            if pos < seq_length:
                layer_logits = logits_per_layer[layer_idx, pos]
                
                for word, token_id in target_token_ids.items():
                    # Store the raw logit value for the target token
                    if image_idx < len(images):  # Safety check
                        heatmap_data[word][f"image_{image_idx}"][layer_idx, patch_idx] = layer_logits[token_id].item()
    
    # Tokenize the prompt to include in results
    tokenization_info = {
        "prompt": prompt,
        "tokenized": processor.tokenizer.encode(prompt),
        "decoded_tokens": [processor.tokenizer.decode([token_id]) for token_id in processor.tokenizer.encode(prompt)],
        "image_token_positions": image_token_positions.tolist(),
        "prompt_without_images": prompt_without_images,
        "tokenized_without_images": tokenized_prompt,
        "decoded_tokens_without_images": [processor.tokenizer.decode([token_id]) for token_id in tokenized_prompt],
        "detected_img_positions_in_tokenized_text": img_token_positions,
        "first_image_tag_char_index": image_tag_index if image_tag_index != -1 else None,
        "text_before_first_image": text_before_image if image_tag_index != -1 else None,
        "instruction_tokens_count": instruction_tokens_count
    }
    
    # Create visualizations
    for word, image_data_dict in heatmap_data.items():
        for image_key, data in image_data_dict.items():
            image_idx = int(image_key.split('_')[1])
            
            # Check if we have valid data before normalizing
            if data.numel() == 0:
                print(f"No data found for word '{word}' in {image_key}, skipping visualization")
                continue
                
            # Basic heatmap for all layers using raw logit values
            plt.figure(figsize=(12, 8))
            plt.imshow(data.numpy(), cmap='viridis', aspect='auto')
            plt.colorbar(label=f'Raw Logit Values for "{word}"')
            plt.xlabel('Image Patch Position (24x24 grid)')
            plt.ylabel('Model Layer')
            plt.title(f'Heatmap of Raw Logit Values for "{word}" across Image Patches and Layers - {image_key}')
            
            # Save the basic heatmap
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'heatmap_{word}_{image_key}_raw.png'))
            plt.close()
            
            # For each layer, create an overlay on the original image
            for layer_idx in range(num_layers):
                layer_data = data[layer_idx]
                
                # Reshape data to 24x24 grid
                grid_data = layer_data.view(grid_size, grid_size)
                
                # Get the corresponding image
                current_image = images[image_idx]
                
                # Resize the image to match our visualization size
                resized_image = current_image.resize((800, 800))
                
                # Convert image to numpy array for plotting
                img_array = np.array(resized_image)
                
                # Resize the grid data to match the image dimensions
                grid_data_upsampled = F.interpolate(
                    grid_data.unsqueeze(0).unsqueeze(0),
                    size=(800, 800),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                # Create figure with two subplots: original and overlay
                plt.figure(figsize=(20, 10))
                
                # Plot original image
                plt.subplot(1, 2, 1)
                plt.imshow(resized_image)
                plt.title(f"Original Image {image_idx}")
                plt.axis('off')
                
                # Plot overlay
                plt.subplot(1, 2, 2)
                plt.imshow(img_array)  # Plot the image
                # Apply the heatmap with alpha transparency using raw logit values
                heatmap = plt.imshow(grid_data_upsampled, cmap='viridis', alpha=0.6)
                plt.colorbar(heatmap, label=f'Raw Logit Values for "{word}"')
                plt.title(f'Layer {layer_idx}: Raw Logit Values for "{word}" - Image {image_idx}')
                plt.axis('off')
                
                # Save the layer-specific overlay
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'overlay_{word}_image{image_idx}_layer{layer_idx}_raw.png'))
                plt.close()
    
    # Return results dictionary
    analysis_results = {
        "tokenization": tokenization_info,
        "layer_predictions": [],
        "instruction_tokens_count": instruction_tokens_count
    }
    
    for layer in range(num_layers):
        layer_results = {
            "layer": layer,
            "position_predictions": []
        }
        
        for pos in range(seq_length):
            layer_logits = logits_per_layer[layer, pos]
            top_tokens = torch.topk(layer_logits, k=5)
            
            pos_results = {
                "position": pos,
                "top_tokens": [
                    {
                        "token": model.tokenizer.decode([idx.item()]),
                        "logit": score.item()
                    }
                    for idx, score in zip(top_tokens.indices, top_tokens.values)
                ]
            }
            layer_results["position_predictions"].append(pos_results)
        
        analysis_results["layer_predictions"].append(layer_results)
    
    # Include heatmap data in results
    analysis_results["heatmap_data"] = {
        word: {img_key: data.tolist() for img_key, data in img_data.items()}
        for word, img_data in heatmap_data.items()
    }
    
    # Print summary of results
    print("\nLogit Lens Analysis and Visualization Summary:")
    print(f"Analyzed {num_layers} layers")
    print(f"Sequence length: {seq_length}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Number of images: {len(images)}")
    print(f"Instruction tokens count: {instruction_tokens_count}")
    print(f"Image token positions start at: {instruction_tokens_count}")
    print(f"Created visualizations for target words: {list(heatmap_data.keys())}")
    
    return analysis_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_json",
        type=str,
        required=True,
        help="Path to JSON file containing prompts/images"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logit_lens_visualizations",
        help="Directory to save results and visualizations"
    )
    parser.add_argument(
        "--target_words",
        type=str,
        nargs="+",
        default=["heart", "sun"],
        help="Target words to visualize in heatmaps"
    )
    args = parser.parse_args()
    
    # Load the JSON data
    with open(args.prompt_json, "r") as f:
        data_list = json.load(f)
    
    # Initialize model and processor
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Clear CUDA cache before model initialization
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    model = VLLM(
        model_name,
        device="cuda",
        dispatch=True
    )
    
    # Process each item
    for item in data_list:
        print(f"\nProcessing item {item['index']}")
        
        # Clear CUDA cache before processing each item
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Load images - support multiple images
        images = []
        if isinstance(item["clean_image_path"], list):
            # Multiple images
            for img_path in item["clean_image_path"]:
                images.append(Image.open(img_path))
        else:
            # Single image
            images.append(Image.open(item["clean_image_path"]))
        
        # Ensure prompt follows template and contains <image> tags
        prompt = item["prompt"]
        if not prompt.startswith("USER:"):
            prompt = f"USER: {prompt}\nASSISTANT:"
            
        # Make sure prompt has <image> tokens for each image
        if "<image>" not in prompt and len(images) > 0:
            # Add image tokens at the beginning if not present
            image_tokens = " ".join(["<image>"] * len(images))
            prompt = prompt.replace("USER:", f"USER: {image_tokens}")
        
        # Create item-specific save directory
        item_save_dir = os.path.join(args.save_dir, f"item_{item['index']}")
        os.makedirs(item_save_dir, exist_ok=True)
        
        # Perform logit lens analysis and visualization
        results = analyze_and_visualize_image_tokens(
            model=model,
            images=images,
            prompt=prompt,
            processor=processor,
            save_dir=item_save_dir,
            target_words=args.target_words
        )
        
        # Save individual results
        output_file = os.path.join(
            item_save_dir,
            f"logit_lens_results_{item['index']}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 