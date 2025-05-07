import torch
import nnsight
from PIL import Image
import argparse
import os
import json
from nnsight.modeling.vllm import VLLM
from transformers import AutoProcessor
from vllm.inputs import TextPrompt
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Perform token-level activation patching for top contributing heads")
    parser.add_argument("--prompt_json", type=str, required=True,
                        help="Path to the JSON file containing prompts/images.")
    parser.add_argument("--output_dir", type=str, default="./data/token_level_patching",
                        help="Directory to save the patching results.")
    parser.add_argument("--top_k_heads", type=int, default=20,
                        help="Number of top heads to analyze (default: 20)")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the JSON data (list of dicts)
    with open(args.prompt_json, "r") as f:
        data_list = json.load(f)

    # Initialize model and processor just once
    model_name = "llava-hf/llava-1.5-7b-hf"  # Using LLaVA 1.5 7B model
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
    config = model.language_model.model.config
    NUM_HEADS = config.num_attention_heads if hasattr(config, 'num_attention_heads') else 32  # Default to 32 if not found
    HEAD_SIZE = 128  # We'll set it once we see the attn shape.

    # Now iterate over each entry in the JSON
    for item in data_list:
        # Extract item data
        index = item["index"]
        clean_image_path = item["clean_image_path"]
        corrupted_image_path = item["corrupted_image_path"]
        prompt = item["prompt"]
        clean_word = item["clean_word"]
        corrupted_word = item["corrupted_word"]
        
        # Create index directory
        index_dir = os.path.join(args.output_dir, f"index_{index}")
        os.makedirs(index_dir, exist_ok=True)
        
        # Load summary to get top heads
        summary_file = f"./data/patching_summary-{index}.json"

        if not os.path.exists(summary_file):
            print(f"No patching summary found for index {index}. Run patching.py first.")
            continue

        with open(summary_file, "r") as f:
            summary = json.load(f)
            
        # Extract the top heads from the summary
        top_heads = []
        for head_info in summary["largest_20_differences"][:args.top_k_heads]:
            head_id, contribution = head_info
            layer, head = map(int, head_id.split("_"))
            top_heads.append((layer, head, contribution))
        
        print(f"\n----- Processing index={index} with {len(top_heads)} top heads -----")
        # Print detailed information about the top heads
        print("Top contributing heads:")
        for i, (layer, head, contribution) in enumerate(top_heads):
            print(f"  {i+1}. Layer {layer}, Head {head}: contribution = {contribution:.6f}")
        print("-------------------------------------")
        
        # Load images
        clean_image = Image.open(clean_image_path)
        corrupted_image = Image.open(corrupted_image_path)

        # Format prompt properly for LLaVA 1.5
        if not prompt.startswith("USER:"):
            prompt = f"USER: {prompt}\nASSISTANT:"
        
        image_tag_index = prompt.find("<image>")
        
        if image_tag_index == -1:
            print("Warning: No <image> tag found in the prompt. Calculating USER: prefix tokens.")
            # Calculate how many tokens "USER: " takes
            user_prefix_tokens = len(model.tokenizer.encode("USER: "))
            instruction_tokens_count = user_prefix_tokens
        else:
            # Tokenize the text up to the first <image> tag
            text_before_image = prompt[:image_tag_index]
            instruction_tokens_count = len(model.tokenizer.encode(text_before_image))
            print(f"Detected {instruction_tokens_count} instruction tokens before first <image> token")
            
        # Figure out the token indices for "clean_word" and "corrupted_word"
        clean_token_ids = model.tokenizer.encode(clean_word)
        corrupted_token_ids = model.tokenizer.encode(corrupted_word)

        # Get the token indices - pick the first token if multi-token word
        if len(clean_token_ids) > 1:
            clean_index = clean_token_ids[1]  # Skip BOS token
        else:
            clean_index = clean_token_ids[0]
            
        if len(corrupted_token_ids) > 1:
            corrupted_index = corrupted_token_ids[1]  # Skip BOS token
        else:
            corrupted_index = corrupted_token_ids[0]
            
        # Get token strings for the sequence
        input_ids = model.tokenizer.encode(prompt.replace("<image>", "[IMAGE]"))
        token_strings = [model.tokenizer.decode([token_id]) for token_id in input_ids]
        
        # Determine sequence length by running the model first to get actual sequence length
        with model.trace() as tracer:
            with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": clean_image})) as result:
                initial_logits = model.logits.output.save()
        
        # Get the actual sequence length from the logits tensor 
        seq_len = initial_logits.size(1)
        print(f"Actual sequence length from model output: {seq_len}")
        
        # Now that we have seq_len, we can iterate over each token_idx
        # Use seq_len-1 to avoid out of bounds indexing
        sample_results_file = os.path.join(index_dir, "sample_level_results.json")
            # Always initialize a new empty list for sample-level data
        sample_level_data = []

        for token_idx in range(seq_len - 1):
            # Check if token_idx is valid
            if token_idx >= seq_len:
                print(f"Skipping token_idx {token_idx} as it's out of bounds (seq_len = {seq_len})")
                continue
            
            # Capture the attention & logits on the "clean" version
            with model.trace() as tracer:
                with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": clean_image})) as result:
                    clean_attn = [
                        model.language_model.model.layers[layer].self_attn.attn.output.save()
                        for layer, _, _ in top_heads
                    ]
                    # Save the final layer output for the current token
                    model.language_model.model.layers[-1].output[0][-1] = model.language_model.model.layers[-1].output[0][token_idx]  # Hidden states 
                    model.language_model.model.layers[-1].output[1][-1] = model.language_model.model.layers[-1].output[1][token_idx]  # Attention mask if available
                    clean_logits = model.logits.output.save()
                    
            # Capture the attention & logits on the "corrupted" version
            with model.trace() as tracer:
                with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": corrupted_image})) as result:
                    corrupted_attn = [
                        model.language_model.model.layers[layer].self_attn.attn.output.save()
                        for layer, _, _ in top_heads
                    ]
                    # Save the final layer output for the current token
                    model.language_model.model.layers[-1].output[0][-1] = model.language_model.model.layers[-1].output[0][token_idx]  # Hidden states 
                    model.language_model.model.layers[-1].output[1][-1] = model.language_model.model.layers[-1].output[1][token_idx]  # Attention mask if available
                    corrupted_logits = model.logits.output.save()
            
            # Create a file to store sample-level results
            
            #clean_logits = clean_logits[0]
            #corrupted_logits = corrupted_logits[0]
            
            try:
                # Access model outputs for this specific token index
                clean_output = model.tokenizer.decode(clean_logits.argmax(dim=-1).item())
                corrupted_output = model.tokenizer.decode(corrupted_logits.argmax(dim=-1).item())
                
                # Calculate raw logit diffs for this token
                clean_diff = clean_logits[clean_index].item() - clean_logits[corrupted_index].item()
                corrupted_diff = corrupted_logits[clean_index].item() - corrupted_logits[corrupted_index].item()
                
                # Create token data for sample-level results
                token_data = {
                    "token_idx": token_idx,
                    "token_string": token_strings[token_idx] if token_idx < len(token_strings) else "[UNK]",
                    "clean_output": clean_output,
                    "corrupted_output": corrupted_output,
                    "clean_logit_diff": clean_diff,
                    "corrupted_logit_diff": corrupted_diff
                }
                
                # Add the token data to our new list
                sample_level_data.append(token_data)
                
                # Save sample-level data
                with open(sample_results_file, "w") as f:
                    json.dump(sample_level_data, f, indent=2)
                
                # Print details for first few tokens only to avoid overwhelming output
                if token_idx < 5:
                    print(f"Token {token_idx} ({token_data['token_string']}) - Clean Output: {clean_output}, Corrupted Output: {corrupted_output}")
                    print(f"  Clean diff: {clean_diff:.4f}, Corrupted diff: {corrupted_diff:.4f}")
            except IndexError as e:
                print(f"Index error when processing token {token_idx}: {e}")
            except Exception as e:
                print(f"Error when processing token {token_idx}: {e}")
            
            # Calculate the patch grid size for image tokens
            num_patches = 576  # 24x24 patches per image for LLaVA 1.5
            patch_grid_size = int(np.sqrt(num_patches))
            
            # Set HEAD_SIZE if not set already
            if HEAD_SIZE is None:
                # Try to determine HEAD_SIZE based on model configuration
                print(f"Using default HEAD_SIZE based on model configuration")
                HEAD_SIZE = 128  # Default for LLaVa-1.5-7b
            
            # Token-level patching for the current token
            for head_idx, (layer, head, contribution) in enumerate(top_heads):
                print(f"Processing head {head_idx+1}/{len(top_heads)}: Layer {layer} Head {head}")
                
                # Create a file to store token-level patching results 
                head_file = os.path.join(index_dir, f"L{layer}H{head}_token_level_patching.json")
                
                # Initialize or load existing token-level data
                if os.path.exists(head_file):
                    with open(head_file, 'r') as f:
                        token_level_data = json.load(f)
                else:
                    token_level_data = []
                
                # Determine if this is an image token
                is_image_token = (token_idx >= instruction_tokens_count and 
                                  token_idx < instruction_tokens_count + num_patches)

                # Get token string
                try:
                    token_string = token_strings[token_idx] if token_idx < len(token_strings) else "[UNK]"
                except IndexError:
                    token_string = "[UNK]"
                    print(f"Warning: token_idx {token_idx} is out of bounds for token_strings with length {len(token_strings)}")
                except Exception as e:
                    token_string = "[UNK]"
                    print(f"Error getting token string for token_idx {token_idx}: {e}")
                
                # Apply patching for this specific token and head
                with model.trace() as tracer:
                    try:
                        # Get the corrupted attention and clone it to avoid modifying original
                        patched_attn = corrupted_attn[head_idx].clone()
                        
                        # Get shapes for debugging
                        clean_shape = clean_attn[head_idx].shape
                        corrupted_shape = corrupted_attn[head_idx].shape
                        
                        # Print debug info for first few tokens only
                        if token_idx < 2:
                            print(f"  Clean attention shape: {clean_shape}, Corrupt attention shape: {corrupted_shape}")
                        
                        # Check if shapes match before patching
                        if clean_attn[head_idx].shape == corrupted_attn[head_idx].shape:
                            # Attempt patching only the specific head
                            patched_attn[:, head] = clean_attn[head_idx][:, head]
                        else:
                            print(f"  Warning: Shape mismatch for L{layer}H{head}. " 
                                  f"Clean: {clean_shape}, Corrupted: {corrupted_shape}")
                            continue
                            
                        # Replace attention for that layer 
                        model.language_model.model.layers[layer].self_attn.attn.output = patched_attn

                        # Invoke the model with the corrupted image
                        with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": corrupted_image})) as result:
                            # Grab the final logits after patching
                            model.language_model.model.layers[-1].output[0][-1] = model.language_model.model.layers[-1].output[0][token_idx]  # Hidden states 
                            model.language_model.model.layers[-1].output[1][-1] = model.language_model.model.layers[-1].output[1][token_idx]  # Attention mask if available
                            patched_logits = model.logits.output.save()
                    except Exception as e:
                        print(f"  Error in patching for L{layer}H{head}: {e}")
                        continue
                
                # Calculate the raw logit values for the clean token
                try:
                    clean_token_logit = clean_logits[0][clean_index].item()
                    corrupted_token_logit = corrupted_logits[0][clean_index].item()
                    
                    # Get the patched logit
                    patched_token_logit = patched_logits[0][clean_index].item()
                    
                    # Calculate patching effect using raw logit differences
                    patch_diff = patched_token_logit - corrupted_token_logit
                    
                    # Create token data entry
                    token_data = {
                        "token_idx": token_idx,
                        "token_string": token_string,
                        "is_image_token": is_image_token,
                        "clean_token_logit": clean_token_logit,
                        "corrupted_token_logit": corrupted_token_logit,
                        "patch_diff": patch_diff
                    }
                    
                    # Check if this token_idx already exists in token_level_data
                    exists = False
                    for entry_idx, entry in enumerate(token_level_data):
                        if entry["token_idx"] == token_idx:
                            # Update existing entry instead of creating a new one
                            token_level_data[entry_idx] = token_data
                            exists = True
                            break
                    
                    # If entry doesn't exist, append it
                    if not exists:
                        token_level_data.append(token_data)
                    
                    # Save token-level data
                    with open(head_file, "w") as f:
                        json.dump(token_level_data, f, indent=2)
                except IndexError as e:
                    print(f"  Index error when processing head L{layer}H{head} for token {token_idx}: {e}")
                except Exception as e:
                    print(f"  Error when processing head L{layer}H{head} for token {token_idx}: {e}")
                
                # Only print completion for final token to avoid log spam
                if token_idx == seq_len - 2:  # Changed from seq_len-1 since we iterate until seq_len-1
                    print(f"Completed token-level patching for L{layer}H{head}.")
            
            # Create a summary file for this index if we're at the last token
            if token_idx == seq_len - 1:
                # Check if summary file already exists
                summary_file = os.path.join(index_dir, "token_level_patching_summary.json")
                
                if os.path.exists(summary_file):
                    # Load existing summary data if file exists
                    with open(summary_file, "r") as f:
                        summary_data = json.load(f)
                        
                    # Update the existing summary with any new information
                    summary_data.update({
                        "instruction_token_count": instruction_tokens_count,
                        "num_image_tokens": num_patches,
                        "sequence_length": seq_len,
                        "heads_analyzed": [f"L{layer}H{head}" for layer, head, _ in top_heads],
                        "sample_level_results": "sample_level_results.json"
                    })
                else:
                    # Create new summary data if file doesn't exist
                    summary_data = {
                        "prompt": prompt,
                        "clean_word": clean_word,
                        "corrupted_word": corrupted_word,
                        "instruction_token_count": instruction_tokens_count,
                        "num_image_tokens": num_patches,
                        "sequence_length": seq_len,
                        "heads_analyzed": [f"L{layer}H{head}" for layer, head, _ in top_heads],
                        "sample_level_results": "sample_level_results.json"
                    }
                
                # Write the summary data back to file
                with open(summary_file, "w") as f:
                    json.dump(summary_data, f, indent=4)
                
                print(f"Finished processing index {index}")

if __name__ == "__main__":
    main() 