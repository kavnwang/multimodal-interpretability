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
from typing import List, Tuple, Dict, Optional, Any
from vllm.inputs import TextPrompt

def analyze_image_tokens(
    model: VLLM,
    image: Image.Image,
    prompt: str,
    processor: Any,
    save_dir: str = "./logit_lens_results"
) -> Dict:
    """
    Perform logit lens analysis on image tokens.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Process image and get image token positions
    model_inputs = processor(images=image, text=prompt, return_tensors="pt")
    input_ids = model_inputs["input_ids"][0]
    
    # Find positions of image patch tokens (non-text tokens)
    text_vocab_size = len(processor.tokenizer)
    text_token_mask = input_ids < text_vocab_size
    image_token_positions = (~text_token_mask).nonzero().flatten()
    
    results = {}
    hidden_states = []
    for layer in range(len(model.language_model.model.layers)):
        with model.trace(temperature=0.0, top_p=1.0) as tracer:
            with tracer.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": image})) as invoke:
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
    
    # Get logit lens predictions for all positions and layers
    results = {
        "layer_predictions": []
    }
    
    num_layers = len(model.language_model.model.layers)
    seq_length = logits_per_layer.size(1)  # Get sequence length
    
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
        
        results["layer_predictions"].append(layer_results)
    
    # Print summary of results
    print("\nLogit Lens Analysis Summary:")
    print(f"Analyzed {num_layers} layers")
    print(f"Sequence length: {seq_length}")
    print(f"Top 5 predictions saved for each position")
    
    return results

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
        default="./logit_lens_results",
        help="Directory to save results"
    )
    args = parser.parse_args()
    
    # Load the JSON data
    with open(args.prompt_json, "r") as f:
        data_list = json.load(f)
    
    # Initialize model and processor
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
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
        
        # Load image
        image = Image.open(item["clean_image_path"])
        prompt = item["prompt"]
        
        # Perform logit lens analysis
        results = analyze_image_tokens(
            model=model,
            image=image,
            prompt=prompt,
            processor=processor,
            save_dir=args.save_dir
        )
        
        # Save individual results
        output_file = os.path.join(
            args.save_dir,
            f"logit_lens_results_{item['index']}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 