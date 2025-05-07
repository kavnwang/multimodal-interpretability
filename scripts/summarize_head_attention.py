import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_attention_file(filepath):
    """Parse an attention_to_tokens.txt file to extract token attention patterns"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract head info from the first line
    first_line = content.split('\n')[0]
    head_info = first_line.strip()
    
    # Parse image token attention if available
    image_attention_file = filepath.replace('attention_to_tokens.txt', 'image_token_attention.txt')
    image_attention = {}
    
    if os.path.exists(image_attention_file):
        with open(image_attention_file, 'r') as f:
            img_content = f.read()
            
        # Extract image token range
        img_range_line = None
        for line in img_content.split('\n'):
            if line.startswith("Image tokens:"):
                img_range_line = line
                break
        
        if img_range_line:
            # Extract image token positions
            parts = img_range_line.split("positions ")[1].split(" to ")
            img_start = int(parts[0])
            img_end = int(parts[1])
            
            # Extract total attention to image for each token
            token_pos = None
            total_img_attn = None
            
            for line in img_content.split('\n'):
                if line.startswith("TOKEN POSITION"):
                    token_pos = int(line.split("TOKEN POSITION ")[1].split(" ")[0])
                elif line.strip().startswith("Total attention to image:"):
                    total_img_attn = float(line.strip().split(": ")[1])
                    if token_pos is not None:
                        image_attention[token_pos] = total_img_attn
    
    return {
        'head_info': head_info,
        'image_attention': image_attention,
    }

def main():
    parser = argparse.ArgumentParser(description="Summarize token attention analysis for top heads")
    parser.add_argument("--index", type=int, required=True,
                        help="Index of the example to analyze")
    args = parser.parse_args()
    
    # Path to the token attention directory for this index
    token_attn_dir = f"./data/token_attention/{args.index}"
    
    if not os.path.exists(token_attn_dir):
        print(f"No token attention data found for index {args.index}")
        print("Please run scripts/analyze_top_heads.py first")
        return
    
    # Get all head directories
    head_dirs = [d for d in os.listdir(token_attn_dir) if os.path.isdir(os.path.join(token_attn_dir, d))]
    
    # Load summary for this index
    summary_file = f"./data/patching_summary-{args.index}.json"
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        prompt = summary.get('prompt', 'Unknown prompt')
    else:
        prompt = 'Unknown prompt'
    
    # Extract and aggregate attention patterns
    head_results = {}
    image_attention_by_head = {}
    
    for head_dir in head_dirs:
        head_path = os.path.join(token_attn_dir, head_dir)
        attn_file = os.path.join(head_path, 'attention_to_tokens.txt')
        
        if os.path.exists(attn_file):
            results = parse_attention_file(attn_file)
            layer, head = head_dir[1:].split('H')
            layer = int(layer)
            head = int(head)
            
            head_results[(layer, head)] = results
            
            # Calculate average attention to image tokens
            if results['image_attention']:
                avg_img_attn = sum(results['image_attention'].values()) / len(results['image_attention'])
                image_attention_by_head[(layer, head)] = avg_img_attn
    
    # Sort heads by their attention to image tokens
    image_focused_heads = sorted(image_attention_by_head.items(), key=lambda x: x[1], reverse=True)
    
    # Print summary
    print(f"Attention Analysis Summary for Example {args.index}")
    print(f"Prompt: {prompt}")
    print(f"Number of heads analyzed: {len(head_results)}")
    print("\nHeads with most attention to image tokens:")
    
    for i, ((layer, head), avg_attn) in enumerate(image_focused_heads[:10]):
        print(f"{i+1}. L{layer}H{head}: {avg_attn:.6f} average attention to image tokens")
    
    # Create visualization of image attention by head
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plot
    head_labels = [f"L{l}H{h}" for (l, h) in image_attention_by_head.keys()]
    attention_values = list(image_attention_by_head.values())
    
    # Sort by attention value
    sorted_indices = np.argsort(attention_values)[::-1]  # Descending order
    head_labels = [head_labels[i] for i in sorted_indices]
    attention_values = [attention_values[i] for i in sorted_indices]
    
    # Plot the top 20 only for readability
    plt.bar(head_labels[:20], attention_values[:20], color='skyblue')
    plt.title(f'Top 20 Heads by Image Token Attention - Example {args.index}')
    plt.xlabel('Head')
    plt.ylabel('Average Attention to Image Tokens')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"./data/image_attention_by_head_{args.index}.png")
    print(f"\nVisualization saved to ./data/image_attention_by_head_{args.index}.png")
    
    # Create output summary file
    output_summary = {
        'index': args.index,
        'prompt': prompt,
        'top_image_focused_heads': [
            {'layer': l, 'head': h, 'avg_attention': float(a)}
            for (l, h), a in image_focused_heads[:10]
        ]
    }
    
    with open(f"./data/head_attention_summary_{args.index}.json", 'w') as f:
        json.dump(output_summary, f, indent=2)
    
    print(f"Summary saved to ./data/head_attention_summary_{args.index}.json")

if __name__ == "__main__":
    main() 