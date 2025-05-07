#!/usr/bin/env python
import os, json, argparse, random, math, torch, numpy as np
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import trange, tqdm
from transformers import AutoProcessor
from vllm.inputs import TextPrompt
from nnsight.modeling.vllm import VLLM
from collections import OrderedDict


normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9) if torch.is_tensor(x) \
                   else [((y - min(x)) / (max(x) - min(x) + 1e-9)) /
                           sum((y - min(x)) / (max(x) - min(x) + 1e-9) for y in x) for y in x]
                          
def set_seed(seed: int = 42):
   """Seed Python, NumPy and PyTorch – nothing else."""
   import random, numpy as np, torch
   random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def head_view(x: torch.Tensor, n_heads: int):
   """(seq, h*d) ⇄ (seq, h, d)
   Transforms between flattened and per-head attention representations.
   Useful for isolating specific attention head outputs.
   """
   if x.dim() == 3:
       seq, h, d = x.shape
       return x.reshape(seq, h * d)
   elif x.dim() == 2:
       seq, hd = x.shape
       return x.view(seq, n_heads, hd // n_heads)
   raise ValueError("Unexpected tensor rank")


def _ordered_tuples(scores_flat, k, model):
   """
   Return the *k* (layer, head, score) triples with the **highest** score.
   Orders attention heads by importance based on their scores.
   """
   vals, idx = torch.topk(scores_flat, k=k, largest=True)
   out = []
   for v, i in zip(vals.tolist(), idx.tolist()):
       L, H = divmod(i, model.h)
       out.append((int(L), int(H), float(v)))
   return out


def _fill_to_k(heads, k, exclude, model, scores=None):
   """
   Ensure the caller gets **exactly `k` distinct (layer, head) pairs**.


   1.  Drop any duplicates that might already be in `heads`.
   2.  If we still have fewer than `k`, pull from remaining heads
       in score-descending order (or layer-major if scores not provided),
       skipping anything in `exclude`.
   """
   # --- 1. make `heads` unique in‑place, preserving order -------------
   seen, uniq = set(), []
   for LH in heads:
       if LH not in seen and LH not in exclude:
           seen.add(LH)
           uniq.append(LH)


   # --- 2. fast path: we now have ≥ k unique heads  -------------------
   if len(uniq) >= k:
       return uniq[:k]


   # --- 3. Get remaining heads, sorted by score if available ----------
   remaining = []
   for L in range(model.L):
       for H in range(model.h):
           cand = (L, H)
           if cand in exclude or cand in seen:
               continue
           remaining.append(cand)
  
   # Sort remaining by score if scores provided, otherwise keep layer-major order
   if scores is not None:
       remaining.sort(key=lambda LH: -scores[LH[0] * model.h + LH[1]].item())


   uniq.extend(remaining[:k - len(uniq)])
   return uniq[:k]


# ───────────────── unified tracing helper ───────────────── #
@torch.no_grad()
def run_once(model, prompt, image):
   """
   Run the model once and capture attention outputs and logits.
   This is the core function for running inference with the model.
   """
   with model.trace(temperature=0., top_p=1.) as tr, \
        tr.invoke(TextPrompt(prompt=prompt,
                             multi_modal_data={'image': image})):
       # Save attention outputs from all layers
       attn = [ly.self_attn.attn.output.save()
               for ly in model.language_model.model.layers]
       # Save final logits
       logits = model.logits.output.save()
   return {'attn': attn, 'logits': logits}


# ────────────────────── core helpers ───────────────────── #
@torch.no_grad()
def cache_runs(model, prompt, clean_img, corrupt_img, n_layers):
   """
   Run the model on both clean and corrupted images and cache the results.
   Creates a cache of model outputs for both clean and corrupted versions of the image.
   """
   cache = {}
   for tag, img in (('clean', clean_img), ('corrupt', corrupt_img)):
       run = run_once(model, prompt, img)
       cache[f'{tag}_attn']   = torch.stack(run['attn'])
       cache[f'{tag}_logits'] = run['logits']
   return cache


def single_patch(model, L, H, cache, prompt, corrupt_img):
   """
   Patch a single attention head from clean to corrupt run.
   Replaces the output of a specific attention head in layer L, head H with clean outputs.
   """
   with model.trace(TextPrompt(prompt=prompt,
                               multi_modal_data={'image': corrupt_img})) as tr:
       patched = head_view(cache['corrupt_attn'][L], model.h).clone()
       src     = head_view(cache['clean_attn'][L],   model.h)
       # Replace the output of head H with the clean version
       patched[:, H] = src[:, H]
       model.language_model.model.layers[L].self_attn.attn.output = head_view(patched, model.h)
       return model.logits.output.save()


# ───────────── single‑patch with reference head ─────────── #
def single_patch_ref(model, L, H, cache, prompt, corrupt_img, ref_LH=None):
   """
   Patch a single head while keeping a reference head clean.
   Allows comparing importance of heads when a reference head is already patched.
   """
   with model.trace(TextPrompt(prompt=prompt,
                               multi_modal_data={'image': corrupt_img})) as tr:
       # candidate head
       patched = head_view(cache['corrupt_attn'][L], model.h).clone()
       patched[:, H] = head_view(cache['clean_attn'][L], model.h)[:, H]
       model.language_model.model.layers[L].self_attn.attn.output = head_view(patched, model.h)


       # reference head (kept clean)
       if ref_LH is not None and (L, H) != ref_LH:
           rL, rH = ref_LH
           patched_r = head_view(cache['corrupt_attn'][rL], model.h).clone()
           patched_r[:, rH] = head_view(cache['clean_attn'][rL], model.h)[:, rH]
           model.language_model.model.layers[rL].self_attn.attn.output = head_view(patched_r, model.h)


       return model.logits.output.save()


# ───────────────────── path‑patch scores ─────────────────── #
@torch.no_grad()
def path_scores(model, prompt, clean_img, corrupt_img, clean_tok, _):
   """
   Calculate path patching scores for all attention heads.
   Measures how much each head contributes to the model's ability to correctly predict tokens.
   """
   cache = cache_runs(model, prompt, clean_img, corrupt_img, model.L)
   # Base probability of correct token when using corrupted image
   p_c = torch.softmax(cache['corrupt_logits'], -1)[0, clean_tok]
   S = torch.zeros(model.L, model.h, device='cuda')
   for L in range(model.L):
       for H in range(model.h):
           # Patch each head individually and measure effect
           logits = single_patch(model, L, H, cache, prompt, corrupt_img)
           p = torch.softmax(logits, -1)[0, clean_tok]
           # Score is normalized improvement in probability
           S[L, H] = (p - p_c) / (p_c + 1e-12)
           print(f"Layer {L}, Head {H}: Score = {S[L, H].item():.6f}")
   return S.cpu()


@torch.no_grad()
def path_scores_with_reference(model, prompt, clean_img, corrupt_img,
                              clean_tok, _, ref_LH=None, exclude=set()):
   """
   Calculate path patching scores with a reference head already patched.
   Used for discovering hierarchical relationships between heads.
   """
   cache = cache_runs(model, prompt, clean_img, corrupt_img, model.L)
   p_c = torch.softmax(cache['corrupt_logits'], -1)[0, clean_tok]
   S = torch.full((model.L, model.h), float('-inf'), device='cuda')
   for L in range(model.L):
       for H in range(model.h):
           if (L, H) in exclude: continue
           # Patch head while keeping reference head patched
           logits = single_patch_ref(model, L, H, cache, prompt, corrupt_img, ref_LH)
           p = torch.softmax(logits, -1)[0, clean_tok]
           S[L, H] = (p - p_c) / (p_c + 1e-12)
           print(f"Layer {L}, Head {H}: Score = {S[L, H].item():.6f}")
   return S.cpu()


# ─────────── greedy A→D group discovery ─────────── #
def discover_groups(model, dataset,
                   k_A=40, k_B=35, k_C=30, k_D=30,
                   max_examples=None):
   """
   Discover functional groups of attention heads using the paper's methodology.
   Groups:
   A - Value fetchers: heads that directly read image information
   B - Position transmitters: heads that process spatial info
   C - Position detectors: heads that detect object positions
   D - Structural readers: heads that understand scene structure
   """
   groups, exclude, ref_LH = {}, set(), None
   ks = {'A': k_A, 'B': k_B, 'C': k_C, 'D': k_D}
   subset = dataset[:max_examples] if (max_examples and len(dataset) > max_examples) else dataset


   for tag in 'ABCD':
       k = ks[tag]
       if k == 0:                           # ← skip empty groups cleanly
           groups[tag] = []
           continue
          
       print(f'Group {tag} scoring {k} heads')
       S_sum, n = torch.zeros(model.L, model.h, device='cuda'), 0
       for ex in tqdm(subset, desc=f'Group {tag} scoring'):
           prompt = ex['prompt']
           if not prompt.startswith('USER:'):
               prompt = f'USER: {prompt}\nASSISTANT:'
           if '<image>' not in prompt:
               prompt = prompt.replace('USER:', 'USER: <image>')
           cl_img, c_img = Image.open(ex['image']), Image.open(ex['noise_image'])
           tok = ex['correct_token']
           if isinstance(tok, str):
               tok = model.tokenizer(tok).input_ids[1]
           # Calculate scores with reference to the previously found group
           S = path_scores_with_reference(model, prompt, cl_img, c_img,
                                          tok, 0, ref_LH, exclude)
           S_sum += S.to(S_sum.device); n += 1
       S_avg = S_sum / max(n, 1)
      
       # Print average scores for all heads
       print("\nAverage scores for Group", tag)
       for L in range(model.L):
           for H in range(model.h):
               if (L, H) not in exclude:
                   print(f"Layer {L}, Head {H}: Avg Score = {S_avg[L, H].item():.6f}")
      
       for L, H in exclude:
           S_avg[L, H] = float('-inf')
       S_flat = S_avg.flatten()
       triples = _ordered_tuples(S_flat, k, model)
       heads = [(L, H) for (L, H, _) in triples]


       heads = _fill_to_k(heads, k, exclude, model, S_flat)
       triples = [(L, H, S_avg[L, H].item()) for (L, H) in heads]
       triples.sort(key=lambda t: -t[2])  # Sort descending by score


       groups[tag] = triples
       exclude.update(heads)
       ref_LH = heads[0]  # Use the top head from this group as reference for next group
       print(f'⇢ Group {tag}: {heads}')
   return groups


def discover_receivers(model, dataset, senders, k,
                      exclude=set(), tag='B'):
   """
   Given a fixed set of `senders` (heads already known to matter),
   pick the top‑k *receiver* heads that most increase performance
   when they are ALSO patched.
  
   Used to find causal chains through the network - heads that receive
   information from other important heads.
   """
   S_sum = torch.zeros(model.L, model.h, device='cuda')
   n = 0
   for ex in tqdm(dataset, desc=f'Group {tag} scoring'):
       prompt = ex['prompt']
       if not prompt.startswith('USER:'):
           prompt = f'USER: {prompt}\nASSISTANT:'
       if '<image>' not in prompt:
           prompt = prompt.replace('USER:', 'USER: <image>')


       clean_img   = Image.open(ex['image'])
       corrupt_img = Image.open(ex['noise_image'])
       tok = ex['correct_token']
       if isinstance(tok, str):
           tok = model.tokenizer(tok).input_ids[1]


       for sender_triple in senders:
           sender = (sender_triple[0], sender_triple[1])  # Extract just the (L, H) tuple from the triple
           for Lr in range(model.L):
               for Hr in range(model.h):
                   if (Lr, Hr) in exclude or Lr >= sender[0]:
                       continue
                   # Calculate score improvement when patching both sender and receiver
                   S_sum[Lr, Hr] += two_hop_score(
                       model, prompt, clean_img, corrupt_img,
                       sender, (Lr, Hr), tok
                   )
       n += 1


   S_avg = S_sum / max(n, 1)
  
   # Print average scores for all candidate receiver heads
   print(f"\nAverage scores for Group {tag} receivers:")
   for L in range(model.L):
       for H in range(model.h):
           if (L, H) not in exclude:
               print(f"Layer {L}, Head {H}: Avg Score = {S_avg[L, H].item():.6f}")
  
   for L, H in exclude:
       S_avg[L, H] = float('-inf')
  
   S_flat = S_avg.flatten()
   triples = _ordered_tuples(S_flat, k, model)
   heads = [(L, H) for (L, H, _) in triples]
   heads = _fill_to_k(heads, k, exclude, model, S_flat)
   triples = [(L, H, S_avg[L, H].item()) for (L, H) in heads]
   triples.sort(key=lambda t: -t[2])  # Sort descending by score
   return triples


# ──────────────────── new multi‑patch helpers ─────────────────── #
def patch_heads(model, cache, targets, src='clean', dst='corrupt'):
   """
   Replace the *value vectors* of every (layer,head) in `targets`
   taken from the `src` run into the forward‑pass of the `dst` run.
  
   Utility for patching multiple heads at once.
   """
   for (L, H) in targets:
       patched = head_view(cache[f'{dst}_attn'][L], model.h).clone()
       patched[:, H] = head_view(cache[f'{src}_attn'][L], model.h)[:, H]
       model.language_model.model.layers[L].self_attn.attn.output = head_view(
           patched, model.h
       )


@torch.no_grad()
def two_hop_score(model, prompt, clean_img, corrupt_img,
                 sender, receiver, clean_tok):
   """
   Score a *single* sender → receiver path.
  
   Measures how much a receiver head improves performance when a
   sender head is already patched - identifying causal chains.


      sender   = (L_s, H_s)
      receiver = (L_r, H_r)
   """
   cache = cache_runs(model, prompt, clean_img, corrupt_img, model.L)


   # 1) baseline  (sender only)
   with model.trace(TextPrompt(prompt=prompt,
                               multi_modal_data={'image': corrupt_img})) as tr:
       patch_heads(model, cache, [sender], 'clean', 'corrupt')
       logits_sender = model.logits.output.save()
   p_sender = torch.softmax(logits_sender, -1)[0, clean_tok]


   # 2) candidate (sender + receiver)
   with model.trace(TextPrompt(prompt=prompt,
                               multi_modal_data={'image': corrupt_img})) as tr:
       patch_heads(model, cache, [sender, receiver], 'clean', 'corrupt')
       logits_pair = model.logits.output.save()
   p_pair = torch.softmax(logits_pair, -1)[0, clean_tok]


   # Score is normalized improvement from adding receiver
   score = (p_pair - p_sender) / (p_sender + 1e-12)
   print(f"Sender {sender} → Receiver {receiver}: Score = {score.item():.6f}")
   return score


@torch.no_grad()
def group_patch_scores(model, prompt, cl_img, c_img,
                      cl_id, _, groups):
   """
   Calculate scores for patching entire groups of heads.
   Evaluates both individual head contributions and full group effect.
   """
   cache = cache_runs(model, prompt, cl_img, c_img, model.L)
   p_c = torch.softmax(cache['corrupt_logits'], -1)[0, cl_id]


   def apply_one(L, H):
       with model.trace(TextPrompt(prompt=prompt,
                                   multi_modal_data={'image': c_img})) as tr:
           patched = head_view(cache['corrupt_attn'][L], model.h).clone()
           patched[:, H] = head_view(cache['clean_attn'][L], model.h)[:, H]
           model.language_model.model.layers[L].self_attn.attn.output = head_view(patched, model.h)
           p = torch.softmax(model.logits.output.save(), -1)[0, cl_id]
           return (p - p_c) / (p_c + 1e-12)


   out = {}
   for name, heads in groups.items():
       # Calculate individual scores for each head in group
       g_scores = {(L, H): apply_one(L, H) for L, H in heads}
       with model.trace(TextPrompt(prompt=prompt,
                                   multi_modal_data={'image': c_img})) as tr:
           # Patch all heads in the group
           for (L, H) in heads:
               patched = head_view(cache['corrupt_attn'][L], model.h).clone()
               patched[:, H] = head_view(cache['clean_attn'][L], model.h)[:, H]
               model.language_model.model.layers[L].self_attn.attn.output = head_view(patched, model.h)
           p_all = torch.softmax(model.logits.output.save(), -1)[0, cl_id]
       # Score for patching all heads in group together
       g_scores['_ALL_'] = (p_all - p_c) / (p_c + 1e-12)
       out[name] = g_scores
   return out


# ──────────────────────── I/O helpers ─────────────────────── #
def dump(obj, base, fmt):
   """
   Save results in multiple formats (pt, json, csv).
   Utility for exporting results for further analysis.
   """
   os.makedirs(os.path.dirname(base), exist_ok=True)
   if fmt in ('pt','all'): torch.save(obj, f'{base}.pt')
   if fmt in ('json','all'):
       with open(f'{base}.json','w') as f: json.dump(obj if isinstance(obj, dict)
                                                   else obj.tolist(), f, indent=2)
   if fmt in ('csv','all'):
       with open(f'{base}.csv','w') as f:
           if isinstance(obj, dict):
               f.write('group,score\n')
               for k,v in obj.items(): f.write(f'{k},{v}\n')
           else:
               f.write('layer,head,score\n')
               for L in range(obj.shape[0]):
                   for H in range(obj.shape[1]):
                       f.write(f'{L},{H},{obj[L,H].item()}\n')


def heatmap(S, title, path):
   """
   Create a heatmap visualization of path patching scores.
   Useful for visualizing which heads are most important.
   """
   plt.figure(figsize=(10,6))
   plt.imshow(S, aspect='auto', cmap='coolwarm')
   plt.colorbar(); plt.title(title); plt.xlabel('Head'); plt.ylabel('Layer')
   plt.tight_layout(); plt.savefig(path); plt.close()


# ───────────────────────────── main ──────────────────────────── #
def main():
   """
   Main execution function that:
   1. Parses arguments
   2. Sets up the model
   3. Discovers functional groups of attention heads (A,B,C,D)
   4. Saves and visualizes results
   """
   p = argparse.ArgumentParser()
   p.add_argument("--prompt_json",        required=True)
   p.add_argument("--seed",               type=int, default=42)
   p.add_argument("--out_dir",            default="./data/path_patching")


   # group discovery & usage
   p.add_argument("--discover_groups",    action="store_true")
   p.add_argument("--use_groups",         action="store_true")
   p.add_argument("--manual_groups",      action="store_true")
   p.add_argument("--box_tracking_dataset")


   # group sizes
   p.add_argument("--n_value_fetcher", type=int, default=40)
   p.add_argument("--n_pos_trans",    type=int, default=35)
   p.add_argument("--n_pos_detect",   type=int, default=30)
   p.add_argument("--n_struct_read",  type=int, default=30)


   # misc
   p.add_argument("--output_format", choices=["json","pt","csv","all"], default="all")
   p.add_argument("--visualize",     action="store_true")
   args = p.parse_args(); set_seed(args.seed)


   # Early-argument validation
   if args.n_value_fetcher <= 0 or args.n_pos_trans <= 0 \
      or args.n_pos_detect <= 0 or args.n_struct_read <= 0:
       p.error("All group sizes must be positive. "
               "Received: "
               f"A={args.n_value_fetcher}, B={args.n_pos_trans}, "
               f"C={args.n_pos_detect}, D={args.n_struct_read}")


   # Initialize the model (LLaVA-1.5-7B)
   model_name = "llava-hf/llava-1.5-7b-hf"
   model = VLLM(model_name, device='cuda', dispatch=True)
   model.L, model.h, model.d = 32, 32, 128   # fixed for LLaVA‑1·5‑7B
   processor = AutoProcessor.from_pretrained(model_name)


   # ───────────────── load data ───────────────── #
   data = json.load(open(args.prompt_json))
   if args.box_tracking_dataset:
       box_data = json.load(open(args.box_tracking_dataset))
       for e in box_data:
           e.setdefault('image', e.get('clean_image_path'))


   # ────────────────── STAGE 0: value‑fetchers (Group A) ───────────────── #
   # Group A: Heads that directly read visual information from the image
   groups = {}
   groups['A'] = discover_groups(
       model, data,
       k_A=args.n_value_fetcher,
       k_B=0, k_C=0, k_D=0,
       max_examples=100
   )['A']


   # ────────────────── STAGE 1: pos‑transmitters (Group B, Q‑path) ─────── #
   # Group B: Heads that process and transmit position information (Q-path)
   groups['B'] = discover_receivers(
       model, data, groups['A'],
       k=args.n_pos_trans,
       exclude=set((L, H) for L, H, _ in groups['A']), tag='B'
   )


   # ────────────────── STAGE 2: pos‑detectors (Group C, V‑path) ────────── #
   # Group C: Heads that detect object positions (V-path)
   groups['C'] = discover_receivers(
       model, data, groups['B'],
       k=args.n_pos_detect,
       exclude=set((L, H) for L, H, _ in groups['A'] + groups['B']), tag='C'
   )


   # ────────────────── STAGE 3: structural‑readers (Group D) ───────────── #
   # Group D: Heads that understand structural relationships between objects
   groups['D'] = discover_receivers(
       model, data, groups['C'],
       k=args.n_struct_read,
       exclude=set((L, H) for L, H, _ in groups['A'] + groups['B'] + groups['C']), tag='D'
   )


   # Pretty print and save with scores
   pretty = OrderedDict()
   for g in 'ABCD':
       pretty[g] = [{'layer': L, 'head': H, 'score': s}
                   for (L, H, s) in groups[g]]


   # save results with scores in readable format
   json.dump(pretty, open(os.path.join(args.out_dir,
           "groups_with_scores.json"), "w"), indent=2)


   # Print final groups with their scores
   print("⇢ FINAL GROUPS (most → least important)")
   for g, triples in pretty.items():
       print(f"\nGroup {g}:")
       for rank, d in enumerate(triples, 1):
           L, H, s = d['layer'], d['head'], d['score']
           print(f"  {rank:2d}. layer {L:2d} · head {H:2d}   score = {s:+.6f}")


   # save and show raw groups data
   os.makedirs(args.out_dir, exist_ok=True)
   json.dump(groups, open(os.path.join(args.out_dir, "groups.json"), "w"), indent=2)
   print("⇢ FINAL GROUPS")
   for g, hs in groups.items():
       print(f"  {g}: {hs}")


if __name__ == "__main__": main()



