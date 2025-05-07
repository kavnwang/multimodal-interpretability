#!/usr/bin/env python
"""
Faithfulness / circuit‑patch experiment
– single‑token evaluation
– no hooks or lambdas: only direct `.value` edits inside NNSight traces
"""

import argparse
import collections
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from nnsight.modeling.vllm import VLLM
from transformers import AutoProcessor
from vllm.inputs import TextPrompt


# ───────────────────────── helpers ──────────────────────────
def get_single_token_id(text: str, tokenizer) -> int:
    ids = tokenizer(text.strip(), add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"Label '{text}' tokenises into {len(ids)} tokens – must be 1.")
    return ids[0]


def argmax_token(logits: torch.Tensor) -> int:
    return logits.argmax(dim=-1).item()


def normalise(tok: str) -> str:
    return tok.strip().lower()


# ─────────────────────────── main ───────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_json", required=True)
    p.add_argument("--groups_json")                # optional: overrides the default A‑D dict
    p.add_argument("--csv_path", default="./data/ablation_summary.csv")
    args = p.parse_args()

    # 1. data ────────────────────────────────────────────────
    data = json.load(open(args.prompt_json))

    # layer‑head groups ──────────────────────────────────────
    default_groups = {
        "A": [[17, 28], [14, 0], [17, 8], [13, 30], [15, 10], [17, 0], [13, 9], [14, 5],
              [12, 0], [9, 19], [14, 11], [15, 29], [11, 19], [8, 6], [13, 5], [10, 4],
              [6, 6], [9, 6], [6, 15], [8, 11]],
        "B": [[10, 31], [10, 24], [12, 10], [15, 2], [14, 15]],
        "C": [[28, 26], [19, 15], [8, 25], [4, 1], [11, 4], [8, 15], [9, 31], [9, 23],
              [7, 7], [7, 14]],
        "D": [[9, 5], [8, 23], [10, 5], [10, 13], [10, 6]],
    }
    if args.groups_json:
        default_groups = json.load(open(args.groups_json))

    allowed = {(layer, head) for g in default_groups.values() for layer, head in g}

    # 2. model + tokenizer ───────────────────────────────────
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    model = VLLM(model_name, device="cuda", dispatch=True)

    L = len(model.language_model.model.layers)
    H = model.language_model.model.config.num_attention_heads
    D_HEAD = model.language_model.model.config.hidden_size // H

    # running sums / counts for per‑head means
    head_sums  = [torch.zeros(H, D_HEAD) for _ in range(L)]  # stay on cpu
    head_counts = [collections.Counter() for _ in range(L)]

    # logs
    total = correct_full = correct_circ = correct_ablate = 0
    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()

    # 3. evaluation loop ─────────────────────────────────────
    for item in tqdm(data, desc="Evaluating"):
        idx           = item["index"]
        clean_img     = Image.open(item["image"])
        corrupt_img   = Image.open(item["noise_image"])
        prompt        = item["prompt"]
        good_tok      = item["correct_token"]
        bad_tok       = item["corrupted_token"]

        good_id = get_single_token_id(good_tok, tokenizer)
        bad_id  = get_single_token_id(bad_tok,  tokenizer)

        # ── pass 1 : clean image – gather means + full accuracy ──
        with model.trace(temperature=0.0, top_p=1.0) as tr:
            with tr.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": clean_img})):
                logits_clean = model.logits.output.save()               # (1, vocab)
                attn_nodes   = [layer.self_attn.attn.output.save() for layer in
                                model.language_model.model.layers]      # Node list

            # accumulate running mean per head
        for l, node in enumerate(attn_nodes):   
            val = node.detach().cpu()                    # (seq, d_model)
            heads = val.reshape(val.size(0), H, D_HEAD).mean(0)  # (H, D_HEAD)
            head_sums[l]  += heads
            for h in range(H):
                head_counts[l][h] += 1

        # accuracy full
        pred_full = argmax_token(logits_clean[0])
        correct_full += int(normalise(tokenizer.decode([pred_full]))
                            == normalise(good_tok))
        total += 1

        # Pre‑compute per‑layer replacement vectors (on GPU) once
        means_gpu = []
        for l in range(L):
            counts = torch.tensor([head_counts[l][h] for h in range(H)],
                                  dtype=torch.float32).unsqueeze(-1)
            means_gpu.append((head_sums[l] / counts).to("cuda"))

        # ── 1) RUN on the corrupted image *just to capture* the raw attentions ──
        with model.trace(temperature=0.0, top_p=1.0) as tr:
            with tr.invoke(TextPrompt(prompt=prompt,
                                    multi_modal_data={"image": corrupt_img})):

                # save logits and every layer's attention output
                logits_corr = model.logits.output.save()
                attn_corr   = [
                    model.language_model.model.layers[l]
                        .self_attn.attn.output.save()          # (seq, d_model)
                    for l in range(L)
                ]

        # ── 2) "Circuit‑only" run: open a **new** trace and patch disallowed heads ──
        with model.trace(temperature=0.0, top_p=1.0) as tr:
            with tr.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": corrupt_img})):

                for l in range(L):
                    seq_len, d_model = attn_corr[l].shape
                    heads = attn_corr[l].reshape(seq_len, H, D_HEAD).clone()

                    keep_mask = torch.tensor(
                        [(l, h) in allowed for h in range(H)],
                        dtype=torch.bool,
                    )
                    disallowed = torch.logical_not(keep_mask)

                    if disallowed.any():
                        # ‑‑ broadcast layer‑wise running means over the sequence length
                        heads[:, disallowed, :] = (
                            means_gpu[l][disallowed]
                            .unsqueeze(0)                       # (1, #bad, D_HEAD)
                            .expand(seq_len, -1, -1)            # (seq, #bad, D_HEAD)
                            .to(heads.dtype)
                        )

                    # write the patched tensor back to the model for this trace
                    model.language_model.model.layers[l].self_attn.attn.output = \
                        heads.reshape(seq_len, d_model)

                logits_circ = model.logits.output.save()

        with model.trace(temperature=0.0, top_p=1.0) as tr:
            with tr.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": corrupt_img})):
                pred_circ_id = argmax_token(logits_circ[0])
                correct_circ += int(
                    normalise(tokenizer.decode([pred_circ_id])) == normalise(good_tok)
                )


        # ── 3) capture CLEAN attentions first ──
        with model.trace(temperature=0.0, top_p=1.0) as tr:
            with tr.invoke(TextPrompt(prompt=prompt,
                                    multi_modal_data={"image": clean_img})):

                attn_clean = [
                    model.language_model.model.layers[l]
                        .self_attn.attn.output.save()
                    for l in range(L)
                ]

        # ── 4) "Ablation‑on‑clean" run: open a **new** trace and ablate disallowed heads ──
        with model.trace(temperature=0.0, top_p=1.0) as tr:
            with tr.invoke(TextPrompt(prompt=prompt,
                                    multi_modal_data={"image": clean_img})):

                for l in range(L):
                    seq_len, d_model = attn_clean[l].shape
                    heads = attn_clean[l].reshape(seq_len, H, D_HEAD).clone()

                    keep_mask = torch.tensor(
                        [(l, h) in allowed for h in range(H)],
                        dtype=torch.bool,
                    )
                    ablate_mask = torch.logical_not(keep_mask)

                    if ablate_mask.any():
                        heads[:, ablate_mask, :] = (
                            means_gpu[l][ablate_mask]
                            .unsqueeze(0)
                            .expand(seq_len, -1, -1)
                            .to(heads.dtype)
                        )

                    model.language_model.model.layers[l].self_attn.attn.output = \
                        heads.reshape(seq_len, d_model)

                logits_ablate = model.logits.output.save()

        with model.trace(temperature=0.0, top_p=1.0) as tr:
            with tr.invoke(TextPrompt(prompt=prompt, multi_modal_data={"image": corrupt_img})):
                pred_ablate_id = argmax_token(logits_ablate[0])
                correct_ablate += int(
                    normalise(tokenizer.decode([pred_ablate_id])) == normalise(good_tok)
                )

        # ── write CSV line ──────────────────────────────────────
        with csv_path.open("a") as f:
            if csv_path.stat().st_size == 0:
                f.write("index,pred_full,pred_circuit,pred_ablate,agree_circ,agree_ablate\n")
            f.write(f"{idx},{tokenizer.decode([pred_full])},"
                    f"{tokenizer.decode([pred_circ_id])},"
                    f"{tokenizer.decode([pred_ablate_id])},"
                    f"{int(pred_full == pred_circ_id)},"
                    f"{int(pred_full == pred_ablate_id)}\n")

    # 4. summary ───────────────────────────────────────────────
    F_full   = correct_full   / total
    F_circ   = correct_circ   / total
    F_ablate = correct_ablate / total
    faith_circ   = float("nan") if F_full == 0 else F_circ   / F_full
    faith_ablate = float("nan") if F_full == 0 else F_ablate / F_full

    print("\n──────── Faithfulness Summary ────────")
    print(f"Dataset size (N)         : {total}")
    print(f"Full‑model accuracy F(M) : {F_full:.4f}")
    print(f"Circuit accuracy  F(Cir) : {F_circ:.4f}")
    print(f"Ablation accuracy F(Abl) : {F_ablate:.4f}")
    if F_full == 0:
        print("Faithfulness  F(Cir)/F(M): undefined (F(M)=0)")
        print("Faithfulness  F(Abl)/F(M): undefined (F(M)=0)")
    else:
        print(f"Faithfulness  F(Cir)/F(M): {faith_circ:.4f}")
        print(f"Faithfulness  F(Abl)/F(M): {faith_ablate:.4f}")
    print("───────────────────────────────────────")


if __name__ == "__main__":
    main()
