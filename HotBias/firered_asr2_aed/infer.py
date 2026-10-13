"""Greedy FireRedASR2-AED decoding with an aligned Top-50 HotBias adapter.

Uses raw-logit Top-50 and prefix matching, without CTC gating or a hotword prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from firered_asr2_aed.features import (
    build_online_sample, gated_fusion_output, load_bridge, prepare_candidates,
)


@torch.no_grad()
def decode(bridge, encoded, hotwords=None, fusion=None, rank_gate=50, max_candidates=40, max_tokens=256):
    if max_tokens < 1 or rank_gate < 1:
        raise ValueError("max_tokens and rank_gate must be positive")
    decoder = bridge.model.decoder
    encoder = encoded["encoder_outputs"]
    encoder_mask = encoded["encoder_mask"]
    if encoder.shape[0] != 1:
        raise ValueError("This greedy pilot accepts one utterance per decode call")
    bridge.model.eval()
    candidates = prepare_candidates(bridge, hotwords, max_candidates) if fusion is not None else None
    use_bias = candidates is not None and bool(candidates.token_ids)
    statistics = {
        "rank_gate": int(rank_gate), "candidate_cap": int(max_candidates),
        "ctc_gate_enabled": False, "beam_size": 1, "softmax_smoothing": 1.25,
        "eos_penalty": 1.0, "external_lm_enabled": False,
        "candidate_rows": 0, "allowed_rows": 0, "rejected_rows": 0,
        "zero_eligible_steps": 0, "applied_candidate_rows": 0, "biased_steps": 0,
        "input_hotwords": 0, "kept_hotwords": 0, "invalid_hotwords": 0,
        "duplicate_hotwords": 0, "cap_dropped_hotwords": 0,
    }
    if candidates is not None:
        statistics.update(candidates.statistics)
    was_training = bool(fusion.training) if fusion is not None else False
    if fusion is not None:
        fusion.eval()
    generated = []
    ended = False
    steps = 0
    ys = torch.tensor([[int(decoder.sos_id)]], dtype=torch.long, device=encoder.device)
    caches = [None] * int(decoder.n_layers)
    try:
        for step in range(max_tokens):
            target_mask = decoder.ignored_target_position_is_0(ys, decoder.pad_id)
            hidden = decoder.dropout(decoder.tgt_word_emb(ys) * decoder.scale + decoder.positional_encoding(ys))
            for index, layer in enumerate(decoder.layer_stack):
                hidden = layer(hidden, encoder, target_mask, encoder_mask, cache=caches[index])
                caches[index] = hidden
            hidden = decoder.layer_norm_out(hidden)
            raw_logits = decoder.tgt_word_prj(hidden[:, -1])
            adjusted = raw_logits
            if use_bias:
                sample = build_online_sample(
                    bridge, encoded, generated, hidden[0, -1], raw_logits[0], candidates, rank_gate,
                )
                output = gated_fusion_output(fusion, sample)
                adjusted = output["final_logits"]
                gate = sample["candidate_gate_mask"]
                statistics["candidate_rows"] += int(gate.numel())
                statistics["allowed_rows"] += int(gate.sum())
                statistics["rejected_rows"] += int((~gate).sum())
                statistics["zero_eligible_steps"] += int(not bool(gate.any()))
                statistics["applied_candidate_rows"] += int(output["candidate_bias"].gt(0).sum())
                statistics["biased_steps"] += int(bool(output["delta_logits"].gt(0).any()))
            # Match official greedy beam=1: temperature before log_softmax,
            # topk(k=1), EOS penalty=1, no external LM or post-hoc replacement.
            scores = F.log_softmax(adjusted / 1.25, dim=-1)
            token = int(torch.topk(scores, k=1, dim=1).indices[0, 0])
            steps = step + 1
            if token == int(decoder.eos_id):
                ended = True
                break
            generated.append(token)
            ys = torch.cat([ys, torch.tensor([[token]], dtype=torch.long, device=ys.device)], dim=1)
    finally:
        if fusion is not None:
            fusion.train(was_training)
    text = bridge.tokenizer.detokenize(generated)
    text = re.sub(r"(<blank>)|(<sil>)", "", text).lower()
    return {"text": text, "token_ids": generated, "ended_with_eos": ended,
            "steps": steps, "truncated": not ended and steps >= max_tokens,
            "gate_statistics": statistics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="Official FireRedASR2-AED weights directory")
    parser.add_argument("--firered-repo", type=Path, help="FireRedASR2S checkout; optional if installed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Completed FireRed adapter checkpoint")
    parser.add_argument("--hotword", action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rank-gate", type=int, choices=[50], default=50)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--verify-native-parity", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.max_candidates < 1:
        parser.error("--max-tokens and --max-candidates must be positive")
    bridge = load_bridge(args.model_dir, args.firered_repo, args.device)
    encoded = bridge.encode(args.audio)
    baseline = decode(bridge, encoded, max_tokens=args.max_tokens)
    result = {"baseline": baseline}
    if args.verify_native_parity:
        with torch.no_grad():
            official = bridge.model.decoder.batch_beam_search(
                encoded["encoder_outputs"], encoded["encoder_mask"],
                beam_size=1, nbest=1, decode_max_len=args.max_tokens,
                softmax_smoothing=1.25, length_penalty=0.0, eos_penalty=1.0,
                elm=None, elm_weight=0.0,
            )
        official_ids = [int(value) for value in official[0][0]["yseq"].tolist()]
        assert baseline["token_ids"] == official_ids, "native greedy token mismatch"
        result["native_greedy_parity"] = True
    from hotbias import AttentionFusionConfig, FrozenFeatureAttentionFusion
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("initialization_only", False) or not payload.get("alignment_completed", False):
        raise ValueError("--checkpoint must be a completed FireRed adapter, not transfer initialization")
    fusion = FrozenFeatureAttentionFusion(AttentionFusionConfig(**payload["config"]))
    fusion.load_state_dict(payload["model_state"], strict=True)
    if fusion.config.audio_dim != 1280 or fusion.config.text_dim != bridge.model.decoder.tgt_word_emb.embedding_dim:
        raise ValueError("FireRed adapter and ASR dimensions do not match")
    fusion.to(device=bridge.device, dtype=torch.float32).eval()
    empty = decode(bridge, encoded, hotwords=[], fusion=fusion, max_tokens=args.max_tokens)
    assert empty["token_ids"] == baseline["token_ids"], "empty-hotword path is not a no-op"
    result["empty_hotword_parity"] = True
    result["biased"] = decode(
        bridge, encoded, hotwords=args.hotword, fusion=fusion, rank_gate=args.rank_gate,
        max_candidates=args.max_candidates, max_tokens=args.max_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
