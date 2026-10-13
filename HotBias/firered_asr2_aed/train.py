#!/usr/bin/env python3
"""Fixed-budget FireRedASR2-AED alignment of six input projections only.

The Qwen3 130-tensor shared core and the complete ASR stay frozen. Training
uses reference text solely for teacher forcing and mixed positive/ordinary
position sampling, never for candidate construction or rank eligibility.
Only raw-base Top50 + token-Trie gating is used; no CTC gate, prompt, or ELM.
No test manifest is accepted and no checkpoint is selected using test scores.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from hotbias.attention_fusion import (  # noqa: E402
    AttentionFusionConfig,
    FrozenFeatureAttentionFusion,
)
from hotbias.transfer import is_backbone_adapter_key  # noqa: E402
from hotbias.transfer_alignment import (  # noqa: E402
    alignment_weight_audit,
    set_trainable_scope,
    state_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path, help="Official FireRedASR2-AED weights directory")
    parser.add_argument("--firered-repo", type=Path, help="FireRedASR2S checkout; optional if installed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init", required=True, type=Path, help="Output of initialize_adapter.py")
    parser.add_argument("--max-samples", type=int, default=0, help="First N manifest rows before seeded shuffle; 0 = all.")
    parser.add_argument("--max-steps", type=int, default=0, help="Total optimizer-step budget, including resume; 0 = one epoch.")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-positions", type=int, default=16)
    parser.add_argument("--rank-gate", type=int, choices=[50], default=50)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--bias-l2-weight", type=float, default=1e-3)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50, help="Optimizer steps; checkpoints are resumable accumulation boundaries.")
    parser.add_argument("--log-every", type=int, default=8, help="Consumed manifest rows; per-row details always go to samples.jsonl.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--amp-bfloat16", action="store_true", help="Explicit opt-in; use the same precision for paired evaluation.")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    for key in ("max_samples", "max_steps"):
        if getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} cannot be negative")
    for key in ("max_positions", "max_candidates", "gradient_accumulation", "save_every", "log_every", "num_threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("learning_rate", "clip_grad_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.weight_decay, args.bias_l2_weight)):
        parser.error("Regularization weights must be finite and nonnegative")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def emit(path: Path, value: dict[str, Any], *, stdout: bool = False) -> None:
    line = json.dumps(value, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if stdout:
        print(line, flush=True)


def read_manifest(path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    audio_paths: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "audio_path", "text", "hotwords", "duration"} - row.keys()
            if missing:
                raise ValueError(f"Manifest line {line_no} missing {sorted(missing)}")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"Manifest line {line_no} has empty/nonstring text")
            if not isinstance(row["hotwords"], list) or any(not isinstance(word, str) for word in row["hotwords"]):
                raise ValueError(f"Manifest line {line_no}: hotwords must be a string list")
            duration = float(row["duration"])
            if not math.isfinite(duration) or not 0 < duration <= 60:
                raise ValueError(f"Manifest line {line_no}: require 0 < duration <= 60 s, got {duration}")
            audio = Path(row["audio_path"])
            if not audio.is_absolute():
                audio = path.parent / audio
            audio = audio.resolve()
            identifier = str(row["id"])
            if identifier in identifiers or str(audio) in audio_paths:
                raise ValueError(f"Duplicate id/audio in selected training manifest at line {line_no}")
            identifiers.add(identifier)
            audio_paths.add(str(audio))
            if not audio.is_file():
                raise FileNotFoundError(f"Manifest line {line_no}: {audio}")
            rows.append(dict(row, id=identifier, audio_path=str(audio), duration=duration, manifest_line=line_no))
            if max_samples and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError("Empty selected training manifest")
    return rows


def hotword_positions(target_ids: list[int], word_ids: list[list[int]]) -> set[int]:
    """Training-only contiguous token spans; this never modifies candidates."""
    positions: set[int] = set()
    for tokens in word_ids:
        if not tokens:
            continue
        width = len(tokens)
        for start in range(len(target_ids) - width + 1):
            if target_ids[start:start + width] == tokens:
                positions.update(range(start, start + width))
    return positions


def choose_positions(length: int, positives: set[int], maximum: int, rng: random.Random) -> list[int]:
    """Include ordinary/EOS positions, not just desired hotword corrections."""
    available = set(range(length))
    if not positives <= available:
        raise ValueError("Positive position outside target+EOS sequence")
    if length <= maximum:
        return list(range(length))
    positive_pool = sorted(positives)
    ordinary_pool = sorted(available - positives)
    positive_count = min(len(positive_pool), maximum // 2)
    ordinary_count = min(len(ordinary_pool), maximum - positive_count)
    selected = set(rng.sample(positive_pool, positive_count))
    selected.update(rng.sample(ordinary_pool, ordinary_count))
    # With unusually dense hotwords, use remaining positives to fill the budget.
    remaining = maximum - len(selected)
    if remaining:
        selected.update(rng.sample(sorted(available - selected), remaining))
    return sorted(selected)


def compute_loss(output: dict[str, torch.Tensor], sample: dict[str, torch.Tensor], l2_weight: float) -> tuple[torch.Tensor, dict[str, Any]]:
    """CE includes ordinary positions; L2 applies only to gate-legal biases."""
    labels = sample["label_ids"]
    gate = sample["candidate_gate_mask"].bool()
    bias = output["candidate_bias"].float()
    if torch.count_nonzero(bias.masked_select(~gate)).item():
        raise RuntimeError("Rank-rejected candidates received nonzero bias")
    active_rows = gate.any(-1)
    if not active_rows.any():
        raise ValueError("compute_loss requires at least one gate-eligible candidate")
    final_logits = output["final_logits"].float()
    base_logits = sample["base_logits"].float()
    if not torch.equal(final_logits[~active_rows], base_logits[~active_rows]):
        raise RuntimeError("All-rejected rows must remain exact base-logit no-ops")
    ce_per_position = F.cross_entropy(final_logits, labels, reduction="none")
    base_ce = F.cross_entropy(base_logits, labels)
    bias_penalty = bias.masked_select(gate).square().mean()
    loss = ce_per_position.mean() + l2_weight * bias_penalty
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite alignment objective")
    return loss, {
        "loss": float(loss.detach()),
        "base_ce": float(base_ce),
        "biased_ce": float(ce_per_position.detach().mean()),
        "bias_l2": float(bias_penalty.detach()),
        "active_mean_bias": float(bias.detach().masked_select(gate).mean()),
        "active_max_bias": float(bias.detach().masked_select(gate).max()),
    }


def new_progress() -> dict[str, Any]:
    return {
        "next_order_index": 0, "optimizer_step": 0, "rows_consumed": 0,
        "encoded_samples": 0, "trained_samples": 0,
        "encoded_seconds": 0.0, "trained_seconds": 0.0,
        "consumed_declared_seconds": 0.0,
        "skipped_empty_target": 0, "skipped_unknown_target": 0,
        "skipped_empty_candidates": 0, "skipped_all_rank_rejected": 0,
        "selected_positions": 0, "selected_hotword_positions": 0,
        "selected_ordinary_positions": 0, "rank_active_positions": 0,
        "candidate_target_positions": 0, "rank_eligible_target_positions": 0,
        "hotword_candidate_target_positions": 0, "hotword_rank_eligible_target_positions": 0,
        "candidate_entries": 0, "rank_eligible_candidate_entries": 0,
        "all_rejected_noop_positions": 0,
        "loss_sum": 0.0, "base_ce_sum": 0.0, "biased_ce_sum": 0.0,
        "bias_l2_sum": 0.0, "wall_seconds_prior": 0.0,
    }


def main() -> None:
    args = parse_args()
    # Import after argparse: --help needs neither native ASR dependencies nor weights.
    from firered_asr2_aed.features import build_training_sample, gated_fusion_output, load_bridge, prepare_candidates

    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    args.init = args.init.resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    if args.firered_repo is not None:
        args.firered_repo = args.firered_repo.expanduser().resolve()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Output is not empty; choose a new directory or explicitly --resume a checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.num_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu only for bounded smoke checks")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.amp_bfloat16 and device.type != "cuda":
        raise ValueError("--amp-bfloat16 is only supported on CUDA")
    rows = read_manifest(args.manifest, args.max_samples)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    settings = {
        "manifest_sha256": sha256(args.manifest), "initial_checkpoint_sha256": sha256(args.init),
        "model_directory": str(args.model_dir),
        "firered_repository": str(args.firered_repo) if args.firered_repo else "installed_package",
        "selected_manifest_rows": len(rows), "selected_manifest_seconds": sum(row["duration"] for row in rows),
        "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        "epochs": 1, "max_samples": args.max_samples, "max_steps": args.max_steps,
        "seed": args.seed, "max_positions": args.max_positions,
        "rank_gate": 50, "max_candidates": args.max_candidates,
        "trainable_scope": "backbone_adapters", "ctc_gate_enabled": False,
        "prompt_enabled": False, "external_lm_enabled": False,
        "gate": "reference-independent raw-base Top50 + token Trie; apply gate AFTER fusion, amax token aggregation",
        "candidate_policy": "fixed original list order; retain at most max_candidates mappable distinct sequences; no reference filtering",
        "position_sampling": "up to half reference hotword-span positions, fill ordinary/EOS then unused positions; training only",
        "coverage_scope": "sampled teacher-forced positions, not autoregressive recall and not full-transcript coverage",
        "objective": "mean full-vocabulary next-token CE over mixed positions + bias_l2_weight * mean squared gate-eligible candidate bias",
        "ordinary_noop_constraint": "ordinary/EOS CE suppresses wrong admitted candidates; rejected candidates have zero bias and zero bias regularization",
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "bias_l2_weight": args.bias_l2_weight, "gradient_accumulation": args.gradient_accumulation,
        "clip_grad_norm": args.clip_grad_norm,
        "precision": "bfloat16_autocast" if args.amp_bfloat16 else "float32_no_tf32",
        "fusion_training_mode": "train (original dropout retained); ASR eval and no_grad",
        "test_used_for_training_or_checkpoint_selection": False,
    }
    initial = torch.load(args.init, map_location="cpu", weights_only=True)
    if not initial.get("initialization_only") or initial.get("alignment_completed", False):
        raise ValueError("--init must be the audited initialization-only target checkpoint")
    fusion = FrozenFeatureAttentionFusion(AttentionFusionConfig(**initial["config"]))
    fusion.load_state_dict(initial["model_state"], strict=True)
    initial_hashes = state_hashes(fusion.state_dict())
    set_trainable_scope(fusion, "backbone_adapters")
    trainable = {name: p.numel() for name, p in fusion.named_parameters() if p.requires_grad}
    if len(trainable) != 18 or sum(trainable.values()) != 1969152:
        raise RuntimeError(f"Unexpected six-projection parameter scope: {trainable}")
    initial_audit = alignment_weight_audit(fusion, initial_hashes)
    if initial_audit["shared_core_tensors"] != 130:
        raise RuntimeError("Expected precisely 130 inherited shared-core tensors")
    core_parameters = sum(value.numel() for name, value in fusion.state_dict().items() if not is_backbone_adapter_key(name))
    if core_parameters != 10432001:
        raise RuntimeError(f"Unexpected shared core parameter count: {core_parameters}")
    fusion.to(device).train()
    parameters = [p for p in fusion.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    progress = new_progress()
    restored_rng: dict[str, Any] | None = None
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not previous.get("resume_ready") or previous.get("alignment_completed"):
            raise ValueError("Resume requires an incomplete checkpoint at a clean optimizer boundary")
        if previous.get("training_settings") != settings:
            raise ValueError("Resume training settings/manifest/init/budget differ; a fixed-budget run cannot silently change")
        if previous.get("initial_state_sha256") != initial_hashes:
            raise ValueError("Resume checkpoint initial state hashes differ")
        fusion.load_state_dict(previous["model_state"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state"])
        progress = previous["progress"]
        restored_rng = previous["rng_state"]
        if not alignment_weight_audit(fusion, initial_hashes)["shared_core_unchanged"]:
            raise RuntimeError("Resume changed a frozen source core tensor")
    atomic_json(args.output / "run_config.json", dict(settings, manifest=str(args.manifest), initial_checkpoint=str(args.init), device=str(device), trainable_parameters=trainable, shared_core_parameters=core_parameters, cli={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}))
    emit(args.output / "train.jsonl", {"event": "loading_frozen_asr", "device": str(device), "resume": bool(args.resume)}, stdout=True)
    bridge = load_bridge(str(args.model_dir), str(args.firered_repo) if args.firered_repo else None, str(device))
    if fusion.config.audio_dim != 1280 or fusion.config.text_dim != bridge.model.decoder.tgt_word_emb.embedding_dim:
        raise ValueError("FireRed model dimensions differ from the initialized adapter")
    if any(parameter.requires_grad for parameter in bridge.model.parameters()) or bridge.model.training:
        raise RuntimeError("ASR bridge must freeze every parameter and remain in eval mode")
    if restored_rng:
        random.setstate(restored_rng["python"])
        torch.set_rng_state(restored_rng["torch"])
        if device.type == "cuda" and restored_rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(restored_rng["cuda"])
    else:
        # ASR loading must not change alignment dropout's initial random stream.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    started = time.monotonic()

    def amp_context():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.amp_bfloat16 else contextlib.nullcontext()

    def summary(event: str) -> dict[str, Any]:
        denominator = max(progress["trained_samples"], 1)
        return {
            "event": event, **progress,
            "actual_seen_hours": progress["encoded_seconds"] / 3600,
            "actual_trained_hours": progress["trained_seconds"] / 3600,
            "loss_mean_per_trained_sample": progress["loss_sum"] / denominator,
            "base_ce_mean_per_trained_sample": progress["base_ce_sum"] / denominator,
            "biased_ce_mean_per_trained_sample": progress["biased_ce_sum"] / denominator,
            "sampled_target_gate_coverage": (progress["rank_eligible_target_positions"] / progress["candidate_target_positions"] if progress["candidate_target_positions"] else None),
            "sampled_hotword_target_gate_coverage": (progress["hotword_rank_eligible_target_positions"] / progress["hotword_candidate_target_positions"] if progress["hotword_candidate_target_positions"] else None),
            "wall_seconds": progress["wall_seconds_prior"] + time.monotonic() - started,
        }

    def save_checkpoint(name: str, *, completed: bool, reason: str, resume_ready: bool = True) -> dict[str, Any]:
        audit = alignment_weight_audit(fusion, initial_hashes)
        if not audit["shared_core_unchanged"] or not audit["frozen_parameters_unchanged"]:
            raise RuntimeError("Shared core or frozen weights changed")
        if any(p.grad is not None for p in bridge.model.parameters()):
            raise RuntimeError("ASR accumulated gradients")
        if any(p.grad is not None for p in fusion.parameters() if not p.requires_grad):
            raise RuntimeError("Shared core accumulated gradients")
        if completed and (not audit["all_six_projection_modules_changed"] or progress["optimizer_step"] == 0):
            raise RuntimeError("Cannot mark alignment completed: all six projections must have updated")
        state = summary("alignment_completed" if completed else reason)
        saved_progress = dict(progress, wall_seconds_prior=state["wall_seconds"])
        payload = {
            "config": asdict(fusion.config),
            "model_state": {key: value.detach().cpu() for key, value in fusion.state_dict().items()},
            "optimizer_state": optimizer.state_dict(), "progress": saved_progress,
            "optimizer_step": progress["optimizer_step"], "next_global_sample": progress["next_order_index"],
            "initialization_only": progress["optimizer_step"] == 0,
            "alignment_completed": completed, "alignment_stop_reason": reason,
            "resume_ready": resume_ready and pending == 0 and not completed,
            "trainable_scope": "backbone_adapters", "training_settings": settings,
            "weight_audit": audit, "initial_state_sha256": initial_hashes,
            "source_package_sha256": initial.get("source_package_sha256"),
            "target_model_revision": initial.get("target_model_revision"),
            "transfer_report": initial.get("transfer_report"), "target_name": initial.get("target_name"),
            "actual_seen_hours": state["actual_seen_hours"], "actual_trained_hours": state["actual_trained_hours"],
            "manifest": str(args.manifest), "status": state,
            "rng_state": {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
            "note": "Frozen Qwen3 core + six FireRed projections; raw-base rank Top50 and Trie only; no CTC gate/prompt/ELM; no held-out tuning.",
        }
        destination = args.output / name
        temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        atomic_json(args.output / "weight_audit.json", audit)
        atomic_json(args.output / "training_state.json", dict(state, alignment_completed=completed, checkpoint=str(destination), resume_ready=payload["resume_ready"], stop_reason=reason, asr_gradients_absent=True, shared_core_gradients_absent=True))
        emit(args.output / "train.jsonl", dict(state, checkpoint=str(destination), alignment_completed=completed), stdout=True)
        return audit

    def update_optimizer() -> None:
        nonlocal pending
        # Each backward used / accumulation. Correct a partial final bucket.
        if pending != args.gradient_accumulation:
            correction = args.gradient_accumulation / pending
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("Missing or nonfinite projection gradient")
        squared_by_module: dict[str, float] = {}
        for name, parameter in fusion.named_parameters():
            if parameter.requires_grad:
                module = name.split(".", 1)[0]
                squared_by_module[module] = squared_by_module.get(module, 0.0) + float(parameter.grad.detach().float().square().sum())
        # Report data-gradient activity separately from AdamW weight decay:
        # changed tensor hashes alone cannot prove a useful training signal.
        progress["last_projection_gradient_norms"] = {key: math.sqrt(value) for key, value in squared_by_module.items()}
        previous_max = progress.setdefault("max_projection_gradient_norms", {})
        for key, value in progress["last_projection_gradient_norms"].items():
            previous_max[key] = max(previous_max.get(key, 0.0), value)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        progress["optimizer_step"] += 1
        progress["last_grad_norm_before_clip"] = float(grad_norm)

    try:
        for position in range(progress["next_order_index"], len(order)):
            if args.max_steps and progress["optimizer_step"] >= args.max_steps:
                break
            row_index = order[position]
            row = rows[row_index]
            record: dict[str, Any] = {"order_index": position, "manifest_line": row["manifest_line"], "id": row["id"], "declared_seconds": row["duration"]}
            tokenized = bridge.tokenize(row["text"])
            target_ids = list(tokenized["ids"])
            skip = None
            if not target_ids:
                skip = "empty_target"
            elif tokenized["unknown_positions"]:
                skip = "unknown_target"
                record["unknown_positions"] = tokenized["unknown_positions"]
            # Reuse inference's exact mapping/forbidden-token/cap policy. The
            # reference is consulted only after candidate retention is fixed.
            candidates = prepare_candidates(bridge, row["hotwords"], args.max_candidates)
            word_ids = candidates.token_ids
            record["candidate_statistics"] = candidates.statistics
            if skip is None and not word_ids:
                skip = "empty_candidates"
            if skip is None:
                positive_steps = hotword_positions(target_ids, word_ids)
                step_indices = choose_positions(len(target_ids) + 1, positive_steps, args.max_positions, random.Random(args.seed + row_index))
                with torch.no_grad(), amp_context():
                    encoded = bridge.encode(row["audio_path"])
                    actual_seconds = float(encoded["duration_seconds"])
                    progress["encoded_samples"] += 1
                    progress["encoded_seconds"] += actual_seconds
                    sample = build_training_sample(bridge, encoded, target_ids, row["hotwords"], rank_gate=50, max_candidates=args.max_candidates, step_indices=step_indices)
                record.update(actual_seconds=actual_seconds, audio_duration_mismatch_seconds=actual_seconds - row["duration"], selected_step_indices=step_indices, sampled_hotword_steps=[step for step in step_indices if step in positive_steps])
                gate = sample["candidate_gate_mask"].bool()
                target_mask = sample["target_candidate_mask"].bool()
                if gate.shape != target_mask.shape or gate.shape[0] != len(step_indices):
                    raise RuntimeError("Feature gate/target/step shapes do not match")
                target_before_gate = target_mask.any(-1)
                target_after_gate = (target_mask & gate).any(-1)
                hotword_selected = torch.tensor([step in positive_steps for step in step_indices], dtype=torch.bool, device=gate.device)
                counters = {
                    "selected_positions": len(step_indices),
                    "selected_hotword_positions": int(hotword_selected.sum()),
                    "selected_ordinary_positions": int((~hotword_selected).sum()),
                    "rank_active_positions": int(gate.any(-1).sum()),
                    "candidate_target_positions": int(target_before_gate.sum()),
                    "rank_eligible_target_positions": int(target_after_gate.sum()),
                    "hotword_candidate_target_positions": int((target_before_gate & hotword_selected).sum()),
                    "hotword_rank_eligible_target_positions": int((target_after_gate & hotword_selected).sum()),
                    "candidate_entries": gate.numel(), "rank_eligible_candidate_entries": int(gate.sum()),
                    "all_rejected_noop_positions": int((~gate.any(-1)).sum()),
                }
                for key, value in counters.items():
                    progress[key] += value
                record.update(counters)
                if not gate.any():
                    skip = "all_rank_rejected"
                else:
                    with amp_context():
                        output = gated_fusion_output(fusion, sample)
                    loss, metrics = compute_loss(output, sample, args.bias_l2_weight)
                    (loss / args.gradient_accumulation).backward()
                    pending += 1
                    progress["trained_samples"] += 1
                    progress["trained_seconds"] += actual_seconds
                    for key in ("loss", "base_ce", "biased_ce", "bias_l2"):
                        progress[key + "_sum"] += metrics[key]
                    record.update(metrics)
                    record["status"] = "trained"
                    del loss, output
                del sample, encoded
            if skip is not None:
                progress["skipped_" + skip] += 1
                record.update(status="skipped", skip_reason=skip)
            progress["rows_consumed"] += 1
            progress["consumed_declared_seconds"] += row["duration"]
            progress["next_order_index"] = position + 1
            emit(args.output / "samples.jsonl", record)
            stepped = pending == args.gradient_accumulation
            if stepped:
                update_optimizer()
            if progress["rows_consumed"] % args.log_every == 0:
                emit(args.output / "train.jsonl", summary("progress"), stdout=True)
            if stepped and progress["optimizer_step"] % args.save_every == 0:
                save_checkpoint(f"checkpoint_step_{progress['optimizer_step']:06d}.pt", completed=False, reason="periodic")
        if pending:
            update_optimizer()
        stop_reason = "declared_optimizer_budget_reached" if args.max_steps and progress["optimizer_step"] >= args.max_steps else "one_selected_manifest_epoch_completed"
        save_checkpoint("checkpoint_final.pt", completed=True, reason=stop_reason)
    except BaseException as error:
        # Failed/interrupted runs are never labeled completed. Only an optimizer
        # boundary is resumable: partially executed backward must not be replayed.
        emit(args.output / "train.jsonl", {"event": "failed_or_interrupted", "type": type(error).__name__, "error": str(error), "pending_accumulation_samples": pending}, stdout=True)
        save_checkpoint("checkpoint_incomplete.pt", completed=False, reason=type(error).__name__, resume_ready=False)
        raise


if __name__ == "__main__":
    main()
