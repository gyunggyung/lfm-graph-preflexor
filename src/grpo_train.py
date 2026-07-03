#!/usr/bin/env python3
"""Graph-GRPO with no external API (H200 x8 multi-rank + vLLM server mode).

Forked from upstream src/run_grpo_graph.py. Differences:
  - Reward: 6-component, but correctness + graph_utility use embedding-based
    scoring (BGE cosine) instead of OpenAI/Grok judge. Paper Eq. (4) weights.
  - Judge queue: every N steps, sample M completions, write to judge/pending/.
    Claude reads these when invoked, writes judge/done/. The trainer drains
    done entries and blends them with embedding reward (calibration).
  - vLLM mode: 'server' by default (4 external replicas), not colocate, so
    train ranks (GPU 4-7) and rollout ranks (GPU 0-3) are fully decoupled.
  - Distributed: torchrun across TRAIN_GPUS.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Stub vllm if ABI mismatch breaks import — lets GRPOTrainer fall back to HF generate
import sys
import types
try:
    import vllm  # noqa: F401
except Exception:
    print("[grpo] vllm import failed, using stub (HF generate rollout)", flush=True)
    sys.modules['vllm'] = types.ModuleType('vllm')

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Make `from src.xxx import yyy` work when launched as a script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.combined_reward import CombinedReward, RewardWeights
from src.programmatic_rewards import (
    extract_graph_json_str,
    extract_post_think,
)


DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_targets(s: str):
    s = (s or "").strip()
    if not s or s.lower() == "default":
        return list(DEFAULT_LORA_TARGETS)
    if s.lower() == "all-linear":
        return "all-linear"
    return [x.strip() for x in s.split(",") if x.strip()]


def load_local_jsonl(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def apply_chat(tokenizer, prompt: str, enable_thinking) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except Exception:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt


def load_orpo_merged(base_model_dir: str, dtype, tokenizer_src: Optional[str] = None):
    """Load ORPO checkpoint. If it's a PEFT adapter, merge into base."""
    adapter_cfg = os.path.join(base_model_dir, "adapter_config.json")
    is_peft = os.path.exists(adapter_cfg)
    if not is_peft and not os.path.isdir(base_model_dir):
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(base_model_dir, "adapter_config.json")
            is_peft = True
        except Exception:
            pass

    if is_peft:
        if os.path.exists(adapter_cfg):
            with open(adapter_cfg) as f:
                cfg = json.load(f)
        else:
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(base_model_dir, "adapter_config.json")) as f:
                cfg = json.load(f)
        base = cfg["base_model_name_or_path"]
        print(f"[load] PEFT adapter detected. base={base} adapter={base_model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=dtype, device_map=None, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, base_model_dir)
        model = model.merge_and_unload()
        if hasattr(model, "peft_config"):
            delattr(model, "peft_config")
    else:
        print(f"[load] Full ORPO model: {base_model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir, dtype=dtype, device_map=None, trust_remote_code=True,
        )
    return model


class StepContext:
    """Mutable state passed through the reward fn."""
    def __init__(self):
        self.global_step = 0
        self.checkpoint_label = "step_0"
        self.model_name = "unknown"
        self.last_queue_step = -1


def make_reward_fn(rewarder: CombinedReward, ctx: StepContext, queue_every: int, queue_batch: int):
    """Build a reward_fn with signature (completions, question, gold_answer, **kwargs) -> List[float]."""

    def reward_fn(completions, prompts=None, question=None, gold_answer=None, **kwargs) -> List[float]:
        n = len(completions)
        q_list = question if question is not None else (prompts or [""] * n)
        ga_list = gold_answer if gold_answer is not None else [""] * n

        items = []
        for i, out in enumerate(completions):
            q = q_list[i] if i < len(q_list) else ""
            ga = ga_list[i] if i < len(ga_list) else ""
            items.append({
                "id": f"{ctx.checkpoint_label}|{i}",
                "question": q,
                "gold_answer": ga,
                "full_output": out,
            })

        breakdowns = rewarder.compute_batch(items)

        # Submit a sample to the judge queue every N steps
        if (rewarder.queue is not None
                and ctx.global_step >= 0
                and (ctx.global_step - ctx.last_queue_step) >= queue_every
                and queue_batch > 0):
            sample = items[:min(queue_batch, len(items))]
            sample_payload = []
            for it, br in zip(sample, breakdowns[:len(sample)]):
                full = it["full_output"]
                sample_payload.append({
                    "id": it["id"],
                    "question": it["question"],
                    "gold_answer": it["gold_answer"],
                    "candidate_answer": extract_post_think(full) or full[-1500:],
                    "graph_json": extract_graph_json_str(full) or "",
                    "embedding_scores": br.as_dict(),
                })
            try:
                rewarder.submit_pending(sample_payload, ctx.checkpoint_label, ctx.model_name)
                ctx.last_queue_step = ctx.global_step
            except Exception as e:
                print(f"[reward_fn] queue submit failed: {e}")

        # Drain any Claude judgments (non-blocking); they'll blend on the *next* call
        try:
            n_new = rewarder.refresh_claude_cache()
            if n_new:
                print(f"[reward_fn] ingested {n_new} Claude judgments")
        except Exception as e:
            print(f"[reward_fn] drain failed: {e}")

        return [b.total for b in breakdowns]

    return reward_fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_dir", required=True)
    p.add_argument("--tokenizer_model", default=None)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name_label", default="unknown")
    p.add_argument("--lora_target_modules", default="default")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--resume_grpo_checkpoint", default=None)
    p.add_argument("--add_new_special_tokens", action="store_true")

    # GRPO hparams (paper 8B Table 2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_prompt_length", type=int, default=1536)
    p.add_argument("--max_completion_length", type=int, default=3500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--scale_rewards", default="batch")
    p.add_argument("--loss_type", default="dapo")

    # Reward weights
    p.add_argument("--weight_correctness", type=float, default=0.30)
    p.add_argument("--weight_format", type=float, default=0.15)
    p.add_argument("--weight_graph_utility", type=float, default=0.25)
    p.add_argument("--weight_graph_networkx", type=float, default=0.10)
    p.add_argument("--weight_graph_diversity", type=float, default=0.10)
    p.add_argument("--weight_graph_structure", type=float, default=0.10)
    p.add_argument("--embed_model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--claude_blend_alpha", type=float, default=0.5)
    p.add_argument("--judge_queue_dir", default=None)
    p.add_argument("--judge_queue_every_steps", type=int, default=50)
    p.add_argument("--judge_queue_batch_size", type=int, default=16)

    # vLLM server mode
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--vllm_mode", default="server", choices=["server", "colocate"])
    p.add_argument("--vllm_server_host", default="127.0.0.1")
    p.add_argument("--vllm_server_port", type=int, default=8123)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.6)

    # chat template
    p.add_argument("--chat_template_enable_thinking", default="auto")

    # hub
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", default=None)
    p.add_argument("--hub_public", action="store_true")
    p.add_argument("--hf_token", default=None)
    args = p.parse_args()

    if args.hf_token:
        login(token=args.hf_token, add_to_git_credential=False)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if local_rank == 0:
        print(f"GRPO | world_size={world_size} | vllm_mode={args.vllm_mode}")

    enable_thinking_arg = args.chat_template_enable_thinking
    enable_thinking = None
    if enable_thinking_arg == "true":
        enable_thinking = True
    elif enable_thinking_arg == "false":
        enable_thinking = False
    # else None -> auto via tokenizer

    # Dataset
    ds = load_local_jsonl(args.dataset_path)
    required = {"prompt", "answer"}
    missing = required - set(ds.column_names)
    if missing:
        raise SystemExit(f"Dataset missing columns: {missing}")

    tokenizer_src = args.tokenizer_model or args.base_model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def map_fn(ex):
        q = (ex.get("prompt") or "").strip()
        ga = (ex.get("answer") or "").strip()
        if not q:
            return {"prompt": None, "gold_answer": None, "question": None}
        return {
            "prompt": apply_chat(tokenizer, q, enable_thinking),
            "question": q,
            "gold_answer": ga,
        }
    ds_mapped = ds.map(map_fn).filter(lambda x: x["prompt"] is not None)
    split = ds_mapped.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    if local_rank == 0:
        print(f"train={len(train_ds)} eval={len(eval_ds)}")

    # Model
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = load_orpo_merged(args.base_model_dir, dtype, tokenizer_src)
    model.config.use_cache = False

    if args.add_new_special_tokens:
        special = [
            "<think>", "</think>", "<brainstorm>", "</brainstorm>",
            "<graph>", "</graph>", "<graph_json>", "</graph_json>",
            "<patterns>", "</patterns>", "<synthesis>", "</synthesis>",
        ]
        n = tokenizer.add_special_tokens({"additional_special_tokens": special})
        if n > 0:
            model.resize_token_embeddings(len(tokenizer))

    if not args.no_lora:
        targets = parse_targets(args.lora_target_modules)
        peft_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
        )
        if args.resume_grpo_checkpoint:
            if local_rank == 0:
                print(f"Resuming GRPO LoRA from {args.resume_grpo_checkpoint}")
            model = get_peft_model(model, peft_cfg)
            model = PeftModel.from_pretrained(model, args.resume_grpo_checkpoint, is_trainable=True)
        else:
            model = get_peft_model(model, peft_cfg)
        if local_rank == 0:
            model.print_trainable_parameters()

    # Reward
    weights = RewardWeights(
        correctness=args.weight_correctness,
        format=args.weight_format,
        graph_utility=args.weight_graph_utility,
        graph_networkx=args.weight_graph_networkx,
        graph_diversity=args.weight_graph_diversity,
        graph_structure=args.weight_graph_structure,
    )
    rewarder = CombinedReward(
        weights=weights,
        embed_model=args.embed_model,
        judge_queue_dir=args.judge_queue_dir,
        claude_blend_alpha=args.claude_blend_alpha,
    )
    ctx = StepContext()
    ctx.model_name = args.model_name_label
    reward_fn = make_reward_fn(rewarder, ctx, args.judge_queue_every_steps, args.judge_queue_batch_size)

    # GRPO config
    cfg_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        remove_unused_columns=False,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_private_repo=(not args.hub_public) if args.push_to_hub else None,
        ddp_find_unused_parameters=False,
    )

    if args.use_vllm and args.vllm_mode == "server":
        cfg_kwargs["use_vllm"] = True
        cfg_kwargs["vllm_mode"] = "server"
        cfg_kwargs["vllm_server_host"] = args.vllm_server_host
        # TRL GRPOConfig expects a single port; if you run multiple replicas
        # behind a load-balancer, point this at the LB.
        cfg_kwargs["vllm_server_port"] = args.vllm_server_port
    elif args.use_vllm and args.vllm_mode == "colocate":
        cfg_kwargs["use_vllm"] = True
        cfg_kwargs["vllm_mode"] = "colocate"
        cfg_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization

    sig = set(inspect_signature(GRPOConfig))
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in sig}
    cfg = GRPOConfig(**cfg_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    # Hook step counter for queue cadence
    orig_step = trainer._maybe_log_save_evaluate if hasattr(trainer, "_maybe_log_save_evaluate") else None

    def patched(*a, **kw):
        ctx.global_step = int(trainer.state.global_step)
        ctx.checkpoint_label = f"step_{ctx.global_step}"
        # Refresh cache right before logging so wandb reflects calibrations
        try:
            rewarder.refresh_claude_cache()
        except Exception:
            pass
        if orig_step is not None:
            return orig_step(*a, **kw)

    if orig_step is not None:
        try:
            trainer._maybe_log_save_evaluate = patched
        except Exception:
            pass

    trainer.train()
    if local_rank == 0:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        if args.push_to_hub and args.hub_model_id:
            trainer.push_to_hub()
        print(f"Done. Saved to {args.output_dir}")


def inspect_signature(cls):
    try:
        import inspect
        return set(inspect.signature(cls.__init__).parameters)
    except Exception:
        return set()


if __name__ == "__main__":
    main()
