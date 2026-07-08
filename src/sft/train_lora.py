"""Train coherence SFT adapters with QLoRA rsLoRA.

The preferred path uses Unsloth. If Unsloth is unavailable, the script falls
back to the same 4-bit LoRA setup with Transformers + PEFT and keeps the
response-only label mask in a local collator.
"""
import argparse
import glob
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE = "/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models"
UNSLOTH_4BIT_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
FALLBACK_MODELS = (
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
)
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="ShareGPT/messages JSONL file")
    parser.add_argument("--out", required=True, help="Adapter output directory")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--base_model", default=UNSLOTH_4BIT_MODEL)
    parser.add_argument("--hf_cache", default=os.environ.get("HF_HOME", DEFAULT_HF_CACHE))
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--backend", choices=["auto", "unsloth", "peft"], default="auto")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--per_device_batch_size",
        "--per_device_train_batch_size",
        dest="per_device_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pack short sequences when the selected backend supports it.",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset_num_proc", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Only used if save_strategy is changed by the caller.",
    )
    return parser.parse_args()


def _cache_dirs(hf_cache):
    dirs = [Path(hf_cache)]
    nested = Path(hf_cache) / "huggingface"
    if nested.exists():
        dirs.append(nested)
    for item in os.environ.get("COHERENCE_EXTRA_CACHE", "").split(":"):
        if item:
            dirs.append(Path(item))
    return dirs


def _snapshot_for_repo(repo_id, hf_cache):
    cache_name = "models--" + repo_id.replace("/", "--")
    for cache_dir in _cache_dirs(hf_cache):
        pattern = cache_dir / cache_name / "snapshots" / "*"
        for snapshot in sorted(glob.glob(str(pattern)), reverse=True):
            snapshot = Path(snapshot)
            if (snapshot / "config.json").exists():
                return str(snapshot)
    return None


def resolve_model(base_model, hf_cache, allow_download):
    """Resolve an already-cached snapshot before letting HF try the network."""
    if Path(base_model).is_dir():
        return base_model, True

    cached = _snapshot_for_repo(base_model, hf_cache)
    if cached:
        return cached, True

    for repo_id in FALLBACK_MODELS:
        cached = _snapshot_for_repo(repo_id, hf_cache)
        if cached:
            print(f"[model] {base_model} not cached; using cached {repo_id}: {cached}")
            return cached, True

    if allow_download:
        print(f"[model] no cached snapshot found; allowing HF download for {base_model}")
        return base_model, False

    raise FileNotFoundError(
        f"No cached snapshot found for {base_model} or {FALLBACK_MODELS}. "
        "Pass --allow_download only on a networked machine."
    )


def configure_hf_cache(hf_cache, local_snapshot):
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "out" / "hf_datasets_cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "out" / "triton_cache"))
    os.environ.setdefault(
        "HF_XET_CACHE",
        os.path.join("/tmp", f"huggingface_{os.environ.get('USER', 'user')}", "xet"),
    )
    if local_snapshot:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_text_dataset(data_path, tokenizer, num_proc):
    from datasets import load_dataset

    data_path = str(Path(data_path))
    dataset = load_dataset("json", data_files=data_path, split="train")
    if "messages" not in dataset.column_names:
        raise ValueError(f"{data_path} must contain a 'messages' column")

    def format_batch(batch):
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in batch["messages"]
        ]
        return {"text": texts}

    return dataset.map(
        format_batch,
        batched=True,
        num_proc=num_proc,
        desc="Formatting llama-3.1 chat",
    )


def make_training_args(args, packing):
    out = str(Path(args.out))

    common = dict(
        output_dir=str(Path(out) / "trainer_state"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        do_train=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    try:
        import inspect

        from trl import SFTConfig

        sft_kwargs = dict(
            dataset_text_field="text",
            dataset_num_proc=args.dataset_num_proc,
        )
        signature = inspect.signature(SFTConfig)
        if "max_length" in signature.parameters:
            sft_kwargs["max_length"] = args.max_seq_length
        elif "max_seq_length" in signature.parameters:
            sft_kwargs["max_seq_length"] = args.max_seq_length
        if "packing" in signature.parameters:
            sft_kwargs["packing"] = packing

        return SFTConfig(**common, **sft_kwargs)
    except ImportError:
        from transformers import TrainingArguments

        return TrainingArguments(**common)


def _extract_final_loss(trainer):
    losses = [
        row.get("loss")
        for row in trainer.state.log_history
        if isinstance(row, dict) and row.get("loss") is not None
    ]
    return float(losses[-1]) if losses else None


def _write_metrics(out_dir, backend, model_name, trainer, final_loss):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "backend": backend,
        "model": model_name,
        "final_train_loss": final_loss,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "log_history": trainer.state.log_history,
    }
    metrics_path = out / "training_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] wrote {metrics_path}")


def _save_adapter_only(model, tokenizer, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    expected = out / "adapter_config.json"
    if not expected.exists():
        raise FileNotFoundError(f"adapter_config.json was not saved under {out}")
    print(f"[save] adapter-only checkpoint saved to {out}")


def train_with_unsloth(args, model_name):
    import unsloth  # noqa: F401 - must be imported before trl/transformers
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=TARGET_MODULES,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=True,
    )

    dataset = load_text_dataset(args.data, tokenizer, args.dataset_num_proc)
    training_args = make_training_args(args, packing=args.packing)
    trainer = _build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=None,
        training_args=training_args,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part=ASSISTANT_HEADER,
        tokenizer=tokenizer,
        num_proc=args.dataset_num_proc,
    )
    trainer.train()
    final_loss = _extract_final_loss(trainer)
    _save_adapter_only(model, tokenizer, args.out)
    _write_metrics(args.out, "unsloth", model_name, trainer, final_loss)
    return final_loss


class ResponseOnlyCollator:
    """Tokenize chat text and mask everything before the assistant response."""

    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.response_ids = tokenizer(
            ASSISTANT_HEADER,
            add_special_tokens=False,
        )["input_ids"]
        if not self.response_ids:
            raise ValueError("Could not tokenize assistant header for response masking")

    def _response_start(self, input_ids):
        n = len(self.response_ids)
        for idx in range(0, len(input_ids) - n + 1):
            if input_ids[idx : idx + n] == self.response_ids:
                return idx + n
        return None

    def __call__(self, examples):
        if "text" in examples[0]:
            batch = self.tokenizer(
                [example["text"] for example in examples],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
        else:
            import torch
            from torch.nn.utils.rnn import pad_sequence

            input_ids = [
                torch.tensor(example["input_ids"][: self.max_seq_length], dtype=torch.long)
                for example in examples
            ]
            attention_mask = [
                torch.ones_like(input_id, dtype=torch.long) for input_id in input_ids
            ]
            batch = {
                "input_ids": pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                ),
                "attention_mask": pad_sequence(
                    attention_mask,
                    batch_first=True,
                    padding_value=0,
                ),
            }
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for row_idx, input_ids in enumerate(batch["input_ids"].tolist()):
            start = self._response_start(input_ids)
            if start is None:
                labels[row_idx, :] = -100
            else:
                labels[row_idx, :start] = -100
        batch["labels"] = labels
        return batch


def _build_sft_trainer(
    model,
    tokenizer,
    dataset,
    data_collator,
    training_args,
    max_seq_length,
    packing,
):
    from trl import SFTTrainer

    try:
        return SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            data_collator=data_collator,
            args=training_args,
        )
    except (AttributeError, TypeError):
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            data_collator=data_collator,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            packing=packing,
        )


def train_with_peft(args, model_name):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_text_dataset(args.data, tokenizer, args.dataset_num_proc)
    if args.packing:
        print("[warn] disabling packing for PEFT fallback to preserve response-only masking.")
    training_args = make_training_args(args, packing=False)
    data_collator = ResponseOnlyCollator(tokenizer, args.max_seq_length)
    trainer = _build_sft_trainer(
        model,
        tokenizer,
        dataset,
        data_collator,
        training_args,
        args.max_seq_length,
        False,
    )
    trainer.train()
    final_loss = _extract_final_loss(trainer)
    _save_adapter_only(model, tokenizer, args.out)
    _write_metrics(args.out, "peft", model_name, trainer, final_loss)
    return final_loss


def main():
    args = parse_args()
    model_name, local_snapshot = resolve_model(
        args.base_model,
        args.hf_cache,
        args.allow_download,
    )
    configure_hf_cache(args.hf_cache, local_snapshot)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.backend in ("auto", "unsloth"):
        try:
            final_loss = train_with_unsloth(args, model_name)
            print(f"[done] backend=unsloth final_train_loss={final_loss}")
            return
        except ImportError as exc:
            if args.backend == "unsloth":
                raise
            print(f"[warn] Unsloth path unavailable ({exc}); falling back to PEFT.")
        except Exception:
            if args.backend == "unsloth":
                raise
            raise

    final_loss = train_with_peft(args, model_name)
    print(f"[done] backend=peft final_train_loss={final_loss}")


if __name__ == "__main__":
    main()
