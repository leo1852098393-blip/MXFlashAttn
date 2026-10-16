"""Exercise the pinned vLLM adapter with a near-16K real prompt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=15000)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("MXFLASHATTN_GRAPH_OFFICIAL_SPLIT", "1")
    os.environ.setdefault("MXFA_NATIVE_DECODE", "0")
    os.environ.setdefault("MXFA_M1_DECODE", "0")
    os.environ.setdefault(
        "MXFLASHATTN_DISPATCH_LOG",
        str(args.output.with_suffix(".dispatch.jsonl").resolve()),
    )
    os.environ.setdefault(
        "MXFLASHATTN_PLUGIN_IMPORT_LOG",
        str(args.output.with_suffix(".plugin.log").resolve()),
    )

    from integrations.vllm import backend as dispatch_module

    registration = dispatch_module.register_vllm_v1_backend()
    if registration.get("registered") != "true":
        raise RuntimeError(registration)
    os.environ["MXFLASHATTN_AUTOREGISTER"] = "1"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed_ids = tokenizer.encode(
        "Long-context attention compatibility probe. ",
        add_special_tokens=False,
    )
    prompt_ids = (seed_ids * ((args.prompt_tokens // len(seed_ids)) + 1))[
        : args.prompt_tokens
    ]

    engine = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=16384,
        max_num_seqs=1,
        enforce_eager=True,
    )
    outputs = engine.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(temperature=0.0, max_tokens=32),
    )
    output = outputs[0]
    result = {
        "kind": "c500_long_prompt_fallback",
        "model": args.model,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(output.outputs[0].token_ids),
        "registration": registration,
        "dispatch_log": str(
            args.output.with_suffix(".dispatch.jsonl").resolve()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
