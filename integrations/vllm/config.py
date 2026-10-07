VLLM_TARGET_VERSION = "0.17.0"
QUICK_MODEL = "/mnt/moark-models/Qwen3-0.6B"
SHOWCASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def validate_vllm_version(version: str) -> None:
    if version != VLLM_TARGET_VERSION:
        raise RuntimeError(
            f"MXFlashAttn currently targets vLLM {VLLM_TARGET_VERSION}; found {version}. "
            "Confirm compatibility before selecting an attention backend."
        )
