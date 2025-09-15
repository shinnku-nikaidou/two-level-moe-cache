from enum import Enum


class ModelType(Enum):
    """
    Supported model types for expert weight caching.

    Based on popular models from HuggingFace and other sources.
    """

    # GPT-OSS models
    GPT_OSS_20B = "gpt-oss-20b"
    GPT_OSS_120B = "gpt-oss-120b"

    # GLaM (Google Language Model)
    GLAM_64B = "glam-64b"
    GLAM_540B = "glam-540b"

    # PaLM-2 MoE variants
    PALM2_GECKO_MOE = "palm2-gecko-moe"
    PALM2_OTTER_MOE = "palm2-otter-moe"
    PALM2_BISON_MOE = "palm2-bison-moe"
    PALM2_UNICORN_MOE = "palm2-unicorn-moe"

    # Mixtral models
    MIXTRAL_8X7B = "mixtral-8x7b-instruct-v0.1"
    MIXTRAL_8X22B = "mixtral-8x22b-instruct-v0.1"

    # DeepSeek MoE
    DEEPSEEK_MOE_16B = "deepseek-moe-16b"
    DEEPSEEK_MOE_67B = "deepseek-moe-67b"

    # Qwen MoE
    QWEN_MOE_A2_7B = "qwen-moe-a2.7b"
    QWEN_MOE_A14_7B = "qwen-moe-a14.7b"

    CHATGLM3_6B_MOE = "chatglm3-6b-moe"

    PHI_TINY_MOE = "phi-tiny-moe"
