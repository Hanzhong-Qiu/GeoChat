from .language_model.geochat_llama import GeoChatLlamaForCausalLM, GeoChatConfig

# MPT support depends on transformers internals (_expand_mask) removed in
# transformers >= 4.32. Skip MPT import if it fails — GeoChat-7B only needs LLaMA.
try:
    from .language_model.geochat_mpt import GeoChatMPTForCausalLM, GeoChatMPTConfig
except ImportError:
    pass
