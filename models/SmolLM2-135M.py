"""SmolLM2-135M: HuggingFace's 30-layer Llama."""
source = "hf_llama"
repo = "HuggingFaceTB/SmolLM2-135M"
weights = None
dtype = "q4_0"
head_dtype = "q8_0"
finer = ("wq", "wk", "wv", "blk.0.", "blk.1.", "blk.12.", "blk.19.", "blk.28.", "blk.29.")   # +44% perplexity -> +4.5%
