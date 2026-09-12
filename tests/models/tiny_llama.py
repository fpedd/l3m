"""tiny_llama: a random 2-layer GQA Llama with an untied head, for the tests."""
source = "random"
arch = "llama"
dim, n_layers, n_heads, n_kv_heads, ffn, vocab, ctx = 64, 2, 4, 2, 128, 260, 64
tied, seed = False, 1
dtype = "f32"
head_dtype = "f32"
