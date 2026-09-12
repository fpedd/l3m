"""tiny_gpt2: a random 2-layer GPT-2 with a tied head, for the tests."""
source = "random"
arch = "gpt2"
dim, n_layers, n_heads, n_kv_heads, ffn, vocab, ctx = 64, 2, 4, 4, 256, 260, 64
tied, seed = True, 2
dtype = "f32"
head_dtype = "f32"
