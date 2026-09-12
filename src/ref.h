// The reference forward pass: one thread, fp32, no kernel table.
#pragma once
#include <stdint.h>
#include "file.h"

typedef struct l3m_ref l3m_ref;

l3m_ref     *l3m_ref_load(const l3m_file *f, int ctx);              // ctx = KV cache length
const float *l3m_ref_forward(l3m_ref *r, int32_t tok, int pos);     // logits[vocab] for this position
void         l3m_ref_free(l3m_ref *r);
