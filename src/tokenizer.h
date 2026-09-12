// The tokenizer section of a .l3m file: byte-table decoder and merge encoder.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "model.h"

typedef struct {
    int             n, bos, eos, max_len, sp_prefix, sp_words, digits;   // bos -1: none
    const uint8_t **bytes;
    const uint16_t *len;
    const float    *score;
    int            *sorted;   // ids by bytes
} l3m_tokenizer;

int  l3m_tokenizer_init(l3m_tokenizer *t, const void *section, size_t size, unsigned flags);   // points into the mmap; -1 if malformed
void l3m_tokenizer_free(l3m_tokenizer *t);
int  l3m_tokenizer_encode(const l3m_tokenizer *t, const char *text, int add_bos, int32_t *out, int max);  // returns count
// Not NUL-terminated; a SentencePiece prefix space is kept.
const uint8_t *l3m_tokenizer_decode(const l3m_tokenizer *t, int32_t tok, int *len);
