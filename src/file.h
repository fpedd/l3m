// A mapped .l3m file: header, table of contents, tokenizer.
#pragma once
#include <stddef.h>
#include "model.h"
#include "tokenizer.h"

typedef struct {
    void             *map;
    size_t            size;
    const l3m_header *h;
    const l3m_tensor *toc;
    l3m_tokenizer     tok;
} l3m_file;

int               l3m_file_open(l3m_file *f, const char *path);           // 0 on success, message on stderr otherwise
void              l3m_file_close(l3m_file *f);
const l3m_tensor *l3m_file_tensor(const l3m_file *f, const char *name);    // NULL if absent
const l3m_tensor *l3m_file_blk(const l3m_file *f, int layer, const char *name);   // "blk.<layer>.<name>"
const void       *l3m_file_data(const l3m_file *f, const l3m_tensor *t);
const float      *l3m_file_f32(const l3m_file *f, const l3m_tensor *t);    // NULL if t is; open checked the dtype
