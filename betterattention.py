#!/usr/bin/env python3
"""
BetterAttention — Hybrid Sliding-Window + Holographic Cognitive Map LM
=======================================================================

The "winning" architecture from the attention-mechanism research sessions:

  Per layer, attention is split across head groups:

    LOCAL heads (majority): exact sliding-window attention over the last
    `--window` tokens. O(W) per decode step, bounded KV cache.

    GLOBAL heads (minority): holographic cognitive map. Learned "lenses"
    (linear projections) map token CONTENT (pre-RoPE) to a low-dim
    content-addressable space. Routing is by MEANING, not position:
      - training: top-k token retrieval inside the block, with the selection
        scores fed back as attention-logit bias (straight-through).
      - decode: tokens are grouped into shelves of `--seg` bytes; each
        completed shelf is summarized by the mean of its lens-positions.
        A query scores shelves (not tokens), picks the top `--top-seg`
        shelves, then reads those tokens with exact attention ("skim the
        page"). The book (KV cache) is never compressed — only the lookup.

  Per-decode-step attention cost ~ O(window + top_seg*seg + n/seg*map_dim):
  constant in practice, sublinear in the shelf-scoring term. RoPE is applied
  only to the slices actually attended (window + selected shelves), never to
  the whole cache.

Usage
-----
  Train (coherence check every 1000 steps, checkpoint at the end):
      python betterattention.py --epochs 1

  Chat (automatic if a checkpoint exists; top-p/top-k/temperature/rep-penalty):
      python betterattention.py

  Force behaviour:
      python betterattention.py --train
      python betterattention.py --chat --ckpt ckpt/betterattention.pt

  Chat commands: /temp X  /topk X  /topp X  /rep X  /len X  /params  /new  /exit

Requires: the GPU-env python (torch+cu121) and input.txt in this folder.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_KEYS = ("n_embd", "n_layer", "n_head", "n_local_heads", "window",
             "map_dim", "top_k_tok", "seg", "top_seg", "block_size",
             "rope_scale")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class Config:
    def __init__(self, **kw):
        # model
        self.n_embd = kw.get("n_embd", 256)
        self.n_layer = kw.get("n_layer", 6)
        self.n_head = kw.get("n_head", 8)
        self.n_local_heads = kw.get("n_local_heads", 6)   # sliding-window heads
        self.head_dim = self.n_embd // self.n_head
        self.window = kw.get("window", 256)               # local attention window W
        self.map_dim = kw.get("map_dim", 32)              # holographic map dimension
        self.top_k_tok = kw.get("top_k_tok", 32)          # global top-k tokens (training)
        self.seg = kw.get("seg", 64)                      # segment ("shelf") size
        self.top_seg = kw.get("top_seg", 4)               # shelves skimmed per step
        self.block_size = kw.get("block_size", 1024)      # training context
        self.dropout = kw.get("dropout", 0.0)
        # data / training
        self.batch_size = kw.get("batch_size", 12)
        self.lr = kw.get("lr", 3e-4)
        self.min_lr = kw.get("min_lr", 3e-5)
        self.warmup = kw.get("warmup", 200)
        self.grad_clip = kw.get("grad_clip", 1.0)
        self.epochs = kw.get("epochs", 1)
        self.max_steps = kw.get("max_steps", 0)           # 0 = until epochs done
        self.sample_every = kw.get("sample_every", 1000)  # coherence check interval
        self.sample_len = kw.get("sample_len", 250)
        self.max_ctx = kw.get("max_ctx", 16384)           # chat/inference context cap
        self.rope_scale = kw.get("rope_scale", 1.0)       # NTK scale (0 = auto)
        self.seed = kw.get("seed", 1337)
        # io
        self.input_file = kw.get("input_file", "input.txt")
        self.ckpt = kw.get("ckpt", os.path.join("ckpt", "betterattention.pt"))

    def to_dict(self):
        return dict(self.__dict__)


# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------

def rope_cos_sin(max_len, head_dim, device, dtype, base=10000.0, scale=1.0):
    # NTK-aware scaling: stretch the base so position ctx*scale lands in the
    # angle range seen during training — fixes global-head extrapolation.
    if scale and scale > 1.0:
        base = base * (scale ** (head_dim / (head_dim - 2)))
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device,
                                       dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)                       # [L, hd/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x: [B,H,L,hd] even/odd pairs; cos/sin broadcastable to [.., L, hd/2]."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = o1, o2
    return out


# --------------------------------------------------------------------------
# Hybrid attention layer
# --------------------------------------------------------------------------

class HybridAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H, HL, hd, m = cfg.n_head, cfg.n_local_heads, cfg.head_dim, cfg.map_dim
        self.H, self.HL, self.HG, self.hd, self.m = H, HL, H - HL, hd, m
        assert self.HG > 0, "need at least one holographic (global) head"
        self.q = nn.Linear(cfg.n_embd, H * hd, bias=False)
        self.k = nn.Linear(cfg.n_embd, H * hd, bias=False)
        self.v = nn.Linear(cfg.n_embd, H * hd, bias=False)
        self.out = nn.Linear(H * hd, cfg.n_embd, bias=False)
        # holographic lenses — operate on CONTENT (pre-RoPE): content-addressable
        self.lens_q = nn.Linear(hd * self.HG, m, bias=False)
        self.lens_k = nn.Linear(hd * self.HG, m, bias=False)
        for lin in (self.lens_q, self.lens_k):
            nn.init.normal_(lin.weight, std=0.02)

    # ---------------- training forward ----------------
    def forward_train(self, x, causal, cos, sin):
        B, T, C = x.shape
        cfg = self.cfg
        q = self.q(x).view(B, T, self.H, self.hd).transpose(1, 2)   # [B,H,T,hd]
        k = self.k(x).view(B, T, self.H, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.H, self.hd).transpose(1, 2)

        q_content, k_content = q, k                  # pre-RoPE content for lenses
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ---- local heads: exact sliding-window attention ----
        t_idx = torch.arange(T, device=x.device)
        win = (t_idx[None, :] - t_idx[:, None]) < cfg.window
        mask = (causal & win)[None, None]                         # [1,1,T,T]
        out_l = F.scaled_dot_product_attention(
            q[:, :self.HL], k[:, :self.HL], v[:, :self.HL], attn_mask=mask)

        # ---- global heads: holographic top-k retrieval ----
        q_g, k_g, v_g = q[:, self.HL:], k[:, self.HL:], v[:, self.HL:]
        pq = self.lens_q(q_content[:, self.HL:].permute(0, 2, 1, 3)
                         .reshape(B, T, -1))                      # [B,T,m]
        pk = self.lens_k(k_content[:, self.HL:].permute(0, 2, 1, 3)
                         .reshape(B, T, -1))                      # [B,T,m]
        scores = torch.einsum("btm,bsm->bts", pq, pk) / math.sqrt(self.m)
        scores = scores.masked_fill(~causal, float("-inf"))
        K = min(cfg.top_k_tok, T)
        topv, topi = scores.topk(K, dim=-1)                       # [B,T,K]

        # selection is shared across global heads -> expand per head, then gather
        topi_h = topi[:, None].expand(-1, self.HG, -1, -1)        # [B,HG,T,K]
        flat_idx = topi_h.reshape(B * self.HG, T * K, 1).expand(-1, -1, self.hd)
        k_sel = k_g.reshape(B * self.HG, T, self.hd).gather(1, flat_idx) \
                   .view(B, self.HG, T, K, self.hd)
        v_sel = v_g.reshape(B * self.HG, T, self.hd).gather(1, flat_idx) \
                   .view(B, self.HG, T, K, self.hd)
        logits = torch.einsum("bhtd,bhtkd->bhtk", q_g, k_sel) / math.sqrt(self.hd)
        logits = logits + topv[:, None].detach()  # straight-through selection bias
        attn = F.softmax(logits, dim=-1)
        out_g = torch.einsum("bhtk,bhtkd->bhtd", attn, v_sel)

        out = torch.cat([out_l, out_g], dim=1).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)

    # ---------------- incremental decode step ----------------
    @staticmethod
    def init_cache(cap, device=None, dtype=None):
        # ring buffer: preallocated KV slots, written in place at index n.
        return {"k": None, "v": None, "pk": None, "n": 0, "cap": cap,
                "seg_means": None, "frozen": 0}

    def step(self, x_t, cache, cos1, sin1, pos):
        """x_t: [1,1,C]; pos: absolute position of this token. Returns (out, cache)."""
        cfg = self.cfg
        q = self.q(x_t).view(1, self.H, 1, self.hd)                 # [1,H,1,hd]
        k = self.k(x_t).view(1, self.H, 1, self.hd)
        v = self.v(x_t).view(1, self.H, 1, self.hd)
        hg_content = q[:, self.HL:]                                 # [1,HG,1,hd]
        pq = self.lens_q(hg_content.permute(0, 2, 1, 3).reshape(1, -1))  # [1,m]

        # append to cache — in-place ring-buffer write, no per-step copy
        if cache["k"] is None:
            dt = k.dtype
            cache["k"] = torch.zeros(1, self.H, cache["cap"], self.hd,
                                     device=k.device, dtype=dt)
            cache["v"] = torch.zeros(1, self.H, cache["cap"], self.hd,
                                     device=k.device, dtype=dt)
            cache["pk"] = torch.zeros(1, self.HG, cache["cap"], self.m,
                                      device=k.device, dtype=dt)
        n = cache["n"]
        cache["k"][:, :, n, :] = k[:, :, 0, :]
        cache["v"][:, :, n, :] = v[:, :, 0, :]
        cache["pk"][:, :, n, :] = pq       # shared lens position, broadcast over heads
        cache["n"] = n + 1
        n += 1
        kc, vc, pkc = cache["k"], cache["v"], cache["pk"]

        # RoPE for the query only; cache slices get RoPE lazily below
        q = apply_rope(q, cos1[pos].view(1, 1, 1, -1), sin1[pos].view(1, 1, 1, -1))

        # ---- local heads: exact attention over the last W tokens ----
        start = max(0, n - cfg.window)
        k_l = apply_rope(kc[:, :self.HL, start:n, :],
                         cos1[start:n].view(1, 1, n - start, -1),
                         sin1[start:n].view(1, 1, n - start, -1)).to(q.dtype)
        v_l = vc[:, :self.HL, start:n, :].to(q.dtype)
        out_l = F.scaled_dot_product_attention(q[:, :self.HL], k_l, v_l)

        # ---- global heads: shelf routing (the "skim") ----
        frozen = n // cfg.seg                       # completed shelves
        if cache["seg_means"] is None:
            cache["seg_means"] = torch.zeros(1, self.HG, 0, self.m,
                                             device=kc.device, dtype=kc.dtype)
        while cache["frozen"] < frozen:             # freeze newly completed shelves
            s = cache["frozen"]
            seg_mean = pkc[:, :, s * cfg.seg:(s + 1) * cfg.seg, :].mean(dim=2, keepdim=True)
            cache["seg_means"] = torch.cat([cache["seg_means"], seg_mean], dim=2)
            cache["frozen"] += 1
        seg_means = cache["seg_means"][0].to(pq.dtype)                  # [HG,S,m]
        if seg_means.size(1) > 0:
            seg_scores = torch.einsum("hm,hsm->hs", pq.expand(self.HG, -1),
                                      seg_means) / math.sqrt(self.m)
        else:
            seg_scores = pq.new_zeros(self.HG, 0)
        part = pkc[:, :, frozen * cfg.seg:n, :][0].to(pq.dtype)         # [HG,P,m]
        part_scores = torch.einsum("hm,hpm->hp", pq.expand(self.HG, -1), part) \
            / math.sqrt(self.m)
        all_scores = torch.cat([seg_scores, part_scores], dim=1)        # [HG,S+P]
        n_sel = min(cfg.top_seg, all_scores.size(1))
        top = all_scores.topk(n_sel, dim=1).indices                     # [HG,ns]

        # shelf index -> token positions (shelves are contiguous ranges)
        pos_lists = []
        for h in range(self.HG):
            toks = []
            for j in top[h].tolist():
                if j < frozen:
                    toks.extend(range(j * cfg.seg, (j + 1) * cfg.seg))
                else:
                    toks.extend(range(frozen * cfg.seg, n))
            pos_lists.append(toks)
        L = max(len(p) for p in pos_lists)
        tok_idx = torch.full((self.HG, L), n - 1, dtype=torch.long, device=kc.device)
        pad_mask = torch.zeros(self.HG, L, dtype=torch.bool, device=kc.device)
        for h, p in enumerate(pos_lists):
            if p:
                tok_idx[h, :len(p)] = torch.tensor(p, device=kc.device)
                pad_mask[h, :len(p)] = True

        # gather raw keys, then RoPE with each token's own absolute position
        g_idx = tok_idx[:, :, None].expand(1, -1, -1, self.hd)          # [1,HG,L,hd]
        k_sel = kc[:, self.HL:].gather(2, g_idx)
        k_sel = apply_rope(k_sel, cos1[tok_idx].unsqueeze(0),
                           sin1[tok_idx].unsqueeze(0)).to(q.dtype)
        v_sel = vc[:, self.HL:].gather(2, g_idx).to(q.dtype)
        logits = (q[:, self.HL:] * k_sel).sum(-1) / math.sqrt(self.hd)  # [1,HG,L]
        logits = logits.masked_fill(~pad_mask[None], float("-inf"))
        attn = F.softmax(logits, dim=-1)
        out_g = (attn.unsqueeze(-1) * v_sel).sum(2, keepdim=True)       # [1,HG,1,hd]

        out = torch.cat([out_l, out_g], dim=1).transpose(1, 2).reshape(1, 1, -1)
        return self.out(out), cache

    # ---------------- chunked prefill (fast context ingestion) ----------------
    @torch.no_grad()
    def prefill_chunk(self, x, cache, pos0, cos1, sin1):
        """Process a chunk x:[1,T,C] starting at absolute position pos0.
        Writes KV into the ring buffer once per chunk, then every query in the
        chunk attends to the full past (cache + chunk prefix) — exact, and
        O(1) buffer writes per chunk instead of per byte."""
        B, T, C = x.shape
        cfg = self.cfg
        q = self.q(x).view(1, T, self.H, self.hd).transpose(1, 2)   # [1,H,T,hd]
        k = self.k(x).view(1, T, self.H, self.hd).transpose(1, 2)
        v = self.v(x).view(1, T, self.H, self.hd).transpose(1, 2)
        pq = self.lens_q(q[:, self.HL:].permute(0, 2, 1, 3).reshape(T, -1))  # [T,m]

        if cache["k"] is None:
            dt = k.dtype
            cache["k"] = torch.zeros(1, self.H, cache["cap"], self.hd,
                                     device=k.device, dtype=dt)
            cache["v"] = torch.zeros(1, self.H, cache["cap"], self.hd,
                                     device=k.device, dtype=dt)
            cache["pk"] = torch.zeros(1, self.HG, cache["cap"], self.m,
                                      device=k.device, dtype=dt)
        n0 = cache["n"]
        n = n0 + T
        cache["k"][:, :, n0:n, :] = k
        cache["v"][:, :, n0:n, :] = v
        cache["pk"][:, :, n0:n, :] = pq             # broadcast over heads
        cache["n"] = n
        kc, vc, pkc = cache["k"], cache["v"], cache["pk"]

        q_rope = apply_rope(q, cos1[pos0:pos0 + T].view(1, 1, T, -1),
                            sin1[pos0:pos0 + T].view(1, 1, T, -1))

        # ---- local heads: attend over cache[0:n] (includes this chunk) ----
        k_all = apply_rope(kc[:, :self.HL, :n, :], cos1[:n].view(1, 1, n, -1),
                           sin1[:n].view(1, 1, n, -1)).to(q_rope.dtype)
        v_all = vc[:, :self.HL, :n, :].to(q_rope.dtype)
        lmask = torch.arange(n, device=x.device)[None, :] <= \
            torch.arange(n0, n, device=x.device)[:, None]           # [T,n]
        out_l = F.scaled_dot_product_attention(
            q_rope[:, :self.HL], k_all, v_all, attn_mask=lmask[None, None])

        # ---- global heads: shelf routing for all T queries ----
        frozen = n // cfg.seg
        if cache["seg_means"] is None:
            cache["seg_means"] = torch.zeros(1, self.HG, 0, self.m,
                                             device=x.device, dtype=k.dtype)
        while cache["frozen"] < frozen:
            s = cache["frozen"]
            seg_mean = pkc[:, :, s * cfg.seg:(s + 1) * cfg.seg, :].mean(dim=2, keepdim=True)
            cache["seg_means"] = torch.cat([cache["seg_means"], seg_mean], dim=2)
            cache["frozen"] += 1
        sm = cache["seg_means"][0, 0]                                   # [S,m] shared
        P = n - frozen * cfg.seg
        S = frozen
        js = torch.arange(S + P, device=x.device)
        starts = torch.where(js < S, js * cfg.seg, S * cfg.seg + (js - S))
        lens = torch.where(js < S, torch.full_like(js, cfg.seg),
                           torch.ones_like(js))
        last = starts + lens - 1                                        # row's last token
        Lm = int(lens.max().item())
        tbl = (starts[:, None] + torch.arange(Lm, device=x.device)[None, :]
               ).clamp(max=n - 1)                                       # [S+P,Lm]
        tvalid = torch.arange(Lm, device=x.device)[None, :] < lens[:, None]

        parts = [(pq @ sm.to(pq.dtype).T) / math.sqrt(self.m)]          # [T,S]
        if P > 0:
            part = pkc[0, 0, S * cfg.seg:n, :].to(pq.dtype)             # [P,m]
            parts.append((pq @ part.T) / math.sqrt(self.m))             # [T,P]
        all_scores = torch.cat(parts, dim=-1)
        qpos = torch.arange(n0, n, device=x.device)                     # [T]
        # a shelf is selectable iff at least one of its tokens is in the past
        allow = starts[None, :] <= qpos[:, None]                        # [T,S+P]
        all_scores = all_scores.masked_fill(~allow, float("-inf"))      # [T,S+P]
        ns = min(cfg.top_seg, S + P)
        top = all_scores.topk(ns, dim=-1).indices                       # [T,ns]

        sel_tbl = tbl[top]                                              # [T,ns,Lm]
        # per-token causal visibility inside each selected shelf
        sel_val = tvalid[top] & (sel_tbl <= qpos[:, None, None])        # [T,ns,Lm]
        K = ns * Lm
        tok_idx = sel_tbl.reshape(T, K)                                 # [T,K]
        val = sel_val.reshape(T, K)                                     # [T,K]

        tok_h = tok_idx[None].expand(self.HG, -1, -1)                   # [HG,T,K]
        g_idx = tok_h[:, :, :, None].expand(self.HG, T, K, self.hd)
        k_sel = kc[0, self.HL:].gather(1, g_idx.reshape(self.HG, T * K, self.hd)
                                       ).reshape(self.HG, T, K, self.hd)
        k_sel = apply_rope(k_sel, cos1[tok_h], sin1[tok_h])
        v_sel = vc[0, self.HL:].gather(1, g_idx.reshape(self.HG, T * K, self.hd)
                                       ).reshape(self.HG, T, K, self.hd)
        logits = (q_rope[0, self.HL:].unsqueeze(2) * k_sel).sum(-1) \
            / math.sqrt(self.hd)                                        # [HG,T,K]
        logits = logits.masked_fill(~val, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        out_g = (attn.unsqueeze(-1) * v_sel).sum(2)                     # [HG,T,hd]

        out = torch.cat([out_l, out_g.unsqueeze(0)], dim=1)             # [1,H,T,hd]
        out = out.transpose(1, 2).reshape(1, T, -1)
        return self.out(out), cache


# --------------------------------------------------------------------------
# Transformer block / LM
# --------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = HybridAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward_train(self, x, causal, cos, sin):
        x = x + self.attn.forward_train(self.ln1(x), causal, cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

    def step(self, x_t, cache, cos1, sin1, pos):
        a, cache = self.attn.step(self.ln1(x_t), cache, cos1, sin1, pos)
        x_t = x_t + a
        x_t = x_t + self.mlp(self.ln2(x_t))
        return x_t, cache

    def prefill_chunk(self, x, cache, pos0, cos1, sin1):
        a, cache = self.attn.prefill_chunk(self.ln1(x), cache, pos0, cos1, sin1)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, cache


class BetterAttentionLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(256, cfg.n_embd)          # byte-level LM
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, 256, bias=False)
        self.head.weight = self.emb.weight                # weight tying
        self.apply(self._init)
        for name, p in self.named_parameters():           # scaled residual init
            if name.endswith("attn.out.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward_train(self, idx):
        B, T = idx.shape
        cfg = self.cfg
        x = self.emb(idx)
        causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        cos, sin = rope_cos_sin(T, cfg.head_dim, idx.device, x.dtype,
                                scale=cfg.rope_scale)
        for blk in self.blocks:
            x = blk.forward_train(x, causal, cos, sin)
        x = self.ln_f(x)
        return self.head(x)

    def init_caches(self, cap, device=None, dtype=None):
        return [self.blocks[i].attn.init_cache(cap, device, dtype)
                for i in range(self.cfg.n_layer)]

    def step(self, tok, caches, pos, cos1, sin1):
        x = self.emb(tok).view(1, 1, -1)
        for i, blk in enumerate(self.blocks):
            x, caches[i] = blk.step(x, caches[i], cos1, sin1, pos)
        return self.head(self.ln_f(x))[:, -1, :], caches

    @torch.no_grad()
    def prefill_chunk(self, toks, caches, pos0, cos1, sin1, chunk=512):
        """toks: 1D LongTensor on device. Feed bytes in chunks; each query sees
        the full past exactly. Returns the last byte's logits (or None if empty)."""
        logits = None
        for i in range(0, len(toks), chunk):
            c = toks[i:i + chunk]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = self.emb(c).unsqueeze(0)
                for li, blk in enumerate(self.blocks):
                    x, _ = blk.prefill_chunk(x, caches[li], pos0 + int(i),
                                             cos1, sin1)
                x = self.head(self.ln_f(x))
            logits = x[:, -1, :].float()
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# Sampling (temperature / top-k / top-p / repetition penalty)
# --------------------------------------------------------------------------

def sample_token(logits, temperature=0.8, top_k=50, top_p=0.9,
                 rep_penalty=1.1, rep_window=192, rep_tokens=None):
    logits = logits[0].float()
    if rep_penalty != 1.0 and rep_tokens:
        for t in list(rep_tokens)[-rep_window:]:
            t = int(t)
            if logits[t] > 0:
                logits[t] /= rep_penalty
            else:
                logits[t] *= rep_penalty
    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        vk, ik = logits.topk(min(top_k, logits.size(-1)))
        probs = torch.zeros_like(logits).scatter_(0, ik, F.softmax(vk, dim=-1))
    else:
        probs = F.softmax(logits, dim=-1)
    if top_p and top_p < 1.0:
        sp, si = probs.sort(descending=True)
        cum = sp.cumsum(-1)
        cut = cum > top_p
        cut[1:] = cut[:-1].clone()
        cut[0] = False
        probs[si[cut]] = 0.0
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1).item()


# --------------------------------------------------------------------------
# Data — byte-level, memmap (514 MB corpus; int64 offsets: n > 2**24)
# --------------------------------------------------------------------------

class ByteData:
    def __init__(self, path):
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.n = self.mm.shape[0]

    def batch(self, block_size, bs, device, rng):
        starts = rng.integers(0, self.n - block_size - 1, size=bs, dtype=np.int64)
        x = np.stack([self.mm[s:s + block_size].astype(np.int64) for s in starts])
        y = np.stack([self.mm[s + 1:s + block_size + 1].astype(np.int64) for s in starts])
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


# --------------------------------------------------------------------------
# Coherence check — fixed seed prompt so checks are comparable across steps
# --------------------------------------------------------------------------

@torch.no_grad()
def coherence_sample(model, cfg, device, prompt=b"\nThe ", length=None, temperature=0.8):
    was_training = model.training
    model.eval()
    length = length or cfg.sample_len
    total = len(prompt) + length + 8
    caches = model.init_caches(total, device, torch.bfloat16)
    cos, sin = rope_cos_sin(total, cfg.head_dim, device, torch.bfloat16,
                            scale=getattr(cfg, "rope_scale", 1.0))
    toks = torch.tensor(list(prompt), dtype=torch.long, device=device)
    logits = model.prefill_chunk(toks, caches, 0, cos, sin) if len(toks) else None
    out = []
    for i in range(length):
        nt = sample_token(logits, temperature=temperature, top_k=50, top_p=0.9,
                          rep_penalty=1.1)
        out.append(nt)
        logits, caches = model.step(torch.tensor([nt], device=device), caches,
                                    len(toks) + i, cos, sin)
    if was_training:
        model.train()
    return (prompt + bytes(out)).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def save_ckpt(model, cfg, step, path, final=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                "step": step, "final": final}, path)


def train(model, cfg, data, device):
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    steps_per_epoch = math.ceil((data.n - 1) / (cfg.batch_size * cfg.block_size))
    total = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    print(f"[train] corpus bytes={data.n:,}  steps/epoch={steps_per_epoch:,}  "
          f"total steps={total:,}", flush=True)

    def lr_at(step):
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        p = (step - cfg.warmup) / max(1, total - cfg.warmup)
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * p))

    model.train()
    t0 = time.time()
    loss_ema = None
    for step in range(total):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = data.batch(cfg.block_size, cfg.batch_size, device, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.forward_train(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        l = loss.item()
        loss_ema = l if loss_ema is None else 0.9 * loss_ema + 0.1 * l
        if step % 50 == 0:
            print(f"step {step:>7}/{total}  loss {loss_ema:.4f}  lr {lr:.2e}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        # ---- coherence check every `sample_every` steps ----
        if (step + 1) % cfg.sample_every == 0:
            txt = coherence_sample(model, cfg, device)
            print(f"\n--- coherence check @ step {step + 1} ---\n{txt}\n"
                  f"--- end sample ---\n", flush=True)
        # rolling safety checkpoint
        if (step + 1) % 5000 == 0:
            save_ckpt(model, cfg, step, cfg.ckpt + ".last")

    save_ckpt(model, cfg, total, cfg.ckpt, final=True)
    print(f"[train] done. checkpoint saved to {cfg.ckpt}")


# --------------------------------------------------------------------------
# Chat mode
# --------------------------------------------------------------------------

@torch.no_grad()
def chat(model, cfg, device):
    model.eval()
    params = {"temperature": 0.8, "top_k": 50, "top_p": 0.9,
              "rep": 1.1, "gen_len": 400}
    max_ctx = cfg.max_ctx
    caches = model.init_caches(max_ctx + 64, device, torch.bfloat16)
    pos = 0
    cos, sin = rope_cos_sin(max_ctx + 64, cfg.head_dim, device, torch.bfloat16,
                            scale=cfg.rope_scale)
    print("\n[chat] BetterAttention ready. Commands: /temp X /topk X /topp X "
          "/rep X /len X /params /new /exit")
    print("[chat] (empty input = keep generating)\n")

    def prefill(text):
        """Append text (str or bytes) to the context via chunked prefill."""
        nonlocal pos
        data = text.encode("utf-8") if isinstance(text, str) else text
        if not data:
            return None
        toks = torch.tensor(list(data), dtype=torch.long, device=device)
        logits = model.prefill_chunk(toks, caches, pos, cos, sin)
        pos += len(toks)
        return logits

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.startswith("/"):
            parts = user.split()
            cmd = parts[0].lower()
            if cmd in ("/exit", "/quit"):
                break
            try:
                if cmd == "/temp":
                    params["temperature"] = float(parts[1])
                elif cmd == "/topk":
                    params["top_k"] = int(parts[1])
                elif cmd == "/topp":
                    params["top_p"] = float(parts[1])
                elif cmd == "/rep":
                    params["rep"] = float(parts[1])
                elif cmd == "/len":
                    params["gen_len"] = int(parts[1])
                elif cmd == "/params":
                    print("   " + "  ".join(f"{k}={v}" for k, v in params.items())
                          + f"  ctx_bytes={pos}")
                    continue
                elif cmd == "/new":
                    caches = model.init_caches(max_ctx + 64, device, torch.bfloat16)
                    pos = 0
                    print("   [context cleared]")
                    continue
                else:
                    print("   unknown command")
                    continue
            except (IndexError, ValueError):
                print("   bad command value")
                continue
            print("   ok")
            continue

        prompt = (user + "\n").encode("utf-8")
        if pos + len(prompt) + params["gen_len"] + 8 > max_ctx:
            print("   [context limit hit — clearing]")
            caches = model.init_caches(max_ctx + 64, device, torch.bfloat16)
            pos = 0
        logits = prefill(prompt)

        gen = []
        rep_tokens = list(range(max(0, pos - 192), pos))
        t0 = time.time()
        for i in range(params["gen_len"]):
            nt = sample_token(logits, temperature=params["temperature"],
                              top_k=params["top_k"], top_p=params["top_p"],
                              rep_penalty=params["rep"], rep_tokens=rep_tokens)
            gen.append(nt)
            rep_tokens.append(nt)
            logits, caches = model.step(torch.tensor([nt], device=device),
                                        caches, pos, cos, sin)
            pos += 1
            if nt == 0x0A and len(gen) >= 40:
                break
        dt = time.time() - t0
        print(f"model> {bytes(gen).decode('utf-8', errors='replace')}")
        print(f"       [{len(gen)} bytes, {dt:.1f}s, "
              f"{len(gen) / max(dt, 1e-9):.0f} b/s, ctx={pos}]\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="BetterAttention hybrid LM")
    ap.add_argument("--input", default="input.txt")
    ap.add_argument("--ckpt", default=os.path.join("ckpt", "betterattention.pt"))
    ap.add_argument("--train", action="store_true", help="force training")
    ap.add_argument("--chat", action="store_true", help="force chat mode")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--max-ctx", type=int, default=16384,
                    help="chat/inference context cap in bytes (e.g. 131072 for 128K)")
    ap.add_argument("--rope-scale", type=float, default=1.0,
                    help="NTK RoPE scale for long-context generalization "
                         "(0 = auto: max_ctx / block_size; 1 = off)")
    # model knobs
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-local-heads", type=int, default=6)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--map-dim", type=int, default=32)
    ap.add_argument("--top-k-tok", type=int, default=32)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--top-seg", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=1024)
    # training knobs
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = torch.device("cuda:0")          # GPU 0 only — GPU discipline
    assert torch.cuda.is_available(), "CUDA required"

    cfg = Config(
        epochs=args.epochs, max_steps=args.max_steps,
        sample_every=args.sample_every, block_size=args.block_size,
        batch_size=args.batch_size, lr=args.lr, max_ctx=args.max_ctx,
        rope_scale=args.rope_scale,
        n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
        n_local_heads=args.n_local_heads, window=args.window,
        map_dim=args.map_dim, top_k_tok=args.top_k_tok, seg=args.seg,
        top_seg=args.top_seg,
    )
    cfg.input_file = args.input
    cfg.ckpt = args.ckpt
    if cfg.rope_scale <= 0:     # auto: stretch trained range across target ctx
        cfg.rope_scale = max(1.0, cfg.max_ctx / cfg.block_size)

    # checkpoint found -> chat mode with the SAVED architecture
    if os.path.exists(cfg.ckpt) and not args.train:
        print(f"[chat] checkpoint found: {cfg.ckpt}")
        sd = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
        for k in ARCH_KEYS:
            if k in sd.get("cfg", {}):
                setattr(cfg, k, sd["cfg"][k])
        cfg.head_dim = cfg.n_embd // cfg.n_head
        model = BetterAttentionLM(cfg).to(device)
        model.load_state_dict(sd["model"])
        print(f"[model] {model.num_params() / 1e6:.2f}M params "
              f"({cfg.n_local_heads} local + {cfg.n_head - cfg.n_local_heads} "
              f"holographic heads x {cfg.n_layer} layers)")
        chat(model, cfg, device)
        return

    if not os.path.exists(cfg.input_file):
        sys.exit(f"input file not found: {cfg.input_file}")
    if args.chat:
        sys.exit("no checkpoint found — train first: python betterattention.py")

    torch.manual_seed(cfg.seed)
    model = BetterAttentionLM(cfg).to(device)
    print(f"[model] {model.num_params() / 1e6:.2f}M params  "
          f"({cfg.n_local_heads} local + {cfg.n_head - cfg.n_local_heads} "
          f"holographic heads x {cfg.n_layer} layers, window={cfg.window}, "
          f"seg={cfg.seg}, map_dim={cfg.map_dim})")
    data = ByteData(cfg.input_file)
    train(model, cfg, data, device)

    # checkpoint now exists -> enter chat mode
    sd = torch.load(cfg.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    chat(model, cfg, device)


if __name__ == "__main__":
    main()
