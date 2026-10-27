import os
import sys
import glob
import json
import math
import time
import random
import argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# BetterAttention hybrid backbone (sliding-window + holographic cognitive map)
from betterattention import Block, Config as BAConfig, rope_cos_sin, sample_token

# ==========================================
# 1. Configuration & CLI Arguments
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Transformer with Active Space-Folding Bottleneck + BetterAttention hybrid backbone (tiktoken gpt2)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--predict_every", type=int, default=2000, help="Predict every N steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (tokens)")
    parser.add_argument("--d_model", type=int, default=1024, help="Embedding dimension (winner: 1024)")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_local_heads", type=int, default=0, help="Sliding-window heads (0 = auto: n_heads - max(1, n_heads//4))")
    parser.add_argument("--window", type=int, default=0, help="Local attention window in tokens (0 = auto: seq_len // 2)")
    parser.add_argument("--max_ctx", type=int, default=8192, help="Chat context cap in tokens")
    parser.add_argument("--rope_scale", type=float, default=0.0, help="NTK RoPE scale (0 = auto: max_ctx / seq_len)")
    parser.add_argument("--recur_steps", type=int, default=6, help="Recurrent folding iterations (T)")
    parser.add_argument("--fold_dim", type=int, default=2048, help="Fold width (winner: 2048 = 2x d_model; d_model//4 = true bottleneck)")
    parser.add_argument("--fold_weight", type=float, default=0.30, help="Max fold-loss weight after warmup (was hardcoded 0.30)")
    parser.add_argument("--fold_mode", type=str, default="future",
                        choices=["recon", "future", "none", "mlp", "dual"],
                        help="recon: reconstruct present state (legacy). future: predict the NEXT state — concept-shaped objective. none: no fold (pure backbone control). mlp: capacity-matched plain MLP branch (control for the fold geometry). dual: BOTH recon and future expander heads in the same model")
    parser.add_argument("--fold_placement", type=str, default="readout", choices=["readout", "per_layer"],
                        help="readout: fold branch after the last block only (legacy). per_layer: shared fold head after every block")
    parser.add_argument("--freq", type=float, default=1.0, help="Initial folding frequency (omega)")
    # CHANGE 2: Controlled gradient flow parameter (alpha)
    parser.add_argument("--grad_flow", type=float, default=0.20, help="Gradient flow ratio to backbone (0.0 to 1.0)")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Steps to ramp fold loss and LR from 0 (winner: 200)")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate (winner: 6e-5)")
    parser.add_argument("--min_lr", type=float, default=5e-6, help="Final LR (cosine decay floor)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to checkpoint file")
    parser.add_argument("--input", type=str, default="input.txt", help="Training text file (legacy single-file mode)")
    # ---- Mixed-dataset (parquet) training; active only when --datasets is given ----
    parser.add_argument("--datasets", nargs="+", default=None, metavar="NAME=PATH[:WEIGHT]",
                        help="Mixed parquet training: one or more NAME=DIR[:WEIGHT] specs (weight default 1.0; "
                             "Windows drive colons are fine). Without this flag the legacy --input mode runs.")
    parser.add_argument("--mix_temperature", type=float, default=1.0, help="Mixing temperature: probs ~ weight^(1/T)")
    parser.add_argument("--mix_chunk", type=int, default=2048, help="Target tokens per mixer chunk (rounded to whole windows)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Parquet text column: 'col' for all datasets, or 'name=col,name2=col2' (others use 'text')")
    parser.add_argument("--grad_accum", type=int, default=4, help="Micro-batches per optimizer step (mixed mode)")
    parser.add_argument("--gsci", action=argparse.BooleanOptionalAction, default=True,
                        help="Gradient Sign Chaos Injection over groups of accumulation micro-batches (mixed mode)")
    parser.add_argument("--gsci_shards", type=int, default=4, help="GSCI gradient groups (clamped to --grad_accum; <2 disables)")
    parser.add_argument("--gsci_lambda", type=float, default=0.1, help="GSCI: g = mean - lambda * var * sign(mean)")
    parser.add_argument("--forced_lr", type=float, default=0.0, help="If > 0, pin the LR to this value every step (ignores schedule)")
    parser.add_argument("--save_interval", type=int, default=20000, help="Save checkpoint (+completed-files log) every N optimizer steps")
    parser.add_argument("--max_iters", type=int, default=100000, help="Mixed mode: stop at this absolute global step (also the LR schedule length)")
    parser.add_argument("--completed_log", type=str, default=None,
                        help="Completed-parquet-files log (default: <checkpoint>.completed_files.json)")
    parser.add_argument("--reset_completed", action="store_true", help="Ignore/clear the completed-files list and train every file again")
    parser.add_argument("--stats_interval", type=int, default=5000, help="Print mixer stats every N optimizer steps")
    return parser.parse_args()

# ==========================================
# 2. Geometric Space-Folding Bottleneck Head
# ==========================================
class RecurrentPeriodicFoldingHead(nn.Module):
    """mode='recon'  : legacy — reconstruct the present state (h_recon ≈ h).
    mode='future' : predict the NEXT backbone state from the folded z.
                    Fold loss = MSE(h_recon[:, :-1], h[:, 1:]) — the fold must
                    encode predictive (concept) content, not surface detail.
                    LM decodes the simulated future: h + h_recon at position t
                    is 'the state after the next token'."""
    def __init__(self, d_model, fold_dim=256, recur_steps=6, init_freq=1.0, mode="recon"):
        super().__init__()
        self.recur_steps = recur_steps
        self.fold_dim = fold_dim
        self.mode = mode

        if mode == "dual":
            # BOTH branches in one model: recon (proven LM ensemble gain)
            # + future (concept-shaped predictive pathway)
            self.recon_head = RecurrentPeriodicFoldingHead(
                d_model, fold_dim, recur_steps, init_freq, "recon")
            self.future_head = RecurrentPeriodicFoldingHead(
                d_model, fold_dim, recur_steps, init_freq, "future")
            return

        if mode == "mlp":
            # capacity-matched control: same param count and I/O shape as the
            # fold_dim=1024 head (~2.1M), plain GELU MLP — no recurrence,
            # no periodicity, no orthogonality constraint
            self.mlp = nn.Sequential(
                nn.Linear(d_model, fold_dim), nn.GELU(),
                nn.Linear(fold_dim, fold_dim), nn.GELU(),
                nn.Linear(fold_dim, d_model))
            return

        self.proj_in = nn.Linear(d_model, fold_dim)

        # Tied recurrent orthogonal weights
        self.weight = nn.Parameter(torch.empty(fold_dim, fold_dim))
        nn.init.orthogonal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(fold_dim))

        # Bounded learnable frequency
        self.omega = nn.Parameter(torch.tensor(init_freq))
        # FIX (dead-fold bug): omega has no gradient pressure keeping it away from 0,
        # and sin(omega*z) -> 0 kills the whole head. Enforce |omega| >= omega_floor
        # via a hinge penalty in compute_loss.
        self.omega_floor = 0.25

        # Unfolding projection
        self.proj_out = nn.Linear(fold_dim, d_model)

        if mode == "future":
            # re-inject the source state at every recurrence step (iterative refinement)
            self.cycle_proj = nn.Linear(d_model, fold_dim, bias=False)
            nn.init.normal_(self.cycle_proj.weight, std=0.02)
            # per-step learnable frequencies (multi-rate reasoning)
            self.omegas = nn.Parameter(torch.full((recur_steps,), float(init_freq)))
            # learned step mixing: the head chooses which iteration(s) to read
            self.step_mix = nn.Parameter(torch.zeros(recur_steps))
            # decodes the predicted FUTURE state from the folded representation
            self.fut_out = nn.Linear(fold_dim, d_model)

    def forward(self, h, grad_flow=0.20):
        # CHANGE 2: Damped Gradient Flow
        # Allows a controlled percentage of folding gradients to shape the backbone
        # while keeping the remaining percentage detached to prevent chaotic blowout.
        h_blended = (1.0 - grad_flow) * h.detach() + grad_flow * h

        if self.mode == "dual":
            _, hr = self.recon_head(h_blended, grad_flow)
            _, hf = self.future_head(h_blended, grad_flow)
            combined = hr + hf               # ensemble copy + simulated future
            self._last = (hr, hf)            # kept for compute_loss
            return combined, combined

        if self.mode == "mlp":
            z = self.mlp[:-1](h_blended)       # hidden activations (for geometry probes)
            h_recon = self.mlp[-1](z)
            return z, h_recon

        z = self.proj_in(h_blended)

        # FIX (dead-fold bug #1: double contraction).
        # weight is orthogonal-init, so ||W z|| == ||z|| already: the isometry is free.
        # The old code ALSO multiplied by 1/sqrt(fold_dim), contracting every direction
        # by ~32x per step; over recur_steps=6 that is ~1e9 total contraction, which
        # annihilates all input-dependence in z (measured: per-unit std = 0 in the
        # epoch-10 checkpoint). Orthogonal init needs scale = 1.0.
        scale = 1.0

        if self.mode == "future":
            # iterative refinement: each step re-reads the source state, each step
            # runs at its own frequency, and the output mixes across steps
            h_reinj = self.cycle_proj(h_blended)
            mix = torch.softmax(self.step_mix, dim=0)
            z_out = torch.zeros_like(z)
            for i in range(self.recur_steps):
                z = torch.sin(self.omegas[i] * (F.linear(z, self.weight * scale, self.bias)
                                                + h_reinj))
                z_out = z_out + mix[i] * z
            h_recon = self.fut_out(z_out)      # predicted next backbone state
            return z_out, h_recon

        # legacy: fixed-input recurrence
        for _ in range(self.recur_steps):
            z = F.linear(z, self.weight * scale, self.bias)
            z = torch.sin(self.omega * z)

        h_recon = self.proj_out(z)
        return z, h_recon

    def compute_loss(self, h, h_recon):
        if self.mode == "dual":
            # each branch is scored against its own objective using the outputs
            # stored by the last forward pass
            hr, hf = self._last
            return self.recon_head.compute_loss(h, hr) + \
                   self.future_head.compute_loss(h, hf)

        if self.mode == "mlp":
            # plain reconstruction: isolates the objective from the geometry
            return F.mse_loss(h_recon, h.detach())

        if self.mode == "future":
            # predict the next state: position t's fold output targets h at t+1
            recon_loss = F.mse_loss(h_recon[:, :-1], h[:, 1:].detach())

            # Orthogonal isometry regularization
            ortho_penalty = torch.norm(
                torch.mm(self.weight, self.weight.t()) - torch.eye(self.fold_dim, device=h.device)
            )

            # omega floor on EVERY step frequency
            omega_penalty = F.relu(self.omega_floor - self.omegas.abs()).mean()

            return recon_loss + 0.01 * ortho_penalty + 1.0 * omega_penalty

        # Information preservation: ensure the folded manifold doesn't collapse
        recon_loss = F.mse_loss(h_recon, h.detach())

        # Orthogonal isometry regularization
        ortho_penalty = torch.norm(
            torch.mm(self.weight, self.weight.t()) - torch.eye(self.fold_dim, device=h.device)
        )

        # FIX (dead-fold bug #2: omega collapse). sin(omega*z) is the only
        # nonlinearity; omega -> 0 makes it identically ~0 and the head outputs a
        # constant (measured: omega drifted 1.0 -> -5.3e-4 in the epoch-10 ckpt).
        # Hinge penalty keeps |omega| above omega_floor.
        omega_penalty = F.relu(self.omega_floor - self.omega.abs())

        return recon_loss + 0.01 * ortho_penalty + 1.0 * omega_penalty

# ==========================================
# 3. Transformer with Fold-Driven Prediction
#    Backbone: BetterAttention hybrid —
#    sliding-window local heads + holographic
#    content-addressable global heads + RoPE
# ==========================================
class FoldingTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_local_heads, n_layers, seq_len,
                 recur_steps, freq, grad_flow, window=0, rope_scale=1.0,
                 map_dim=32, top_k_tok=32, seg=64, top_seg=4, fold_dim=0,
                 fold_mode="recon", fold_placement="readout"):
        super().__init__()
        self.seq_len = seq_len
        self.grad_flow = grad_flow
        self.rope_scale = rope_scale
        self.head_dim = d_model // n_heads
        self.fold_dim = fold_dim if fold_dim > 0 else d_model * 2   # legacy expander
        self.fold_placement = fold_placement

        window = window if window > 0 else max(32, seq_len // 2)
        # Attention-layer config (n_layer unused here — the stack is built below)
        ba_cfg = BAConfig(
            n_embd=d_model, n_layer=1, n_head=n_heads, n_local_heads=n_local_heads,
            window=min(window, seq_len), map_dim=map_dim,
            top_k_tok=min(top_k_tok, seq_len),
            seg=min(seg, max(1, seq_len)), top_seg=top_seg,
            block_size=seq_len, rope_scale=rope_scale,
        )
        self.blocks = nn.ModuleList([Block(ba_cfg) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Auxiliary Space-Folding Head (omit entirely for the no-fold control)
        self.folding_head = None if fold_mode == "none" else \
            RecurrentPeriodicFoldingHead(
                d_model=d_model, fold_dim=self.fold_dim, recur_steps=recur_steps,
                init_freq=freq, mode=fold_mode
            )

        # Primary Language Modeling Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Scaled residual init (same scheme as betterattention)
        for name, p in self.named_parameters():
            if name.endswith("attn.out.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, idx):
        B, T = idx.shape
        h = self.token_emb(idx)

        causal_mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        cos, sin = rope_cos_sin(T, self.head_dim, idx.device, h.dtype,
                                scale=self.rope_scale)
        self._fold_pairs = []
        for i, blk in enumerate(self.blocks):
            h = blk.forward_train(h, causal_mask, cos, sin)
            if self.fold_placement == "per_layer" and self.folding_head is not None:
                _, h_recon_i = self.folding_head(h, grad_flow=1.0)
                self._fold_pairs.append((h, h_recon_i))
                h = h + h_recon_i           # fold contribution enters the stream
        h = self.ln_f(h)

        if self.folding_head is None:
            # pure backbone control: no fold, no simulated future
            return self.lm_head(h), h, h

        if self.fold_placement == "per_layer":
            # last per-layer application doubles as the readout
            h_last, h_recon = self._fold_pairs[-1]
            logits = self.lm_head(h)        # h already carries fold contributions
        else:
            # FIX (dead-fold bug #3: detached reconstruction target).
            # Pass a LIVE h into the head input path so the recon objective actually
            # pressures the fold to preserve information.
            z, h_recon = self.folding_head(h, grad_flow=1.0)
            self._fold_pairs.append((h, h_recon))
            # CHANGE 1: Route predictions through the folded space!
            # The LM head is forced to decode the folded/reconstructed manifold.
            h_fused = h + h_recon
            logits = self.lm_head(h_fused)

        return logits, h, h_recon

    def fold_loss(self):
        """Summed (and averaged) fold loss over all placements; None if no fold."""
        if self.folding_head is None or not getattr(self, "_fold_pairs", []):
            return None
        total = None
        for h_i, hr_i in self._fold_pairs:
            l = self.folding_head.compute_loss(h_i, hr_i)
            total = l if total is None else total + l
        return total / len(self._fold_pairs)

    # ---- incremental generation helpers (ring-buffer caches) ----
    def init_gen_caches(self, cap, device):
        return [blk.attn.init_cache(cap, device, torch.bfloat16)
                for blk in self.blocks]

    @torch.no_grad()
    def gen_step(self, tok, caches, pos, cos, sin):
        x = self.token_emb(tok).view(1, 1, -1)
        for i, blk in enumerate(self.blocks):
            x, caches[i] = blk.step(x, caches[i], cos, sin, pos)
            if self.fold_placement == "per_layer" and self.folding_head is not None:
                _, hr_i = self.folding_head(x, grad_flow=1.0)
                x = x + hr_i
        x = self.ln_f(x)
        if self.folding_head is None or self.fold_placement == "per_layer":
            return self.lm_head(x)[:, -1, :], caches
        z, h_recon = self.folding_head(x, grad_flow=1.0)
        return self.lm_head(x + h_recon)[:, -1, :], caches

# ==========================================
# 4. Helpers: Dataset & Text Generation
# ==========================================
def load_tokens(path, enc):
    """Encode `path` with tiktoken gpt2, caching as a uint16 memmap.
    Streams in 8MB chunks so the 514MB corpus never materializes in RAM."""
    cache_path = path + ".tokens.gpt2.npy"
    if os.path.exists(cache_path):
        data = np.memmap(cache_path, dtype=np.uint16, mode="r")
        print(f"[*] Token cache hit: {len(data):,} tokens from '{cache_path}'")
        return data

    print(f"[*] Encoding '{path}' with tiktoken gpt2 (first run only)...")
    tmp = cache_path + ".tmp"
    n = 0
    with open(path, "rb") as f, open(tmp, "wb") as out:
        buf = b""
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk
            # cut at a UTF-8 boundary (hold back any trailing partial multibyte seq)
            cut = len(buf)
            while cut > 0 and (buf[cut - 1] & 0xC0) == 0x80:
                cut -= 1
            if cut > 0 and (buf[cut - 1] & 0x80):
                cut -= 1
            ids = enc.encode_ordinary(buf[:cut].decode("utf-8", errors="ignore"))
            buf = buf[cut:]
            np.asarray(ids, dtype=np.uint16).tofile(out)
            n += len(ids)
    os.replace(tmp, cache_path)
    data = np.memmap(cache_path, dtype=np.uint16, mode="r")
    print(f"[*] Encoded {n:,} tokens -> '{cache_path}'")
    return data


def get_batch(data, batch_size, seq_len, device, rng):
    ix = rng.integers(0, len(data) - seq_len - 1, size=batch_size, dtype=np.int64)
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i + 1:i + seq_len + 1].astype(np.int64)) for i in ix]).to(device)
    return x, y


@torch.no_grad()
def generate_sample(model, enc, prompt="To be", max_new_tokens=150, device="cuda"):
    model.eval()
    enc_ids = enc.encode_ordinary(prompt)
    cap = max(len(enc_ids) + max_new_tokens + 8, model.seq_len + 8)
    caches = model.init_gen_caches(cap, device)
    cos, sin = rope_cos_sin(cap, model.head_dim, device, torch.bfloat16,
                            scale=model.rope_scale)
    logits = None
    pos = 0
    gen = []
    for t in enc_ids:
        logits, caches = model.gen_step(torch.tensor([t], device=device), caches, pos, cos, sin)
        pos += 1
    for _ in range(max_new_tokens):
        nt = sample_token(logits, temperature=0.8, top_k=50, top_p=0.9, rep_penalty=1.1)
        gen.append(nt)
        logits, caches = model.gen_step(torch.tensor([nt], device=device), caches, pos, cos, sin)
        pos += 1
    model.train()
    return enc.decode(gen)


@torch.no_grad()
def generate_chat(
    model,
    prompt,
    enc,
    caches,
    pos,
    cos,
    sin,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
):
    """
    Incremental generator (ring-buffer caches) supporting temperature, top-k,
    top-p (nucleus), and repetition penalty. The prompt is appended to the
    existing context; returns (text, caches, new_pos).
    """
    model.eval()
    if not prompt:
        prompt = " "
    enc_ids = enc.encode_ordinary(prompt)
    dev = next(model.parameters()).device

    logits = None
    for t in enc_ids:
        logits, caches = model.gen_step(torch.tensor([t], device=dev), caches, pos, cos, sin)
        pos += 1

    gen = []
    rep_tokens = list(range(max(0, pos - 192), pos))
    for _ in range(max_new_tokens):
        nt = sample_token(logits, temperature=temperature, top_k=top_k,
                          top_p=top_p, rep_penalty=repetition_penalty,
                          rep_tokens=rep_tokens)
        gen.append(nt)
        rep_tokens.append(nt)
        logits, caches = model.gen_step(torch.tensor([nt], device=dev), caches, pos, cos, sin)
        pos += 1

    model.train()
    return enc.decode(gen), caches, pos


def chat_interface(model, enc, device, max_ctx):
    """
    Interactive chat session prompting the user for generation hyper-parameters.
    Context persists across turns (ring-buffer caches, capped at max_ctx tokens).
    """
    print("\n" + "=" * 50)
    print("INTERACTIVE CHAT MODE")
    print("=" * 50)
    print("Configure generation parameters (press Enter to use default values):")

    def get_input(prompt_text, default_val, type_fn):
        raw = input(f"{prompt_text} [default {default_val}]: ").strip()
        if not raw:
            return default_val
        try:
            return type_fn(raw)
        except ValueError:
            print(f"[!] Invalid input. Using default: {default_val}")
            return default_val

    temperature = get_input("Temperature (e.g. 0.8)", 0.8, float)
    top_k = get_input("Top-K (0 to disable)", 50, int)
    top_p = get_input("Top-P (e.g. 0.9, 1.0 to disable)", 0.9, float)
    rep_penalty = get_input("Repetition Penalty (e.g. 1.1)", 1.1, float)
    max_tokens = get_input("Max new tokens per response", 150, int)

    caches = model.init_gen_caches(max_ctx + 8, device)
    pos = 0
    cos, sin = rope_cos_sin(max_ctx + 8, model.head_dim, device, torch.bfloat16,
                            scale=model.rope_scale)

    print("\nChat session active. Type your prompt and press Enter.")
    print("Type 'exit' or 'quit' to leave chat mode.\n" + "-" * 50)

    while True:
        try:
            user_prompt = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat mode.")
            break

        if user_prompt.strip().lower() in ["exit", "quit"]:
            print("Exiting chat mode.")
            break
        if not user_prompt.strip():
            continue

        n_prompt = len(enc.encode_ordinary(user_prompt)) + 1
        if pos + n_prompt + max_tokens + 8 > max_ctx:
            print("[!] Context limit reached — clearing context.")
            caches = model.init_gen_caches(max_ctx + 8, device)
            pos = 0

        try:
            response, caches, pos = generate_chat(
                model=model,
                prompt=user_prompt + "\n",
                enc=enc,
                caches=caches,
                pos=pos,
                cos=cos,
                sin=sin,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
            )
        except RuntimeError as e:
            print(f"[!] Generation error: {e}")
            caches = model.init_gen_caches(max_ctx + 8, device)
            pos = 0
            continue
        print(f"Model: {response}")

# ==========================================
# 4b. Mixed-dataset streaming (parquet) + completed-file ledger
#     pyarrow is imported lazily (legacy --input mode never needs it)
# ==========================================
MIX_LOOKAHEAD = 8          # chunks buffered per dataset (bounded lookahead)
LOG_VERSION = 1


def parse_dataset_specs(specs, temperature=1.0, check_dirs=True):
    """Parse NAME=PATH[:WEIGHT] specs -> [(name, path, weight)].
    The weight is split off with rsplit(':', 1) and only if numeric, so Windows
    drive-letter paths (C:\\data\\x) parse correctly."""
    if not (isinstance(temperature, (int, float)) and math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"--mix_temperature must be > 0 (got {temperature})")
    out, seen = [], set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"bad dataset spec '{spec}' (expected NAME=PATH[:WEIGHT])")
        name, rest = spec.split("=", 1)
        name, rest = name.strip(), rest.strip()
        if not name or not rest:
            raise ValueError(f"bad dataset spec '{spec}' (empty name or path)")
        path, weight = rest, 1.0
        if ":" in rest:
            head, tail = rest.rsplit(":", 1)
            try:
                w = float(tail)
            except ValueError:
                w = None
            if w is not None:
                path, weight = head, w
        if not path:
            raise ValueError(f"bad dataset spec '{spec}' (empty path)")
        if not (math.isfinite(weight) and weight > 0):
            raise ValueError(f"dataset '{name}': weight must be a finite number > 0 (got {weight})")
        if name in seen:
            raise ValueError(f"duplicate dataset name '{name}'")
        if check_dirs and not os.path.isdir(path):
            raise ValueError(f"dataset '{name}': directory not found: {path}")
        seen.add(name)
        out.append((name, path, weight))
    if not out:
        raise ValueError("no datasets given")
    return out


def parse_text_columns(arg):
    """'col' -> (col, {}); 'a=c1,b=c2' -> ('text', {'a': 'c1', 'b': 'c2'})."""
    if "=" not in arg:
        return arg.strip() or "text", {}
    col_map = {}
    for item in arg.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"bad --text_column item '{item}' (expected name=col)")
        n, c = item.split("=", 1)
        col_map[n.strip()] = c.strip()
    return "text", col_map


def mix_probs(weights, temperature):
    """Normalized sampling probabilities: w^(1/T) / sum."""
    p = np.array([w ** (1.0 / temperature) for w in weights], dtype=np.float64)
    return p / p.sum()


def file_key(path):
    """Identity of a file version: (normalized abs path, size, mtime_ns); None if unreadable."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.normcase(os.path.abspath(path)), int(st.st_size), int(st.st_mtime_ns))


class Ledger:
    """Per-file state machine: read -> consumed -> applied -> committed.
    read      : first chunk of the file entered the mixer
    consumed  : last window of the file was emitted to a (not yet applied) micro-batch
    applied   : the optimizer step containing that window completed
    committed : a checkpoint holding that step (and the log) was written"""
    def __init__(self, entries=()):
        self.completed = {}              # committed (persisted) entries
        for e in entries:
            self.completed[(e["path"], e["size"], e["mtime_ns"])] = dict(e)
        self.consumed, self.applied = {}, {}
        self.read, self.failed = set(), set()

    def is_done(self, key):
        return key in self.completed

    def consume(self, key, dataset, tokens):
        self.consumed[key] = {"path": key[0], "size": key[1], "mtime_ns": key[2],
                              "dataset": dataset, "tokens": int(tokens)}

    def apply(self):
        for key in self.consumed:        # a retrained (touched/modified) file supersedes its old version
            for store in (self.completed, self.applied):
                for old in [k for k in store if k[0] == key[0] and k != key]:
                    del store[old]
        self.applied.update(self.consumed)
        self.consumed.clear()

    def discard_unapplied(self):
        n = len(self.consumed)
        self.consumed.clear()
        return n

    def snapshot(self):
        return list(self.completed.values()) + list(self.applied.values())

    def commit(self):
        self.completed.update(self.applied)
        self.applied.clear()

    def done_count(self, dataset):
        return sum(1 for e in self.snapshot() if e["dataset"] == dataset)


def _valid_entry(e):
    return (isinstance(e, dict) and isinstance(e.get("path"), str) and isinstance(e.get("size"), int)
            and isinstance(e.get("mtime_ns"), int) and isinstance(e.get("dataset"), str)
            and isinstance(e.get("tokens"), int))


def load_completed_log(path):
    """-> (committed_step, entries) or None (missing/corrupt; corrupt file is renamed *.corrupt)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if (not isinstance(obj, dict) or obj.get("version") != LOG_VERSION
                or not isinstance(obj.get("committed_step"), int)
                or not isinstance(obj.get("files"), list)
                or not all(_valid_entry(e) for e in obj["files"])):
            raise ValueError("schema mismatch")
        return obj["committed_step"], obj["files"]
    except (OSError, ValueError) as e:
        print(f"[!] Completed-files log '{path}' is unreadable ({e}); treating as empty.")
        try:
            os.replace(path, path + ".corrupt")
            print(f"    (moved to '{path}.corrupt')")
        except OSError:
            pass
        return None


def write_completed_log(path, entries, step):
    """Atomic write (tmp + fsync + os.replace)."""
    obj = {"version": LOG_VERSION, "committed_step": int(step),
           "files": sorted((dict(e) for e in entries), key=lambda e: e["path"])}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def reconcile_completed(args, ckpt, log_path):
    """Decide the completed-files list at startup. The checkpoint is authoritative;
    the log mirrors it (and is repaired from it). Returns a list of entries."""
    if args.reset_completed:
        print("[*] --reset_completed: ignoring the completed-files list (rewritten at the next save).")
        return []
    log = load_completed_log(log_path)
    if ckpt is None:
        if log is not None:
            print(f"[!] Stale completed-files log '{log_path}' ignored: no checkpoint to resume from "
                  f"(it is overwritten at the first save).")
        return []
    ck_step = int(ckpt.get("global_step", 0))
    ck_files = ckpt.get("completed_files")
    if isinstance(ck_files, list) and all(_valid_entry(e) for e in ck_files):
        ck_keys = {(e["path"], e["size"], e["mtime_ns"]) for e in ck_files}
        log_keys = {(e["path"], e["size"], e["mtime_ns"]) for e in log[1]} if log else None
        if log is None or log[0] != ck_step or log_keys != ck_keys:
            print(f"[!] Completed-files log disagrees with the checkpoint (step {ck_step}); "
                  f"repairing log from the checkpoint ({len(ck_files)} files).")
            write_completed_log(log_path, ck_files, ck_step)
        return list(ck_files)
    if log is not None and log[0] == ck_step:
        print(f"[*] Checkpoint has no embedded file list; using log '{log_path}' (step {ck_step}).")
        return list(log[1])
    if log is not None:
        print(f"[!] Log step {log[0]} != checkpoint step {ck_step} and checkpoint has no file list; log ignored.")
    return []


class _Chunk:
    __slots__ = ("dataset", "key", "arr", "last", "file_tokens")

    def __init__(self, dataset, key, arr, last, file_tokens):
        self.dataset, self.key, self.arr, self.last, self.file_tokens = dataset, key, arr, last, file_tokens


def iter_file_chunks(path, column, enc, chunk_len):
    """Stream one parquet file -> (uint16 array, is_last). Chunks never span files; the
    final chunk (possibly short) is held back so it can be flagged last. Documents are
    tokenized with encode_ordinary (literal '<|endoftext|>' in text is plain text) and
    separated by the EOT id."""
    import pyarrow.parquet as pq          # lazy: legacy mode never imports pyarrow
    eot = enc.eot_token
    pf = pq.ParquetFile(path)
    try:
        names = pf.schema_arrow.names
        if column not in names:
            raise KeyError(f"column '{column}' not found (available: {names})")
        buf = np.empty(0, dtype=np.uint16)
        held = None
        for batch in pf.iter_batches(batch_size=500, columns=[column]):
            for text in batch.column(0).to_pylist():
                if text is None:
                    continue
                ids = np.asarray(enc.encode_ordinary(str(text)) + [eot], dtype=np.uint16)
                buf = ids if buf.size == 0 else np.concatenate([buf, ids])
                pos = 0
                while buf.size - pos >= chunk_len:
                    arr = buf[pos:pos + chunk_len].copy()
                    pos += chunk_len
                    if held is not None:
                        yield held, False
                    held = arr
                if pos:
                    buf = buf[pos:].copy()
        if buf.size:
            if held is not None:
                yield held, False
            held = buf
        if held is not None:
            yield held, True
    finally:
        close = getattr(pf, "close", None)
        if close:
            close()


class DatasetMixer:
    """Weighted multi-dataset chunk mixer with an exact consumption ledger.
    prob_i = w_i^(1/T) / sum_j w_j^(1/T) over datasets that still have data
    (exhausted datasets are dropped and probabilities renormalized)."""
    def __init__(self, specs, enc, seq_len, mix_chunk, temperature, default_col, col_map,
                 ledger, seed=1337, lookahead=MIX_LOOKAHEAD):
        self.enc, self.seq_len, self.ledger = enc, seq_len, ledger
        self.chunk_len = max(1, mix_chunk // seq_len) * seq_len + 1
        self.temperature, self.lookahead = temperature, lookahead
        self.rng = np.random.default_rng(seed)
        self.weights, self.queues, self.files, self.gens, self.cur = {}, {}, {}, {}, {}
        self.cols, self.files_total, self.files_skipped = {}, {}, {}
        self.tokens_sampled, self.chunks_sampled, self.batches_sampled = {}, {}, {}
        self.active, self.dropped = [], []
        self._cur = None                                   # (chunk, window offset)
        shuf = random.Random(seed)
        for name, path, w in specs:
            self.weights[name] = w
            self.cols[name] = col_map.get(name, default_col)
            self.tokens_sampled[name] = self.chunks_sampled[name] = self.batches_sampled[name] = 0
            found = sorted(glob.glob(os.path.join(glob.escape(path), "**", "*.parquet"), recursive=True))
            shuf.shuffle(found)
            todo, skipped = deque(), 0
            for fp in found:
                key = file_key(fp)
                if key is None:
                    print(f"[!] {name}: cannot stat '{fp}', skipped.")
                elif ledger.is_done(key):
                    skipped += 1
                else:
                    todo.append((fp, key))
            self.files_total[name], self.files_skipped[name] = len(found), skipped
            if not todo:
                why = ("no parquet files found" if not found else
                       f"all {len(found)} files already completed")
                print(f"[*] Dataset '{name}' dropped: {why}.")
                self.dropped.append(name)
                continue
            if skipped:
                print(f"[*] Dataset '{name}': skipping {skipped} completed file(s); {len(todo)} to train.")
            self.files[name], self.queues[name] = todo, deque()
            self.gens[name], self.cur[name] = None, None
            self.active.append(name)
        self.target = dict(zip(self.active, mix_probs([self.weights[n] for n in self.active], temperature))) \
            if self.active else {}

    # ---- per-dataset lookahead refill ----
    def _fill(self, ds):
        q = self.queues[ds]
        while len(q) < self.lookahead:
            if self.gens[ds] is None:
                if not self.files[ds]:
                    return
                path, key = self.files[ds].popleft()
                self.cur[ds] = {"path": path, "key": key, "tokens": 0, "any": False}
                self.gens[ds] = iter_file_chunks(path, self.cols[ds], self.enc, self.chunk_len)
            info = self.cur[ds]
            try:
                arr, last = next(self.gens[ds])
            except StopIteration:
                self.gens[ds] = None
                if not info["any"]:                         # file without any text
                    print(f"[!] {ds}: '{info['path']}' has no text; marked complete (0 tokens).")
                    self.ledger.consume(info["key"], ds, 0)
                continue
            except Exception as e:                           # unreadable file / missing column
                print(f"[!] {ds}: skipping '{info['path']}': {type(e).__name__}: {e}")
                self.ledger.failed.add(info["key"])
                self.gens[ds] = None
                continue
            if not info["any"]:
                self.ledger.read.add(info["key"])
            info["any"] = True
            info["tokens"] += len(arr)
            q.append(_Chunk(ds, info["key"], arr, last, info["tokens"] if last else 0))
            if last:
                self.gens[ds] = None

    def _active_with_data(self):
        for ds in list(self.active):
            self._fill(ds)
            if not self.queues[ds]:
                print(f"[*] Dataset '{ds}' exhausted; dropping it and renormalizing the mixing probabilities.")
                self.active.remove(ds)
                self.dropped.append(ds)
        return self.active

    def _next_chunk(self):
        names = self._active_with_data()
        if not names:
            return None
        probs = mix_probs([self.weights[n] for n in names], self.temperature)
        ds = names[int(self.rng.choice(len(names), p=probs))]
        ch = self.queues[ds].popleft()
        self.tokens_sampled[ds] += len(ch.arr)
        self.chunks_sampled[ds] += 1
        return ch

    def _finish(self, ch):
        self._cur = None
        if ch.last:
            self.ledger.consume(ch.key, ch.dataset, ch.file_tokens)

    def next_window(self):
        """-> (uint16 array of seq_len+1 tokens, dataset) or None when all data is consumed."""
        while True:
            if self._cur is not None:
                ch, off = self._cur
                if off + self.seq_len + 1 <= len(ch.arr):
                    w = ch.arr[off:off + self.seq_len + 1]
                    off += self.seq_len
                    self._cur = (ch, off)
                    if off + self.seq_len + 1 > len(ch.arr):
                        self._finish(ch)
                    return w, ch.dataset
                self._finish(ch)                             # tail shorter than a window: dropped, counted consumed
            ch = self._next_chunk()
            if ch is None:
                return None
            self._cur = (ch, 0)

    def get_batch(self, batch_size, device):
        """-> (x, y) int64 tensors, possibly smaller than batch_size at the end; None if exhausted."""
        ws = []
        for _ in range(batch_size):
            r = self.next_window()
            if r is None:
                break
            ws.append(r[0])
            self.batches_sampled[r[1]] += 1
        if not ws:
            return None
        a = torch.from_numpy(np.stack(ws).astype(np.int64))
        return a[:, :-1].contiguous().to(device), a[:, 1:].contiguous().to(device)

    def stats_state(self):
        return {"tokens_sampled": dict(self.tokens_sampled), "chunks_sampled": dict(self.chunks_sampled),
                "batches_sampled": dict(self.batches_sampled)}

    def load_stats(self, st):
        if not isinstance(st, dict):
            return
        for k in ("tokens_sampled", "chunks_sampled", "batches_sampled"):
            for n, v in (st.get(k) or {}).items():
                if n in getattr(self, k) and isinstance(v, int):
                    getattr(self, k)[n] += v


def print_mix_stats(mixer, ledger, step):
    tot = max(1, sum(mixer.tokens_sampled.values()))
    print(f"\n[Mixer stats @ step {step}] (target = normalized w^(1/T) over datasets present at start)")
    print(f"  {'dataset':<14}{'files done':>12}{'tokens':>14}{'share%':>9}{'target%':>9}{'drift':>8}{'windows':>10}")
    for n in mixer.weights:
        share = 100.0 * mixer.tokens_sampled[n] / tot
        tgt = 100.0 * mixer.target.get(n, 0.0)
        print(f"  {n:<14}{ledger.done_count(n):>6}/{mixer.files_total[n]:<5}{mixer.tokens_sampled[n]:>14,}"
              f"{share:>9.1f}{tgt:>9.1f}{share - tgt:>+8.1f}{mixer.batches_sampled[n]:>10,}")
    print(f"  failed files: {len(ledger.failed)} | active datasets: {', '.join(mixer.active) or '-'}")


# ==========================================
# 4c. Training mechanics (loss, GSCI, checkpointing, mixed loop)
# ==========================================
def model_loss(model, x, y, step_no, args, vocab_size, device):
    """CLM + warmed-up fold loss for one micro-batch -> (total, clm, fold, fold_weight)."""
    logits, h, h_recon = model(x)
    loss_clm = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    if model.folding_head is None:
        return loss_clm, loss_clm, torch.zeros((), device=device), 0.0
    loss_fold = model.fold_loss()
    fold_weight = min(args.fold_weight, args.fold_weight * (step_no / float(args.warmup_steps)))
    return (1.0 - fold_weight) * loss_clm + fold_weight * loss_fold, loss_clm, loss_fold, fold_weight


def gsci_combine(grads, lam):
    """Gradient Sign Chaos Injection over per-group gradients g_k (same shape):
    mean_k(g) - lam * var_k(g, population) * sign(mean_k(g))."""
    s = torch.stack([g.detach() for g in grads])
    mean = s.mean(dim=0)
    var = s.var(dim=0, unbiased=False)
    return mean - lam * var * torch.sign(mean)


def gsci_groups(n_micro, shards):
    """Contiguous groups of micro-batch indices (min(shards, n_micro) groups)."""
    k = min(shards, n_micro)
    return [list(g) for g in np.array_split(np.arange(n_micro), k) if len(g)]


def accumulate_grads(model, micro, step_no, args, vocab_size, device, use_gsci):
    """Backprop over `micro` (list of (x, y)); leaves final grads in p.grad.
    Returns (mean total, mean clm, mean fold, fold_weight) as floats."""
    params = [p for p in model.parameters() if p.requires_grad]
    n = len(micro)
    tot = clm = fold = fw = 0.0

    def run(idx_list, scale):
        nonlocal tot, clm, fold, fw
        for i in idx_list:
            x, y = micro[i]
            t, c, f, fw = model_loss(model, x, y, step_no, args, vocab_size, device)
            (t * scale).backward()
            tot += t.item() / n
            clm += c.item() / n
            fold += f.item() / n

    for p in params:
        p.grad = None
    if not use_gsci:
        run(range(n), 1.0 / n)
        return tot, clm, fold, fw
    group_grads = []
    for grp in gsci_groups(n, args.gsci_shards):
        for p in params:
            p.grad = None
        run(grp, 1.0 / len(grp))
        group_grads.append([p.grad.detach().clone() if p.grad is not None else None for p in params])
    for i, p in enumerate(params):
        gs = [g[i] for g in group_grads]
        if all(g is None for g in gs):
            p.grad = None
        else:
            p.grad = gsci_combine([g if g is not None else torch.zeros_like(p) for g in gs], args.gsci_lambda)
    return tot, clm, fold, fw


def make_lr_fn(args, total_steps):
    """Warmup + cosine decay (clamped); --forced_lr > 0 pins the LR."""
    def lr_at(step):
        if args.forced_lr > 0:
            return args.forced_lr
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))
    return lr_at


def save_checkpoint(path, model, optimizer, global_step, epoch, model_cfg, args, extra=None):
    """Atomic checkpoint write (tmp + os.replace). Holds only primitives/tensors."""
    data = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "enc_name": "gpt2",
        "model_cfg": dict(model_cfg),
        "args": vars(args),
    }
    if extra:
        data.update(extra)
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def run_mixed_training(args, specs, model, optimizer, enc, device, vocab_size, seq_len,
                       model_cfg, global_step, epoch, ckpt):
    """Mixed-dataset training. Returns (last_x or None, interrupted)."""
    log_path = args.completed_log or (args.checkpoint + ".completed_files.json")
    ledger = Ledger(reconcile_completed(args, ckpt, log_path))
    default_col, col_map = parse_text_columns(args.text_column)
    mixer = DatasetMixer(specs, enc, seq_len, args.mix_chunk, args.mix_temperature,
                         default_col, col_map, ledger)
    if ckpt is not None:
        mixer.load_stats(ckpt.get("mixer_stats"))
    if not mixer.active:
        print("[*] Nothing to train: every dataset is fully completed (or has no files). "
              "Use --reset_completed to train them again. No checkpoint written.")
        return None, False

    accum = max(1, args.grad_accum)
    use_gsci = bool(args.gsci)
    if use_gsci and min(args.gsci_shards, accum) < 2:
        print(f"[!] GSCI disabled: needs >= 2 gradient groups (gsci_shards={args.gsci_shards}, "
              f"grad_accum={accum}).")
        use_gsci = False
    lr_at = make_lr_fn(args, args.max_iters)
    start_step = global_step
    if global_step >= args.max_iters:
        print(f"[*] global_step {global_step} >= --max_iters {args.max_iters}: nothing to do. "
              f"Raise --max_iters to continue. No checkpoint written.")
        return None, False

    print(f"[*] Mixed training: {len(mixer.active)} dataset(s) | chunk {mixer.chunk_len} tokens | "
          f"batch {args.batch_size} x accum {accum} | GSCI "
          f"{'on (groups=%d, lambda=%g)' % (min(args.gsci_shards, accum), args.gsci_lambda) if use_gsci else 'off'}")
    print(f"[*] LR: " + (f"FORCED {args.forced_lr:.2e}" if args.forced_lr > 0 else
          f"{args.lr:.0e} -> {args.min_lr:.0e} cosine over {args.max_iters:,} steps (warmup {args.warmup_steps})")
          + f" | steps {global_step + 1}..{args.max_iters} | save every {args.save_interval} | log '{log_path}'")
    print("-" * 75)

    def do_save(tag):
        snap = ledger.snapshot()
        save_checkpoint(args.checkpoint, model, optimizer, global_step, epoch, model_cfg, args,
                        {"completed_files": snap, "mixer_stats": mixer.stats_state(), "data_mode": "mixed"})
        write_completed_log(log_path, snap, global_step)      # log only after the checkpoint is in place
        ledger.commit()
        print(f"[*] Checkpoint saved ({tag}) at step {global_step}: '{args.checkpoint}' "
              f"| completed files logged: {len(snap)}")

    last_x, interrupted, stop_reason = None, False, f"reached --max_iters {args.max_iters}"
    tot = clm = fold = fw = float("nan")
    model.train()
    try:
        while global_step < args.max_iters:
            micro = []
            for _ in range(accum):
                b = mixer.get_batch(args.batch_size, device)
                if b is None:
                    break
                micro.append(b)
            if not micro:
                stop_reason = "all datasets exhausted"
                ledger.apply()    # tails consumed by this empty batch: all their windows were trained in earlier steps
                break
            step_no = global_step + 1
            lr = lr_at(step_no)
            for g in optimizer.param_groups:
                g["lr"] = lr
            tot, clm, fold, fw = accumulate_grads(model, micro, step_no, args, vocab_size, device, use_gsci)
            if not all(math.isfinite(v) for v in (tot, clm, fold)):
                raise RuntimeError(f"non-finite loss at step {step_no} (total={tot}, clm={clm}, fold={fold})")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step = step_no
            ledger.apply()                                    # files finished in this step are now trained
            last_x = micro[-1][0][:1].detach().clone()

            if global_step % args.predict_every == 0 or global_step == 1:
                sample = generate_sample(model, enc, prompt="To be", max_new_tokens=150, device=device)
                print(f"\n[Step {global_step} | lr {lr:.2e}]")
                print(f"  Total Loss: {tot:.4f} | CLM ({(1 - fw) * 100:.1f}%): {clm:.4f} | Fold ({fw * 100:.1f}%): {fold:.4f}")
                print(f"  Model Sample: {repr(sample[:140])}...\n")
                print("-" * 75)
            if global_step % args.stats_interval == 0:
                print(f"[Step {global_step}] loss {tot:.4f} (clm {clm:.4f}, fold {fold:.4f}) lr {lr:.2e}")
                print_mix_stats(mixer, ledger, global_step)
            if global_step % args.save_interval == 0:
                do_save("periodic")
    except KeyboardInterrupt:
        interrupted = True
        dropped = ledger.discard_unapplied()
        print(f"\n[!] Interrupted: partial accumulation discarded ({dropped} unapplied file(s) not logged); "
              f"saving at last applied step {global_step}.")

    if global_step > start_step or ledger.applied:
        print(f"[*] Training stopped: {stop_reason} (step {global_step}) | last loss {tot:.4f} "
              f"(clm {clm:.4f}, fold {fold:.4f})")
        print_mix_stats(mixer, ledger, global_step)
        do_save("final" if not interrupted else "interrupt")
    else:
        print("[*] No optimizer step was run; no checkpoint written.")
    return last_x, interrupted


# ==========================================
# 5. Training Loop
# ==========================================
def main():
    args = parse_args()
    for _s in (sys.stdout, sys.stderr):          # generated text may not fit the console codepage
        try:
            _s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    mixed = args.datasets is not None
    specs = None
    if mixed:
        try:
            specs = parse_dataset_specs(args.datasets, args.mix_temperature)
            if args.mix_chunk < 1 or args.save_interval < 1 or args.stats_interval < 1 or args.max_iters < 1:
                raise ValueError("--mix_chunk, --save_interval, --stats_interval and --max_iters must be >= 1")
            if args.forced_lr < 0:
                raise ValueError("--forced_lr must be >= 0")
            parse_text_columns(args.text_column)
        except ValueError as e:
            print(f"[!] {e}")
            sys.exit(2)
        print("[*] Mixed-dataset mode: " + ", ".join(f"{n}(w={w:g})" for n, _, w in specs)
              + f" | T={args.mix_temperature:g} | --epochs/--input ignored")
    elif not os.path.exists(args.input):
        print(f"[!] '{args.input}' not found. Creating sample dataset...")
        sample_text = (
            "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
            "The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles "
            "And by opposing end them. To die—to sleep, No more; and by a sleep to say we end "
            "The heart-ache and the thousand natural shocks That flesh is heir to.\n"
        ) * 150
        with open(args.input, "w", encoding="utf-8") as f:
            f.write(sample_text)

    enc = tiktoken.get_encoding("gpt2")     # ~50K vocab (50,257)
    vocab_size = enc.n_vocab
    data = None if mixed else load_tokens(args.input, enc)

    # Architecture follows the checkpoint when one exists, so chat/continue
    # works regardless of the CLI values used at launch.
    ckpt = None
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
    mc = ckpt.get("model_cfg") if ckpt else None

    d_model = mc["d_model"] if mc else args.d_model
    n_heads = mc["n_heads"] if mc else args.n_heads
    n_layers = mc["n_layers"] if mc else args.n_layers
    seq_len = mc["seq_len"] if mc else args.seq_len
    recur_steps = mc["recur_steps"] if mc else args.recur_steps
    freq = mc["freq"] if mc else args.freq
    grad_flow = mc["grad_flow"] if mc else args.grad_flow
    if mc:
        window = mc.get("window") or max(32, seq_len // 2)
        n_local = mc["n_local_heads"]
        rope_scale = mc["rope_scale"]
        fold_dim = mc.get("fold_dim", 0)
        fold_mode = mc.get("fold_mode", "recon")
        fold_placement = mc.get("fold_placement", "readout")
    else:
        window = args.window if args.window > 0 else max(32, seq_len // 2)
        n_local = args.n_local_heads if args.n_local_heads > 0 \
            else n_heads - max(1, n_heads // 4)
        rope_scale = args.rope_scale if args.rope_scale > 0 \
            else max(1.0, args.max_ctx / seq_len)
        fold_dim = args.fold_dim
        fold_mode = args.fold_mode
        fold_placement = args.fold_placement

    # Resolved architecture (what was actually built) -- saved in checkpoints
    model_cfg = {
        "vocab_size": vocab_size, "d_model": d_model, "n_heads": n_heads,
        "n_local_heads": n_local, "n_layers": n_layers, "seq_len": seq_len,
        "recur_steps": recur_steps, "freq": freq, "grad_flow": grad_flow,
        "window": window, "rope_scale": rope_scale, "fold_dim": fold_dim,
        "fold_mode": fold_mode, "fold_placement": fold_placement,
    }

    model = FoldingTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_local_heads=n_local,
        n_layers=n_layers,
        seq_len=seq_len,
        recur_steps=recur_steps,
        freq=freq,
        grad_flow=grad_flow,
        window=window,
        rope_scale=rope_scale,
        fold_dim=fold_dim,
        fold_mode=fold_mode,
        fold_placement=fold_placement,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Total Model Parameters: {total_params:,}")
    if model.folding_head is None:
        print("[*] Fold: DISABLED (pure backbone control)")
    else:
        print(f"[*] Fold bottleneck: {model.fold_dim} dims "
              f"({'true bottleneck (compressed)' if model.fold_dim < d_model else 'legacy expander'})"
              f" | mode: {model.folding_head.mode} | placement: {model.fold_placement}")
    print(f"[*] Backbone: BetterAttention hybrid "
          f"({n_local} sliding-window + {n_heads - n_local} holographic heads "
          f"x {n_layers} layers, window={window})")
    print(f"[*] Tokenizer: tiktoken gpt2 (vocab {vocab_size:,})")
    print(f"[*] Gradient Flow to Backbone: {grad_flow * 100:.1f}%")
    print(f"[*] Sequence Horizon: {seq_len} tokens | RoPE scale: {rope_scale:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    global_step = 0

    # Checkpoint menu (architecture already resolved from the checkpoint)
    if ckpt is not None:
        print(f"\n[?] Checkpoint found at '{args.checkpoint}'.")
        print("  [1] Chat (interactive generation with top_p, top_k, temperature, repetition penalty)")
        print("  [2] Continue training")

        choice = ""
        while choice not in ["1", "2", "chat", "continue"]:
            choice = input("Select an option (1/2): ").strip().lower()

        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[*] Loaded weights from '{args.checkpoint}'.")

        if choice in ["1", "chat"]:
            chat_interface(model, enc, device, args.max_ctx)
            return

        # Option 2: Resume training
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"[!] Optimizer state could not be restored ({e}); continuing with a fresh optimizer.")
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)
        print(f"[*] Resuming training from epoch {start_epoch + 1}, global step {global_step}.\n")

    if mixed:
        last_x, interrupted = run_mixed_training(
            args, specs, model, optimizer, enc, device, vocab_size, seq_len,
            model_cfg, global_step, start_epoch, ckpt)
        if last_x is None or interrupted:
            return
        geometric_analysis(model, args, last_x)
        return

    rng = np.random.default_rng(1337)
    steps_per_epoch = max(1, len(data) // (args.batch_size * seq_len))

    # Cosine LR decay with warmup (clamped, so resumed runs past the schedule
    # floor sit at min_lr instead of going negative)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))

    print(f"[*] Starting training: {args.epochs} epochs (~{steps_per_epoch} steps/epoch, "
          f"LR {args.lr:.0e} -> {args.min_lr:.0e} cosine over {total_steps:,} steps)")
    print("-" * 75)

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        for step in range(steps_per_epoch):
            global_step += 1
            lr = lr_at(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr
            x, y = get_batch(data, args.batch_size, seq_len, device, rng)

            logits, h, h_recon = model(x)

            # 1. Primary Language Modeling Loss (derived from folded space)
            loss_clm = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            # 2. Auxiliary Space-Folding Loss (skipped for the no-fold control)
            if model.folding_head is None:
                fold_weight, clm_weight = 0.0, 1.0
                loss_fold = torch.zeros((), device=device)
            else:
                loss_fold = model.fold_loss()
                # 3. Dynamic Weight Warmup (Ramping from 0% -> fold_weight over warmup_steps)
                fold_weight = min(args.fold_weight,
                                  args.fold_weight * (global_step / float(args.warmup_steps)))
                clm_weight = 1.0 - fold_weight

            total_loss = clm_weight * loss_clm + fold_weight * loss_fold

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Predict and evaluate every N steps
            if global_step % args.predict_every == 0 or global_step == 1:
                sample = generate_sample(model, enc, prompt="To be", max_new_tokens=150, device=device)
                print(f"\n[Step {global_step} | Epoch {epoch}/{start_epoch + args.epochs} | lr {lr:.2e}]")
                print(f"  Total Loss: {total_loss.item():.4f} | CLM ({clm_weight*100:.1f}%): {loss_clm.item():.4f} | Fold ({fold_weight*100:.1f}%): {loss_fold.item():.4f}")
                print(f"  Model Sample: {repr(sample[:140])}...\n")
                print("-" * 75)

    print("[*] Training Complete!")

    # Save Checkpoint at the end of training (resolved architecture, atomic write)
    save_checkpoint(args.checkpoint, model, optimizer, global_step, start_epoch + args.epochs,
                    model_cfg, args)
    print(f"[*] Checkpoint saved successfully to '{args.checkpoint}'.")

    x_test, _ = get_batch(data, batch_size=1, seq_len=seq_len, device=device, rng=rng)
    geometric_analysis(model, args, x_test)


# ==========================================
# 6. Geometric Probing: Inspecting the Folds
# ==========================================
def geometric_analysis(model, args, x_test):
    print("\n" + "=" * 50)
    print("GEOMETRIC FOLDING ANALYSIS")
    print("=" * 50)

    head = model.folding_head
    if head is None:
        print("[*] Fold disabled — no folding analysis for the control model.")
        return

    if head.mode == "mlp":
        print("[*] MLP control branch — no recurrent frequencies to report.")
    elif head.mode == "dual":
        print(f"[*] DUAL head — recon omega: {head.recon_head.omega.item():.4f}")
        print(f"[*] DUAL head — future omegas: "
              + ", ".join(f"{o:.3f}" for o in head.future_head.omegas.detach().tolist()))
    elif head.mode == "future":
        omegas = head.omegas.detach().tolist()
        print(f"[*] Per-step Frequencies (omega): "
              + ", ".join(f"{o:.3f}" for o in omegas))
    else:
        print(f"[*] Initial Frequency (omega): {args.freq:.4f}")
        print(f"[*] Learned Frequency (omega): {head.omega.item():.4f}")

    with torch.no_grad():
        _, h_test, _ = model(x_test)
        z_test, _ = model.folding_head(h_test, grad_flow=0.0)

        h_flat = h_test.squeeze(0)
        z_flat = z_test.squeeze(0)

        dist_h = torch.cdist(h_flat, h_flat)
        dist_z = torch.cdist(z_flat, z_flat)

        # FIX (nan guard): corrcoef of a constant vector is nan. If the fold has
        # collapsed (all pairwise z-distances ~0), report 0.0 instead of nan.
        dh_v, dz_v = dist_h.view(-1), dist_z.view(-1)
        if dz_v.std() < 1e-10:
            corr = 0.0
        else:
            corr = torch.corrcoef(torch.stack([dh_v, dz_v]))[0, 1].item()

        # Exclude diagonal zeros to compute true non-zero dynamic range
        dist_h_nonzero = dist_h[dist_h > 0]
        dist_z_nonzero = dist_z[dist_z > 0]
        comp_h = (dist_h_nonzero.max() / dist_h_nonzero.min()).item() if len(dist_h_nonzero) > 0 else 1.0
        comp_z = (dist_z_nonzero.max() / dist_z_nonzero.min()).item() if len(dist_z_nonzero) > 0 else 1.0

    print(f"[*] Distance Correlation (Euclidean vs Folded): {corr:.4f}")
    print(f"    -> (1.0 = linear/unfolded, < 0.6 = heavily folded manifold)")
    print(f"[*] Euclidean Distance Dynamic Range: {comp_h:.2f}x")
    print(f"[*] Folded Space Dynamic Range:      {comp_z:.2f}x")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
