"""
JAX Transformer implementation mirroring the torch.nn.Transformer API.

Classes
-------
MultiHeadAttention
    Scaled dot-product multi-head attention.
TransformerEncoderLayer
    Single encoder layer (self-attention + FFN).
TransformerDecoderLayer
    Single decoder layer (self-attention + cross-attention + FFN).
TransformerEncoder
    Stack of N encoder layers with optional norm.
TransformerDecoder
    Stack of N decoder layers with optional norm.
Transformer
    Full encoder-decoder transformer (mirrors torch.nn.Transformer).

All modules are plain dataclasses whose parameters live in ordinary
Python / NumPy / JAX arrays — no magic framework state.  Call
`module.init_params(rng_key)` to get an initial parameter dict, then
pass that dict (together with inputs) to `module.forward(params, ...)`.

Usage example
-------------
    import jax
    import jax.numpy as jnp
    from transformer_jax import Transformer

    rng = jax.random.PRNGKey(0)

    model = Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,          # only applied when training=True
    )

    params = model.init_params(rng)

    src = jax.random.normal(rng, (10, 2, 512))   # (src_len, batch, d_model)
    tgt = jax.random.normal(rng, ( 5, 2, 512))   # (tgt_len, batch, d_model)

    out = model.forward(params, src, tgt, training=False)
    print(out.shape)   # (5, 2, 512)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
from jax import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_heads(x: jnp.ndarray, nhead: int) -> jnp.ndarray:
    """(seq, batch, d_model) -> (batch, nhead, seq, head_dim)"""
    seq, batch, d_model = x.shape
    head_dim = d_model // nhead
    x = x.reshape(seq, batch, nhead, head_dim)
    return x.transpose(1, 2, 0, 3)           # (batch, nhead, seq, head_dim)


def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    """(batch, nhead, seq, head_dim) -> (seq, batch, d_model)"""
    batch, nhead, seq, head_dim = x.shape
    x = x.transpose(2, 0, 1, 3)              # (seq, batch, nhead, head_dim)
    return x.reshape(seq, batch, nhead * head_dim)


def _linear(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """y = x @ W^T + b"""
    return x @ params["weight"].T + params["bias"]


def _layer_norm(params: dict, x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = x.mean(-1, keepdims=True)
    var  = x.var( -1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return params["weight"] * x_hat + params["bias"]


def _dropout(rng: Optional[jnp.ndarray], x: jnp.ndarray,
             rate: float, training: bool) -> jnp.ndarray:
    if not training or rate == 0.0:
        return x
    keep = random.bernoulli(rng, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


# ---------------------------------------------------------------------------
# Parameter initialisers
# ---------------------------------------------------------------------------

def _init_linear(rng: jnp.ndarray, in_dim: int, out_dim: int) -> dict:
    k1, _ = random.split(rng)
    std = math.sqrt(2.0 / (in_dim + out_dim))
    return {
        "weight": random.normal(k1, (out_dim, in_dim)) * std,
        "bias":   jnp.zeros(out_dim),
    }


def _init_layer_norm(dim: int) -> dict:
    return {"weight": jnp.ones(dim), "bias": jnp.zeros(dim)}


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    key_padding_mask: Optional[jnp.ndarray] = None,
    dropout_rng: Optional[jnp.ndarray] = None,
    dropout: float = 0.0,
    training: bool = False,
) -> jnp.ndarray:
    """
    Parameters
    ----------
    q, k, v : (batch, nhead, seq, head_dim)
    mask : additive float attention mask broadcastable to
           (batch, nhead, tgt_seq, src_seq) — -inf blocks a position.
           This follows PyTorch's convention; we convert it to JAX's
           boolean mask (True = attend) internally.
    key_padding_mask : bool array (batch, src_seq);
                       True = *ignore* that key (PyTorch convention).
                       Inverted to True = attend before passing to JAX.

    Returns
    -------
    (batch, nhead, tgt_seq, head_dim)
    """
    # --- Build a single boolean mask (True = attend) for JAX ----------

    bool_mask: Optional[jnp.ndarray] = None

    if mask is not None:
        # Additive float mask: 0 = attend, -inf = block.
        # Convert to bool: finite values → True, -inf → False.
        # Shape: broadcastable to (batch, nhead, tgt, src)
        bool_mask = jnp.isfinite(mask)  # True where we should attend

    if key_padding_mask is not None:
        # (batch, src) → (batch, 1, 1, src);  True in PyTorch = *ignore*
        kpm = ~key_padding_mask[:, None, None, :]   # flip: True = attend
        bool_mask = kpm if bool_mask is None else (bool_mask & kpm)

    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    # (batch, nhead, tgt, src)
    logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale

    if bool_mask is not None:
        logits = jnp.where(bool_mask, logits, jnp.finfo(logits.dtype).min)

    weights = jax.nn.softmax(logits, axis=-1)

    if training and dropout > 0.0:
        assert dropout_rng is not None, \
            "dropout_rng must be provided when training with dropout > 0"
        keep_prob = 1.0 - dropout
        keep = random.bernoulli(dropout_rng, keep_prob, weights.shape)
        weights = jnp.where(keep, weights / keep_prob, 0.0)

    # (batch, nhead, tgt, head_dim)
    return jnp.einsum("bhts,bhsd->bhtd", weights, v)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

@dataclass
class MultiHeadAttention:
    """
    Multi-head attention module mirroring torch.nn.MultiheadAttention.

    Parameters
    ----------
    embed_dim : int   – total model dimension (d_model)
    num_heads : int   – number of attention heads
    dropout   : float – attention-weight dropout rate
    bias      : bool  – whether to use bias in projection layers
    kdim      : int   – key projection input dim  (default = embed_dim)
    vdim      : int   – value projection input dim (default = embed_dim)
    """

    embed_dim: int
    num_heads: int
    dropout:   float = 0.0
    bias:      bool  = True
    kdim:      Optional[int] = None
    vdim:      Optional[int] = None

    def __post_init__(self):
        self.kdim = self.kdim or self.embed_dim
        self.vdim = self.vdim or self.embed_dim
        assert self.embed_dim % self.num_heads == 0, \
            "embed_dim must be divisible by num_heads"

    # ------------------------------------------------------------------
    def init_params(self, rng: jnp.ndarray) -> dict:
        rngs = random.split(rng, 4)
        return {
            "q_proj": _init_linear(rngs[0], self.embed_dim, self.embed_dim),
            "k_proj": _init_linear(rngs[1], self.kdim,      self.embed_dim),
            "v_proj": _init_linear(rngs[2], self.vdim,      self.embed_dim),
            "out_proj": _init_linear(rngs[3], self.embed_dim, self.embed_dim),
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        params: dict,
        query: jnp.ndarray,
        key:   jnp.ndarray,
        value: jnp.ndarray,
        attn_mask:       Optional[jnp.ndarray] = None,
        key_padding_mask: Optional[jnp.ndarray] = None,
        rng: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Parameters
        ----------
        query : (tgt_len, batch, embed_dim)
        key   : (src_len, batch, kdim)
        value : (src_len, batch, vdim)

        Returns
        -------
        (tgt_len, batch, embed_dim)
        """
        q = _linear(params["q_proj"], query)   # (tgt, batch, embed_dim)
        k = _linear(params["k_proj"], key)
        v = _linear(params["v_proj"], value)

        q = _split_heads(q, self.num_heads)    # (batch, nhead, tgt, head_dim)
        k = _split_heads(k, self.num_heads)
        v = _split_heads(v, self.num_heads)

        attn_out = scaled_dot_product_attention(
            q, k, v,
            mask=attn_mask,
            key_padding_mask=key_padding_mask,
            dropout_rng=rng,
            dropout=self.dropout,
            training=training,
        )                                       # (batch, nhead, tgt, head_dim)

        attn_out = _merge_heads(attn_out)       # (tgt, batch, embed_dim)
        return _linear(params["out_proj"], attn_out)


# ---------------------------------------------------------------------------
# TransformerEncoderLayer
# ---------------------------------------------------------------------------

@dataclass
class TransformerEncoderLayer:
    """
    Single encoder layer:
        x = LayerNorm(x + SelfAttn(x))
        x = LayerNorm(x + FFN(x))

    norm_first=True applies pre-norm (pre-LN transformer).
    """

    d_model:         int
    nhead:           int
    dim_feedforward: int   = 2048
    dropout:         float = 0.1
    norm_first:      bool  = False  # pre-LN when True

    def __post_init__(self):
        self.self_attn = MultiHeadAttention(self.d_model, self.nhead, self.dropout)

    # ------------------------------------------------------------------
    def init_params(self, rng: jnp.ndarray) -> dict:
        rngs = random.split(rng, 4)
        return {
            "self_attn": self.self_attn.init_params(rngs[0]),
            "linear1":   _init_linear(rngs[1], self.d_model, self.dim_feedforward),
            "linear2":   _init_linear(rngs[2], self.dim_feedforward, self.d_model),
            "norm1":     _init_layer_norm(self.d_model),
            "norm2":     _init_layer_norm(self.d_model),
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        params: dict,
        src: jnp.ndarray,
        src_mask: Optional[jnp.ndarray] = None,
        src_key_padding_mask: Optional[jnp.ndarray] = None,
        rng: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        rng1, rng2, rng3 = (random.split(rng, 3) if rng is not None
                            else (None, None, None))

        if self.norm_first:
            # Pre-LN
            normed = _layer_norm(params["norm1"], src)
            attn_out = self.self_attn.forward(
                params["self_attn"], normed, normed, normed,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                rng=rng1, training=training,
            )
            src = src + _dropout(rng2, attn_out, self.dropout, training)

            normed = _layer_norm(params["norm2"], src)
            ff = jax.nn.relu(_linear(params["linear1"], normed))
            ff = _dropout(rng3, ff, self.dropout, training)
            ff = _linear(params["linear2"], ff)
            src = src + _dropout(rng3, ff, self.dropout, training)
        else:
            # Post-LN
            attn_out = self.self_attn.forward(
                params["self_attn"], src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                rng=rng1, training=training,
            )
            src = _layer_norm(params["norm1"],
                              src + _dropout(rng2, attn_out, self.dropout, training))

            ff = jax.nn.relu(_linear(params["linear1"], src))
            ff = _dropout(rng3, ff, self.dropout, training)
            ff = _linear(params["linear2"], ff)
            src = _layer_norm(params["norm2"],
                              src + _dropout(rng3, ff, self.dropout, training))

        return src


# ---------------------------------------------------------------------------
# TransformerDecoderLayer
# ---------------------------------------------------------------------------

@dataclass
class TransformerDecoderLayer:
    """
    Single decoder layer:
        x = LayerNorm(x + SelfAttn(x))
        x = LayerNorm(x + CrossAttn(x, memory))
        x = LayerNorm(x + FFN(x))
    """

    d_model:         int
    nhead:           int
    dim_feedforward: int   = 2048
    dropout:         float = 0.1
    norm_first:      bool  = False

    def __post_init__(self):
        self.self_attn  = MultiHeadAttention(self.d_model, self.nhead, self.dropout)
        self.cross_attn = MultiHeadAttention(self.d_model, self.nhead, self.dropout)

    # ------------------------------------------------------------------
    def init_params(self, rng: jnp.ndarray) -> dict:
        rngs = random.split(rng, 5)
        return {
            "self_attn":  self.self_attn.init_params(rngs[0]),
            "cross_attn": self.cross_attn.init_params(rngs[1]),
            "linear1":    _init_linear(rngs[2], self.d_model, self.dim_feedforward),
            "linear2":    _init_linear(rngs[3], self.dim_feedforward, self.d_model),
            "norm1":      _init_layer_norm(self.d_model),
            "norm2":      _init_layer_norm(self.d_model),
            "norm3":      _init_layer_norm(self.d_model),
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        params: dict,
        tgt: jnp.ndarray,
        memory: jnp.ndarray,
        tgt_mask:              Optional[jnp.ndarray] = None,
        memory_mask:           Optional[jnp.ndarray] = None,
        tgt_key_padding_mask:  Optional[jnp.ndarray] = None,
        memory_key_padding_mask: Optional[jnp.ndarray] = None,
        rng: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        rngs = (random.split(rng, 6) if rng is not None
                else [None] * 6)

        if self.norm_first:
            # --- self attention ---
            normed = _layer_norm(params["norm1"], tgt)
            sa = self.self_attn.forward(
                params["self_attn"], normed, normed, normed,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                rng=rngs[0], training=training,
            )
            tgt = tgt + _dropout(rngs[1], sa, self.dropout, training)

            # --- cross attention ---
            normed = _layer_norm(params["norm2"], tgt)
            ca = self.cross_attn.forward(
                params["cross_attn"], normed, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                rng=rngs[2], training=training,
            )
            tgt = tgt + _dropout(rngs[3], ca, self.dropout, training)

            # --- feed-forward ---
            normed = _layer_norm(params["norm3"], tgt)
            ff = jax.nn.relu(_linear(params["linear1"], normed))
            ff = _dropout(rngs[4], ff, self.dropout, training)
            ff = _linear(params["linear2"], ff)
            tgt = tgt + _dropout(rngs[5], ff, self.dropout, training)
        else:
            # --- self attention ---
            sa = self.self_attn.forward(
                params["self_attn"], tgt, tgt, tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                rng=rngs[0], training=training,
            )
            tgt = _layer_norm(params["norm1"],
                              tgt + _dropout(rngs[1], sa, self.dropout, training))

            # --- cross attention ---
            ca = self.cross_attn.forward(
                params["cross_attn"], tgt, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                rng=rngs[2], training=training,
            )
            tgt = _layer_norm(params["norm2"],
                              tgt + _dropout(rngs[3], ca, self.dropout, training))

            # --- feed-forward ---
            ff = jax.nn.relu(_linear(params["linear1"], tgt))
            ff = _dropout(rngs[4], ff, self.dropout, training)
            ff = _linear(params["linear2"], ff)
            tgt = _layer_norm(params["norm3"],
                              tgt + _dropout(rngs[5], ff, self.dropout, training))

        return tgt


# ---------------------------------------------------------------------------
# TransformerEncoder
# ---------------------------------------------------------------------------

@dataclass
class TransformerEncoder:
    """Stack of N TransformerEncoderLayers with an optional final LayerNorm."""

    encoder_layer: TransformerEncoderLayer
    num_layers: int
    norm: bool = False    # set True to apply a final LayerNorm

    # ------------------------------------------------------------------
    def init_params(self, rng: jnp.ndarray) -> dict:
        rngs = random.split(rng, self.num_layers + 1)
        params = {
            f"layer_{i}": self.encoder_layer.init_params(rngs[i])
            for i in range(self.num_layers)
        }
        if self.norm:
            params["norm"] = _init_layer_norm(self.encoder_layer.d_model)
        return params

    # ------------------------------------------------------------------
    def forward(
        self,
        params: dict,
        src: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        src_key_padding_mask: Optional[jnp.ndarray] = None,
        rng: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        rngs = (random.split(rng, self.num_layers) if rng is not None
                else [None] * self.num_layers)

        x = src
        for i in range(self.num_layers):
            x = self.encoder_layer.forward(
                params[f"layer_{i}"], x,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                rng=rngs[i], training=training,
            )
        if self.norm:
            x = _layer_norm(params["norm"], x)
        return x


# ---------------------------------------------------------------------------
# TransformerDecoder
# ---------------------------------------------------------------------------

@dataclass
class TransformerDecoder:
    """Stack of N TransformerDecoderLayers with an optional final LayerNorm."""

    decoder_layer: TransformerDecoderLayer
    num_layers: int
    norm: bool = False

    # ------------------------------------------------------------------
    def init_params(self, rng: jnp.ndarray) -> dict:
        rngs = random.split(rng, self.num_layers + 1)
        params = {
            f"layer_{i}": self.decoder_layer.init_params(rngs[i])
            for i in range(self.num_layers)
        }
        if self.norm:
            params["norm"] = _init_layer_norm(self.decoder_layer.d_model)
        return params

    # ------------------------------------------------------------------
    def forward(
        self,
        params: dict,
        tgt: jnp.ndarray,
        memory: jnp.ndarray,
        tgt_mask: Optional[jnp.ndarray] = None,
        memory_mask: Optional[jnp.ndarray] = None,
        tgt_key_padding_mask: Optional[jnp.ndarray] = None,
        memory_key_padding_mask: Optional[jnp.ndarray] = None,
        rng: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        rngs = (random.split(rng, self.num_layers) if rng is not None
                else [None] * self.num_layers)

        x = tgt
        for i in range(self.num_layers):
            x = self.decoder_layer.forward(
                params[f"layer_{i}"], x, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                rng=rngs[i], training=training,
            )
        if self.norm:
            x = _layer_norm(params["norm"], x)
        return x


# ---------------------------------------------------------------------------
# Transformer  (top-level, mirrors torch.nn.Transformer)
# ---------------------------------------------------------------------------

@dataclass
class Transformer:
    """
    Full encoder-decoder Transformer, API-compatible with torch.nn.Transformer.

    Parameters
    ----------
    d_model           : int   – model dimension (default 512)
    nhead             : int   – attention heads (default 8)
    num_encoder_layers: int   – encoder depth (default 6)
    num_decoder_layers: int   – decoder depth (default 6)
    dim_feedforward   : int   – FFN hidden dim (default 2048)
    dropout           : float – dropout probability (default 0.1)
    norm_first        : bool  – use pre-LN if True (default False)
    """

    d_model:            int   = 512
    nhead:              int   = 8
    num_encoder_layers: int   = 6
    num_decoder_layers: int   = 6
    dim_feedforward:    int   = 2048
    dropout:            float = 0.1
    norm_first:         bool  = False

    def __post_init__(self):
        enc_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            norm_first=self.norm_first,
        )
        dec_layer = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            norm_first=self.norm_first,
        )
        self.encoder = TransformerEncoder(enc_layer, self.num_encoder_layers, norm=True)
        self.decoder = TransformerDecoder(dec_layer, self.num_decoder_layers, norm=True)

    # ------------------------------------------------------------------
    def init_params(self, rng: jnp.ndarray) -> dict:
        rng_enc, rng_dec = random.split(rng)
        return {
            "encoder": self.encoder.init_params(rng_enc),
            "decoder": self.decoder.init_params(rng_dec),
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        params: dict,
        src: jnp.ndarray,
        tgt: jnp.ndarray,
        src_mask:               Optional[jnp.ndarray] = None,
        tgt_mask:               Optional[jnp.ndarray] = None,
        memory_mask:            Optional[jnp.ndarray] = None,
        src_key_padding_mask:   Optional[jnp.ndarray] = None,
        tgt_key_padding_mask:   Optional[jnp.ndarray] = None,
        memory_key_padding_mask: Optional[jnp.ndarray] = None,
        rng: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Parameters
        ----------
        src : (src_len, batch, d_model)
        tgt : (tgt_len, batch, d_model)

        Masks follow PyTorch conventions:
        - attn_mask      : additive float mask, -inf blocks attention
        - key_padding_mask: bool, True = ignore that key position

        Returns
        -------
        output : (tgt_len, batch, d_model)
        """
        rng_enc, rng_dec = (random.split(rng) if rng is not None else (None, None))

        memory = self.encoder.forward(
            params["encoder"], src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            rng=rng_enc, training=training,
        )

        output = self.decoder.forward(
            params["decoder"], tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            rng=rng_dec, training=training,
        )

        return output

    # ------------------------------------------------------------------
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> jnp.ndarray:
        """
        Causal mask for autoregressive decoding.
        Returns an additive mask of shape (sz, sz) where upper-triangle
        positions are -inf and the rest are 0.
        Mirrors torch.nn.Transformer.generate_square_subsequent_mask.
        """
        mask = jnp.triu(jnp.full((sz, sz), -jnp.inf), k=1)
        return mask


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)

    model = Transformer(
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )

    params = model.init_params(rng)

    src = jax.random.normal(rng, (10, 2, 64))  # (src_len, batch, d_model)
    tgt = jax.random.normal(rng, ( 5, 2, 64))  # (tgt_len, batch, d_model)

    # Causal mask for the target
    tgt_mask = Transformer.generate_square_subsequent_mask(tgt.shape[0])

    out = model.forward(params, src, tgt, tgt_mask=tgt_mask, training=False)
    print("Output shape:", out.shape)   # (5, 2, 64)
    assert out.shape == tgt.shape, "Shape mismatch!"
    print("Smoke test passed ✓")

    # JIT-compile the forward pass
    @jax.jit
    def jit_forward(params, src, tgt):
        return model.forward(params, src, tgt, tgt_mask=tgt_mask, training=False)

    out_jit = jit_forward(params, src, tgt)
    print("JIT output shape:", out_jit.shape)
    print("JIT smoke test passed ✓")
