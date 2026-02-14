import torch
import types
from typing import Optional, Callable, Tuple, Dict, Any, Union, TYPE_CHECKING, TypeVar, List
import re

import folder_paths
import os
import json
import math

import comfy.samplers
import comfy.sample
import comfy.sampler_helpers
import comfy.utils
import comfy.model_management
import comfy.supported_models
from comfy.ldm.modules.attention import optimized_attention as comfy_optimized_attention, attention_pytorch as comfy_attention_pytorch

from comfy.cli_args import args

from .flux.redux import ReReduxImageEncoder
from comfy.ldm.flux.redux import ReduxImageEncoder

from comfy.ldm.flux.model  import Flux
from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock

from .flux.model  import ReFlux
from .flux.layers import SingleStreamBlock as ReSingleStreamBlock, DoubleStreamBlock as ReDoubleStreamBlock

from comfy.ldm.flux.model  import Flux
from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock

from comfy.ldm.hidream.model import HiDreamImageTransformer2DModel
from comfy.ldm.hidream.model import HiDreamImageBlock, HiDreamImageSingleTransformerBlock, HiDreamImageTransformerBlock, HiDreamAttention

from .hidream.model import HDModel
from .hidream.model import HDBlock, HDBlockDouble, HDBlockSingle, HDAttention, HDMoEGate, HDMOEFeedForwardSwiGLU, HDFeedForwardSwiGLU, HDLastLayer

from comfy.ldm.modules.diffusionmodules.mmdit import OpenAISignatureMMDITWrapper, JointBlock
from .sd35.mmdit import ReOpenAISignatureMMDITWrapper, ReJointBlock

from comfy.ldm.aura.mmdit import MMDiT, DiTBlock, MMDiTBlock, SingleAttention, DoubleAttention
from .aura.mmdit import ReMMDiT, ReDiTBlock, ReMMDiTBlock, ReSingleAttention, ReDoubleAttention

from comfy.ldm.wan.model import WanAttentionBlock, WanI2VCrossAttention, WanModel, WanSelfAttention, WanT2VCrossAttention
from .wan.model import ReWanAttentionBlock, ReWanI2VCrossAttention, ReWanModel, ReWanRawSelfAttention, ReWanSelfAttention, ReWanSlidingSelfAttention, ReWanT2VSlidingCrossAttention, ReWanT2VCrossAttention, ReWanT2VRawCrossAttention

from comfy.ldm.chroma.model import Chroma
from comfy.ldm.chroma.layers import SingleStreamBlock as ChromaSingleStreamBlock, DoubleStreamBlock as ChromaDoubleStreamBlock

from .chroma.model import ReChroma
from .chroma.layers import ReChromaSingleStreamBlock, ReChromaDoubleStreamBlock

from comfy.ldm.lightricks.model import LTXVModel
#from comfy.ldm.chroma.layers import SingleStreamBlock as ChromaSingleStreamBlock, DoubleStreamBlock as ChromaDoubleStreamBlock

from .lightricks.model import ReLTXVModel
#from .chroma.layers import ReChromaSingleStreamBlock, ReChromaDoubleStreamBlock

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel, ResBlock
from comfy.ldm.modules.attention import SpatialTransformer, BasicTransformerBlock, CrossAttention
from .sd.openaimodel import ReUNetModel, ReResBlock
from .sd.attention import ReBasicTransformerBlock, ReCrossAttention, ReSpatialTransformer

from .latents import get_orthogonal, get_cosine_similarity
from .style_transfer import StyleWCT, WaveletStyleWCT, Retrojector, StyleMMDiT_Model
from .res4lyf import RESplain

from .helper import parse_range_string

from comfy.model_sampling import *

ANIMA_MODEL_CLASS = getattr(comfy.supported_models, "Anima", None)
try:
    from comfy.ldm.anima.model import Anima as ANIMA_DIFFUSION_MODEL_CLASS
except Exception:
    ANIMA_DIFFUSION_MODEL_CLASS = None


def _scalar_or_default(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.reshape(-1)[0].item())
    try:
        return float(value)
    except Exception:
        return default


def _extract_anima_cross_attn_mask(mask: Optional[torch.Tensor], text_len: int, k_tokens: int) -> Optional[torch.Tensor]:
    """Extract only the cross-attention columns from a SplitAttentionMask and
    pad/trim to match the actual context length (k_tokens).  SplitAttentionMask
    stores ``[cross_attn(img, text) | self_attn(img, img)]`` concatenated along
    the last dimension.  Taking ``mask[:, :k_tokens]`` when k_tokens > text_len
    would leak self-attention values into cross-attention positions, corrupting
    the regional conditioning signal.  This helper avoids that by slicing to
    ``text_len`` first, then zero-padding to ``k_tokens``."""
    if mask is None or text_len <= 0:
        return None
    cross = mask[..., :text_len]                    # only genuine cross-attn cols
    if cross.shape[-1] >= k_tokens:
        return cross[..., :k_tokens]                # trim if context shrank
    # pad remaining columns (adapter padding tokens) with zeros → blocked
    pad_cols = k_tokens - cross.shape[-1]
    pad_shape = list(cross.shape[:-1]) + [pad_cols]
    pad = torch.zeros(*pad_shape, dtype=cross.dtype, device=cross.device)
    return torch.cat([cross, pad], dim=-1)


def _slice_anima_mask_for_attention(mask: Optional[torch.Tensor], q_tokens: int, k_tokens: int, attn_kind: str):
    if mask is None:
        return None

    # Keep explicit batch masks [B, Q, K] when present.
    while mask.ndim > 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    if mask.ndim not in (2, 3):
        return None

    rows, cols = mask.shape[-2], mask.shape[-1]
    sliced = None

    if rows == q_tokens and cols == k_tokens:
        sliced = mask
    elif attn_kind == "cross":
        if rows == q_tokens and cols >= k_tokens:
            sliced = mask[..., :k_tokens]
        elif rows >= q_tokens and cols == k_tokens:
            sliced = mask[-q_tokens:, :]
        elif rows >= q_tokens and cols >= k_tokens:
            sliced = mask[-q_tokens:, :k_tokens]
    else:
        if rows == q_tokens and cols >= k_tokens:
            sliced = mask[..., -k_tokens:]
        elif rows >= q_tokens and cols == k_tokens:
            sliced = mask[-q_tokens:, :]
        elif rows >= q_tokens and cols >= k_tokens:
            sliced = mask[-q_tokens:, -k_tokens:]

    if sliced is not None and sliced.shape[-2] == q_tokens and sliced.shape[-1] < k_tokens:
        # Anima text preprocessing can pad context length (e.g. to 512 tokens).
        # Regional masks are built from pre-pad token lengths; pad mask columns
        # so masking still applies instead of falling back to unmasked attention.
        pad_cols = k_tokens - sliced.shape[-1]
        pad_shape = list(sliced.shape[:-1]) + [pad_cols]
        pad = torch.zeros(*pad_shape, dtype=sliced.dtype, device=sliced.device)
        sliced = torch.cat([sliced, pad], dim=-1)

    if sliced is None or sliced.shape[-2] != q_tokens or sliced.shape[-1] != k_tokens:
        return None
    return sliced


def _reshape_anima_qkv_for_attention(t: torch.Tensor) -> torch.Tensor:
    # Match comfy.ldm.cosmos.predict2.torch_attention_op layout flattening:
    # [B, ..., H, D] -> [B, H, S, D], where S is flattened sequence axes.
    in_shape = t.shape
    return t.movedim(-2, 1).reshape(in_shape[0], in_shape[-2], -1, in_shape[-1])


def _anima_attention_op(q, k, v, transformer_options=None, attn_kind="cross", fallback_attn_op=None):
    if transformer_options is None:
        transformer_options = {}

    # For Anima, regional masking on cross-attention is the stable path.
    # Self-attention masking can collapse output quality into noise.
    if attn_kind != "cross":
        if fallback_attn_op is not None:
            return fallback_attn_op(q, k, v, transformer_options=transformer_options)

        q_attn = _reshape_anima_qkv_for_attention(q)
        k_attn = _reshape_anima_qkv_for_attention(k)
        v_attn = _reshape_anima_qkv_for_attention(v)
        return comfy_optimized_attention(
            q_attn,
            k_attn,
            v_attn,
            q_attn.shape[1],
            skip_reshape=True,
            transformer_options=transformer_options,
        )

    q_attn = _reshape_anima_qkv_for_attention(q)
    k_attn = _reshape_anima_qkv_for_attention(k)
    v_attn = _reshape_anima_qkv_for_attention(v)

    mask = transformer_options.get("_res4lyf_anima_attn_mask")
    mask = _slice_anima_mask_for_attention(mask, q_attn.shape[2], k_attn.shape[2], attn_kind)

    if mask is None:
        if fallback_attn_op is not None:
            return fallback_attn_op(q, k, v, transformer_options=transformer_options)
        return comfy_optimized_attention(
            q_attn,
            k_attn,
            v_attn,
            q_attn.shape[1],
            skip_reshape=True,
            transformer_options=transformer_options,
        )

    q_dtype = q_attn.dtype if q_attn.dtype.is_floating_point else torch.float32
    mask_device = q_attn.device

    # Preserve soft regional mask strengths (gradient masks) instead of
    # collapsing everything to bool. This lets regional weight/floor schedules
    # affect Anima similarly to other model integrations.
    if mask.dtype == torch.bool:
        allow = mask.to(device=mask_device, dtype=q_dtype)
    else:
        allow = torch.clamp(mask.to(device=mask_device, dtype=q_dtype), 0.0, 1.0)

    weight = abs(_scalar_or_default(transformer_options.get("regional_conditioning_weight"), 1.0))
    floor = abs(_scalar_or_default(transformer_options.get("regional_conditioning_floor"), 0.0))
    weight = min(max(weight, 0.0), 1.0)
    floor = min(max(floor, 0.0), 1.0)

    # Blend toward unmasked attention as weight approaches zero.
    allow = floor + (1.0 - floor) * allow
    allow = 1.0 - weight * (1.0 - allow)
    allow = torch.clamp(allow, 1e-6, 1.0)

    mask_strength = abs(_scalar_or_default(transformer_options.get("_res4lyf_anima_mask_strength"), 4.0))
    mask_strength = max(mask_strength, 1.0)
    mask_bias = torch.log(allow) * mask_strength

    # Blackwell + xformers currently fails when passing tensor attn_bias.
    # Force PyTorch SDPA path for masked regional attention.
    return comfy_attention_pytorch(
        q_attn,
        k_attn,
        v_attn,
        q_attn.shape[1],
        mask=mask_bias,
        skip_reshape=True,
        transformer_options=transformer_options,
    )

class PRED:
    TYPE_VP    = {CONST}
    TYPE_VE    = {EPS}
    TYPE_VPRED = {V_PREDICTION, EDM}
    TYPE_X0    = {X0, IMG_TO_IMG}
    
    TYPE_ALL   = TYPE_VP | TYPE_VE | TYPE_VPRED | TYPE_X0
    
    @classmethod
    def get_type(cls, model_sampling):
        bases = type(model_sampling).__mro__
        return next((v_type for v_type in bases if v_type in cls.TYPE_ALL), None)


def time_snr_shift_exponential(alpha, t):
    return math.exp(alpha) / (math.exp(alpha) + (1 / t - 1) ** 1.0)

def time_snr_shift_linear(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


COMPILE_MODES = ["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"]


class TorchCompileModels: 
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model"                   : ("MODEL",),
                    "backend"                 : (["inductor", "cudagraphs"],),
                    "fullgraph"               : ("BOOLEAN",                    {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode"                    : (COMPILE_MODES,                {"default": "default"}),
                    "dynamic"                 : ("BOOLEAN",                    {"default": False, "tooltip": "Enable dynamic mode"}),
                    "dynamo_cache_size_limit" : ("INT",                        {"default": 64, "min": 0, "max": 1024,       "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                    "triton_max_block_x"      : ("INT",                        {"default": 0,  "min": 0, "max": 4294967296, "step": 1})
                }}
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/model_patches"

    def main(self,
            model,
            backend       = "inductor",
            mode          = "default",
            fullgraph     = False,
            dynamic       = False,
            dynamo_cache_size_limit = 64,
            triton_max_block_x = 0,
            ):
        
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        
        if triton_max_block_x > 0:
            import os
            os.environ["TRITON_MAX_BLOCK_X"] = "4096"
        
        if not self._compiled:
            try:
                if hasattr(diffusion_model, "double_blocks"):
                    for i, block in enumerate(diffusion_model.double_blocks):
                        m.add_object_patch(f"diffusion_model.double_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                    self._compiled = True
                    
                if hasattr(diffusion_model, "single_blocks"):
                    for i, block in enumerate(diffusion_model.single_blocks):
                        m.add_object_patch(f"diffusion_model.single_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                    self._compiled = True
                    
                if hasattr(diffusion_model, "double_layers"):
                    for i, block in enumerate(diffusion_model.double_layers):
                        m.add_object_patch(f"diffusion_model.double_layers.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                    self._compiled = True
                    
                if hasattr(diffusion_model, "single_layers"):
                    for i, block in enumerate(diffusion_model.single_layers):
                        m.add_object_patch(f"diffusion_model.single_layers.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                    self._compiled = True
                    
                    
                    
                if hasattr(diffusion_model, "double_stream_blocks"):
                    for i, block in enumerate(diffusion_model.double_stream_blocks):
                        m.add_object_patch(f"diffusion_model.double_stream_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                    self._compiled = True
                    
                if hasattr(diffusion_model, "single_stream_blocks"):
                    for i, block in enumerate(diffusion_model.single_stream_blocks):
                        m.add_object_patch(f"diffusion_model.single_stream_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                    self._compiled = True
                                        
                                        
                                        
                if hasattr(diffusion_model, "joint_blocks"):
                    for i, block in enumerate(diffusion_model.joint_blocks):
                        m.add_object_patch(f"diffusion_model.joint_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                    self._compiled = True
                    
                if hasattr(diffusion_model, "blocks"):
                    for i, block in enumerate(diffusion_model.blocks):
                        m.add_object_patch(f"diffusion_model.blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                    self._compiled = True
                    
                if self._compiled == False:
                    raise RuntimeError("Model not compiled. Verify that this is a Flux, SD3.5, HiDream, WAN, or Aura model!")
                
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model. Verify that this is a Flux, SD3.5, HiDream, WAN, or Aura model!")
        
        return (m, )


class ReWanPatcherAdvanced:
    def __init__(self):
        self.sliding_window_size = 0
        self.sliding_window_self_attn = "false"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model"                    : ("MODEL",),
                #"self_attn_blocks" : ("STRING",  {"default": "0,1,2,3,4,5,6,7,8,9,", "multiline": True}),
                "self_attn_blocks"         : ("STRING",  {"default": "all", "multiline": True}),
                "cross_attn_blocks"        : ("STRING",  {"default": "all", "multiline": True}),
                "enable"                   : ("BOOLEAN", {"default": True}),
                "sliding_window_self_attn" : (['false', 'standard', 'circular'], {"default": "false"}),
                "sliding_window_frames"    : ("INT",   {"default": 60,   "min": 4,    "max": 0xffffffffffffffff, "step": 4, "tooltip": "How many real frames each frame sees. Divide frames by 4 to get real frames."}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"

    def main(self, model, self_attn_blocks, cross_attn_blocks, sliding_window_self_attn="false", sliding_window_frames=60, style_dtype="float32", enable=True, force=False):
        
        style_dtype = getattr(torch, style_dtype) if style_dtype != "default" else None
        model.model.diffusion_model.style_dtype = style_dtype
        model.model.diffusion_model.proj_weights = None
        model.model.diffusion_model.y0_adain_embed = None
        
        sliding_window_size = sliding_window_frames // 4
        
        self_attn_blocks  = parse_range_string(self_attn_blocks)
        cross_attn_blocks = parse_range_string(cross_attn_blocks)
        
        T2V = type(model.model.model_config) is comfy.supported_models.WAN21_T2V
        
        if (enable or force) and model.model.diffusion_model.__class__ == WanModel:
            m = model.clone()
            m.model.diffusion_model.__class__     = ReWanModel
            m.model.diffusion_model.threshold_inv = False
            
            for i, block in enumerate(m.model.diffusion_model.blocks):
                block.__class__            = ReWanAttentionBlock
                if i in self_attn_blocks:
                    if sliding_window_self_attn != "false":
                        block.self_attn.__class__ = ReWanSlidingSelfAttention
                        block.self_attn.winderz = sliding_window_size
                        block.self_attn.winderz_type = sliding_window_self_attn
                    else:
                        block.self_attn.__class__  = ReWanSelfAttention
                        block.self_attn.winderz_type = "false"
                else:
                    block.self_attn.__class__  = ReWanRawSelfAttention
                if i in cross_attn_blocks:
                    if T2V:
                        if False: #sliding_window_self_attn != "false":
                            block.cross_attn.__class__ = ReWanT2VSlidingCrossAttention
                            block.cross_attn.winderz = sliding_window_size
                            block.cross_attn.winderz_type = sliding_window_self_attn
                        else:
                            block.cross_attn.__class__ = ReWanT2VCrossAttention

                    else:
                        block.cross_attn.__class__ = ReWanI2VCrossAttention
                block.idx            = i
                block.self_attn.idx  = i
                block.cross_attn.idx = i # 40 total blocks (i == 39)
                
        elif enable and (sliding_window_self_attn != self.sliding_window_self_attn or sliding_window_size != self.sliding_window_size) and model.model.diffusion_model.__class__ == ReWanModel:
            m = model.clone()
            
            for i, block in enumerate(m.model.diffusion_model.blocks):
                if i in self_attn_blocks:
                    block.self_attn.winderz = sliding_window_size
                    block.self_attn.winderz_type = sliding_window_self_attn
        
        elif not enable and model.model.diffusion_model.__class__ == ReWanModel:
            m = model.clone()
            m.model.diffusion_model.__class__ = WanModel
            
            for i, block in enumerate(m.model.diffusion_model.blocks):
                block.__class__            = WanAttentionBlock
                block.self_attn.__class__  = WanSelfAttention
                block.cross_attn.__class__ = WanT2VCrossAttention
                block.idx       = i

        elif model.model.diffusion_model.__class__ not in {ReWanModel, WanModel}:
            raise ValueError("This node is for enabling regional conditioning for WAN only!")
            m = model
        
        return (m,)
    
class ReWanPatcher(ReWanPatcherAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"  : ("MODEL",),
                "enable" : ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, model, enable=True, force=False):
        return super().main(
            model             = model,
            self_attn_blocks  = "all",
            cross_attn_blocks = "all",
            enable            = enable,
            force             = force
        )    

class ReDoubleStreamBlockNoMask(ReDoubleStreamBlock):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)
    
class ReSingleStreamBlockNoMask(ReSingleStreamBlock):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)

class ReFluxPatcherAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model"               : ("MODEL",),
                "doublestream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "singlestream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "style_dtype"         : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"              : ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"

    def main(self, model, doublestream_blocks, singlestream_blocks, style_dtype, enable=True, force=False):
        
        doublestream_blocks = parse_range_string(doublestream_blocks)
        singlestream_blocks = parse_range_string(singlestream_blocks)
        
        style_dtype = getattr(torch, style_dtype) if style_dtype != "default" else None
        
        model.model.diffusion_model.style_dtype = style_dtype
        model.model.diffusion_model.proj_weights = None
        model.model.diffusion_model.y0_adain_embed = None
        model.model.diffusion_model.adain_pw_cache = None
        
        model.model.diffusion_model.StyleWCT = StyleWCT()
        model.model.diffusion_model.Retrojector = Retrojector(model.model.diffusion_model.img_in, pinv_dtype=style_dtype, dtype=style_dtype)
        
        if (enable or force) and model.model.diffusion_model.__class__ == Flux:
            m = model.clone()
            m.model.diffusion_model.__class__     = ReFlux
            m.model.diffusion_model.threshold_inv = False
            
            for i, block in enumerate(m.model.diffusion_model.double_blocks):
                if i in doublestream_blocks:
                    block.__class__ = ReDoubleStreamBlock
                else:
                    block.__class__ = ReDoubleStreamBlockNoMask
                block.idx       = i

            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                if i in singlestream_blocks:
                    block.__class__ = ReSingleStreamBlock
                else:
                    block.__class__ = ReSingleStreamBlockNoMask
                block.idx       = i
                
        
        elif not enable and model.model.diffusion_model.__class__ == ReFlux:
            m = model.clone()
            m.model.diffusion_model.__class__ = Flux
            
            for i, block in enumerate(m.model.diffusion_model.double_blocks):
                block.__class__ = DoubleStreamBlock
                block.idx       = i

            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                block.__class__ = SingleStreamBlock
                block.idx       = i
                
        #elif model.model.diffusion_model.__class__ != Flux and model.model.diffusion_model.__class__ != ReFlux:
        elif model.model.diffusion_model.__class__ not in {ReFlux, Flux}:
            raise ValueError("This node is for enabling regional conditioning for Flux only!")
        else:
            m = model
        
        return (m,)
    
class ReFluxPatcher(ReFluxPatcherAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"       : ("MODEL",),
                "style_dtype" : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"      : ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, model, style_dtype="float32", enable=True, force=False):
        return super().main(
            model               = model,
            doublestream_blocks = "all",
            singlestream_blocks = "all",
            style_dtype         = style_dtype,
            enable              = enable,
            force               = force
        )    






class ReReduxPatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "style_model" : ("STYLE_MODEL",),
                "style_dtype" : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"      : ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("STYLE_MODEL",)
    RETURN_NAMES = ("style_model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"
    EXPERIMENTAL = True

    def main(self, style_model, style_dtype, enable=True, force=False):
        
        style_model.model.style_dtype = getattr(torch, style_dtype) if style_dtype != "default" else None
        style_model.model.proj_weights = None
        style_model.model.y0_adain_embed = None
                
        if (enable or force) and style_model.model.__class__ == ReduxImageEncoder:
            m = style_model#.clone()
            m.model.__class__     = ReReduxImageEncoder
            m.model.threshold_inv = False
        
        elif not enable and style_model.model.__class__ == ReReduxImageEncoder:
            m = style_model#.clone()
            m.model.__class__ = ReduxImageEncoder
            
        elif style_model.model.__class__ not in {ReReduxImageEncoder, ReduxImageEncoder}:
            raise ValueError("This node is for enabling style conditioning for Redux only!")
        else:
            m = style_model
        
        return (m,)



class ReChromaDoubleStreamBlockNoMask(ReChromaDoubleStreamBlock):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)
    
class ReChromaSingleStreamBlockNoMask(ReChromaSingleStreamBlock):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)

class ReChromaPatcherAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model"               : ("MODEL",),
                "doublestream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "singlestream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "style_dtype"         : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"              : ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"

    def main(self, model, doublestream_blocks, singlestream_blocks, style_dtype, enable=True, force=False):
        
        doublestream_blocks = parse_range_string(doublestream_blocks)
        singlestream_blocks = parse_range_string(singlestream_blocks)
        
        style_dtype = getattr(torch, style_dtype) if style_dtype != "default" else None
        model.model.diffusion_model.style_dtype = style_dtype
        model.model.diffusion_model.proj_weights = None
        model.model.diffusion_model.y0_adain_embed = None
        
        model.model.diffusion_model.StyleWCT = StyleWCT()
        model.model.diffusion_model.Retrojector = Retrojector(model.model.diffusion_model.img_in, pinv_dtype=style_dtype, dtype=style_dtype)
        
        if (enable or force) and model.model.diffusion_model.__class__ == Chroma:
            m = model.clone()
            m.model.diffusion_model.__class__     = ReChroma
            m.model.diffusion_model.threshold_inv = False
            
            for i, block in enumerate(m.model.diffusion_model.double_blocks):
                if i in doublestream_blocks:
                    block.__class__ = ReChromaDoubleStreamBlock
                else:
                    block.__class__ = ReChromaDoubleStreamBlockNoMask
                block.idx       = i

            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                if i in singlestream_blocks:
                    block.__class__ = ReChromaSingleStreamBlock
                else:
                    block.__class__ = ReChromaSingleStreamBlockNoMask
                block.idx       = i
                
        
        elif not enable and model.model.diffusion_model.__class__ == ReChroma:
            m = model.clone()
            m.model.diffusion_model.__class__ = Chroma
            
            for i, block in enumerate(m.model.diffusion_model.double_blocks):
                block.__class__ = DoubleStreamBlock
                block.idx       = i

            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                block.__class__ = SingleStreamBlock
                block.idx       = i
                
        #elif model.model.diffusion_model.__class__ != Chroma and model.model.diffusion_model.__class__ != ReChroma:
        elif model.model.diffusion_model.__class__ not in {ReChroma, Chroma}:
            raise ValueError("This node is for enabling regional conditioning for Chroma only!")
        else:
            m = model
        
        return (m,)
    
class ReChromaPatcher(ReChromaPatcherAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"       : ("MODEL",),
                "style_dtype" : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"      : ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, model, style_dtype="float32", enable=True, force=False):
        return super().main(
            model               = model,
            doublestream_blocks = "all",
            singlestream_blocks = "all",
            style_dtype         = style_dtype,
            enable              = enable,
            force               = force
        )    







"""class ReLTXVDoubleStreamBlockNoMask(ReLTXVDoubleStreamBlock):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)
    
class ReLTXVSingleStreamBlockNoMask(ReLTXVSingleStreamBlock):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)"""

class ReLTXVPatcherAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model"               : ("MODEL",),
                "doublestream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "singlestream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "style_dtype"         : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"              : ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"

    def main(self, model, doublestream_blocks, singlestream_blocks, style_dtype, enable=True, force=False):
        
        doublestream_blocks = parse_range_string(doublestream_blocks)
        singlestream_blocks = parse_range_string(singlestream_blocks)
        
        style_dtype = getattr(torch, style_dtype) if style_dtype != "default" else None
        model.model.diffusion_model.style_dtype = style_dtype
        model.model.diffusion_model.proj_weights = None
        model.model.diffusion_model.y0_adain_embed = None
        
        model.model.diffusion_model.StyleWCT = StyleWCT()
        model.model.diffusion_model.Retrojector = Retrojector(model.model.diffusion_model.patchify_proj, pinv_dtype=style_dtype, dtype=style_dtype)
        
        if (enable or force) and model.model.diffusion_model.__class__ == LTXVModel:
            m = model.clone()
            m.model.diffusion_model.__class__     = ReLTXVModel
            m.model.diffusion_model.threshold_inv = False
            
            """for i, block in enumerate(m.model.diffusion_model.double_blocks):
                if i in doublestream_blocks:
                    block.__class__ = ReChromaDoubleStreamBlock
                else:
                    block.__class__ = ReChromaDoubleStreamBlockNoMask
                block.idx       = i

            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                if i in singlestream_blocks:
                    block.__class__ = ReChromaSingleStreamBlock
                else:
                    block.__class__ = ReChromaSingleStreamBlockNoMask
                block.idx       = i"""
                
        
        elif not enable and model.model.diffusion_model.__class__ == ReLTXVModel:
            m = model.clone()
            m.model.diffusion_model.__class__ = LTXVModel
            
            """for i, block in enumerate(m.model.diffusion_model.double_blocks):
                block.__class__ = DoubleStreamBlock
                block.idx       = i

            for i, block in enumerate(m.model.diffusion_model.single_blocks):
                block.__class__ = SingleStreamBlock
                block.idx       = i"""
                
        #elif model.model.diffusion_model.__class__ != LTXVModel and model.model.diffusion_model.__class__ != ReLTXVModel:
        elif model.model.diffusion_model.__class__ not in {ReLTXVModel, LTXVModel}:
            raise ValueError("This node is for enabling regional conditioning for LTXV only!")
        else:
            m = model
        
        return (m,)
    
class ReLTXVPatcher(ReLTXVPatcherAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"       : ("MODEL",),
                "style_dtype" : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"      : ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, model, style_dtype="float32", enable=True, force=False):
        return super().main(
            model               = model,
            doublestream_blocks = "all",
            singlestream_blocks = "all",
            style_dtype         = style_dtype,
            enable              = enable,
            force               = force
        )    





class ReSDPatcherAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model"               : ("MODEL",),
                "doublestream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "singlestream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "style_dtype"         : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"              : ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"
    #EXPERIMENTAL = True

    def main(self, model, doublestream_blocks, singlestream_blocks, style_dtype, enable=True, force=False):
        
        doublestream_blocks = parse_range_string(doublestream_blocks)
        singlestream_blocks = parse_range_string(singlestream_blocks)
        
        style_dtype = getattr(torch, style_dtype) if style_dtype != "default" else None
        model.model.diffusion_model.style_dtype = style_dtype
        model.model.diffusion_model.proj_weights = None
        model.model.diffusion_model.y0_adain_embed = None
        
        model.model.diffusion_model.StyleWCT    = StyleWCT()
        model.model.diffusion_model.Retrojector = Retrojector(model.model.diffusion_model.input_blocks[0][0], pinv_dtype=style_dtype, dtype=style_dtype, patch_size=1)
        
        if (enable or force) and model.model.diffusion_model.__class__ == UNetModel:
            m = model.clone()
            m.model.diffusion_model.__class__     = ReUNetModel
            m.model.diffusion_model.threshold_inv = False
                
            for i in range(len(m.model.diffusion_model.input_blocks)):
                for j in range(len(m.model.diffusion_model.input_blocks[i])):
                    if isinstance(m.model.diffusion_model.input_blocks[i][j], ResBlock):
                        m.model.diffusion_model.input_blocks[i][j].__class__ = ReResBlock
                    if isinstance(m.model.diffusion_model.input_blocks[i][j], SpatialTransformer):
                        m.model.diffusion_model.input_blocks[i][j].__class__ = ReSpatialTransformer
                        for k in range(len(m.model.diffusion_model.input_blocks[i][j].transformer_blocks)):
                            m.model.diffusion_model.input_blocks[i][j].transformer_blocks[k].__class__ = ReBasicTransformerBlock
                            m.model.diffusion_model.input_blocks[i][j].transformer_blocks[k].attn1.__class__ = ReCrossAttention
                            m.model.diffusion_model.input_blocks[i][j].transformer_blocks[k].attn2.__class__ = ReCrossAttention
        
            #m.model.diffusion_model.middle_block[1].transformer_blocks[0].__class__ = ReBasicTransformerBlock
            for i in range(len(m.model.diffusion_model.middle_block)):
                if isinstance(m.model.diffusion_model.middle_block[i], ResBlock):
                    m.model.diffusion_model.middle_block[i].__class__ = ReResBlock
                if isinstance(m.model.diffusion_model.middle_block[i], SpatialTransformer):
                    m.model.diffusion_model.middle_block[i].__class__ = ReSpatialTransformer
                    for k in range(len(m.model.diffusion_model.middle_block[i].transformer_blocks)):
                        m.model.diffusion_model.middle_block[i].transformer_blocks[k].__class__ = ReBasicTransformerBlock
                        m.model.diffusion_model.middle_block[i].transformer_blocks[k].attn1.__class__ = ReCrossAttention
                        m.model.diffusion_model.middle_block[i].transformer_blocks[k].attn2.__class__ = ReCrossAttention

            for i in range(len(m.model.diffusion_model.output_blocks)):
                for j in range(len(m.model.diffusion_model.output_blocks[i])):
                    if isinstance(m.model.diffusion_model.output_blocks[i][j], ResBlock):
                        m.model.diffusion_model.output_blocks[i][j].__class__ = ReResBlock
                    if isinstance(m.model.diffusion_model.output_blocks[i][j], SpatialTransformer):
                        m.model.diffusion_model.output_blocks[i][j].__class__ = ReSpatialTransformer
                        for k in range(len(m.model.diffusion_model.output_blocks[i][j].transformer_blocks)):
                            m.model.diffusion_model.output_blocks[i][j].transformer_blocks[k].__class__ = ReBasicTransformerBlock
                            m.model.diffusion_model.output_blocks[i][j].transformer_blocks[k].attn1.__class__ = ReCrossAttention
                            m.model.diffusion_model.output_blocks[i][j].transformer_blocks[k].attn2.__class__ = ReCrossAttention

        elif not enable and model.model.diffusion_model.__class__ == ReUNetModel:
            m = model.clone()
            m.model.diffusion_model.__class__ = UNetModel
            
            for i in range(len(m.model.diffusion_model.input_blocks)):
                for j in range(len(m.model.diffusion_model.input_blocks[i])):
                    if isinstance(m.model.diffusion_model.input_blocks[i][j], ReResBlock):
                        m.model.diffusion_model.input_blocks[i][j].__class__ = ResBlock
                    if isinstance(m.model.diffusion_model.input_blocks[i][j], ReSpatialTransformer):
                        m.model.diffusion_model.input_blocks[i][j].__class__ = SpatialTransformer
                        for k in range(len(m.model.diffusion_model.input_blocks[i][j].transformer_blocks)):
                            m.model.diffusion_model.input_blocks[i][j].transformer_blocks[k].__class__ = BasicTransformerBlock
                            m.model.diffusion_model.input_blocks[i][j].transformer_blocks[k].attn1.__class__ = CrossAttention
                            m.model.diffusion_model.input_blocks[i][j].transformer_blocks[k].attn2.__class__ = CrossAttention
        
            #m.model.diffusion_model.middle_block[1].transformer_blocks[0].__class__ = BasicTransformerBlock
            for i in range(len(m.model.diffusion_model.middle_block)):
                if isinstance(m.model.diffusion_model.middle_block[i], ReResBlock):
                    m.model.diffusion_model.middle_block[i].__class__ = ResBlock
                if isinstance(m.model.diffusion_model.middle_block[i], ReSpatialTransformer):
                    m.model.diffusion_model.middle_block[i].__class__ = SpatialTransformer
                    for k in range(len(m.model.diffusion_model.middle_block[i].transformer_blocks)):
                        m.model.diffusion_model.middle_block[i].transformer_blocks[k].__class__ = BasicTransformerBlock
                        m.model.diffusion_model.middle_block[i].transformer_blocks[k].attn1.__class__ = CrossAttention
                        m.model.diffusion_model.middle_block[i].transformer_blocks[k].attn2.__class__ = CrossAttention

            for i in range(len(m.model.diffusion_model.output_blocks)):
                for j in range(len(m.model.diffusion_model.output_blocks[i])):
                    if isinstance(m.model.diffusion_model.output_blocks[i][j], ReResBlock):
                        m.model.diffusion_model.output_blocks[i[j]].__class__ = ResBlock
                    if isinstance(m.model.diffusion_model.output_blocks[i][j], ReSpatialTransformer):
                        m.model.diffusion_model.output_blocks[i[j]].__class__ = SpatialTransformer
                        for k in range(len(m.model.diffusion_model.output_blocks[i][j].transformer_blocks)):
                            m.model.diffusion_model.output_blocks[i][j].transformer_blocks[k].__class__ = BasicTransformerBlock
                            m.model.diffusion_model.output_blocks[i][j].transformer_blocks[k].attn1.__class__ = CrossAttention
                            m.model.diffusion_model.output_blocks[i][j].transformer_blocks[k].attn2.__class__ = CrossAttention

        #elif model.model.diffusion_model.__class__ != UNetModel and model.model.diffusion_model.__class__ != ReUNetModel:
        elif model.model.diffusion_model.__class__ not in {ReUNetModel, UNetModel}:
            raise ValueError("This node is for enabling regional conditioning for SD1.5 and SDXL only!")
        else:
            m = model
        
        return (m,)
    
class ReSDPatcher(ReSDPatcherAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"       : ("MODEL",),
                "style_dtype" : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"      : ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, model, style_dtype="float32", enable=True, force=False):
        return super().main(
            model               = model,
            doublestream_blocks = "all",
            singlestream_blocks = "all",
            style_dtype         = style_dtype,
            enable              = enable,
            force               = force
        )    








class HDBlockDoubleNoMask(HDBlockDouble):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)
    
class HDBlockSingleNoMask(HDBlockSingle):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)


class ReHiDreamPatcherAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model"                : ("MODEL",),
                "double_stream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "single_stream_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "style_dtype"          : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"               : ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"

    def main(self, model, double_stream_blocks, single_stream_blocks, style_dtype, enable=True, force=False):
        
        double_stream_blocks = parse_range_string(double_stream_blocks)
        single_stream_blocks = parse_range_string(single_stream_blocks)
        
        style_dtype = getattr(torch, style_dtype) if style_dtype != "default" else None
        model.model.diffusion_model.style_dtype = style_dtype
        model.model.diffusion_model.proj_weights = None
        model.model.diffusion_model.y0_adain_embed = None
        
        model.model.diffusion_model.StyleWCT    = StyleWCT()
        model.model.diffusion_model.WaveletStyleWCT = WaveletStyleWCT()
        model.model.diffusion_model.Retrojector = Retrojector(model.model.diffusion_model.x_embedder.proj, pinv_dtype=style_dtype, dtype=style_dtype)
        #model.model.diffusion_model.Endojector  = Retrojector(model.model.diffusion_model.final_layer.linear, pinv_dtype=style_dtype, dtype=style_dtype, ENDO=True)
        
        #model.model.diffusion_model.Style = StyleMMDiT_HiDream()
        #model.model.diffusion_model.Style.Retrojector = Retrojector(model.model.diffusion_model.x_embedder.proj, pinv_dtype=style_dtype, dtype=style_dtype)
        
        sort_buffer = {}
        
        if (enable or force) and model.model.diffusion_model.__class__ == HiDreamImageTransformer2DModel:
            m = model.clone()
            m.model.diffusion_model.__class__     = HDModel
            m.model.diffusion_model.threshold_inv = False
            m.model.diffusion_model.final_layer.__class__ = HDLastLayer
            
            m.model.diffusion_model.final_layer.linear.weight.data = m.model.diffusion_model.final_layer.linear.weight.data.to(torch.bfloat16)
            m.model.diffusion_model.final_layer.linear.bias.data = m.model.diffusion_model.final_layer.linear.bias.data.to(torch.bfloat16)
            
            for i, block in enumerate(m.model.diffusion_model.double_stream_blocks):
                block.__class__             = HDBlock

                if i in double_stream_blocks:
                    block.block.__class__   = HDBlockDouble
                else:
                    block.block.__class__   = HDBlockDoubleNoMask
                    
                block.block.attn1.__class__ = HDAttention
                    
                block.block.ff_i.__class__  = HDMOEFeedForwardSwiGLU
                block.block.ff_i.shared_experts.__class__ = HDFeedForwardSwiGLU
                for j in range(len(block.block.ff_i.experts)):
                    block.block.ff_i.experts[j].__class__ = HDFeedForwardSwiGLU
                block.block.ff_i.gate.__class__ = HDMoEGate
                block.block.ff_t.__class__  = HDFeedForwardSwiGLU
                    
                block.block.attn1.single_stream = False
                block.block.attn1.double_stream = True
                
                block.block.sort_buffer       = sort_buffer
                block.block.attn1.sort_buffer = sort_buffer
                
                block.idx             = i
                block.block.idx       = i
                block.block.attn1.idx = i

            for i, block in enumerate(m.model.diffusion_model.single_stream_blocks):
                block.__class__             = HDBlock

                if i in single_stream_blocks:
                    block.block.__class__       = HDBlockSingle
                else:
                    block.block.__class__       = HDBlockSingleNoMask

                block.block.attn1.__class__ = HDAttention
                block.block.ff_i.__class__  = HDMOEFeedForwardSwiGLU
                block.block.ff_i.shared_experts.__class__ = HDFeedForwardSwiGLU
                for j in range(len(block.block.ff_i.experts)):
                    block.block.ff_i.experts[j].__class__ = HDFeedForwardSwiGLU
                block.block.ff_i.gate.__class__ = HDMoEGate
                
                block.block.attn1.single_stream = True
                block.block.attn1.double_stream = False
                
                block.block.sort_buffer       = sort_buffer
                block.block.attn1.sort_buffer = sort_buffer
                
                block.idx             = i
                block.block.idx       = i
                block.block.attn1.idx = i

        elif not enable and model.model.diffusion_model.__class__ == HDModel:
            m = model.clone()
            m.model.diffusion_model.__class__ = HiDreamImageTransformer2DModel
            
            for i, block in enumerate(m.model.diffusion_model.double_stream_blocks):
                if i in double_stream_blocks:
                    block.__class__             = HiDreamImageBlock
                    block.block.__class__       = HiDreamImageTransformerBlock
                    block.block.attn1.__class__ = HiDreamAttention
                block.idx       = i
            
            for i, block in enumerate(m.model.diffusion_model.single_stream_blocks):
                if i in single_stream_blocks:
                    block.__class__             = HiDreamImageBlock
                    block.block.__class__       = HiDreamImageSingleTransformerBlock
                    block.block.attn1.__class__ = HiDreamAttention
                block.idx       = i
                
        #elif model.model.diffusion_model.__class__ != HDModel and model.model.diffusion_model.__class__ != HiDreamImageTransformer2DModel:
        elif model.model.diffusion_model.__class__ not in {HDModel, HiDreamImageTransformer2DModel}:
            raise ValueError("This node is for enabling regional conditioning for HiDream only!")
        else:
            m = model
        
        return (m,)
    
class ReHiDreamPatcher(ReHiDreamPatcherAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"       : ("MODEL",),
                "style_dtype" : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"      : ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, model, style_dtype="default", enable=True, force=False):
        return super().main(
            model                = model,
            double_stream_blocks = "all",
            single_stream_blocks = "all",
            style_dtype          = style_dtype,
            enable               = enable,
            force                = force
        )    



class ReAnimaPatcherAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "RES4LYF/model_patches"
    FUNCTION = "main"

    original_forward = ANIMA_DIFFUSION_MODEL_CLASS._forward if ANIMA_DIFFUSION_MODEL_CLASS is not None else None

    @staticmethod
    def _supports_anima(model):
        if ANIMA_MODEL_CLASS is None or ANIMA_DIFFUSION_MODEL_CLASS is None:
            return False
        if not isinstance(model.model.model_config, ANIMA_MODEL_CLASS):
            return False
        return hasattr(model.model.diffusion_model, "blocks")

    @staticmethod
    def _make_attn_op(attn_kind: str, fallback_attn_op):
        def _wrapped(q, k, v, transformer_options={}):
            return _anima_attention_op(
                q=q,
                k=k,
                v=v,
                transformer_options=transformer_options,
                attn_kind=attn_kind,
                fallback_attn_op=fallback_attn_op,
            )
        return _wrapped

    @staticmethod
    def _build_base_token_mask(attn_mask_obj, context: torch.Tensor, transformer_options: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
        img_len = getattr(attn_mask_obj, "img_len", 0)
        t = getattr(attn_mask_obj, "t", 1)
        if img_len <= 0:
            return None

        k_tokens = context.shape[1]
        base_start_hint = None
        base_len_hint = None
        if transformer_options is not None:
            base_start_hint = transformer_options.get("_res4lyf_anima_base_context_start")
            base_len_hint = transformer_options.get("_res4lyf_anima_base_context_len")

        try:
            start = int(base_start_hint) if base_start_hint is not None else 0
        except Exception:
            start = 0
        start = min(max(start, 0), k_tokens)

        try:
            span_len = int(base_len_hint) if base_len_hint is not None else 0
        except Exception:
            span_len = 0

        if span_len <= 0:
            start = 0
            end = k_tokens
        else:
            end = min(start + max(span_len, 1), k_tokens)
            if end <= start:
                start = 0
                end = k_tokens

        q_tokens = int(img_len * t)
        mask = torch.zeros((q_tokens, k_tokens), dtype=torch.bool, device=context.device)
        mask[:, start:end] = True
        return mask

    @staticmethod
    def _prepare_regional_context(context: torch.Tensor, transformer_options: Dict[str, Any]) -> torch.Tensor:
        attn_mask_obj = transformer_options.get("AttnMask")
        attn_mask_neg_obj = transformer_options.get("AttnMask_neg")
        weight = _scalar_or_default(transformer_options.get("regional_conditioning_weight"), 0.0)
        weight_neg = _scalar_or_default(transformer_options.get("regional_conditioning_weight_neg"), 0.0)
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        keep_base_mask_when_weight_zero = str(
            os.environ.get("RES4LYF_ANIMA_ZERO_WEIGHT_BASE_MASK", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}

        if attn_mask_obj is None and attn_mask_neg_obj is None:
            transformer_options.pop("_res4lyf_anima_attn_mask", None)
            return context

        k_tokens = context.shape[1]  # actual context length (post LLM-adapter / padding)

        attn_mask_pos = None
        if attn_mask_obj is not None and weight != 0.0:
            attn_mask_obj.attn_mask_recast(context.dtype)
            raw_mask = attn_mask_obj.get(weight=weight)
            text_len = getattr(attn_mask_obj, "text_len", 0)
            if text_len > 0 and raw_mask is not None and raw_mask.shape[-1] != k_tokens:
                attn_mask_pos = _extract_anima_cross_attn_mask(raw_mask, text_len, k_tokens)
            else:
                attn_mask_pos = raw_mask
        elif attn_mask_obj is not None and keep_base_mask_when_weight_zero:
            attn_mask_pos = ReAnimaPatcherAdvanced._build_base_token_mask(attn_mask_obj, context, transformer_options)

        attn_mask_neg = None
        if attn_mask_neg_obj is not None and weight_neg != 0.0:
            attn_mask_neg_obj.attn_mask_recast(context.dtype)
            raw_mask_neg = attn_mask_neg_obj.get(weight=weight_neg)
            text_len_neg = getattr(attn_mask_neg_obj, "text_len", 0)
            if text_len_neg > 0 and raw_mask_neg is not None and raw_mask_neg.shape[-1] != k_tokens:
                attn_mask_neg = _extract_anima_cross_attn_mask(raw_mask_neg, text_len_neg, k_tokens)
            else:
                attn_mask_neg = raw_mask_neg

        if isinstance(cond_or_uncond, (list, tuple)) and len(cond_or_uncond) == context.shape[0]:
            mixed_masks: List[Optional[torch.Tensor]] = []
            for batch_index, cond_flag in enumerate(cond_or_uncond):
                if int(cond_flag) == 1:
                    mixed_masks.append(attn_mask_neg)
                else:
                    mixed_masks.append(attn_mask_pos)

            if any(mask is not None for mask in mixed_masks):
                ref_mask = next(mask for mask in mixed_masks if mask is not None)
                full_mask = torch.ones_like(ref_mask, dtype=ref_mask.dtype, device=ref_mask.device)
                stacked = []
                for mask in mixed_masks:
                    curr = full_mask if mask is None else mask
                    stacked.append(curr)
                transformer_options["_res4lyf_anima_attn_mask"] = torch.stack(stacked, dim=0)
            else:
                transformer_options.pop("_res4lyf_anima_attn_mask", None)
            return context

        if attn_mask_pos is not None:
            transformer_options["_res4lyf_anima_attn_mask"] = attn_mask_pos
        elif attn_mask_neg is not None:
            transformer_options["_res4lyf_anima_attn_mask"] = attn_mask_neg
        else:
            transformer_options.pop("_res4lyf_anima_attn_mask", None)
        return context

    @staticmethod
    def _forward_with_regional(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        if transformer_options is None:
            transformer_options = {}
            kwargs["transformer_options"] = transformer_options

        base_context_start = kwargs.pop("anima_base_context_start", None)
        base_context_len = kwargs.pop("anima_base_context_len", None)

        if base_context_start is None:
            transformer_options.pop("_res4lyf_anima_base_context_start", None)
        else:
            try:
                transformer_options["_res4lyf_anima_base_context_start"] = int(base_context_start)
            except Exception:
                transformer_options.pop("_res4lyf_anima_base_context_start", None)

        if base_context_len is None:
            transformer_options.pop("_res4lyf_anima_base_context_len", None)
        else:
            try:
                transformer_options["_res4lyf_anima_base_context_len"] = int(base_context_len)
            except Exception:
                transformer_options.pop("_res4lyf_anima_base_context_len", None)

        context = ReAnimaPatcherAdvanced._prepare_regional_context(context, transformer_options)
        return ReAnimaPatcherAdvanced.original_forward(self, x, timesteps, context, fps, padding_mask, **kwargs)

    def main(self, model, enable=True, force=False):
        if not self._supports_anima(model):
            raise ValueError("This node is for enabling regional conditioning for Anima only!")

        m = model.clone()
        diffusion_model = m.model.diffusion_model

        if enable or force:
            diffusion_model._res4lyf_anima_regional_enabled = True
            diffusion_model._forward = types.MethodType(ReAnimaPatcherAdvanced._forward_with_regional, diffusion_model)

            for i, block in enumerate(diffusion_model.blocks):
                block.idx = i
                if hasattr(block, "self_attn"):
                    if not hasattr(block.self_attn, "_res4lyf_original_attn_op"):
                        block.self_attn._res4lyf_original_attn_op = block.self_attn.attn_op
                    block.self_attn.attn_op = ReAnimaPatcherAdvanced._make_attn_op(
                        "self",
                        block.self_attn._res4lyf_original_attn_op,
                    )
                if hasattr(block, "cross_attn"):
                    if not hasattr(block.cross_attn, "_res4lyf_original_attn_op"):
                        block.cross_attn._res4lyf_original_attn_op = block.cross_attn.attn_op
                    block.cross_attn.attn_op = ReAnimaPatcherAdvanced._make_attn_op(
                        "cross",
                        block.cross_attn._res4lyf_original_attn_op,
                    )
        else:
            diffusion_model._res4lyf_anima_regional_enabled = False
            if ReAnimaPatcherAdvanced.original_forward is not None:
                diffusion_model._forward = types.MethodType(ReAnimaPatcherAdvanced.original_forward, diffusion_model)

            for block in diffusion_model.blocks:
                if hasattr(block, "self_attn") and hasattr(block.self_attn, "_res4lyf_original_attn_op"):
                    block.self_attn.attn_op = block.self_attn._res4lyf_original_attn_op
                if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "_res4lyf_original_attn_op"):
                    block.cross_attn.attn_op = block.cross_attn._res4lyf_original_attn_op

            diffusion_model._res4lyf_anima_attn_mask = None

        return (m,)


class ReAnimaPatcher(ReAnimaPatcherAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, model, enable=True, force=False):
        return super().main(model=model, enable=enable, force=force)


class ReJointBlockNoMask(ReJointBlock):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)

class ReSD35PatcherAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model"        : ("MODEL",),
                "joint_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "style_dtype"  : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"       : ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"

    def main(self, model, joint_blocks, style_dtype, enable=True, force=False):
        
        style_dtype = getattr(torch, style_dtype) if style_dtype != "default" else None
        model.model.diffusion_model.style_dtype = style_dtype
        model.model.diffusion_model.proj_weights = None
        model.model.diffusion_model.y0_adain_embed = None
        
        model.model.diffusion_model.StyleWCT    = StyleWCT()
        model.model.diffusion_model.Retrojector = Retrojector(model.model.diffusion_model.x_embedder.proj, pinv_dtype=style_dtype, dtype=style_dtype)
        
        joint_blocks = parse_range_string(joint_blocks)
        
        if (enable or force) and model.model.diffusion_model.__class__ == OpenAISignatureMMDITWrapper:
            m = model.clone()
            m.model.diffusion_model.__class__     = ReOpenAISignatureMMDITWrapper
            m.model.diffusion_model.threshold_inv = False
            
            for i, block in enumerate(m.model.diffusion_model.joint_blocks):
                if i in joint_blocks:
                    block.__class__ = ReJointBlock
                else:
                    ReJointBlockNoMask
                block.idx       = i

        elif not enable and model.model.diffusion_model.__class__ == ReOpenAISignatureMMDITWrapper:
            m = model.clone()
            m.model.diffusion_model.__class__ = OpenAISignatureMMDITWrapper
            
            for i, block in enumerate(m.model.diffusion_model.joint_blocks):
                block.__class__ = JointBlock
                block.idx       = i

        elif model.model.diffusion_model.__class__ not in {ReOpenAISignatureMMDITWrapper, OpenAISignatureMMDITWrapper}:
            raise ValueError("This node is for enabling regional conditioning for SD3.5 only!")
            m = model
        
        return (m,)
    
class ReSD35Patcher(ReSD35PatcherAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"       : ("MODEL",),
                "style_dtype" : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"      : ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, model, style_dtype="float32", enable=True, force=False):
        return super().main(
            model        = model,
            joint_blocks = "all",
            style_dtype  = style_dtype,
            enable       = enable,
            force        = force
        )    

class ReDoubleAttentionNoMask(ReDoubleAttention):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)
    
class ReSingleAttentionNoMask(ReSingleAttention):
    def forward(self, c, mask=None):
        return super().forward(c, mask=None)

class ReAuraPatcherAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model"              : ("MODEL",),
                "doublelayer_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "singlelayer_blocks" : ("STRING",  {"default": "all", "multiline": True}),
                "style_dtype"        : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"             : ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"

    def main(self, model, doublelayer_blocks, singlelayer_blocks, style_dtype, enable=True, force=False):
        
        doublelayer_blocks = parse_range_string(doublelayer_blocks)
        singlelayer_blocks = parse_range_string(singlelayer_blocks)
        
        style_dtype = getattr(torch, style_dtype) if style_dtype != "default" else None
        model.model.diffusion_model.style_dtype = style_dtype
        model.model.diffusion_model.proj_weights = None
        model.model.diffusion_model.y0_adain_embed = None
        
        model.model.diffusion_model.StyleWCT    = StyleWCT()
        model.model.diffusion_model.Retrojector = Retrojector(model.model.diffusion_model.init_x_linear, pinv_dtype=style_dtype, dtype=style_dtype)
        
        if (enable or force) and model.model.diffusion_model.__class__ == MMDiT:
            m = model.clone()
            m.model.diffusion_model.__class__     = ReMMDiT
            m.model.diffusion_model.threshold_inv = False
            
            for i, block in enumerate(m.model.diffusion_model.double_layers):
                block.__class__ = ReMMDiTBlock
                if i in doublelayer_blocks:
                    block.attn.__class__ = ReDoubleAttention
                else:
                    block.attn.__class__ = ReDoubleAttentionNoMask
                block.idx       = i
                
            for i, block in enumerate(m.model.diffusion_model.single_layers):
                block.__class__ = ReDiTBlock
                if i in singlelayer_blocks:
                    block.attn.__class__ = ReSingleAttention
                else:
                    block.attn.__class__ = ReSingleAttentionNoMask
                block.idx       = i

        elif not enable and model.model.diffusion_model.__class__ == ReMMDiT:
            m = model.clone()
            m.model.diffusion_model.__class__ = MMDiT
            
            for i, block in enumerate(m.model.diffusion_model.double_layers):
                block.__class__ = MMDiTBlock
                block.attn.__class__ = DoubleAttention
                block.idx       = i
                
            for i, block in enumerate(m.model.diffusion_model.single_layers):
                block.__class__ = DiTBlock
                block.attn.__class__ = SingleAttention
                block.idx       = i

        elif model.model.diffusion_model.__class__ not in {ReMMDiT, MMDiT}:
            raise ValueError("This node is for enabling regional conditioning for AuraFlow only!")
            m = model
        
        return (m,)

class ReAuraPatcher(ReAuraPatcherAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model"       : ("MODEL",),
                "style_dtype" : (["default", "bfloat16", "float16", "float32", "float64"],  {"default": "float64"}),
                "enable"      : ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, model, style_dtype="float32", enable=True, force=False):
        return super().main(
            model              = model,
            doublelayer_blocks = "all",
            singlelayer_blocks = "all",
            style_dtype         = style_dtype,
            enable             = enable,
            force              = force
        )    


class FluxOrthoCFGPatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model":        ("MODEL",),
            "enable":       ("BOOLEAN", {"default": True}),
            "ortho_T5":     ("BOOLEAN", {"default": True}),
            "ortho_clip_L": ("BOOLEAN", {"default": True}),
            "zero_clip_L":  ("BOOLEAN", {"default": True}),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY     = "RES4LYF/model_patches"
    FUNCTION     = "main"
    EXPERIMENTAL = True
    
    original_forward = Flux.forward

    @staticmethod
    def new_forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):

        for _ in range(500):
            if self.ortho_T5 and get_cosine_similarity(context[0], context[1]) != 0:
                context[0] = get_orthogonal(context[0], context[1])
            if self.ortho_clip_L and get_cosine_similarity(y[0], y[1]) != 0:
                y[0] = get_orthogonal(y[0].unsqueeze(0), y[1].unsqueeze(0)).squeeze(0)
                
        RESplain("postcossim1: ", get_cosine_similarity(context[0], context[1]))
        RESplain("postcossim2: ", get_cosine_similarity(y[0], y[1]))
        
        if self.zero_clip_L:
            y[0] = torch.zeros_like(y[0])
        
        return FluxOrthoCFGPatcher.original_forward(self, x, timestep, context, y, guidance, control, transformer_options, **kwargs)

    def main(self, model, enable=True, ortho_T5=True, ortho_clip_L=True, zero_clip_L=True):
        m = model.clone()

        if enable:
            m.model.diffusion_model.ortho_T5     = ortho_T5
            m.model.diffusion_model.ortho_clip_L = ortho_clip_L
            m.model.diffusion_model.zero_clip_L  = zero_clip_L
            Flux.forward = types.MethodType(FluxOrthoCFGPatcher.new_forward, m.model.diffusion_model)
        else:
            Flux.forward = FluxOrthoCFGPatcher.original_forward

        return (m,)
    
    
    
    
class FluxGuidanceDisable:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model":       ("MODEL",),
                "disable":     ("BOOLEAN", {"default": True}),
                "zero_clip_L": ("BOOLEAN", {"default": True}),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/model_patches"

    original_forward = Flux.forward

    @staticmethod
    def new_forward(self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs):

        y = torch.zeros_like(y)
        
        return FluxGuidanceDisable.original_forward(self, x, timestep, context, y, guidance, control, transformer_options, **kwargs)

    def main(self, model, disable=True, zero_clip_L=True):
        m = model.clone()
        if disable:
            m.model.diffusion_model.params.guidance_embed = False
        else:
            m.model.diffusion_model.params.guidance_embed = True
            
        #m.model.diffusion_model.zero_clip_L = zero_clip_L
        if zero_clip_L:
            Flux.forward = types.MethodType(FluxGuidanceDisable.new_forward, m.model.diffusion_model)

        return (m,)



class ModelSamplingAdvanced:
    # this is used to set the "shift" using either exponential scaling (default for SD3.5M and Flux) or linear scaling (default for SD3.5L and SD3 2B beta)
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model":   ("MODEL",),
                    "scaling": (["exponential", "linear"], {"default": 'exponential'}), 
                    "shift":   ("FLOAT",                   {"default": 3.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    }
                }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/model_shift"

    def sigma_exponential(self, timestep):
        return time_snr_shift_exponential(self.timestep_shift, timestep / self.multiplier)

    def sigma_linear(self, timestep):
        return time_snr_shift_linear(self.timestep_shift, timestep / self.multiplier)

    def main(self, model, scaling, shift):
        m = model.clone()
        
        self.timestep_shift = shift
        self.multiplier     = 1000
        timesteps           = 1000
        sampling_base       = None
        
        if isinstance(m.model.model_config, comfy.supported_models.Flux) or isinstance(m.model.model_config, comfy.supported_models.FluxSchnell) or isinstance(m.model.model_config, comfy.supported_models.Chroma):
            self.multiplier = 1
            timesteps = 10000
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.AuraFlow):
            self.multiplier = 1
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
        
        elif isinstance(m.model.model_config, comfy.supported_models.SD3):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.HiDream):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.HunyuanVideo):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST

        elif ANIMA_MODEL_CLASS is not None and isinstance(m.model.model_config, ANIMA_MODEL_CLASS):
            self.multiplier = 1
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        if isinstance(m.model.model_config, comfy.supported_models.WAN21_T2V) or isinstance(m.model.model_config, comfy.supported_models.WAN21_I2V):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.CosmosT2V) or isinstance(m.model.model_config, comfy.supported_models.CosmosI2V):
            self.multiplier = 1
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingContinuousEDM
            sampling_type = comfy.model_sampling.CONST

        elif isinstance(m.model.model_config, comfy.supported_models.LTXV):
            self.multiplier = 1000 # incorrect?
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST
            
        if sampling_base is None:
            raise ValueError("Model not supported by ModelSamplingAdvanced")

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        m.object_patches['model_sampling'] = m.model.model_sampling = ModelSamplingAdvanced(m.model.model_config)

        m.model.model_sampling.__dict__['shift']      = self.timestep_shift
        m.model.model_sampling.__dict__['multiplier'] = self.multiplier

        s_range = torch.arange(1, timesteps + 1, 1).to(torch.float64)
        if scaling == "exponential": 
            ts = self.sigma_exponential((s_range / timesteps) * self.multiplier)
        elif scaling == "linear": 
            ts = self.sigma_linear((s_range / timesteps) * self.multiplier)

        m.model.model_sampling.register_buffer('sigmas', ts)
        m.object_patches['model_sampling'].sigmas = m.model.model_sampling.sigmas
        
        return (m,)



class ModelSamplingAdvancedResolution:
    # this is used to set the "shift" using either exponential scaling (default for SD3.5M and Flux) or linear scaling (default for SD3.5L and SD3 2B beta)
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model":        ("MODEL",),
                    "scaling":      (["exponential", "linear"], {"default": 'exponential'}), 
                    "max_shift":    ("FLOAT",                   {"default": 1.35, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    "base_shift":   ("FLOAT",                   {"default": 0.85, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    "latent_image": ("LATENT",),
                }
                }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/model_shift"

    def sigma_exponential(self, timestep):
        return time_snr_shift_exponential(self.timestep_shift, timestep / self.multiplier)

    def sigma_linear(self, timestep):
        return time_snr_shift_linear(self.timestep_shift, timestep / self.multiplier)

    def main(self, model, scaling, max_shift, base_shift, latent_image):
        m = model.clone()
        
        height, width = latent_image['samples'].shape[-2:]
        frames = latent_image['samples'].shape[-3] if latent_image['samples'].ndim == 5 else 1
        
        x1    = 256
        x2    = 4096
        mm    = (max_shift - base_shift) / (x2 - x1)
        b     = base_shift - mm * x1
        shift = (1 * width * height / (8 * 8 * 2 * 2)) * mm + b
        
        self.timestep_shift = shift
        self.multiplier     = 1000
        timesteps           = 1000
        
        if isinstance(m.model.model_config, comfy.supported_models.Flux) or isinstance(m.model.model_config, comfy.supported_models.FluxSchnell) or isinstance(m.model.model_config, comfy.supported_models.Chroma):
            self.multiplier = 1
            timesteps = 10000
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.AuraFlow):
            self.multiplier = 1
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.SD3):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.HiDream):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.HunyuanVideo):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST

        elif ANIMA_MODEL_CLASS is not None and isinstance(m.model.model_config, ANIMA_MODEL_CLASS):
            self.multiplier = 1
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        if isinstance(m.model.model_config, comfy.supported_models.WAN21_T2V) or isinstance(m.model.model_config, comfy.supported_models.WAN21_I2V):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST
            
        elif isinstance(m.model.model_config, comfy.supported_models.CosmosT2V) or isinstance(m.model.model_config, comfy.supported_models.CosmosI2V):
            self.multiplier = 1
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingContinuousEDM
            sampling_type = comfy.model_sampling.CONST

        elif isinstance(m.model.model_config, comfy.supported_models.LTXV):
            self.multiplier = 1000
            timesteps = 1000
            sampling_base = comfy.model_sampling.ModelSamplingFlux
            sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        m.object_patches['model_sampling'] = m.model.model_sampling = ModelSamplingAdvanced(m.model.model_config)

        m.model.model_sampling.__dict__['shift'] = self.timestep_shift
        m.model.model_sampling.__dict__['multiplier'] = self.multiplier

        s_range = torch.arange(1, timesteps + 1, 1).to(torch.float64)
        if scaling == "exponential": 
            ts = self.sigma_exponential((s_range / timesteps) * self.multiplier)
        elif scaling == "linear": 
            ts = self.sigma_linear((s_range / timesteps) * self.multiplier)

        m.model.model_sampling.register_buffer('sigmas', ts)
        m.object_patches['model_sampling'].sigmas = m.model.model_sampling.sigmas
        
        return (m,)
    
# Code adapted from https://github.com/comfyanonymous/ComfyUI/
class UNetSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":           ("MODEL",),
                "filename_prefix": ("STRING", {"default": "models/ComfyUI"}),
                },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
                },
            }
        
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "RES4LYF/model_merging"
    DESCRIPTION = "Save a .safetensors containing only the model data."

    def save(self, model, filename_prefix, prompt=None, extra_pnginfo=None):
        save_checkpoint(
            model, 
            clip            = None,
            vae             = None,
            filename_prefix = filename_prefix,
            output_dir      = self.output_dir,
            prompt          = prompt,
            extra_pnginfo   = extra_pnginfo,
            )
        
        return {}


def save_checkpoint(
        model,
        clip            = None,
        vae             = None,
        clip_vision     = None,
        filename_prefix = None,
        output_dir      = None,
        prompt          = None,
        extra_pnginfo   = None,
        ):
    
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
    prompt_info = ""
    if prompt is not None:
        prompt_info = json.dumps(prompt)

    metadata = {}

    enable_modelspec = True
    if isinstance(model.model, comfy.model_base.SDXL):
        if isinstance(model.model, comfy.model_base.SDXL_instructpix2pix):
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-edit"
        else:
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base"
    elif isinstance(model.model, comfy.model_base.SDXLRefiner):
        metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-refiner"
    elif isinstance(model.model, comfy.model_base.SVD_img2vid):
        metadata["modelspec.architecture"] = "stable-video-diffusion-img2vid-v1"
    elif isinstance(model.model, comfy.model_base.SD3):
        metadata["modelspec.architecture"] = "stable-diffusion-v3-medium" #TODO: other SD3 variants
    else:
        enable_modelspec = False

    if enable_modelspec:
        metadata["modelspec.sai_model_spec"] = "1.0.0"
        metadata["modelspec.implementation"] = "sgm"
        metadata["modelspec.title"] = "{} {}".format(filename, counter)

    #TODO:
    # "stable-diffusion-v1", "stable-diffusion-v1-inpainting", "stable-diffusion-v2-512",
    # "stable-diffusion-v2-768-v", "stable-diffusion-v2-unclip-l", "stable-diffusion-v2-unclip-h",
    # "v2-inpainting"

    extra_keys = {}
    model_sampling = model.get_model_object("model_sampling")
    if isinstance(model_sampling, comfy.model_sampling.ModelSamplingContinuousEDM):
        if isinstance(model_sampling, comfy.model_sampling.V_PREDICTION):
            extra_keys["edm_vpred.sigma_max"] = torch.tensor(model_sampling.sigma_max).float()
            extra_keys["edm_vpred.sigma_min"] = torch.tensor(model_sampling.sigma_min).float()

    if model.model.model_type == comfy.model_base.ModelType.EPS:
        metadata["modelspec.predict_key"] = "epsilon"
    elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
        metadata["modelspec.predict_key"] = "v"

    if not args.disable_metadata:
        metadata["prompt"] = prompt_info
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

    output_checkpoint = f"{filename}_{counter:05}_.safetensors"
    output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

    sd_save_checkpoint(output_checkpoint, model, clip, vae, clip_vision, metadata=metadata, extra_keys=extra_keys)


def sd_save_checkpoint(output_path, model, clip=None, vae=None, clip_vision=None, metadata=None, extra_keys={}):
    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.get_sd()

    comfy.model_management.load_models_gpu(load_models, force_patch_weights=True)
    clip_vision_sd = clip_vision.get_sd() if clip_vision is not None else None
    vae_sd = vae.get_sd() if vae is not None else None                             #THIS ALLOWS SAVING UNET ONLY
    sd = model.model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    for k in sd:
        t = sd[k]
        if not t.is_contiguous():
            sd[k] = t.contiguous()

    comfy.utils.save_torch_file(sd, output_path, metadata=metadata)



# Code adapted from https://github.com/kijai/ComfyUI-KJNodes
class TorchCompileModelFluxAdvanced: 
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model":         ("MODEL",),
                    "backend":       (["inductor", "cudagraphs"],),
                    "fullgraph":     ("BOOLEAN",                                                                    {"default": False, "tooltip": "Enable full graph mode"}),
                    "mode":          (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                    "double_blocks": ("STRING",                                                                     {"default": "0-18", "multiline": True}),
                    "single_blocks": ("STRING",                                                                     {"default": "0-37", "multiline": True}),
                    "dynamic":       ("BOOLEAN",                                                                    {"default": False, "tooltip": "Enable dynamic mode"}),
                }}
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/model_patches"

    def parse_blocks(self, blocks_str):
        blocks = []
        for part in blocks_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                blocks.extend(range(start, end + 1))
            else:
                blocks.append(int(part))
        return blocks

    def main(self,
            model,
            backend       = "inductor",
            mode          = "default",
            fullgraph     = False,
            single_blocks = "0-37",
            double_blocks = "0-18",
            dynamic       = False,
            ):
        
        single_block_list = self.parse_blocks(single_blocks)
        double_block_list = self.parse_blocks(double_blocks)
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        
        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.double_blocks):
                    if i in double_block_list:
                        m.add_object_patch(f"diffusion_model.double_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                for i, block in enumerate(diffusion_model.single_blocks):
                    if i in single_block_list:
                        m.add_object_patch(f"diffusion_model.single_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model. Verify that this is a Flux model!")
        
        return (m, )
        # rest of the layers that are not patched
        # diffusion_model.final_layer = torch.compile(diffusion_model.final_layer, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.guidance_in = torch.compile(diffusion_model.guidance_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.img_in = torch.compile(diffusion_model.img_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.time_in = torch.compile(diffusion_model.time_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.txt_in = torch.compile(diffusion_model.txt_in, mode=mode, fullgraph=fullgraph, backend=backend)
        # diffusion_model.vector_in = torch.compile(diffusion_model.vector_in, mode=mode, fullgraph=fullgraph, backend=backend)
        
        #   @torch.compile(mode="default", dynamic=False, fullgraph=False, backend="inductor")
        

class TorchCompileModelAura:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model":                   ("MODEL",),
                    "backend":                 (["inductor", "cudagraphs"],),
                    "fullgraph":               ("BOOLEAN",                    {"default": False,                                "tooltip": "Enable full graph mode"}),
                    "mode":                    (COMPILE_MODES               , {"default": "default"}),
                    "dynamic":                 ("BOOLEAN",                    {"default": False,                                "tooltip": "Enable dynamic mode"}),
                    "dynamo_cache_size_limit": ("INT",                        {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                }}
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/model_patches"

    def main(self,
            model,
            backend       = "inductor",
            mode          = "default",
            fullgraph     = False,
            dynamic       = False,
            dynamo_cache_size_limit = 64,
            ):

        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        
        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.double_layers):
                    m.add_object_patch(f"diffusion_model.double_layers.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                for i, block in enumerate(diffusion_model.single_layers):
                    m.add_object_patch(f"diffusion_model.single_layers.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model. Verify that this is an AuraFlow model!")
        
        return (m, )

class TorchCompileModelSD35:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "model":                   ("MODEL",),
                    "backend":                 (["inductor", "cudagraphs"],),
                    "fullgraph":               ("BOOLEAN",                    {"default": False,                                "tooltip": "Enable full graph mode"}),
                    "mode":                    (COMPILE_MODES               , {"default": "default"}),
                    "dynamic":                 ("BOOLEAN",                    {"default": False,                                "tooltip": "Enable dynamic mode"}),
                    "dynamo_cache_size_limit": ("INT",                        {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                }}
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION     = "main"
    CATEGORY     = "RES4LYF/model_patches"

    def main(self,
            model,
            backend       = "inductor",
            mode          = "default",
            fullgraph     = False,
            dynamic       = False,
            dynamo_cache_size_limit = 64,
            ):
        
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        
        if not self._compiled:
            try:
                for i, block in enumerate(diffusion_model.joint_blocks):
                    m.add_object_patch(f"diffusion_model.joint_blocks.{i}", torch.compile(block, mode=mode, dynamic=dynamic, fullgraph=fullgraph, backend=backend))
                self._compiled = True
                compile_settings = {
                    "backend"  : backend,
                    "mode"     : mode,
                    "fullgraph": fullgraph,
                    "dynamic"  : dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model. Verify that this is a SD3.5 model!")
        
        return (m, )


class ClownpileModelWanVideo:
    def __init__(self):
        self._compiled = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model"                     : ("MODEL",),
                "backend"                   : (["inductor","cudagraphs"], {"default" : "inductor"}),
                "fullgraph"                 : ("BOOLEAN",                 {"default"                : False, "tooltip"                   : "Enable full graph mode"}),
                "mode"                      : (COMPILE_MODES,             {"default": "default"}),
                "dynamic"                   : ("BOOLEAN",                 {"default"                : False, "tooltip"                   : "Enable dynamic mode"}),
                "dynamo_cache_size_limit"   : ("INT",                     {"default"                : 64, "min"                          : 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                #"compile_self_attn_blocks" : ("INT",                     {"default"                : 0, "min"                           : 0, "max": 100, "step" : 1, "tooltip": "Maximum blocks to compile. These use huge amounts of VRAM with large attention masks."}),
                "skip_self_attn_blocks"     : ("STRING",                  {"default"                 : "0,1,2,3,4,5,6,7,8,9,", "multiline": True, "tooltip": "For WAN only: select self-attn blocks to disable. Due to the size of the self-attn masks, VRAM required to compile blocks using regional WAN is excessive. List any blocks selected in the ReWanPatcher node."}),
                "compile_transformer_blocks": ("BOOLEAN",                 {"default"                : True,  "tooltip"                    : "Compile all transformer blocks"}),
                "force_recompile"           : ("BOOLEAN",                 {"default": False, "tooltip": "Force recompile."}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "RES4LYF/model"
    EXPERIMENTAL = True

    def patch(self, model, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, skip_self_attn_blocks, compile_transformer_blocks, force_recompile):
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")
        torch._dynamo.config.cache_size_limit = dynamo_cache_size_limit
        
        skip_self_attn_blocks = parse_range_string(skip_self_attn_blocks)
        
        if force_recompile:
            self._compiled = False
        
        if not self._compiled:
            try:
                if compile_transformer_blocks:
                    for i, block in enumerate(diffusion_model.blocks):
                        #if i % 2 == 1:
                        if i not in skip_self_attn_blocks:
                            compiled_block = torch.compile(block, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                            m.add_object_patch(f"diffusion_model.blocks.{i}", compiled_block)
                        #block.self_attn = torch.compile(block.self_attn, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                        #block.cross_attn = torch.compile(block.cross_attn, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                        #if i < compile_self_attn_blocks:
                        #    block.self_attn = torch.compile(block.self_attn, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                        #    #compiled_block = torch.compile(block, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                        #    #m.add_object_patch(f"diffusion_model.blocks.{i}", compiled_block)
                        #block.cross_attn = torch.compile(block.cross_attn, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode)
                self._compiled = True
                compile_settings = {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": fullgraph,
                    "dynamic": dynamic,
                }
                setattr(m.model, "compile_settings", compile_settings)
            except:
                raise RuntimeError("Failed to compile model. Verify that this is a WAN model!")
        return (m, )

