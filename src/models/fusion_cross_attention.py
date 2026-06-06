from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.branch_audio_wav2vec import AudioBranch
from src.models.branch_visual_optflow import TemporalFlowBranch
from src.models.branch_visual_vit import SpatialViTBranch


FACE_VALID_EPS = 1e-6


def _face_valid_tokens(face_valid: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | None:
    if face_valid is None:
        return None
    valid = face_valid.to(device=like.device, dtype=like.dtype).clamp(0.0, 1.0)
    if valid.ndim == 1:
        return valid.view(-1, 1).expand(-1, like.size(1))
    if valid.ndim == 2:
        if valid.size(1) == like.size(1):
            return valid
        if valid.size(1) == 1:
            return valid.expand(-1, like.size(1))
    raise ValueError(f"face_valid must have shape [B] or [B,T], got {tuple(face_valid.shape)}")


def _face_valid_score(face_valid: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor | None:
    valid = _face_valid_tokens(face_valid, like)
    if valid is None:
        return None
    return valid.mean(dim=1)


def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return sequence.mean(dim=1)
    weights = mask.to(device=sequence.device, dtype=sequence.dtype).unsqueeze(-1).clamp(0.0, 1.0)
    denom = weights.sum(dim=1)
    summed = (sequence * weights).sum(dim=1)
    pooled = summed / denom.clamp_min(FACE_VALID_EPS)
    return torch.where(denom > 0.0, pooled, torch.zeros_like(pooled))


def _key_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    valid = mask > 0.5
    if bool(valid.all()):
        return None
    padding = ~valid
    all_invalid = padding.all(dim=1)
    if bool(all_invalid.any()):
        padding = padding.clone()
        padding[all_invalid] = False
    return padding


def _sinusoidal_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


def _add_temporal_positional_encoding(sequence: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        return sequence
    pe = _sinusoidal_encoding(sequence.size(1), sequence.size(2), sequence.device, sequence.dtype)
    return sequence + scale * pe.unsqueeze(0)


def _local_allowed(query_len: int, key_len: int, radius: int, device: torch.device) -> torch.Tensor:
    if query_len <= 0 or key_len <= 0:
        raise ValueError("query_len and key_len must be positive for local attention.")
    query_positions = torch.linspace(0.0, float(key_len - 1), query_len, device=device)
    key_positions = torch.arange(key_len, device=device, dtype=torch.float32)
    return (key_positions.unsqueeze(0) - query_positions.unsqueeze(1)).abs() <= float(radius)


def _local_attention_mask(
    valid_tokens: torch.Tensor | None,
    query_len: int,
    key_len: int,
    num_heads: int,
    radius: int | None,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if radius is None or radius < 0:
        return None, None
    local = _local_allowed(query_len, key_len, radius, device)
    if valid_tokens is None:
        return ~local, None

    valid = valid_tokens.to(device=device) > 0.5
    allowed_by_validity = local.unsqueeze(0) & valid.unsqueeze(1)
    query_has_valid_key = allowed_by_validity.any(dim=-1)
    allowed = torch.where(query_has_valid_key.unsqueeze(-1), allowed_by_validity, local.unsqueeze(0))
    mask = (~allowed).repeat_interleave(num_heads, dim=0)
    return mask, query_has_valid_key


def _temporal_gaussian_bias(
    query_len: int,
    key_len: int,
    sigma: float | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if sigma is None or sigma <= 0.0:
        return None
    if query_len <= 0 or key_len <= 0:
        raise ValueError("query_len and key_len must be positive for temporal attention bias.")
    query_positions = torch.linspace(0.0, float(key_len - 1), query_len, device=device, dtype=torch.float32)
    key_positions = torch.arange(key_len, device=device, dtype=torch.float32)
    distance = key_positions.unsqueeze(0) - query_positions.unsqueeze(1)
    bias = -(distance.square()) / (2.0 * float(sigma) * float(sigma))
    return bias.to(dtype=dtype)


def _combine_attention_masks(local_mask: torch.Tensor | None, temporal_bias: torch.Tensor | None) -> torch.Tensor | None:
    if local_mask is None:
        return temporal_bias
    if temporal_bias is None:
        return local_mask
    if local_mask.ndim == 3:
        combined = temporal_bias.unsqueeze(0).expand(local_mask.size(0), -1, -1).clone()
    else:
        combined = temporal_bias.clone()
    return combined.masked_fill(local_mask, -1e4)


def _add_key_validity_to_attention_mask(
    attention_mask: torch.Tensor | None,
    valid_tokens: torch.Tensor | None,
    query_len: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if valid_tokens is None:
        return attention_mask
    valid = valid_tokens.to(device=device) > 0.5
    if bool(valid.all()):
        return attention_mask

    invalid_keys = ~valid
    all_invalid = invalid_keys.all(dim=1)
    if bool(all_invalid.any()):
        invalid_keys = invalid_keys.clone()
        invalid_keys[all_invalid] = False
    invalid_mask = invalid_keys.unsqueeze(1).expand(-1, query_len, -1).repeat_interleave(num_heads, dim=0)

    if attention_mask is None:
        combined = torch.zeros(invalid_mask.shape, device=device, dtype=dtype)
    elif attention_mask.dtype == torch.bool:
        if attention_mask.ndim == 2:
            bool_mask = attention_mask.unsqueeze(0).expand(invalid_mask.size(0), -1, -1)
        else:
            bool_mask = attention_mask
        combined = torch.zeros(bool_mask.shape, device=device, dtype=dtype).masked_fill(bool_mask, -1e4)
    elif attention_mask.ndim == 2:
        combined = attention_mask.unsqueeze(0).expand(invalid_mask.size(0), -1, -1).clone().to(device=device, dtype=dtype)
    else:
        combined = attention_mask.clone().to(device=device, dtype=dtype)
    return combined.masked_fill(invalid_mask, -1e4)


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        use_temporal_positional_encoding: bool = False,
        local_attention_radius: int | None = None,
        temporal_position_scale: float = 1.0,
        temporal_attention_sigma: float | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.use_temporal_positional_encoding = bool(use_temporal_positional_encoding)
        self.local_attention_radius = None if local_attention_radius is None else int(local_attention_radius)
        self.temporal_position_scale = float(temporal_position_scale)
        self.temporal_attention_sigma = None if temporal_attention_sigma is None else float(temporal_attention_sigma)
        self.visual_projection = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.audio_attention_norm = nn.LayerNorm(dim)
        self.visual_attention_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attention_dropout = nn.Dropout(0.1)
        self.post_attention_norm = nn.LayerNorm(dim)
        self.fusion_projection = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, 2)

    def forward(
        self,
        spatial: torch.Tensor,
        flow: torch.Tensor,
        audio: torch.Tensor,
        face_valid: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        visual = self.visual_projection(torch.cat([spatial, flow], dim=-1))
        valid_tokens = _face_valid_tokens(face_valid, visual)
        audio_query = self.audio_attention_norm(audio)
        visual_context = self.visual_attention_norm(visual)
        if self.use_temporal_positional_encoding:
            audio_query = _add_temporal_positional_encoding(audio_query, self.temporal_position_scale)
            visual_context = _add_temporal_positional_encoding(visual_context, self.temporal_position_scale)
        local_mask, query_has_valid_key = _local_attention_mask(
            valid_tokens,
            audio_query.size(1),
            visual_context.size(1),
            self.num_heads,
            self.local_attention_radius,
            visual_context.device,
        )
        temporal_bias = _temporal_gaussian_bias(
            audio_query.size(1),
            visual_context.size(1),
            self.temporal_attention_sigma,
            visual_context.device,
            audio_query.dtype,
        )
        attention_mask = _combine_attention_masks(local_mask, temporal_bias)
        key_padding_mask = None if local_mask is not None else _key_padding_mask(valid_tokens)
        if local_mask is None and temporal_bias is not None:
            attention_mask = _add_key_validity_to_attention_mask(
                attention_mask,
                valid_tokens,
                audio_query.size(1),
                self.num_heads,
                visual_context.device,
                audio_query.dtype,
            )
            key_padding_mask = None
        attention_delta, _ = self.cross_attention(
            audio_query,
            visual_context,
            visual_context,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        valid_score = _face_valid_score(face_valid, visual)
        if query_has_valid_key is not None:
            attention_delta = attention_delta * query_has_valid_key.to(attention_delta.dtype).unsqueeze(-1)
        elif valid_score is not None:
            attention_delta = attention_delta * (valid_score > 0.0).to(attention_delta.dtype).view(-1, 1, 1)
        attended = self.post_attention_norm(audio + self.attention_dropout(attention_delta))
        fused = self.fusion_projection(torch.cat([audio, attended, visual], dim=-1))
        lstm_out, _ = self.lstm(fused)
        pooled = _masked_mean(lstm_out, valid_tokens)
        logits = self.classifier(pooled)
        if return_features:
            return logits, {
                "visual": visual,
                "attention_delta": attention_delta,
                "attended": attended,
                "fused": fused,
                "pooled": pooled,
                "logits": logits,
            }
        return logits


class SingleStreamTemporalHead(nn.Module):
    def __init__(self, dim: int = 256, lstm_hidden: int = 128, lstm_layers: int = 2) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, 2)

    def forward(
        self,
        features: torch.Tensor,
        face_valid: torch.Tensor | None = None,
        return_pooled: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        normalized = self.input_norm(features)
        lstm_out, _ = self.lstm(normalized)
        pooled = _masked_mean(lstm_out, _face_valid_tokens(face_valid, features))
        logits = self.classifier(pooled)
        if return_pooled:
            return logits, pooled
        return logits


class GatedLogitFusion(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        gate_hidden: int = 128,
        stream_dropout: float = 0.0,
        gate_init_weights: tuple[float, float, float] | None = None,
        stream_logit_temperature: tuple[float, float, float] | None = None,
        logit_fusion_mode: str = "weighted_logits",
        margin_clip: float | None = None,
        reliability_confidence_gamma: float = 1.0,
        reliability_confidence_floor: float = 0.0,
        reliability_confidence_max_weight: float | None = None,
        majority_consistency_margin: float | None = None,
        vote_consensus_bonus: float = 0.0,
        audio_reliability: float = 1.0,
        confidence_detach: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_head = SingleStreamTemporalHead(dim, lstm_hidden, lstm_layers)
        self.flow_head = SingleStreamTemporalHead(dim, lstm_hidden, lstm_layers)
        self.audio_head = SingleStreamTemporalHead(dim, lstm_hidden, lstm_layers)
        self.gate_norm_spatial = nn.LayerNorm(dim)
        self.gate_norm_flow = nn.LayerNorm(dim)
        self.gate_norm_audio = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gate_hidden, 3),
        )
        self._init_gate(gate_init_weights)
        self.stream_dropout = float(stream_dropout)
        temperatures = torch.tensor(stream_logit_temperature or (1.0, 1.0, 1.0), dtype=torch.float32)
        if temperatures.numel() != 3 or torch.any(temperatures <= 0):
            raise ValueError("stream_logit_temperature must contain three positive values.")
        self.register_buffer("stream_logit_temperature", temperatures.view(1, 3, 1), persistent=False)
        self.logit_fusion_mode = logit_fusion_mode.lower()
        if self.logit_fusion_mode not in {"weighted_logits", "normalized_margin", "reliability_confidence", "vote_score_sum"}:
            raise ValueError(f"Unsupported logit_fusion_mode: {logit_fusion_mode}")
        self.margin_clip = margin_clip
        self.reliability_confidence_gamma = float(reliability_confidence_gamma)
        if self.reliability_confidence_gamma <= 0.0:
            raise ValueError("reliability_confidence_gamma must be positive.")
        self.reliability_confidence_floor = float(reliability_confidence_floor)
        if self.reliability_confidence_floor < 0.0:
            raise ValueError("reliability_confidence_floor must be non-negative.")
        self.reliability_confidence_max_weight = (
            None if reliability_confidence_max_weight is None else float(reliability_confidence_max_weight)
        )
        if self.reliability_confidence_max_weight is not None:
            if self.reliability_confidence_max_weight < (1.0 / 3.0) or self.reliability_confidence_max_weight > 1.0:
                raise ValueError("reliability_confidence_max_weight must be in [1/3, 1].")
        self.majority_consistency_margin = None if majority_consistency_margin is None else float(majority_consistency_margin)
        if self.majority_consistency_margin is not None and self.majority_consistency_margin < 0.0:
            raise ValueError("majority_consistency_margin must be non-negative.")
        self.vote_consensus_bonus = float(vote_consensus_bonus)
        if self.vote_consensus_bonus < 0.0:
            raise ValueError("vote_consensus_bonus must be non-negative.")
        self.audio_reliability = float(audio_reliability)
        if self.audio_reliability < 0.0:
            raise ValueError("audio_reliability must be non-negative.")
        self.confidence_detach = bool(confidence_detach)

    def _init_gate(self, gate_init_weights: tuple[float, float, float] | None) -> None:
        final = self.gate[-1]
        assert isinstance(final, nn.Linear)
        weights = gate_init_weights or (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        prior = torch.tensor(weights, dtype=torch.float32)
        if prior.numel() != 3 or torch.any(prior <= 0):
            raise ValueError("gate_init_weights must contain three positive values.")
        prior = prior / prior.sum()
        with torch.no_grad():
            final.weight.zero_()
            final.bias.copy_(prior.log())

    def _pooled_for_gate(
        self,
        spatial: torch.Tensor,
        flow: torch.Tensor,
        audio: torch.Tensor,
        visual_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_spatial = _masked_mean(self.gate_norm_spatial(spatial), visual_valid)
        pooled_flow = _masked_mean(self.gate_norm_flow(flow), visual_valid)
        pooled_audio = self.gate_norm_audio(audio).mean(dim=1)
        return torch.cat([pooled_spatial, pooled_flow, pooled_audio], dim=-1)

    def _apply_stream_dropout(self, gate_logits: torch.Tensor) -> torch.Tensor:
        if not self.training or self.stream_dropout <= 0.0:
            return gate_logits
        keep = torch.rand(gate_logits.shape, device=gate_logits.device) >= self.stream_dropout
        all_dropped = ~keep.any(dim=1)
        if all_dropped.any():
            fallback = torch.randint(0, keep.size(1), (int(all_dropped.sum()),), device=keep.device)
            keep[all_dropped] = False
            keep[all_dropped, fallback] = True
        return gate_logits.masked_fill(~keep, -1e4)

    def _calibrate_stream_logits(self, stream_logits: torch.Tensor) -> torch.Tensor:
        return stream_logits / self.stream_logit_temperature.to(stream_logits.device, stream_logits.dtype)

    def _apply_reliability_confidence_cap(self, weights: torch.Tensor) -> torch.Tensor:
        if self.reliability_confidence_max_weight is None:
            return weights
        cap = torch.full_like(weights, self.reliability_confidence_max_weight)
        capped = weights
        locked = torch.zeros_like(weights, dtype=torch.bool)
        zeros = torch.zeros_like(weights)
        for _ in range(weights.size(1)):
            active = ~locked
            locked_sum = torch.where(locked, capped, zeros).sum(dim=1, keepdim=True)
            remaining = (1.0 - locked_sum).clamp_min(0.0)
            active_base = torch.where(active, weights, zeros)
            active_sum = active_base.sum(dim=1, keepdim=True)
            active_count = active.sum(dim=1, keepdim=True).to(dtype=weights.dtype).clamp_min(1.0)
            redistributed = active_base / active_sum.clamp_min(FACE_VALID_EPS) * remaining
            uniform = active.to(dtype=weights.dtype) * (remaining / active_count)
            capped = torch.where(active, torch.where(active_sum > FACE_VALID_EPS, redistributed, uniform), capped)
            over_cap = (capped > cap) & active
            capped = torch.where(over_cap, cap, capped)
            locked = locked | over_cap
        return capped / capped.sum(dim=1, keepdim=True).clamp_min(FACE_VALID_EPS)

    def _apply_majority_consistency(
        self,
        logits: torch.Tensor,
        calibrated_stream_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.majority_consistency_margin is None:
            return logits, {}
        lie_votes = (calibrated_stream_logits[:, :, 1] >= calibrated_stream_logits[:, :, 0]).to(logits.dtype).sum(dim=1)
        majority_sign = torch.where(
            lie_votes >= 2.0,
            torch.ones_like(lie_votes),
            -torch.ones_like(lie_votes),
        )
        fused_margin = logits[:, 1] - logits[:, 0]
        adjusted_margin = majority_sign * fused_margin.abs().clamp_min(self.majority_consistency_margin)
        adjusted_logits = torch.stack([-0.5 * adjusted_margin, 0.5 * adjusted_margin], dim=1)
        applied = (fused_margin.sign() != majority_sign).to(logits.dtype)
        return adjusted_logits, {
            "majority_lie_votes": lie_votes,
            "majority_consistency_applied": applied,
            "majority_margin": adjusted_margin,
        }

    def _fuse_stream_logits(
        self,
        calibrated_stream_logits: torch.Tensor,
        gate_weights: torch.Tensor,
        visual_reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.logit_fusion_mode == "weighted_logits":
            logits = torch.sum(gate_weights.unsqueeze(-1) * calibrated_stream_logits, dim=1)
            return logits, {}

        if self.logit_fusion_mode in {"reliability_confidence", "vote_score_sum"}:
            probabilities = torch.softmax(calibrated_stream_logits, dim=-1)
            if visual_reliability is None:
                visual = torch.ones(
                    (calibrated_stream_logits.size(0), 1),
                    device=calibrated_stream_logits.device,
                    dtype=calibrated_stream_logits.dtype,
                )
            else:
                visual = visual_reliability.to(calibrated_stream_logits.device, calibrated_stream_logits.dtype).view(-1, 1).clamp(0.0, 1.0)
            audio = torch.full_like(visual, self.audio_reliability)
            reliability = torch.cat([visual, visual, audio], dim=1)

        if self.logit_fusion_mode == "reliability_confidence":
            confidence = (probabilities[:, :, 1] - 0.5).abs()
            confidence = confidence.pow(self.reliability_confidence_gamma)
            confidence_for_weight = confidence + self.reliability_confidence_floor
            if self.confidence_detach:
                confidence = confidence.detach()
                confidence_for_weight = confidence_for_weight.detach()
                reliability = reliability.detach()
            raw_weights = reliability * confidence_for_weight
            weight_sum = raw_weights.sum(dim=1, keepdim=True)
            confidence_weights = raw_weights / weight_sum.clamp_min(FACE_VALID_EPS)
            capped_weights = self._apply_reliability_confidence_cap(confidence_weights)
            fallback = weight_sum <= FACE_VALID_EPS
            fusion_weights = torch.where(fallback, gate_weights, capped_weights)
            logits = torch.sum(fusion_weights.unsqueeze(-1) * calibrated_stream_logits, dim=1)
            logits, majority_features = self._apply_majority_consistency(logits, calibrated_stream_logits)
            return logits, {
                "fusion_weights": fusion_weights,
                "fusion_weights_uncapped": confidence_weights,
                "fusion_cap_active": (capped_weights - confidence_weights).abs().sum(dim=1),
                "fusion_confidence": confidence,
                "fusion_confidence_for_weight": confidence_for_weight,
                "fusion_reliability": reliability,
                "fusion_raw_weights": raw_weights,
                "fusion_weight_sum": weight_sum.squeeze(1),
                "learned_gate_weights": gate_weights,
                **majority_features,
            }

        if self.logit_fusion_mode == "vote_score_sum":
            vote_class = probabilities.argmax(dim=-1)
            vote_probability = probabilities.gather(2, vote_class.unsqueeze(-1)).squeeze(-1)
            # Keep support differentiable; in this mode support itself forms the final logit.
            vote_support = reliability.detach() * (self.reliability_confidence_floor + vote_probability.pow(self.reliability_confidence_gamma))
            lie_mask = (vote_class == 1).to(calibrated_stream_logits.dtype)
            truth_mask = 1.0 - lie_mask
            lie_support = (vote_support * lie_mask).sum(dim=1)
            truth_support = (vote_support * truth_mask).sum(dim=1)
            lie_vote_count = lie_mask.sum(dim=1)
            truth_vote_count = truth_mask.sum(dim=1)
            if self.vote_consensus_bonus > 0.0:
                lie_bonus = 1.0 + self.vote_consensus_bonus * (lie_vote_count - 1.0).clamp_min(0.0)
                truth_bonus = 1.0 + self.vote_consensus_bonus * (truth_vote_count - 1.0).clamp_min(0.0)
                lie_support = lie_support * lie_bonus
                truth_support = truth_support * truth_bonus
            vote_margin = lie_support - truth_support
            logits = torch.stack([-0.5 * vote_margin, 0.5 * vote_margin], dim=1)
            support_sum = vote_support.sum(dim=1, keepdim=True)
            support_weights = vote_support / support_sum.clamp_min(FACE_VALID_EPS)
            return logits, {
                "fusion_weights": support_weights,
                "vote_class": vote_class.to(calibrated_stream_logits.dtype),
                "vote_probability": vote_probability,
                "vote_support": vote_support,
                "vote_lie_support": lie_support,
                "vote_truth_support": truth_support,
                "vote_lie_count": lie_vote_count,
                "vote_truth_count": truth_vote_count,
                "vote_margin": vote_margin,
                "fusion_reliability": reliability.detach(),
                "learned_gate_weights": gate_weights,
            }

        stream_margins = calibrated_stream_logits[:, :, 1] - calibrated_stream_logits[:, :, 0]
        margin_scale = torch.sqrt(torch.mean(stream_margins.square(), dim=1, keepdim=True)).clamp_min(1e-4)
        normalized_margins = stream_margins / margin_scale
        if self.margin_clip is not None and self.margin_clip > 0.0:
            normalized_margins = normalized_margins.clamp(-self.margin_clip, self.margin_clip)
        fused_margin = torch.sum(gate_weights * normalized_margins, dim=1)
        logits = torch.stack([-0.5 * fused_margin, 0.5 * fused_margin], dim=1)
        return logits, {
            "stream_margins": stream_margins,
            "normalized_stream_margins": normalized_margins,
            "fused_margin": fused_margin,
            "margin_scale": margin_scale.squeeze(1),
        }

    def forward(
        self,
        spatial: torch.Tensor,
        flow: torch.Tensor,
        audio: torch.Tensor,
        face_valid: torch.Tensor | None = None,
        return_features: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid_tokens = _face_valid_tokens(face_valid, spatial)
        valid_score = _face_valid_score(face_valid, spatial)
        logits_spatial, pooled_spatial = self.spatial_head(spatial, face_valid=valid_tokens, return_pooled=True)
        logits_flow, pooled_flow = self.flow_head(flow, face_valid=valid_tokens, return_pooled=True)
        logits_audio, pooled_audio = self.audio_head(audio, return_pooled=True)
        gate_logits = self.gate(self._pooled_for_gate(spatial, flow, audio, valid_tokens))
        valid: torch.Tensor | None = None
        if valid_score is not None:
            valid = valid_score.to(device=gate_logits.device, dtype=gate_logits.dtype).view(-1, 1).clamp(0.0, 1.0)
            gate_logits = gate_logits.clone()
            gate_logits[:, :2] = gate_logits[:, :2] + (valid + FACE_VALID_EPS).log()
        dropped_gate_logits = self._apply_stream_dropout(gate_logits)
        if valid is not None:
            invalid = valid <= FACE_VALID_EPS
            dropped_gate_logits = dropped_gate_logits.clone()
            dropped_gate_logits[:, :2] = dropped_gate_logits[:, :2].masked_fill(invalid, -1e4)
            dropped_gate_logits[:, 2:3] = torch.where(invalid, gate_logits[:, 2:3], dropped_gate_logits[:, 2:3])
        gate_weights = torch.softmax(dropped_gate_logits, dim=-1)
        stream_logits = torch.stack([logits_spatial, logits_flow, logits_audio], dim=1)
        calibrated_stream_logits = self._calibrate_stream_logits(stream_logits)
        logits, margin_features = self._fuse_stream_logits(calibrated_stream_logits, gate_weights, valid)
        effective_gate_weights = margin_features.get("fusion_weights", gate_weights)

        aux = {
            "logits": logits,
            "logits_spatial": logits_spatial,
            "logits_flow": logits_flow,
            "logits_audio": logits_audio,
            "gate_weights": effective_gate_weights,
            "gate_logits": gate_logits,
        }
        if "learned_gate_weights" in margin_features:
            aux["learned_gate_weights"] = margin_features["learned_gate_weights"]
        if valid_score is not None:
            aux["face_valid"] = valid_score.to(device=gate_logits.device, dtype=gate_logits.dtype).view(-1)
        if return_features:
            features = {
                "pooled_spatial": pooled_spatial,
                "pooled_flow": pooled_flow,
                "pooled_audio": pooled_audio,
                "logits": logits,
                "logits_spatial": logits_spatial,
                "logits_flow": logits_flow,
                "logits_audio": logits_audio,
                "calibrated_logits_spatial": calibrated_stream_logits[:, 0],
                "calibrated_logits_flow": calibrated_stream_logits[:, 1],
                "calibrated_logits_audio": calibrated_stream_logits[:, 2],
                "gate_logits": gate_logits,
                "gate_weights": effective_gate_weights,
                "gate_spatial": effective_gate_weights[:, 0],
                "gate_flow": effective_gate_weights[:, 1],
                "gate_audio": effective_gate_weights[:, 2],
                **margin_features,
            }
            if valid_score is not None:
                features["face_valid"] = valid_score.to(device=gate_logits.device, dtype=gate_logits.dtype).view(-1)
                features["face_valid_tokens"] = valid_tokens.to(device=gate_logits.device, dtype=gate_logits.dtype)
            return logits, features
        if return_aux:
            return aux
        return logits


class ThreeStreamModel(nn.Module):
    def __init__(
        self,
        vit_name: str,
        wav2vec_name: str,
        projection_dim: int = 256,
        attention_heads: int = 4,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        freeze_vit: bool = True,
        freeze_wav2vec: bool = True,
        stream_mode: str = "all",
        gate_hidden_dim: int = 128,
        stream_dropout: float = 0.0,
        gate_init_weights: tuple[float, float, float] | None = None,
        stream_logit_temperature: tuple[float, float, float] | None = None,
        logit_fusion_mode: str = "weighted_logits",
        margin_clip: float | None = None,
        reliability_confidence_gamma: float = 1.0,
        reliability_confidence_floor: float = 0.0,
        reliability_confidence_max_weight: float | None = None,
        majority_consistency_margin: float | None = None,
        vote_consensus_bonus: float = 0.0,
        audio_reliability: float = 1.0,
        confidence_detach: bool = True,
        use_temporal_positional_encoding: bool = False,
        local_attention_radius: int | None = None,
        temporal_position_scale: float = 1.0,
        temporal_attention_sigma: float | None = None,
    ) -> None:
        super().__init__()
        self.stream_mode = stream_mode.lower()
        if self.stream_mode not in {"all", "spatial", "rgb", "flow", "audio", "gated"}:
            raise ValueError(f"Unsupported stream_mode: {stream_mode}")
        self.spatial = (
            SpatialViTBranch(vit_name, projection_dim, freeze_vit)
            if self.stream_mode in {"all", "spatial", "rgb", "gated"}
            else None
        )
        self.flow = TemporalFlowBranch(projection_dim) if self.stream_mode in {"all", "flow", "gated"} else None
        self.audio = (
            AudioBranch(wav2vec_name, projection_dim, t_out=16, num_heads=attention_heads, freeze=freeze_wav2vec)
            if self.stream_mode in {"all", "audio", "gated"}
            else None
        )
        self.fusion = (
            CrossModalFusion(
                projection_dim,
                attention_heads,
                lstm_hidden_dim,
                lstm_layers,
                use_temporal_positional_encoding,
                local_attention_radius,
                temporal_position_scale,
                temporal_attention_sigma,
            )
            if self.stream_mode == "all"
            else None
        )
        self.gated_fusion = (
            GatedLogitFusion(
                projection_dim,
                lstm_hidden_dim,
                lstm_layers,
                gate_hidden_dim,
                stream_dropout,
                gate_init_weights,
                stream_logit_temperature,
                logit_fusion_mode,
                margin_clip,
                reliability_confidence_gamma,
                reliability_confidence_floor,
                reliability_confidence_max_weight,
                majority_consistency_margin,
                vote_consensus_bonus,
                audio_reliability,
                confidence_detach,
            )
            if self.stream_mode == "gated"
            else None
        )
        self.single_head = SingleStreamTemporalHead(projection_dim, lstm_hidden_dim, lstm_layers) if self.stream_mode not in {"all", "gated"} else None

    def forward(
        self,
        rgb_frames: torch.Tensor,
        optflow_frames: torch.Tensor,
        audio_waveform: torch.Tensor,
        face_valid: torch.Tensor | None = None,
        return_features: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.stream_mode == "all":
            assert self.spatial is not None and self.flow is not None and self.audio is not None and self.fusion is not None
            spatial = self.spatial(rgb_frames)
            flow = self.flow(optflow_frames)
            audio = self.audio(audio_waveform)
            if return_features:
                logits, features = self.fusion(spatial, flow, audio, face_valid=face_valid, return_features=True)
                features.update({"spatial": spatial, "flow": flow, "audio": audio})
                return logits, features
            return self.fusion(spatial, flow, audio, face_valid=face_valid)

        if self.stream_mode == "gated":
            assert self.spatial is not None and self.flow is not None and self.audio is not None and self.gated_fusion is not None
            spatial = self.spatial(rgb_frames)
            flow = self.flow(optflow_frames)
            audio = self.audio(audio_waveform)
            if return_features:
                logits, features = self.gated_fusion(spatial, flow, audio, face_valid=face_valid, return_features=True)
                features.update({"spatial": spatial, "flow": flow, "audio": audio})
                return logits, features
            return self.gated_fusion(spatial, flow, audio, face_valid=face_valid, return_aux=return_aux)

        if self.stream_mode in {"spatial", "rgb"}:
            assert self.spatial is not None
            features = self.spatial(rgb_frames)
            feature_name = "spatial"
        elif self.stream_mode == "flow":
            assert self.flow is not None
            features = self.flow(optflow_frames)
            feature_name = "flow"
        else:
            assert self.audio is not None
            features = self.audio(audio_waveform)
            feature_name = "audio"

        assert self.single_head is not None
        stream_face_valid = face_valid if self.stream_mode in {"spatial", "rgb", "flow"} else None
        logits = self.single_head(features, face_valid=stream_face_valid)
        if return_features:
            return logits, {feature_name: features, "logits": logits}
        return logits
