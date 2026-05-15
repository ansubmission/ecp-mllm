from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SiteProfile:
    site_id: str
    default_variant: str = "sff3c"
    allowed_variants: tuple[str, ...] = ("sff3c", "raw", "cfc_3channel")
    default_transport: str = "stitched_mp4"
    expected_regime: tuple[str, ...] = ()
    enabled_branches: tuple[str, ...] = ("custom_trickle_reduction", "custom_high_throughput", "custom_low_count")
    stream_primary_direction: str | None = None
    direction_prior_strength: float = 1.0
    recount_guardrail_level: str = "balanced"
    calibration_profile_id: str = "default"
    risk_thresholds: Mapping[str, float | int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SITE_PROFILES: dict[str, SiteProfile] = {
    "kenai": SiteProfile(
        site_id="kenai",
        expected_regime=("high_throughput", "low_count_sparse", "far_view"),
        enabled_branches=("custom_trickle_reduction", "custom_high_throughput", "custom_low_count"),
        calibration_profile_id="kenai_default",
        risk_thresholds={
            "low_count_total_max": 2,
            "low_count_candidate_max": 1,
            "low_count_peak_max": 1.0,
        },
    ),
    "rightbank": SiteProfile(
        site_id="rightbank",
        expected_regime=("low_count_sparse", "far_view"),
        enabled_branches=("custom_trickle_reduction", "custom_low_count"),
        calibration_profile_id="rightbank_sparse",
        risk_thresholds={
            "low_count_total_max": 2,
            "low_count_candidate_max": 1,
            "low_count_peak_max": 1.0,
        },
    ),
    "channel": SiteProfile(
        site_id="channel",
        expected_regime=("high_throughput",),
        enabled_branches=("custom_trickle_reduction", "custom_high_throughput"),
        calibration_profile_id="channel_dense",
        risk_thresholds={
            "low_count_total_max": 0,
        },
    ),
    "elwha": SiteProfile(
        site_id="elwha",
        expected_regime=("low_count_sparse", "echo_prone"),
        enabled_branches=("custom_trickle_reduction", "custom_low_count"),
        calibration_profile_id="elwha_echo",
        risk_thresholds={
            "low_count_total_max": 2,
            "low_count_candidate_max": 1,
            "low_count_peak_max": 1.0,
        },
    ),
    "nushagak": SiteProfile(
        site_id="nushagak",
        expected_regime=("stream_throughput",),
        enabled_branches=("custom_stream_throughput",),
        stream_primary_direction="left",
        calibration_profile_id="nushagak_stream",
        risk_thresholds={
            "low_count_total_max": 2,
            "low_count_candidate_max": 1,
            "low_count_peak_max": 1.0,
        },
    ),
    "generic": SiteProfile(site_id="generic"),
}


def infer_site_id(domain: str) -> str:
    value = str(domain or "").strip().lower()
    if value == "kenai-rightbank":
        return "rightbank"
    if value == "kenai-channel":
        return "channel"
    if value.startswith("kenai"):
        return "kenai"
    if value == "elwha":
        return "elwha"
    if value == "nushagak":
        return "nushagak"
    return value or "generic"


def infer_clip_hints(clip_key: str | None) -> tuple[str, ...]:
    value = str(clip_key or "").lower()
    hints: list[str] = []
    if "leftfar" in value or "rightfar" in value or "_lo_" in value or "stratum2" in value:
        hints.append("far_view")
    if "leftnear" in value or "rightnear" in value or "_ln_" in value:
        hints.append("near_view")
    if "stratum2" in value or "leftfar" in value or "rightfar" in value:
        hints.append("low_count_sparse")
    if "echo" in value or "persist" in value:
        hints.append("echo_prone")
    return tuple(dict.fromkeys(hints))


def resolve_site_profile(domain: str, *, clip_key: str | None = None) -> SiteProfile:
    site_id = infer_site_id(domain)
    base = DEFAULT_SITE_PROFILES.get(site_id) or DEFAULT_SITE_PROFILES["generic"]
    hints = infer_clip_hints(clip_key)
    if not hints:
        return base
    expected_regime = tuple(dict.fromkeys([*base.expected_regime, *hints]))
    return SiteProfile(
        site_id=base.site_id,
        default_variant=base.default_variant,
        allowed_variants=base.allowed_variants,
        default_transport=base.default_transport,
        expected_regime=expected_regime,
        enabled_branches=base.enabled_branches,
        stream_primary_direction=base.stream_primary_direction,
        direction_prior_strength=base.direction_prior_strength,
        recount_guardrail_level=base.recount_guardrail_level,
        calibration_profile_id=base.calibration_profile_id,
        risk_thresholds=dict(base.risk_thresholds),
    )
