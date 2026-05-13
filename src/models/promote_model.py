from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.models.register_model import _load_registry, _save_registry
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _candidate_aliases(preferred_alias: str) -> tuple[str, ...]:
    aliases = [preferred_alias]
    for fallback in ("Candidate", "Challenger"):
        if fallback not in aliases:
            aliases.append(fallback)
    return tuple(aliases)


def _metric(metadata: dict[str, Any], name: str) -> float:
    return float(metadata.get("metrics", {}).get(name, float("inf")))


def _load_promotion_gate() -> dict[str, Any]:
    try:
        return get_settings().promotion_gate()
    except Exception:
        return {}


def promotion_checks(
    candidate: dict[str, Any],
    champion: dict[str, Any] | None,
    max_mae_degradation_pct: float | None = None,
    max_rmse_degradation_pct: float | None = None,
) -> tuple[bool, list[str]]:
    gate = _load_promotion_gate()
    if max_mae_degradation_pct is None:
        max_mae_degradation_pct = gate.get("max_mae_degradation_pct", 0.05)
    if max_rmse_degradation_pct is None:
        max_rmse_degradation_pct = gate.get("max_rmse_degradation_pct", 0.05)
    max_terminal_deg = gate.get("max_terminal_mae_degradation_pct", 0.10)

    reasons: list[str] = []
    if candidate.get("validation_status") != "passed":
        reasons.append("candidate validation_status is not passed")
    if not candidate.get("feature_columns"):
        reasons.append("candidate feature metadata is missing")

    candidate_terminal_metrics = candidate.get("terminal_metrics", {})
    if "D+1" not in candidate_terminal_metrics or "D+2" not in candidate_terminal_metrics:
        reasons.append("candidate D+1/D+2 metrics are missing")

    # Absolute error limits from thresholds.yml (agreed with Camila/Ana)
    for horizon, key in [("D+1", "max_mae_d1"), ("D+2", "max_mae_d2"), ("D+3", "max_mae_d3")]:
        limit = gate.get(key)
        if limit is None:
            continue
        horizon_metrics = candidate_terminal_metrics.get(horizon, {})
        candidate_h_mae = float(horizon_metrics.get("mae", float("inf")))
        if candidate_h_mae != float("inf") and candidate_h_mae > limit:
            reasons.append(
                f"candidate MAE for {horizon} ({candidate_h_mae:.2f} t/h) exceeds "
                f"business limit of {limit} t/h (see thresholds.yml)"
            )

    max_mape = gate.get("max_mape")
    if max_mape is not None:
        candidate_mape = _metric(candidate, "mape")
        if candidate_mape != float("inf") and candidate_mape > max_mape:
            reasons.append(
                f"candidate MAPE {candidate_mape:.3f} exceeds limit of {max_mape:.3f}"
            )

    min_r2 = gate.get("min_r2")
    if min_r2 is not None:
        candidate_r2 = float(candidate.get("metrics", {}).get("r2", 0))
        if candidate_r2 < min_r2:
            reasons.append(
                f"candidate R² {candidate_r2:.3f} is below minimum of {min_r2:.3f}"
            )

    if not champion:
        return len(reasons) == 0, reasons

    # Relative degradation vs current Champion
    candidate_mae = _metric(candidate, "mae")
    champion_mae = _metric(champion, "mae")
    candidate_rmse = _metric(candidate, "rmse")
    champion_rmse = _metric(champion, "rmse")
    if candidate_mae > champion_mae * (1 + max_mae_degradation_pct):
        reasons.append(
            f"candidate MAE {candidate_mae:.3f} degrades beyond "
            f"{max_mae_degradation_pct * 100:.0f}% allowed threshold vs champion {champion_mae:.3f}"
        )
    if candidate_rmse > champion_rmse * (1 + max_rmse_degradation_pct):
        reasons.append(
            f"candidate RMSE {candidate_rmse:.3f} degrades beyond "
            f"{max_rmse_degradation_pct * 100:.0f}% allowed threshold vs champion {champion_rmse:.3f}"
        )

    champion_terminal_metrics = champion.get("terminal_metrics", {})
    for key, metrics in candidate_terminal_metrics.items():
        if key not in champion_terminal_metrics:
            continue
        candidate_key_mae = float(metrics.get("mae", float("inf")))
        champion_key_mae = float(champion_terminal_metrics[key].get("mae", float("inf")))
        if candidate_key_mae > champion_key_mae * (1 + max_terminal_deg):
            reasons.append(
                f"{key} MAE degrades by more than {max_terminal_deg * 100:.0f}% vs champion"
            )
    return len(reasons) == 0, reasons


def _record_promotion_history(
    registry: dict[str, Any],
    candidate_version: str,
    champion_alias: str,
    previous_production_version: str | None,
) -> None:
    history = registry.setdefault("promotion_history", [])
    history.append({
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "new_production_version": str(candidate_version),
        "previous_production_version": str(previous_production_version) if previous_production_version else None,
        "champion_alias": champion_alias,
    })
    # Keep last 20 entries so the file doesn't grow unbounded
    registry["promotion_history"] = history[-20:]


def promote_local_candidate(candidate_alias: str = "Candidate", champion_alias: str = "Champion") -> dict[str, Any]:
    registry = _load_registry()
    aliases = registry.get("aliases", {})
    versions = registry.get("versions", {})
    selected_alias = next((alias for alias in _candidate_aliases(candidate_alias) if aliases.get(alias)), None)
    candidate_version = aliases.get(selected_alias) if selected_alias else None
    if not candidate_version:
        raise RuntimeError(f"No local model alias named one of {_candidate_aliases(candidate_alias)}")
    candidate = versions[str(candidate_version)]
    champion_version = aliases.get(champion_alias) or aliases.get("Production")
    champion = versions.get(str(champion_version)) if champion_version else None
    passed, reasons = promotion_checks(candidate, champion)
    if not passed:
        raise RuntimeError("Promotion checks failed:\n  - " + "\n  - ".join(reasons))

    previous_production = aliases.get("Production")
    aliases[champion_alias] = str(candidate_version)
    aliases["Production"] = str(candidate_version)
    registry["aliases"] = aliases
    _record_promotion_history(registry, candidate_version, champion_alias, previous_production)
    _save_registry(registry)
    LOGGER.info("Promoted local %s model version %s to %s/Production", selected_alias, candidate_version, champion_alias)
    return candidate


def rollback_to_previous_champion(target_version: str | None = None) -> dict[str, Any]:
    """Restore Production to the previous Champion version (or a specific version).

    Usage:
        python -m src.models.promote_model --rollback
        python -m src.models.promote_model --rollback --target-version 3
    """
    registry = _load_registry()
    aliases = registry.get("aliases", {})
    versions = registry.get("versions", {})

    if target_version:
        if str(target_version) not in versions:
            raise RuntimeError(f"Version {target_version} not found in local registry")
        rollback_version = str(target_version)
    else:
        history = registry.get("promotion_history", [])
        if len(history) < 2:
            raise RuntimeError(
                "No previous promotion history found. "
                "Cannot roll back — set --target-version explicitly."
            )
        previous = history[-2]
        rollback_version = previous.get("new_production_version")
        if not rollback_version or rollback_version not in versions:
            raise RuntimeError(
                f"Previous production version {rollback_version!r} not found in local registry. "
                "Use --target-version to specify an available version."
            )

    current_production = aliases.get("Production")
    aliases["Production"] = rollback_version
    aliases["Champion"] = rollback_version
    registry["aliases"] = aliases
    _record_promotion_history(registry, rollback_version, "Champion (rollback)", current_production)
    _save_registry(registry)
    rolled_back = versions[rollback_version]
    LOGGER.warning(
        "ROLLBACK: Production reverted from version %s to version %s",
        current_production, rollback_version,
    )
    return rolled_back


def promote_mlflow_candidate(candidate_alias: str = "Candidate", champion_alias: str = "Champion") -> None:
    settings = get_settings()
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        LOGGER.warning("MLflow is not installed; skipped MLflow alias promotion")
        return
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    try:
        candidate_version = None
        selected_alias = None
        for alias in _candidate_aliases(candidate_alias):
            try:
                candidate_version = client.get_model_version_by_alias(settings.model_name, alias)
                selected_alias = alias
                break
            except Exception:
                continue
        if candidate_version is None:
            raise RuntimeError(f"No MLflow model alias named one of {_candidate_aliases(candidate_alias)}")
        try:
            champion_version = client.get_model_version_by_alias(settings.model_name, champion_alias)
        except Exception:
            try:
                champion_version = client.get_model_version_by_alias(settings.model_name, "Production")
            except Exception:
                champion_version = None

        def metadata(model_version) -> dict[str, Any] | None:
            if not model_version:
                return None
            details = client.get_model_version(settings.model_name, model_version.version)
            tags = dict(getattr(details, "tags", {}) or {})
            return {
                "model_name": settings.model_name,
                "model_version": str(model_version.version),
                "validation_status": tags.get("validation_status"),
                "metrics": {
                    "mae": float(tags.get("metric_mae", "inf")),
                    "rmse": float(tags.get("metric_rmse", "inf")),
                    "r2": float(tags.get("metric_r2", "0")),
                    "mape": float(tags.get("metric_mape", "inf")),
                },
                "feature_columns": json.loads(tags.get("feature_columns", "[]")),
                "terminal_metrics": json.loads(tags.get("terminal_metrics", "{}")),
            }

        candidate_metadata = metadata(candidate_version)
        champion_metadata = metadata(champion_version)
        passed, reasons = promotion_checks(candidate_metadata, champion_metadata)
        if not passed:
            raise RuntimeError("MLflow promotion checks failed:\n  - " + "\n  - ".join(reasons))
        client.set_registered_model_alias(settings.model_name, champion_alias, candidate_version.version)
        client.set_registered_model_alias(settings.model_name, "Production", candidate_version.version)
        LOGGER.info(
            "Promoted MLflow %s model version %s to %s/Production",
            selected_alias,
            candidate_version.version,
            champion_alias,
        )
    except Exception as exc:
        LOGGER.warning("MLflow alias promotion failed; local promotion may still have succeeded: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a Candidate model to Champion/Production.")
    parser.add_argument("--candidate-alias", default="Candidate")
    parser.add_argument("--champion-alias", default="Champion")
    parser.add_argument("--output", default=None)
    parser.add_argument("--rollback", action="store_true", help="Roll back Production to the previous version")
    parser.add_argument("--target-version", default=None, help="Specific version to roll back to")
    args = parser.parse_args()

    output = Path(args.output) if args.output else get_settings().models_dir / "latest_promotion.json"

    if args.rollback:
        rolled_back = rollback_to_previous_champion(args.target_version)
        output.write_text(json.dumps(rolled_back, indent=2, sort_keys=True))
        print(f"Rolled back to {rolled_back['model_name']} version {rolled_back['model_version']}")
        return

    promoted = None
    try:
        promoted = promote_local_candidate(args.candidate_alias, args.champion_alias)
    except Exception as exc:
        LOGGER.warning("Local promotion did not complete: %s", exc)
    promote_mlflow_candidate(args.candidate_alias, args.champion_alias)
    if promoted:
        output.write_text(json.dumps(promoted, indent=2, sort_keys=True))
        print(f"Promoted {promoted['model_name']} version {promoted['model_version']}")
    else:
        output.write_text(json.dumps({"status": "mlflow_promotion_attempted"}, indent=2))
        print("Local registry was unavailable; attempted MLflow promotion")


if __name__ == "__main__":
    main()
