from __future__ import annotations

import argparse
import json
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


def promotion_checks(
    candidate: dict[str, Any],
    champion: dict[str, Any] | None,
    max_mae_degradation_pct: float = 0.03,
    max_rmse_degradation_pct: float = 0.05,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate.get("validation_status") != "passed":
        reasons.append("candidate validation_status is not passed")
    if not candidate.get("feature_columns"):
        reasons.append("candidate feature metadata is missing")
    candidate_terminal_metrics = candidate.get("terminal_metrics", {})
    if "D+1" not in candidate_terminal_metrics or "D+2" not in candidate_terminal_metrics:
        reasons.append("candidate D+1/D+2 metrics are missing")
    if not champion:
        return len(reasons) == 0, reasons

    candidate_mae = _metric(candidate, "mae")
    champion_mae = _metric(champion, "mae")
    candidate_rmse = _metric(candidate, "rmse")
    champion_rmse = _metric(champion, "rmse")
    if candidate_mae > champion_mae * (1 + max_mae_degradation_pct):
        reasons.append(
            f"candidate MAE {candidate_mae:.3f} degrades beyond allowed threshold vs champion {champion_mae:.3f}"
        )
    if candidate_rmse > champion_rmse * (1 + max_rmse_degradation_pct):
        reasons.append(
            f"candidate RMSE {candidate_rmse:.3f} degrades beyond allowed threshold vs champion {champion_rmse:.3f}"
        )

    champion_terminal_metrics = champion.get("terminal_metrics", {})
    for key, metrics in candidate_terminal_metrics.items():
        if key not in champion_terminal_metrics:
            continue
        candidate_key_mae = float(metrics.get("mae", float("inf")))
        champion_key_mae = float(champion_terminal_metrics[key].get("mae", float("inf")))
        if candidate_key_mae > champion_key_mae * 1.10:
            reasons.append(f"{key} MAE degrades more than 10 percent")
    return len(reasons) == 0, reasons


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
        raise RuntimeError("Promotion checks failed: " + "; ".join(reasons))

    aliases[champion_alias] = str(candidate_version)
    aliases["Production"] = str(candidate_version)
    registry["aliases"] = aliases
    _save_registry(registry)
    LOGGER.info("Promoted local %s model version %s to %s/Production", selected_alias, candidate_version, champion_alias)
    return candidate


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
            raise RuntimeError("MLflow promotion checks failed: " + "; ".join(reasons))
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
    args = parser.parse_args()
    promoted = None
    try:
        promoted = promote_local_candidate(args.candidate_alias, args.champion_alias)
    except Exception as exc:
        LOGGER.warning("Local promotion did not complete: %s", exc)
    promote_mlflow_candidate(args.candidate_alias, args.champion_alias)
    output = Path(args.output) if args.output else get_settings().models_dir / "latest_promotion.json"
    if promoted:
        output.write_text(json.dumps(promoted, indent=2, sort_keys=True))
        print(f"Promoted {promoted['model_name']} version {promoted['model_version']}")
    else:
        output.write_text(json.dumps({"status": "mlflow_promotion_attempted"}, indent=2))
        print("Local registry was unavailable; attempted MLflow promotion")


if __name__ == "__main__":
    main()
