from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "thesis_ablation_results.png"

STRATEGY_FILES: Dict[str, Path] = {
    "none": BASE_DIR / "Thesis_Metrics_none.xlsx",
    "keyword_expansion": BASE_DIR / "Thesis_Metrics_keyword_expansion.xlsx",
    "hyde": BASE_DIR / "Thesis_Metrics_hyde.xlsx",
}

METRIC_KEYS = {
    "Precision": "precision",
    "Recall@k": "recall_at_k",
    "F1-Score": "f1_score",
    "MRR": "mrr",
    "nDCG@5": "ndcg_at_5",
}


def _load_macro_sheet(path: Path) -> pd.DataFrame:
    """Load the Macro_Averages sheet from a workbook with explicit error context."""
    try:
        return pd.read_excel(path, sheet_name="Macro_Averages")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing workbook: {path.name}") from exc
    except ValueError as exc:
        raise ValueError(f"Workbook {path.name} does not contain 'Macro_Averages' sheet") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path.name}: {exc}") from exc


def _macro_to_dict(df: pd.DataFrame) -> Dict[str, float]:
    """Convert two-column macro table to a dictionary."""
    if "metric" not in df.columns or "value" not in df.columns:
        raise ValueError("Macro_Averages must contain 'metric' and 'value' columns")
    return {str(k): v for k, v in zip(df["metric"], df["value"]) }


def _get_metric_value(macro: Dict[str, float], key: str) -> float:
    """Resolve metric values, including dynamic nDCG key naming."""
    if key in macro:
        return float(macro[key])

    if key.startswith("ndcg_at_"):
        for candidate_key, value in macro.items():
            if str(candidate_key).startswith("ndcg_at_"):
                return float(value)

    raise KeyError(f"Metric '{key}' not found in Macro_Averages")


def load_all_metrics() -> pd.DataFrame:
    """Load and validate all required strategy files before plotting."""
    missing: List[str] = []
    errors: List[str] = []

    for strategy, path in STRATEGY_FILES.items():
        if not path.exists():
            missing.append(f"{strategy}: {path.name}")

    if missing:
        print("Missing required workbook(s):")
        for item in missing:
            print(f"- {item}")
        raise FileNotFoundError("Required strategy workbook(s) missing; aborting chart generation.")

    rows = []
    for strategy, path in STRATEGY_FILES.items():
        try:
            macro_df = _load_macro_sheet(path)
            macro_map = _macro_to_dict(macro_df)
            for metric_label, metric_key in METRIC_KEYS.items():
                rows.append(
                    {
                        "strategy": strategy,
                        "metric": metric_label,
                        "score": _get_metric_value(macro_map, metric_key),
                    }
                )
        except Exception as exc:
            errors.append(f"{strategy} ({path.name}): {exc}")

    if errors:
        print("Error(s) while loading workbook(s):")
        for err in errors:
            print(f"- {err}")
        raise RuntimeError("Workbook loading failed; aborting chart generation.")

    return pd.DataFrame(rows)


def build_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Create a multi-panel bar chart and save high-resolution PNG."""
    sns.set_theme(style="whitegrid", context="talk", palette="deep")

    metrics_order = list(METRIC_KEYS.keys())
    strategy_order = list(STRATEGY_FILES.keys())

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes_flat = axes.flatten()

    for idx, metric_label in enumerate(metrics_order):
        ax = axes_flat[idx]
        subset = df[df["metric"] == metric_label].copy()
        subset["strategy"] = pd.Categorical(
            subset["strategy"], categories=strategy_order, ordered=True
        )
        subset = subset.sort_values("strategy")

        sns.barplot(
            data=subset,
            x="strategy",
            y="score",
            ax=ax,
            order=strategy_order,
            hue="strategy",
            hue_order=strategy_order,
            dodge=False,
            legend=False,
            width=0.6,
        )

        ax.set_title(metric_label, fontsize=14, pad=10)
        ax.set_xlabel("Strategy", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.0)

        for bar in ax.patches:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                (bar.get_x() + bar.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    # Hide the unused 6th panel
    axes_flat[-1].axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=sns.color_palette("deep")[i])
        for i in range(len(strategy_order))
    ]
    fig.legend(
        handles,
        strategy_order,
        title="Retrieval Strategy",
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle("Ablation Study Macro-Metric Comparison", fontsize=18, y=1.06)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        df = load_all_metrics()
        build_chart(df, OUTPUT_FILE)
        print(f"Chart saved: {OUTPUT_FILE}")
        return 0
    except Exception as exc:
        print(f"Chart generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
