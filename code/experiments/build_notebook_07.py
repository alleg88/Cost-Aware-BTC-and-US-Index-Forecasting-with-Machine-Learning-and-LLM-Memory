"""Build the concise artifact-only reader for the final Q2 lockbox."""
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient

from experiments.notebook_hygiene import canonical_colab_setup


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = CODE_ROOT / "notebooks" / "22_Lockbox_Q2_2026.ipynb"
STATE_ROOT = CODE_ROOT / "experiments" / "cache" / "final_q2_lockbox"


def _markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def _code(source: str, *, tags: tuple[str, ...] = ()):
    cell = nbformat.v4.new_code_cell(dedent(source).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def build_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "MSC code",
        "language": "python",
        "name": "msc-code",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.metadata["final_q2_lockbox"] = "COMPLETE"
    notebook.cells = [
        _code(canonical_colab_setup(), tags=("setup",)),
        _markdown(
            """
            # Final Q2 2026 lockbox confirmation

            **Objective.** Evaluate the six policies frozen before Q2 on the untouched interval
            `[2026-04-01, 2026-07-01)` without fitting, calibration, threshold selection or promotion.
            BTC is the sole confirmatory stream; USA500 and USATECH are separate descriptive transport checks.
            """
        ),
        _markdown(
            """
            ## Concise methodology

            The model specification and panel-equivalent reconstructed weights were frozen before Q2; saved
            Forward classes and registered policy decisions had to reproduce exactly before the lockbox opened.

            - **BTC Qualified Union:** LSTM DZ55 trades only at confidence at least 0.75; Linear SVM DZ75 uses
              every directional prediction; an **opposite-signal veto** cancels only simultaneous disagreement.
              The frozen LSTM DZ55 is the control. Trades use TP 200 bps, SL 100 bps, one M15 holding bar and
              **10 bps** all-in round-trip cost.
            - **USA500:** the primary policy is the DeBERTa-matched Linear SVM DZ15 at 0.40 confidence; the
              secondary comparator is the arithmetic mean of all nine probability vectors at 0.55 confidence.
              VIX remains included because it passed the earlier frozen admission gate; cost is **2 bps** round trip.
            - **USATECH:** the primary policy is the LLM-matched LSTM DZ15 at 0.65 confidence; the secondary
              comparator is the arithmetic mean of all nine probability vectors at 0.55 confidence. The frozen
              LLM uses batch size 10 and temperature 0; cost is **3 bps** round trip.

            The nine-model vote contains **Logistic Regression, Decision Tree, Random Forest, Linear SVM,
            XGBoost, CatBoost, MLP, LSTM and GRU** with equal probability weight. Net, Sharpe, Sortino and
            drawdown below already deduct the registered costs; Sharpe and Sortino use all 91 zero-filled UTC days.
            Fresh Q2 sentiment coverage is reported explicitly and is not inferred from a model's feature-arm name.
            """
        ),
        _code(
            """
            from pathlib import Path
            import hashlib, json
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import Markdown, display

            CODE_ROOT = next(
                path for path in (Path.cwd(), *Path.cwd().parents)
                if (path / "pyproject.toml").is_file()
            )
            STATE_ROOT = CODE_ROOT / "experiments" / "cache" / "final_q2_lockbox"

            def sha256(path):
                digest = hashlib.sha256()
                with Path(path).open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                return digest.hexdigest()

            def resolve_bound_path(value):
                path = Path(value)
                if path.anchor or ":" in str(value):
                    raise ValueError("Artifact paths must be relative to the result directory")
                root = RESULT_ROOT.resolve(strict=True)
                resolved = (root / path).resolve(strict=True)
                resolved.relative_to(root)
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                return resolved

            def validate_manifest():
                opened = json.loads((STATE_ROOT / "OPENED.json").read_text(encoding="utf-8"))
                protocol_hash = opened["identity"]["protocol_hash"]
                result_root = STATE_ROOT / protocol_hash
                complete = json.loads((result_root / "COMPLETE.json").read_text(encoding="utf-8"))
                manifest_path = result_root / "manifest.json"
                manifest_sha256 = sha256(manifest_path)
                assert complete["result_hashes"]["manifest"] == manifest_sha256
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                assert manifest["opening_identity"] == opened["identity"]
                return result_root, manifest, manifest_sha256

            def validate_all_artifacts(manifest):
                artifact_hashes = manifest["artifact_hashes"]
                for key, entry in artifact_hashes.items():
                    assert sha256(resolve_bound_path(entry["path"])) == entry["sha256"], key
                return {key: resolve_bound_path(entry["path"]) for key, entry in artifact_hashes.items()}

            RESULT_ROOT, FINAL_MANIFEST, MANIFEST_SHA256 = validate_manifest()
            ARTIFACTS = validate_all_artifacts(FINAL_MANIFEST)
            VALIDATED = True
            """,
            tags=("artifact-validation",),
        ),
        _code(
            """
            LABELS = {
                "btc_qualified_union_v1": "BTC Qualified Union",
                "btc_lstm_dz55": "BTC LSTM DZ55 control",
                "usa500_best_single_deberta_svm": "USA500 DeBERTa SVM",
                "usa500_deberta_soft_vote": "USA500 all-nine soft vote",
                "usatech_best_single_deepseek_lstm": "USATECH LLM LSTM",
                "usatech_deepseek_soft_vote": "USATECH all-nine soft vote",
            }

            def read_table(key):
                assert VALIDATED
                return pd.read_parquet(ARTIFACTS[key])

            def require_table(frame, numeric):
                assert len(frame) > 0
                values = frame.loc[:, numeric].to_numpy(dtype=float)
                assert np.isfinite(values).all()
                assert np.any(values != 0.0)
                return frame

            def economics_table(stream):
                frame = read_table("summaries").loc[lambda x: x["stream"].eq(stream)].copy()
                frame["Policy"] = frame["candidate_id"].map(LABELS)
                frame["Net %"] = 100 * frame["net_return"]
                frame["Sharpe"] = frame["daily_sharpe"]
                frame["Sortino"] = frame["daily_sortino"]
                frame["Max drawdown %"] = 100 * frame["max_drawdown"]
                frame = frame.rename(columns={
                    "trades": "Trades", "trades_per_day": "Trades/day",
                    "long_trades": "LONG", "short_trades": "SHORT",
                })
                columns = ["Policy", "Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Max drawdown %"]
                return require_table(frame[columns], ["Trades", "Net %", "Sharpe", "Sortino"])
            """
        ),
        _markdown(
            """
            Method: The integrity row reports the immutable opening state and verifies every displayed artifact
            against the completed run before any result table is read.
            """
        ),
        _code(
            """
            score_keys = [
                key for key in ARTIFACTS
                if key.startswith("sentiment__scores") and key.endswith(".parquet")
            ]
            fresh_sentiment_rows = sum(len(pd.read_parquet(ARTIFACTS[key])) for key in score_keys)
            integrity = pd.DataFrame([{
                "State": "COMPLETE",
                "Interval": "2026-04-01 to 2026-07-01 (UTC, end exclusive)",
                "Frozen candidates": len(FINAL_MANIFEST["candidate_ids"]),
                "Fresh scored news/direct rows": fresh_sentiment_rows,
                "Retuning": "None",
                "Manifest SHA-256": MANIFEST_SHA256[:16] + "…",
            }])
            display(integrity)
            coverage = "no fresh scored news/direct rows" if fresh_sentiment_rows == 0 else f"{fresh_sentiment_rows} fresh scored rows"
            display(Markdown(f"**Takeaway:** The reader accepted one hash-consistent Q2 run with no retuning and {coverage}."))
            """,
            tags=("result-table",),
        ),
        _markdown(
            """
            Method: BTC policies are evaluated on the same 91-day calendar and sorted by Net after the fixed
            10 bps round-trip deduction; Sharpe and Sortino are daily.
            """
        ),
        _code(
            """
            btc_table = economics_table("btcusdt").sort_values("Net %", ascending=False)
            display(btc_table)
            leader = btc_table.iloc[0]
            display(Markdown(f"**Takeaway:** {leader['Policy']} ranks first on BTC Net at {leader['Net %']:.2f}%."))
            """,
            tags=("result-table",),
        ),
        _markdown(
            """
            Method: The sole confirmatory comparison is the paired BTC Qualified Union minus LSTM total Net;
            the 95% interval uses 5,000 fixed Monday–Sunday block resamples with seed 42.
            """
        ),
        _code(
            """
            contrast = read_table("paired_contrasts").loc[lambda x: x["stream"].eq("btcusdt")].copy()
            summary = read_table("summaries").set_index("candidate_id")
            contrast["Net %"] = 100 * contrast["point_estimate"]
            contrast["Sharpe"] = contrast["policy_id"].map(summary["daily_sharpe"])
            contrast["Sortino"] = contrast["policy_id"].map(summary["daily_sortino"])
            contrast["95% interval, pp"] = contrast.apply(lambda row: f"[{100*row.lower_95:.2f}, {100*row.upper_95:.2f}]", axis=1)
            contrast["Support"] = contrast["primary_confirmatory_support"].map({True: "Yes", False: "No"})
            contrast_view = require_table(contrast[["Net %", "Sharpe", "Sortino", "95% interval, pp", "Support"]], ["Net %", "Sharpe", "Sortino"]).sort_values("Net %", ascending=False)
            display(contrast_view)
            row = contrast.iloc[0]
            display(Markdown(f"**Takeaway:** Confirmatory support is **{row['Support']}**; the paired Net difference is {100*row['point_estimate']:.2f} percentage points."))
            """,
            tags=("result-table",),
        ),
        _markdown(
            """
            Method: USA500 keeps the frozen DeBERTa SVM primary and equal-weight all-nine comparator separate,
            uses admitted VIX features and deducts 2 bps round trip; rows are sorted by Net.
            """
        ),
        _code(
            """
            usa500_table = economics_table("usa500").sort_values("Net %", ascending=False)
            display(usa500_table)
            leader = usa500_table.iloc[0]
            display(Markdown(f"**Takeaway:** {leader['Policy']} has the higher descriptive USA500 Net ({leader['Net %']:.2f}%); no fresh Q2 news/direct rows were available."))
            """,
            tags=("result-table",),
        ),
        _markdown(
            """
            Method: USATECH keeps the frozen LLM LSTM primary and equal-weight all-nine comparator separate,
            uses admitted VIX features and deducts 3 bps round trip; rows are sorted by Net.
            """
        ),
        _code(
            """
            usatech_table = economics_table("usatech").sort_values("Net %", ascending=False)
            display(usatech_table)
            leader = usatech_table.iloc[0]
            display(Markdown(f"**Takeaway:** {leader['Policy']} has the higher descriptive USATECH Net ({leader['Net %']:.2f}%); no fresh Q2 news/direct rows were available."))
            """,
            tags=("result-table",),
        ),
        _markdown(
            """
            Method: Side coverage and the exact double-cost stress use unchanged trades; the normal-cost Net,
            Sharpe and Sortino remain alongside LONG and SHORT counts for interpretation.
            """
        ),
        _code(
            """
            stress = read_table("summaries").copy()
            stress["Policy"] = stress["candidate_id"].map(LABELS)
            stress["Net %"] = 100 * stress["net_return"]
            stress["Sharpe"] = stress["daily_sharpe"]
            stress["Sortino"] = stress["daily_sortino"]
            stress["2x-cost Net %"] = 100 * stress["stress_2x_net_return"]
            stress = stress.rename(columns={"long_trades": "LONG", "short_trades": "SHORT"})
            stress_view = require_table(stress[["Policy", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "2x-cost Net %"]], ["LONG", "SHORT", "Net %", "Sharpe", "Sortino", "2x-cost Net %"]).sort_values("Net %", ascending=False)
            display(stress_view)
            survivors = int(stress_view["2x-cost Net %"].gt(0).sum())
            display(Markdown(f"**Takeaway:** {survivors} of {len(stress_view)} frozen policies remain positive when registered costs are doubled."))
            """,
            tags=("result-table",),
        ),
        _markdown(
            """
            Method: The only figure cumulatively sums each policy's already cost-adjusted daily Net on the same
            zero-filled UTC calendar, with one panel per market and no cross-market pooling.
            """
        ),
        _code(
            """
            streams = {
                "BTC": ["btc_qualified_union_v1", "btc_lstm_dz55"],
                "USA500": ["usa500_best_single_deberta_svm", "usa500_deberta_soft_vote"],
                "USATECH": ["usatech_best_single_deepseek_lstm", "usatech_deepseek_soft_vote"],
            }
            fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
            for axis, (title, candidate_ids) in zip(axes, streams.items()):
                for candidate_id in candidate_ids:
                    daily = read_table(f"daily__{candidate_id}").set_index("date")["net_return"]
                    axis.plot(daily.index, 100 * daily.cumsum(), label=LABELS[candidate_id], linewidth=1.8)
                axis.axhline(0, color="black", linewidth=0.7)
                axis.set_title(title)
                axis.set_ylabel("Cumulative Net (%)")
                axis.tick_params(axis="x", rotation=30)
                axis.legend(fontsize=7)
            fig.tight_layout()
            plt.show()
            display(Markdown("**Takeaway:** The panels show transport paths separately; visual differences do not create any new promotion decision."))
            """,
            tags=("result-figure",),
        ),
        _markdown(
            """
            Method: Final statements reproduce the immutable verdict and keep the index evidence descriptive;
            the result is final whether positive, negative or inconclusive.
            """
        ),
        _code(
            """
            verdict = json.loads(ARTIFACTS["verdict"].read_text(encoding="utf-8"))
            display(Markdown(f"**BTC:** {verdict['interpretation']}"))
            for stream in ("USA500", "USATECH"):
                table = economics_table(stream.lower()).sort_values("Net %", ascending=False)
                display(Markdown(f"**{stream}:** descriptive leader is {table.iloc[0]['Policy']} at {table.iloc[0]['Net %']:.2f}% Net; no promotion is permitted."))
            display(Markdown("**Final:** thresholds, models, ensemble membership and costs remain frozen; no post-Q2 retuning is allowed."))
            """
        ),
    ]
    return notebook


def write_notebook(path: str | Path = NOTEBOOK_PATH) -> Path:
    destination = Path(path)
    opened = STATE_ROOT / "OPENED.json"
    if not opened.is_file():
        raise PermissionError("Notebook 07 is written only after the one-shot Q2 opening")
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), destination)
    return destination


def execute_notebook(path: str | Path = NOTEBOOK_PATH, *, timeout: int = 600) -> Path:
    target = Path(path)
    notebook = nbformat.read(target, as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="msc-code",
        resources={"metadata": {"path": str(CODE_ROOT)}},
    ).execute()
    nbformat.write(executed, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--path", type=Path, default=NOTEBOOK_PATH)
    args = parser.parse_args()
    path = write_notebook(args.path)
    if args.execute:
        execute_notebook(path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
