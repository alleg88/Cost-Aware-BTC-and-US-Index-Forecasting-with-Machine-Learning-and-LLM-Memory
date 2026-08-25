from __future__ import annotations

import pandas as pd

from reflection_agent.v2.transport import SchemaCallResult
from reflection_agent.v4.contracts import RouterChoice
from reflection_agent.v4.router import (
    FullInformationRouter,
    build_weekly_blocks,
)


class _Config:
    model = "deepseek-v4-flash:cloud"


class FakeCaller:
    config = _Config()

    def __init__(
        self,
        indices: list[int],
        *,
        evidence_indices: list[list[int]] | None = None,
        memory_indices: list[list[int]] | None = None,
    ) -> None:
        self.indices = iter(indices)
        self.evidence_indices = iter(evidence_indices or [[] for _ in indices])
        self.memory_indices = iter(memory_indices or [[] for _ in indices])
        self.calls: list[dict[str, object]] = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        value = RouterChoice(
            choice_index=next(self.indices),
            evidence_indices=next(self.evidence_indices),
            memory_indices=next(self.memory_indices),
        )
        return SchemaCallResult(
            status="success",
            value=value,
            raw_content=value.model_dump_json(),
            request_hash="a" * 64,
            response_hash="b" * 64,
            schema_hash="c" * 64,
            attempts=1,
            latency_seconds=0.01,
            metadata={},
            errors=(),
        )


def _payoffs() -> pd.DataFrame:
    rows = []
    for week, start in enumerate(("2024-01-01", "2024-01-08")):
        for choice_index, policy_id, net, trades in (
            (0, "UNION_ONLY", 0.0, 0),
            (1, "LSTM_HIGH", 0.01 if week == 0 else -0.01, 2),
        ):
            rows.append(
                {
                    "block_id": f"b{week}",
                    "block_start": pd.Timestamp(start, tz="UTC"),
                    "block_available_at": pd.Timestamp(start, tz="UTC")
                    + pd.Timedelta(days=1),
                    "stage": "development",
                    "choice_index": choice_index,
                    "policy_id": policy_id,
                    "incremental_net": net,
                    "additional_trades": trades,
                    "additional_long_trades": trades // 2,
                    "additional_short_trades": trades // 2,
                }
            )
    return pd.DataFrame(rows)


def test_blocks_are_chronological_and_feedback_is_strictly_later() -> None:
    blocks = build_weekly_blocks(_payoffs())
    assert [block.block_id for block in blocks] == ["b0", "b1"]
    assert blocks[0].available_at < blocks[1].starts_at


def test_real_router_sees_first_payoff_only_on_second_choice() -> None:
    caller = FakeCaller([1, 0])
    router = FullInformationRouter(caller=caller, memory_mode="real")
    result = router.run(_payoffs())
    assert result["choice_index"].tolist() == [1, 0]
    assert len(caller.calls) == 2
    first_payload = caller.calls[0]["messages"][-1]["content"]
    second_payload = caller.calls[1]["messages"][-1]["content"]
    assert '"memory_cards":[]' in first_payload
    assert '"incremental_net":0.01' in second_payload
    assert '2024-01-' not in second_payload


def test_invalid_evidence_reference_fails_closed_to_union() -> None:
    caller = FakeCaller([1, 1], evidence_indices=[[0], [63]])
    router = FullInformationRouter(caller=caller, memory_mode="real")
    result = router.run(_payoffs())
    assert result["choice_index"].tolist() == [0, 0]
    assert result["call_status"].tolist() == ["invalid_reference", "invalid_reference"]
    assert result["evidence_indices"].tolist() == ["[]", "[]"]


def test_static_and_hedge_controls_make_no_llm_calls() -> None:
    caller = FakeCaller([])
    static = FullInformationRouter(caller=caller, memory_mode="static", static_choice=1)
    assert static.run(_payoffs())["choice_index"].tolist() == [1, 1]
    hedge = FullInformationRouter(caller=caller, memory_mode="hedge")
    assert len(hedge.run(_payoffs())) == 2
    assert caller.calls == []
