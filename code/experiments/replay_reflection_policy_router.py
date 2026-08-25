"""Run the frozen reflection router with provider access disabled."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Iterator

from experiments import run_reflection_policy_router as runner


class OfflineProviderError(RuntimeError):
    """Raised if canonical replay reaches a provider instead of an exact cache hit."""


def _blocked_provider_call(*_args, **_kwargs):
    raise OfflineProviderError("canonical offline cache miss: provider call blocked")


@contextmanager
def offline_provider_guard() -> Iterator[None]:
    if os.environ.get("MSC_CANONICAL_OFFLINE") != "1":
        raise OfflineProviderError("canonical replay requires MSC_CANONICAL_OFFLINE=1")
    original = runner.DeepSeekSchemaCaller.call
    runner.DeepSeekSchemaCaller.call = _blocked_provider_call
    try:
        yield
    finally:
        runner.DeepSeekSchemaCaller.call = original


def main(argv: list[str] | None = None) -> int:
    with offline_provider_guard():
        return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["OfflineProviderError", "main", "offline_provider_guard"]
