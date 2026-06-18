"""
Token & Cost Tracking for Veska (Optional).

OFF by default. User enables manually.
Stores in user's database (not our config files).

Tracks tokens and estimated costs per agent, per task, and total.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional


# Approximate pricing per 1M tokens (as of 2025)
DEFAULT_PRICING = {
    # Claude models
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o1": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 12.0},
}


class UsageRecord:
    """A single usage record."""

    def __init__(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_id: str = "",
    ) -> None:
        self.agent_name = agent_name
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.task_id = task_id
        self.timestamp = time.time()
        self.estimated_cost = 0.0

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "estimated_cost": self.estimated_cost,
        }


class CostTracker:
    """
    Optional token and cost tracking.

    OFF by default. User enables it and provides a storage callback
    to persist data to their own database.

    Usage:
        tracker = CostTracker(enabled=True)

        # Optionally set a storage callback
        tracker.set_storage(my_db_save_function)

        # Record usage (called automatically by providers)
        tracker.record(
            agent_name="backend_agent",
            model="claude-sonnet-4-6",
            input_tokens=1500,
            output_tokens=800,
            task_id="task_1",
        )

        # Get stats
        total = tracker.total_cost
        by_agent = tracker.get_agent_costs()
    """

    def __init__(
        self,
        enabled: bool = False,
        pricing: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self._enabled = enabled
        self._pricing = pricing or dict(DEFAULT_PRICING)
        self._records: list[UsageRecord] = []
        self._storage_callback: Optional[Callable[[UsageRecord], None]] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable cost tracking."""
        self._enabled = True

    def disable(self) -> None:
        """Disable cost tracking."""
        self._enabled = False

    def set_storage(self, callback: Callable[[UsageRecord], None]) -> None:
        """
        Set a callback to persist usage records.

        The callback receives a UsageRecord and should save it
        to the user's database. We don't manage storage ourselves.
        """
        self._storage_callback = callback

    def set_pricing(self, model: str, input_price: float, output_price: float) -> None:
        """Set custom pricing for a model (per 1M tokens)."""
        self._pricing[model] = {"input": input_price, "output": output_price}

    def record(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_id: str = "",
    ) -> Optional[UsageRecord]:
        """
        Record a usage event.

        Args:
            agent_name: Which agent made the call.
            model: Which model was used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            task_id: Which task this was for.

        Returns:
            UsageRecord if tracking is enabled, None otherwise.
        """
        if not self._enabled:
            return None

        record = UsageRecord(
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            task_id=task_id,
        )

        # Calculate estimated cost
        record.estimated_cost = self._calculate_cost(model, input_tokens, output_tokens)

        self._records.append(record)

        # Persist via callback if set
        if self._storage_callback:
            try:
                self._storage_callback(record)
            except Exception:
                pass

        return record

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost in USD."""
        pricing = self._pricing.get(model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    # --- Query methods ---

    @property
    def total_tokens(self) -> dict:
        """Get total token usage."""
        input_total = sum(r.input_tokens for r in self._records)
        output_total = sum(r.output_tokens for r in self._records)
        return {
            "input": input_total,
            "output": output_total,
            "total": input_total + output_total,
        }

    @property
    def total_cost(self) -> float:
        """Get total estimated cost in USD."""
        return sum(r.estimated_cost for r in self._records)

    def get_agent_costs(self) -> dict[str, float]:
        """Get cost breakdown by agent."""
        costs: dict[str, float] = {}
        for record in self._records:
            costs[record.agent_name] = costs.get(record.agent_name, 0) + record.estimated_cost
        return costs

    def get_task_costs(self) -> dict[str, float]:
        """Get cost breakdown by task."""
        costs: dict[str, float] = {}
        for record in self._records:
            if record.task_id:
                costs[record.task_id] = costs.get(record.task_id, 0) + record.estimated_cost
        return costs

    def get_model_usage(self) -> dict[str, dict]:
        """Get usage breakdown by model."""
        usage: dict[str, dict] = {}
        for record in self._records:
            if record.model not in usage:
                usage[record.model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            usage[record.model]["calls"] += 1
            usage[record.model]["input_tokens"] += record.input_tokens
            usage[record.model]["output_tokens"] += record.output_tokens
            usage[record.model]["cost"] += record.estimated_cost
        return usage

    def get_records(
        self,
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[UsageRecord]:
        """Get usage records with optional filters."""
        records = self._records
        if agent_name:
            records = [r for r in records if r.agent_name == agent_name]
        if task_id:
            records = [r for r in records if r.task_id == task_id]
        return records[-limit:]

    def clear(self) -> None:
        """Clear all records (in-memory only)."""
        self._records.clear()

    @property
    def stats(self) -> dict:
        """Get overall statistics."""
        tokens = self.total_tokens
        return {
            "enabled": self._enabled,
            "total_calls": len(self._records),
            "total_tokens": tokens["total"],
            "total_cost_usd": round(self.total_cost, 6),
            "by_agent": self.get_agent_costs(),
        }
