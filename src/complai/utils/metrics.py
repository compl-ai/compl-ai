def mean_with_none(items: list[int | float | None]) -> float | None:
    not_none = [item for item in items if item is not None]

    return sum(not_none) / len(not_none) if not_none else 0


def sum_aggregation(items: list[int | float]) -> int | float:
    return sum(items)
