"""Pure equation builders for the ContractBench contraction order."""

from __future__ import annotations

from string import ascii_letters


def sequential_contraction_steps(
    number_of_resources: int,
) -> tuple[tuple[str, str, str], ...]:
    """Return one einsum step per resource, in environment resource order."""
    _validate_resource_count(number_of_resources)

    labels = iter(ascii_letters)
    metric = next(labels)
    capital = next(labels)
    current_inventory = [next(labels) for _ in range(number_of_resources)]
    orders = [next(labels) for _ in range(number_of_resources)]
    next_inventory = [next(labels) for _ in range(number_of_resources)]

    value_labels = metric + "".join(next_inventory) + capital
    steps: list[tuple[str, str, str]] = []
    for index in range(number_of_resources):
        table_labels = (
            current_inventory[index]
            + orders[index]
            + next_inventory[index]
        )
        output_labels = (
            metric
            + "".join(current_inventory[: index + 1])
            + "".join(next_inventory[index + 1 :])
            + "".join(orders[: index + 1])
            + capital
        )
        steps.append((table_labels, value_labels, output_labels))
        value_labels = output_labels
    return tuple(steps)


def _validate_resource_count(number_of_resources: int) -> None:
    if isinstance(number_of_resources, bool) or not isinstance(
        number_of_resources, int
    ):
        raise TypeError("number_of_resources must be an integer")
    if number_of_resources < 1:
        raise ValueError("number_of_resources must be positive")
    required_labels = 2 + 3 * number_of_resources
    if required_labels > len(ascii_letters):
        raise ValueError("Too many resources for the sequential contraction")
