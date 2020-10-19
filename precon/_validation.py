"""Internal functions for argument validation."""

from typing import Union


def _handle_axis(axis: Union[str, int]) -> int:
    """Handles axis arguments including "columns" and "index" strings."""
    if axis not in {0, 1, 'columns', 'index'}:
        raise ValueError(
            "axis value error: not in {0, 1, 'columns', 'index'}"
        )

    # Map to int if str
    if isinstance(axis, str):
        axis_mapper = {'index': 0, 'columns': 1}
        axis = axis_mapper.get(axis)

    return axis
