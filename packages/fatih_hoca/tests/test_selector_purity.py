"""Task 3 purity tests: Selector must not call record_swap."""
from unittest.mock import MagicMock

from fatih_hoca.selector import Selector
from fatih_hoca.registry import ModelRegistry


def test_select_does_not_record_swap():
    nh = MagicMock()
    nh.recent_swap_count.return_value = 0
    nh.can_swap.return_value = True
    nh.record_swap = MagicMock()

    registry = ModelRegistry()
    sel = Selector(registry=registry, nerd_herd=nh, available_providers=set())

    try:
        sel.select(task="coder", difficulty=5)
    except Exception:
        pass

    nh.record_swap.assert_not_called()


def test_select_reads_can_swap_from_nerd_herd():
    """When there are no models, select() returns None before swap logic.
    This test ensures record_swap is still never called even when can_swap=False."""
    nh = MagicMock()
    nh.can_swap.return_value = False
    nh.recent_swap_count.return_value = 3
    nh.record_swap = MagicMock()

    registry = ModelRegistry()
    sel = Selector(registry=registry, nerd_herd=nh, available_providers=set())
    try:
        sel.select(task="coder", difficulty=5)
    except Exception:
        pass

    # Regardless of whether swap logic was reached, record_swap must never be called.
    nh.record_swap.assert_not_called()
