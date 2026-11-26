import numpy as np
import pytest

from analysis_tools.data import (
    CutFlowEntry,
    EventSelection,
)


def test_event_selection_initial_state() -> None:
    selection = EventSelection(
        event_count=4,
    )

    np.testing.assert_array_equal(
        selection.mask,
        [True, True, True, True],
    )
    np.testing.assert_array_equal(
        selection.indices,
        [0, 1, 2, 3],
    )

    assert selection.event_count == 4
    assert selection.cut_flow == ()


def test_event_selection_add() -> None:
    selection = EventSelection(
        event_count=4,
    )

    returned = selection.add(
        "Even events",
        np.array([True, False, True, False]),
    )

    assert returned is selection
    assert selection.event_count == 2

    np.testing.assert_array_equal(
        selection.mask,
        [True, False, True, False],
    )
    np.testing.assert_array_equal(
        selection.indices,
        [0, 2],
    )

    assert selection.cut_flow == (
        CutFlowEntry(
            name="Even events",
            before=4,
            after=2,
        ),
    )


def test_event_selection_combines_multiple_cuts() -> None:
    selection = EventSelection(
        event_count=5,
    )

    selection.add(
        "First cut",
        np.array([True, True, True, False, False]),
    )

    selection.add(
        "Second cut",
        np.array([False, True, True, True, False]),
    )

    np.testing.assert_array_equal(
        selection.mask,
        [False, True, True, False, False],
    )
    np.testing.assert_array_equal(
        selection.indices,
        [1, 2],
    )

    assert selection.event_count == 2

    assert selection.cut_flow == (
        CutFlowEntry(
            name="First cut",
            before=5,
            after=3,
        ),
        CutFlowEntry(
            name="Second cut",
            before=3,
            after=2,
        ),
    )


def test_event_selection_supports_method_chaining() -> None:
    selection = (
        EventSelection(
            event_count=4,
        )
        .add(
            "Label cut",
            np.array([True, True, False, False]),
        )
        .add(
            "Wall cut",
            np.array([False, True, True, False]),
        )
    )

    np.testing.assert_array_equal(
        selection.indices,
        [1],
    )


def test_cut_flow_entry_removed() -> None:
    entry = CutFlowEntry(
        name="Wall cut",
        before=100,
        after=75,
    )

    assert entry.removed == 25


def test_event_selection_mask_returns_copy() -> None:
    selection = EventSelection(
        event_count=3,
    )

    mask = selection.mask
    mask[0] = False

    np.testing.assert_array_equal(
        selection.mask,
        [True, True, True],
    )


def test_event_selection_cut_flow_returns_tuple() -> None:
    selection = EventSelection(
        event_count=3,
    )

    selection.add(
        "Selection",
        np.array([True, False, True]),
    )

    assert isinstance(
        selection.cut_flow,
        tuple,
    )


def test_event_selection_supports_empty_input() -> None:
    selection = EventSelection(
        event_count=0,
    )

    assert selection.event_count == 0
    assert selection.mask.shape == (0,)
    assert selection.indices.shape == (0,)


def test_event_selection_can_remove_all_events() -> None:
    selection = EventSelection(
        event_count=3,
    )

    selection.add(
        "Remove all",
        np.array([False, False, False]),
    )

    assert selection.event_count == 0

    np.testing.assert_array_equal(
        selection.mask,
        [False, False, False],
    )

    np.testing.assert_array_equal(
        selection.indices,
        np.array(
            [],
            dtype=np.int64,
        ),
    )


def test_event_selection_rejects_negative_event_count() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        EventSelection(
            event_count=-1,
        )


def test_event_selection_rejects_non_one_dimensional_condition() -> None:
    selection = EventSelection(
        event_count=2,
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        selection.add(
            "Invalid condition",
            np.array(
                [
                    [True, False],
                    [False, True],
                ]
            ),
        )


def test_event_selection_rejects_non_boolean_condition() -> None:
    selection = EventSelection(
        event_count=3,
    )

    with pytest.raises(
        ValueError,
        match="Boolean",
    ):
        selection.add(
            "Invalid condition",
            np.array(
                [1, 0, 1],
            ),
        )


def test_event_selection_rejects_length_mismatch() -> None:
    selection = EventSelection(
        event_count=3,
    )

    with pytest.raises(
        ValueError,
        match="2 events.*3",
    ):
        selection.add(
            "Invalid condition",
            np.array([True, False]),
        )


def test_event_selection_records_zero_removed_events() -> None:
    selection = EventSelection(
        event_count=3,
    )

    selection.add(
        "Keep all",
        np.array([True, True, True]),
    )

    entry = selection.cut_flow[0]

    assert entry.before == 3
    assert entry.after == 3
    assert entry.removed == 0


def test_event_selection_print_cut_flow(
    capsys,
) -> None:
    selection = EventSelection(
        event_count=4,
    )

    selection.add(
        "Label cut",
        np.array([True, True, False, False]),
    )

    selection.add(
        "Wall cut",
        np.array([False, True, True, False]),
    )

    selection.print_cut_flow()

    output = capsys.readouterr().out

    assert "Cut flow:" in output
    assert "Initial events" in output
    assert "Label cut" in output
    assert "Wall cut" in output
    assert "4" in output
    assert "2" in output
    assert "1" in output


def test_event_selection_with_label_and_wall() -> None:
    labels = np.array(
        [0, 1, 1, 2, 1],
    )

    wall = np.array(
        [300.0, 100.0, 250.0, 500.0, 400.0],
    )

    selection = EventSelection(
        event_count=len(labels),
    )

    selection.add(
        "Electron",
        labels == 1,
    )

    selection.add(
        "Wall >= 200 cm",
        wall >= 200.0,
    )

    np.testing.assert_array_equal(
        selection.indices,
        [2, 4],
    )

    assert selection.cut_flow == (
        CutFlowEntry(
            name="Electron",
            before=5,
            after=3,
        ),
        CutFlowEntry(
            name="Wall >= 200 cm",
            before=3,
            after=2,
        ),
    )
