"""Plot a timeline of pipeline events."""

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _clean_event_id(event_id: str) -> str:
    """Clean the event id."""
    label = event_id.split(".")
    if len(label) > 1:
        # remove all parts containing BIDS entities like "sub-" and "ses-"
        label = [part for part in label if not ("sub-" in part or "ses-" in part)]

    return ".".join(label)


@dataclass
class Event:
    """A pipeline event."""

    name: str
    start: datetime
    finish: datetime

    start_seconds: float = 0
    finish_seconds: float = 0
    slot: int = -1

    @classmethod
    def from_dict(cls, data: dict) -> Optional["Event"]:
        """Create an event from a dictionary."""
        if "start" not in data or "finish" not in data or "id" not in data:
            return None
        return cls(
            name=_clean_event_id(
                data["id"]
            ),  # Event IDs are not unique and contain garbage
            start=datetime.fromisoformat(data["start"]),
            finish=datetime.fromisoformat(data["finish"]),
        )


def _read_events(callback_log_path: pathlib.Path) -> list[Event]:
    """Read events from a callback log file."""
    events = []
    with open(callback_log_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = Event.from_dict(data)
            if event is not None:
                events.append(event)
    return events


def _events_get_range(events: list[Event]) -> tuple[datetime, datetime]:
    """Get the time range of events."""
    return min(events, key=lambda x: x.start).start, max(
        events, key=lambda x: x.finish
    ).finish


def _events_normalize(events: list[Event], reference_time: datetime) -> None:
    """Normalize event times to seconds from a reference time."""
    for event in events:
        event.start_seconds = (event.start - reference_time).total_seconds()
        event.finish_seconds = (event.finish - reference_time).total_seconds()


def _events_sorted(events: list[Event]) -> list[Event]:
    """Sort events by start time."""
    return sorted(events, key=lambda x: x.start_seconds)


def _events_calculate_slots(events: list[Event], allowed_overlap: int) -> None:
    """Calculate the slot for each event. Sort before calling this function."""
    slots_max_time: list[float] = []
    for event in events:
        slot = -1
        for idx, last_time in enumerate(slots_max_time):
            if event.start_seconds + allowed_overlap >= last_time:
                slot = idx

        if slot == -1:
            slot = len(slots_max_time)
            slots_max_time.append(event.finish_seconds)
        else:
            slots_max_time[slot] = event.finish_seconds

        event.slot = slot


def _minutes_to_hhmm(minutes: int | float) -> str:
    """Convert minutes to HH:MM format."""
    minutes = int(minutes)
    return f"{minutes//60:02d}:{minutes%60:02d}"


def _events_plot(
    events: list[Event],
    first_start: datetime,
    last_end: datetime,
    allowed_overlap: int,
    label_min_percent: float,
    caption: str = "",
) -> None:
    """Plot the timeline of events."""
    assert 0 <= label_min_percent <= 1, "label_min_percent must be between 0 and 1"

    # set seed for reproducibility
    np.random.seed(42)

    total_duration = (last_end - first_start).total_seconds()

    fig, ax = plt.subplots(figsize=(18, 8))

    for event in events:
        # random color
        color = np.random.rand(
            3,
        )
        start = event.start_seconds
        duration = event.finish_seconds - event.start_seconds
        ax.barh(event.slot, duration / 60, left=start / 60, color=color)

        if duration > (label_min_percent * total_duration):
            ax.text(
                (start + 0.5 * duration) / 60,
                event.slot,
                f"{event.name} ({_minutes_to_hhmm(duration/60)})",
                rotation=90,
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.grid(True)
    ax.set_yticks([])  # Disable y labels
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Parallel execution")
    ax.set_title(
        f"Start: {first_start:%Y-%m-%d %H:%M:%S%z}, "
        f"Duration: {_minutes_to_hhmm(total_duration/60)}, "
        f"Overlap: {allowed_overlap} sec, "
        f"Events: {len(events)}"
        f"\n{caption}"
    )

    # minutes -> HH:MM format
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _minutes_to_hhmm(x)))

    ax.invert_yaxis()

    plt.tight_layout()


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(description="Plot a timeline of pipeline events.")
    parser.add_argument(
        "callback_log", type=pathlib.Path, help="Path to the callback.log file."
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Path to save the plot. If not provided, "
        "the plot will be shown in a window.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=5,
        help="Allowed overlap between events in the same slot. Default: 5 seconds.",
    )
    parser.add_argument(
        "--label-min-percent",
        type=float,
        default=0.01,
        help="Minimum percentage of the total timeline duration "
        "for an event to be labeled. Default: 0.01",
    )
    return parser


def main() -> None:
    """Main function."""
    args = _build_parser().parse_args()
    callback_log_path: pathlib.Path = args.callback_log
    output_path: pathlib.Path | None = args.output
    allowed_overlap: int = args.overlap
    label_min_percent: float = args.label_min_percent

    events = _read_events(callback_log_path)

    if len(events) == 0:
        print("No events found in the log file.")
        sys.exit(1)
    print(f"Read {len(events)} events from the log file.")

    first_start, last_end = _events_get_range(events)
    _events_normalize(events, reference_time=first_start)
    events = _events_sorted(events)
    _events_calculate_slots(events, allowed_overlap)
    _events_plot(
        events,
        first_start,
        last_end,
        allowed_overlap,
        label_min_percent,
        caption=f"File: {callback_log_path}",
    )

    if output_path is not None:
        plt.savefig(output_path)
    else:
        print("Showing plot (if nothing happens Ctrl+C to close)")
        plt.show()


if __name__ == "__main__":
    main()
