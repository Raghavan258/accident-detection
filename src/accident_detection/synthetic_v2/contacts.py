"""Public emergency numbers shown with synthetic results.

These are national public numbers for India. They are not a dispatch record
and they are not assigned to the synthetic pins.
"""

from __future__ import annotations

EMERGENCY_CONTACTS: tuple[dict[str, str], ...] = (
    {
        "service": "National emergency",
        "number": "112",
        "note": "Public emergency number for India. Not a unit sent to these synthetic events.",
    },
    {
        "service": "Police",
        "number": "100",
        "note": "Public police number. This table does not look up a local station.",
    },
    {
        "service": "Ambulance",
        "number": "108",
        "note": "Public ambulance number. This table does not look up a hospital.",
    },
)
