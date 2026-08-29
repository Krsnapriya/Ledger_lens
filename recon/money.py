"""
Money is an int. Always. Paise.

A float never touches a ledger value in this system. Every amount that enters
the pipeline is parsed to an integer number of paise at the ingestion boundary
and stays an integer until it is formatted for human display.

This is not pedantry. 0.1 + 0.2 != 0.3 in IEEE-754, and a reconciliation engine
that carries float money will report balance-check failures that are artifacts
of the representation rather than the data. Those failures are indistinguishable
from real ones, which makes the entire exception list untrustworthy.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# Matches the junk that shows up in real bank/gateway CSV exports:
#   "1,23,456.78"  Indian lakh grouping
#   "123,456.78"   Western grouping
#   "INR 1234.50"  currency prefix
#   "(1234.50)"    accounting negative
#   "1234.50 Cr"   / "1234.50 Dr"
_CURRENCY_PREFIX = re.compile(r"^\s*(?:INR|Rs\.?|₹)\s*", re.IGNORECASE)
_CR_DR_SUFFIX = re.compile(r"\s*(Cr|Dr)\s*$", re.IGNORECASE)


class MoneyParseError(ValueError):
    """Raised when a value cannot be parsed. Never silently coerced to zero."""


def parse_paise(raw: object) -> int:
    """Parse an arbitrary CSV money cell into integer paise.

    Raises MoneyParseError rather than returning 0 on failure. A silent zero in
    a reconciliation engine is worse than a crash: it produces a balanced-looking
    ledger that is wrong.
    """
    if raw is None:
        raise MoneyParseError("None is not an amount")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "null", "none", "-"}:
        raise MoneyParseError(f"empty amount cell: {raw!r}")

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()

    m = _CR_DR_SUFFIX.search(s)
    if m:
        if m.group(1).lower() == "dr":
            negative = True
        s = _CR_DR_SUFFIX.sub("", s).strip()

    s = _CURRENCY_PREFIX.sub("", s).strip()
    s = s.replace(",", "").replace(" ", "")

    if s.startswith("-"):
        negative = True
        s = s[1:]

    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError) as exc:
        raise MoneyParseError(f"unparseable amount: {raw!r}") from exc

    # Decimal -> paise with explicit half-up at the 2nd decimal place.
    paise = int((d * 100).to_integral_value(rounding="ROUND_HALF_UP"))
    return -paise if negative else paise


def fmt(paise: int) -> str:
    """Format paise for human display in Indian grouping. Display only."""
    sign = "-" if paise < 0 else ""
    p = abs(int(paise))
    rupees, sub = divmod(p, 100)
    s = str(rupees)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return f"{sign}₹{s}.{sub:02d}"


def pct_of(paise: int, rate_bp: int) -> int:
    """Percentage of an amount in basis points, half-up, in integer paise.

    rate_bp = 200 means 2.00%. Integer arithmetic throughout; the +5000/10000
    is an explicit half-up rounding at the paise boundary.
    """
    num = paise * rate_bp
    return (num + 5000) // 10000 if num >= 0 else -((-num + 5000) // 10000)
