"""Pure EAN-13 parsing for Ravensburger barcode lookups (no I/O).

Ravensburger EAN-13 barcodes embed the article number that keys their public
image CDN (see `app.services.ravensburger_client`). Two GS1 company prefixes
are in use:

- ``4005556`` (standard line, e.g. kids puzzles): the 5-digit payload
  (digits 8-12) *is* the full article number. Example: EAN 4005556050093 ->
  article ``05009``.
- ``4005555`` (adult/1000-piece line): article numbers are 8 digits, and the
  EAN payload carries only their *last 5* digits — the leading 3-digit series
  prefix (e.g. ``120``) is not recoverable from the EAN alone, so callers
  probe a small list of known series prefixes against the CDN.

Any other prefix is not a Ravensburger product.
"""

from typing import List

RAVENSBURGER_STANDARD_PREFIX = "4005556"
RAVENSBURGER_ADULT_PREFIX = "4005555"


def ean_checksum_valid(ean: str) -> bool:
    """Check whether a string is a well-formed EAN-13 with a valid check digit.

    Args:
        ean: The candidate barcode payload.

    Returns:
        True if `ean` is exactly 13 digits and its GS1 mod-10 check digit
        (weights 1,3,1,3,... over the first 12 digits) matches.
    """
    if len(ean) != 13 or not ean.isdigit():
        return False
    digits = [int(char) for char in ean]
    weighted_sum = sum(digit * (1 if index % 2 == 0 else 3) for index, digit in enumerate(digits[:12]))
    return (10 - weighted_sum % 10) % 10 == digits[12]


def candidate_article_numbers(ean: str, series_prefixes: List[str]) -> List[str]:
    """Derive the Ravensburger article numbers a valid EAN-13 could map to.

    Args:
        ean: A well-formed EAN-13 (validate with `ean_checksum_valid` first).
        series_prefixes: Known 3-digit adult-line series prefixes to try, in
            preference order (see `Settings.RAVENSBURGER_SERIES_PREFIXES`).

    Returns:
        Candidate article numbers in preference order: the bare payload for
        the standard line, one prefixed candidate per series for the adult
        line, or an empty list for non-Ravensburger prefixes.
    """
    prefix, payload = ean[:7], ean[7:12]
    if prefix == RAVENSBURGER_STANDARD_PREFIX:
        return [payload]
    if prefix == RAVENSBURGER_ADULT_PREFIX:
        return [f"{series}{payload}" for series in series_prefixes]
    return []
