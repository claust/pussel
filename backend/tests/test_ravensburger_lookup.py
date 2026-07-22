"""Tests for the pure EAN-13 parsing in app/services/ravensburger_lookup.py."""

from app.services.ravensburger_lookup import candidate_article_numbers, ean_checksum_valid

SERIES_PREFIXES = ["120", "130", "132", "150", "160", "170"]


class TestEanChecksumValid:
    """Checksum validation for EAN-13 payloads."""

    def test_valid_standard_line_ean(self) -> None:
        """The Frozen II 2x12 box barcode (article 05009) is valid."""
        assert ean_checksum_valid("4005556050093") is True

    def test_valid_adult_line_ean(self) -> None:
        """A known adult-line barcode (article 12000622) is valid."""
        assert ean_checksum_valid("4005555006220") is True

    def test_invalid_check_digit(self) -> None:
        """A single flipped check digit is rejected."""
        assert ean_checksum_valid("4005556050094") is False

    def test_wrong_length(self) -> None:
        """Too-short and too-long inputs are rejected."""
        assert ean_checksum_valid("400555605009") is False
        assert ean_checksum_valid("40055560500933") is False
        assert ean_checksum_valid("") is False

    def test_non_digit_input(self) -> None:
        """Non-digit characters are rejected even at the right length."""
        assert ean_checksum_valid("400555605009X") is False
        assert ean_checksum_valid("4005556o50093") is False


class TestCandidateArticleNumbers:
    """Derivation of candidate article numbers from a valid EAN."""

    def test_standard_prefix_yields_payload(self) -> None:
        """4005556 EANs map directly to their 5-digit payload."""
        assert candidate_article_numbers("4005556050093", SERIES_PREFIXES) == ["05009"]

    def test_adult_prefix_yields_series_candidates(self) -> None:
        """4005555 EANs yield one candidate per configured series prefix, in order."""
        assert candidate_article_numbers("4005555006220", SERIES_PREFIXES) == [
            "12000622",
            "13000622",
            "13200622",
            "15000622",
            "16000622",
            "17000622",
        ]

    def test_adult_prefix_with_no_series_yields_nothing(self) -> None:
        """An empty series list produces no adult-line candidates."""
        assert candidate_article_numbers("4005555006220", []) == []

    def test_other_prefix_yields_nothing(self) -> None:
        """Non-Ravensburger GS1 prefixes produce no candidates."""
        assert candidate_article_numbers("4006381333931", SERIES_PREFIXES) == []
