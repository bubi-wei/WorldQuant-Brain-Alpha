"""Tests for SkeletonExtractor — AST normalization, slot assignment, and isomorphism.

Run with:
    pytest tests/test_skeleton_extractor.py -v
"""

from __future__ import annotations

import pytest

from alpha_agent.engine.skeleton_extractor import SkeletonExtractor, _classify_number
from alpha_agent.engine.validator import extract_ast, quick_validate

extractor = SkeletonExtractor()


# ── extract_ast ───────────────────────────────────────────────────────────────

class TestExtractAst:
    def test_simple_expr(self):
        tree = extract_ast("rank(close)")
        assert tree is not None

    def test_arithmetic(self):
        tree = extract_ast("(close - open) / open")
        assert tree is not None

    def test_invalid_expr_returns_none(self):
        tree = extract_ast("rank(close")
        assert tree is None

    def test_empty_returns_none(self):
        tree = extract_ast("")
        assert tree is None


# ── SkeletonExtractor.extract ─────────────────────────────────────────────────

class TestExtractSkeleton:
    def test_basic_field_replacement(self):
        tmpl = extractor.extract("rank(close)")
        assert tmpl is not None
        assert "$X1" in tmpl.template_str
        assert "close" not in tmpl.template_str
        assert len(tmpl.field_slots) == 1
        assert tmpl.field_slots[0]["name"] == "$X1"
        assert tmpl.field_slots[0]["literal"] == "close"

    def test_number_replacement(self):
        tmpl = extractor.extract("ts_mean(volume, 20)")
        assert tmpl is not None
        assert "$W1" in tmpl.template_str
        assert "20" not in tmpl.template_str
        assert len(tmpl.param_slots) == 1
        assert tmpl.param_slots[0]["literal"] == "20"

    def test_group_enum_replacement(self):
        tmpl = extractor.extract("group_rank(close, subindustry)")
        assert tmpl is not None
        assert "$G1" in tmpl.template_str
        assert "subindustry" not in tmpl.template_str
        assert len(tmpl.group_slots) == 1
        assert tmpl.group_slots[0]["literal"] == "subindustry"

    def test_operators_captured(self):
        tmpl = extractor.extract("group_rank(ts_std_dev(returns, 20), subindustry)")
        assert tmpl is not None
        assert "group_rank" in tmpl.operators_used
        assert "ts_std_dev" in tmpl.operators_used

    def test_same_field_gets_same_placeholder(self):
        # ts_corr(x, x, n) — same field used twice should reuse placeholder
        tmpl = extractor.extract("ts_corr(close, close, 10)")
        assert tmpl is not None
        # Only one unique field slot
        assert len(tmpl.field_slots) == 1
        assert tmpl.template_str.count("$X1") == 2

    def test_two_different_fields_get_different_placeholders(self):
        tmpl = extractor.extract("ts_corr(volume, returns, 10)")
        assert tmpl is not None
        assert len(tmpl.field_slots) == 2
        names = [s["name"] for s in tmpl.field_slots]
        assert "$X1" in names
        assert "$X2" in names

    def test_invalid_expr_returns_none(self):
        assert extractor.extract("rank(close") is None


# ── Isomorphism ───────────────────────────────────────────────────────────────

class TestIsomorphism:
    def test_structurally_identical_different_fields(self):
        """Two expressions that differ only in field names are isomorphic."""
        tmpl_a = extractor.extract("group_rank(ts_std_dev(returns, 20), subindustry)")
        tmpl_b = extractor.extract("group_rank(ts_std_dev(close, 20), subindustry)")
        assert tmpl_a is not None and tmpl_b is not None
        assert tmpl_a.template_str == tmpl_b.template_str

    def test_structurally_identical_different_params(self):
        """Different window sizes → same skeleton."""
        tmpl_a = extractor.extract("ts_mean(volume, 20)")
        tmpl_b = extractor.extract("ts_mean(volume, 30)")
        assert tmpl_a is not None and tmpl_b is not None
        assert tmpl_a.template_str == tmpl_b.template_str

    def test_structurally_identical_different_group(self):
        """Different group enum values → same skeleton."""
        tmpl_a = extractor.extract("group_rank(close, subindustry)")
        tmpl_b = extractor.extract("group_rank(close, industry)")
        assert tmpl_a is not None and tmpl_b is not None
        assert tmpl_a.template_str == tmpl_b.template_str

    def test_structurally_different_not_isomorphic(self):
        """Different operator structure → different skeleton."""
        tmpl_a = extractor.extract("group_rank(ts_std_dev(returns, 20), subindustry)")
        tmpl_b = extractor.extract("ts_rank(returns, 20)")
        assert tmpl_a is not None and tmpl_b is not None
        assert tmpl_a.template_str != tmpl_b.template_str

    def test_nesting_difference_not_isomorphic(self):
        """Shallower vs deeper nesting → different skeletons."""
        tmpl_a = extractor.extract("rank(close)")
        tmpl_b = extractor.extract("rank(ts_mean(close, 5))")
        assert tmpl_a is not None and tmpl_b is not None
        assert tmpl_a.template_str != tmpl_b.template_str

    def test_full_example_same_window(self):
        """Both windows are 20 → same literal → same placeholder $W1 (preserves same-window pattern)."""
        expr = "group_rank(ts_std_dev(returns, 20) / ts_mean(volume, 20), subindustry)"
        tmpl = extractor.extract(expr)
        assert tmpl is not None
        assert "$X1" in tmpl.template_str
        assert "$X2" in tmpl.template_str
        assert "$W1" in tmpl.template_str
        assert "$G1" in tmpl.template_str
        # Same window 20 → both slots reference $W1 (structurally meaningful)
        assert tmpl.template_str == "group_rank(ts_std_dev($X1, $W1) / ts_mean($X2, $W1), $G1)"

    def test_full_example_different_windows(self):
        """Different windows → distinct $W1 and $W2 placeholders."""
        expr = "group_rank(ts_std_dev(returns, 20) / ts_mean(volume, 30), subindustry)"
        tmpl = extractor.extract(expr)
        assert tmpl is not None
        assert "$W1" in tmpl.template_str
        assert "$W2" in tmpl.template_str
        assert tmpl.template_str == "group_rank(ts_std_dev($X1, $W1) / ts_mean($X2, $W2), $G1)"


# ── Instantiation ─────────────────────────────────────────────────────────────

class TestInstantiation:
    def test_basic_instantiation(self):
        tmpl = extractor.extract("ts_mean(close, 20)")
        assert tmpl is not None
        result = SkeletonExtractor.instantiate(
            tmpl.template_str,
            field_mapping={"$X1": "volume"},
            param_mapping={"$W1": 30},
        )
        assert result == "ts_mean(volume, 30)"
        assert quick_validate(result).ok

    def test_group_instantiation(self):
        tmpl = extractor.extract("group_rank(close, subindustry)")
        assert tmpl is not None
        result = SkeletonExtractor.instantiate(
            tmpl.template_str,
            field_mapping={"$X1": "vwap"},
            param_mapping={},
            group_mapping={"$G1": "sector"},
        )
        assert result == "group_rank(vwap, sector)"
        assert quick_validate(result).ok

    def test_complex_instantiation(self):
        template_str = "group_rank(ts_std_dev($X1, $W1) / ts_mean($X2, $W2), $G1)"
        result = SkeletonExtractor.instantiate(
            template_str,
            field_mapping={"$X1": "returns", "$X2": "turnover"},
            param_mapping={"$W1": 20, "$W2": 10},
            group_mapping={"$G1": "industry"},
        )
        assert result == "group_rank(ts_std_dev(returns, 20) / ts_mean(turnover, 10), industry)"

    def test_roundtrip_extract_instantiate(self):
        """Extract then re-instantiate with original values → same expression."""
        expr = "group_rank(ts_mean(volume, 20), subindustry)"
        tmpl = extractor.extract(expr)
        assert tmpl is not None

        # Re-instantiate with original literal values from slots
        field_mapping = {s["name"]: s["literal"] for s in tmpl.field_slots}
        param_mapping = {p["name"]: p["literal"] for p in tmpl.param_slots}
        group_mapping = {g["name"]: g["literal"] for g in tmpl.group_slots}

        result = SkeletonExtractor.instantiate(
            tmpl.template_str,
            field_mapping=field_mapping,
            param_mapping=param_mapping,
            group_mapping=group_mapping,
        )
        assert result == expr


# ── Number classification ─────────────────────────────────────────────────────

class TestClassifyNumber:
    def test_integer_window(self):
        assert _classify_number("20") == "int_window"
        assert _classify_number("252") == "int_window"
        assert _classify_number("5") == "int_window"

    def test_quantile(self):
        assert _classify_number("0.5") == "quantile"
        assert _classify_number("0.9") == "quantile"

    def test_float_threshold(self):
        assert _classify_number("1.5") == "float_threshold"
        assert _classify_number("-0.5") in ("quantile", "float_threshold")


# ── Param slot seen values ────────────────────────────────────────────────────

class TestParamSlots:
    def test_seen_values_recorded(self):
        tmpl = extractor.extract("ts_std_dev(returns, 30)")
        assert tmpl is not None
        assert len(tmpl.param_slots) == 1
        slot = tmpl.param_slots[0]
        assert 30 in slot["seen"] or "30" in [str(x) for x in slot["seen"]]

    def test_multiple_params_different_slots(self):
        tmpl = extractor.extract("ts_corr(volume, returns, 10)")
        assert tmpl is not None
        assert len(tmpl.param_slots) == 1
        assert tmpl.param_slots[0]["literal"] == "10"

    def test_same_number_gets_same_slot(self):
        # ts_std_dev(returns, 20) / ts_mean(volume, 20) — same window 20 → one param slot
        tmpl = extractor.extract("ts_std_dev(returns, 20) / ts_mean(volume, 20)")
        assert tmpl is not None
        # Both windows should collapse to the same placeholder
        assert tmpl.template_str.count("$W1") == 2
        assert len(tmpl.param_slots) == 1


# ── Regression expressions (known shapes) ────────────────────────────────────

@pytest.mark.parametrize("expr,expected_template", [
    (
        "rank(close)",
        "rank($X1)",
    ),
    (
        "ts_rank(returns, 252)",
        "ts_rank($X1, $W1)",
    ),
    (
        "group_rank(close, subindustry)",
        "group_rank($X1, $G1)",
    ),
    (
        "ts_corr(volume, returns, 10)",
        "ts_corr($X1, $X2, $W1)",
    ),
])
def test_known_templates(expr, expected_template):
    tmpl = extractor.extract(expr)
    assert tmpl is not None, f"Failed to extract: {expr}"
    assert tmpl.template_str == expected_template, (
        f"Expected '{expected_template}', got '{tmpl.template_str}'"
    )
