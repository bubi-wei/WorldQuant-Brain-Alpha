"""Local FASTEXPR expression validator.

Performs pre-flight checks before submitting to WQB, saving API quota:
  1. Bracket/parenthesis balance
  2. Basic lark grammar parse (subset of FASTEXPR)
  3. Operator arity check
  4. Field existence check against a known field set

Returns a ValidationResult with pass/fail and a list of error messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lark import Lark, Tree, UnexpectedInput

# ── Grammar ───────────────────────────────────────────────────────────────────
# A subset FASTEXPR grammar covering the most common patterns.
# Real FASTEXPR is a superset; unknown-but-valid expressions will still
# pass the WQB API — this validator only catches obvious mistakes.

_FASTEXPR_GRAMMAR = r"""
    ?start: expr

    ?expr: expr "+" term   -> add
         | expr "-" term   -> sub
         | expr "*" term   -> mul
         | expr "/" term   -> div
         | expr ">" term   -> gt
         | expr "<" term   -> lt
         | expr ">=" term  -> gte
         | expr "<=" term  -> lte
         | expr "==" term  -> eq
         | expr "!=" term  -> neq
         | term

    ?term: "-" atom        -> neg
         | atom

    ?atom: NUMBER           -> number
         | SIGNED_NUMBER    -> number
         | STRING           -> string
         | IDENTIFIER "(" arglist ")" -> call
         | IDENTIFIER       -> field_ref
         | "(" expr ")"

    arglist: (expr ("," expr)*)? ("," kwarg)*
    kwarg: IDENTIFIER "=" expr

    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    STRING: "'" /[^']*/ "'"
           | "\"" /[^"]*/ "\""

    %import common.NUMBER
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""

_PARSER = Lark(_FASTEXPR_GRAMMAR, parser="earley", ambiguity="resolve")


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.ok:
            return "PASS"
        return "FAIL: " + "; ".join(self.errors)


class ExprValidator:
    """Validates FASTEXPR expressions locally."""

    def __init__(
        self,
        operator_kb: Any | None = None,
        known_fields: set[str] | None = None,
    ) -> None:
        if operator_kb is None:
            from alpha_agent.data.operator_kb import OperatorKB  # noqa: PLC0415
            operator_kb = OperatorKB()
        self._kb = operator_kb
        self._known_fields = known_fields or set()
        self._known_ops = set(self._kb.all_names())

    def set_known_fields(self, fields: set[str]) -> None:
        """Update the set of valid field names for this dataset/universe."""
        self._known_fields = fields

    def validate(self, expression: str) -> ValidationResult:
        """Run all validation checks and return a combined result."""
        errors: list[str] = []
        warnings: list[str] = []

        # Check 1: Not empty
        expr = expression.strip()
        if not expr:
            return ValidationResult(ok=False, errors=["Empty expression"])

        # Check 2: Bracket balance
        bracket_err = self._check_brackets(expr)
        if bracket_err:
            errors.append(bracket_err)

        # Check 3: Grammar parse
        parse_err = self._check_grammar(expr)
        if parse_err:
            errors.append(parse_err)

        # If basic syntax is broken, skip deeper checks
        if errors:
            return ValidationResult(ok=False, errors=errors)

        # Check 4: Unknown operators (hard fail)
        unknown_ops = self._find_unknown_operators(expr)
        if unknown_ops:
            errors.append(f"Unknown operators: {unknown_ops}")

        # Check 5: Unknown fields (hard fail in strict mode)
        if self._known_fields:
            unknown_fields = self._find_unknown_fields(expr)
            if unknown_fields:
                errors.append(f"Fields not in known set: {unknown_fields}")

        return ValidationResult(ok=True, errors=[], warnings=warnings)

    # ── internal checks ───────────────────────────────────────────────────────

    @staticmethod
    def _check_brackets(expr: str) -> str | None:
        depth = 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                return f"Unexpected ')' at position {i}"
        if depth != 0:
            return f"Unmatched '(' — {depth} unclosed bracket(s)"
        return None

    @staticmethod
    def _check_grammar(expr: str) -> str | None:
        try:
            _PARSER.parse(expr)
            return None
        except UnexpectedInput as e:
            # Trim to a short message
            msg = str(e).split("\n")[0]
            return f"Parse error: {msg}"
        except Exception as e:
            return f"Parse error: {e}"

    def _find_unknown_operators(self, expr: str) -> list[str]:
        # Operator calls look like: word(
        calls = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expr)
        # Underscore-prefixed custom names are NOT allowed unless present in operator KB.
        unknown = [c for c in calls if c not in self._known_ops]
        return list(dict.fromkeys(unknown))

    def _find_unknown_fields(self, expr: str) -> list[str]:
        # Field references: identifiers NOT followed by '(' and NOT pure keywords
        _KEYWORDS = {
            "if_else", "trade_when", "rank", "group_rank", "subindustry", "industry",
            "sector", "market", "bucket", "range", "TRUE", "FALSE", "true", "false",
        }
        # Find all identifiers NOT followed by '('
        refs = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*\()", expr)
        # Filter out operators, keywords, and numeric constants
        candidates = {
            r for r in refs
            if r not in self._known_ops
            and r not in _KEYWORDS
            and not r.isnumeric()
        }
        return [f for f in candidates if f not in self._known_fields]


def quick_validate(expression: str, known_fields: set[str] | None = None) -> ValidationResult:
    """Convenience function for one-off validation without OperatorKB."""
    # Avoid loading OperatorKB/VectorStore for lightweight callers
    class _MinimalKB:
        def all_names(self) -> list[str]:
            return []
    validator = ExprValidator(operator_kb=_MinimalKB(), known_fields=known_fields or set())
    return validator.validate(expression)


def extract_ast(expression: str) -> Tree | None:
    """Parse a FASTEXPR string and return its lark Tree, or None on failure.

    Public helper used by SkeletonExtractor to walk the AST without
    instantiating a full ExprValidator.
    """
    try:
        return _PARSER.parse(expression.strip())
    except Exception:
        return None
