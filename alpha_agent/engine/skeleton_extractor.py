"""Skeleton Extractor — hybrid AST + LLM structural normalizer.

Converts a concrete FASTEXPR expression into a reusable skeleton template:
  - Structural skeleton (pure AST): deterministic, no LLM needed
  - Slot semantic hints (LLM annotation): optional enrichment for SkeletonAgent

AST normalization rules:
  - call nodes: operator name + child structure preserved exactly
  - field_ref leaves: replaced by $X1, $X2, ... (same literal → same placeholder)
  - number leaves: replaced by $W1, $W2, ... (typed: int_window / float_threshold)
  - string leaves (enum values like 'subindustry'): replaced by $G1, $G2, ...

Two identical expressions produce the same template_str (deterministic).
The lark Tree is serialized to JSON for storage.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from lark import Token, Tree

from alpha_agent.config import settings
from alpha_agent.llm_utils import supports_json_response_format
from alpha_agent.engine.validator import extract_ast

_WINDOW_RANGES = {
    "int_window": (1, 252),
    "float_threshold": (0.0, 1.0),
    "quantile": (0.0, 1.0),
}

_GROUP_ENUMS = {"subindustry", "industry", "sector", "market"}

_LLM_HINT_PROMPT = """\
You are analyzing a FASTEXPR alpha expression.
Expression: {expression}
Hypothesis (if known): {hypothesis}

For each field slot below, write a short semantic hint describing what kind of
financial data field should fill this position.

Field slots:
{slots_text}

Reply ONLY as valid JSON: {{"$X1": "hint text", "$X2": "hint text", ...}}
Keep each hint under 15 words. Focus on the economic role (e.g. "momentum/return-like",
"fundamental valuation ratio", "volume/liquidity measure").
"""


@dataclass
class SkeletonTemplate:
    """Result of extracting a skeleton from one FASTEXPR expression."""
    template_str: str
    template_ast_json: str
    operators_used: list[str] = field(default_factory=list)
    field_slots: list[dict[str, Any]] = field(default_factory=list)
    param_slots: list[dict[str, Any]] = field(default_factory=list)
    group_slots: list[dict[str, Any]] = field(default_factory=list)


class SkeletonExtractor:
    """Extracts reusable skeleton templates from FASTEXPR expressions."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.llm_model

    # ── public API ────────────────────────────────────────────────────────────

    def extract(self, expression: str) -> SkeletonTemplate | None:
        """Extract a skeleton template (pure AST, no LLM). Returns None if unparseable."""
        tree = extract_ast(expression)
        if tree is None:
            return None
        return self._normalize(tree, expression)

    async def extract_with_hints(
        self,
        expression: str,
        hypothesis: str = "",
    ) -> SkeletonTemplate | None:
        """Extract skeleton + enrich field slots with LLM semantic hints."""
        template = self.extract(expression)
        if template is None:
            return None
        if not template.field_slots:
            return template
        await self._annotate_field_hints(template, expression, hypothesis)
        return template

    # ── normalization ─────────────────────────────────────────────────────────

    def _normalize(self, tree: Tree, original_expr: str) -> SkeletonTemplate:
        """Walk the AST and produce a template with slot placeholders."""
        field_map: dict[str, str] = {}   # literal_name -> $Xn
        param_map: dict[str, str] = {}   # str(value)   -> $Wn
        group_map: dict[str, str] = {}   # literal_str  -> $Gn

        operators: list[str] = []
        field_slots: list[dict[str, Any]] = []
        param_slots: list[dict[str, Any]] = []
        group_slots: list[dict[str, Any]] = []

        def walk(node: Any) -> str:
            if isinstance(node, Tree):
                if node.data == "call":
                    op_name = str(node.children[0])
                    operators.append(op_name)
                    args_tree = node.children[1]  # arglist
                    args_strs = [walk(c) for c in args_tree.children]
                    return f"{op_name}({', '.join(args_strs)})"

                if node.data == "arglist":
                    return ", ".join(walk(c) for c in node.children)

                if node.data == "kwarg":
                    k = str(node.children[0])
                    v = walk(node.children[1])
                    return f"{k}={v}"

                if node.data in ("add", "sub", "mul", "div"):
                    op_sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[node.data]
                    return f"{walk(node.children[0])} {op_sym} {walk(node.children[1])}"

                if node.data in ("gt", "lt", "gte", "lte", "eq", "neq"):
                    op_sym = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<=",
                              "eq": "==", "neq": "!="}[node.data]
                    return f"{walk(node.children[0])} {op_sym} {walk(node.children[1])}"

                if node.data == "neg":
                    return f"-{walk(node.children[0])}"

                # fallback: join children
                return " ".join(walk(c) for c in node.children)

            if isinstance(node, Token):
                tok_type = node.type
                tok_val = str(node)

                if tok_type == "IDENTIFIER":
                    # field reference (not a call — calls are handled above)
                    if tok_val in _GROUP_ENUMS:
                        if tok_val not in group_map:
                            idx = len(group_map) + 1
                            ph = f"$G{idx}"
                            group_map[tok_val] = ph
                            group_slots.append({
                                "name": ph,
                                "literal": tok_val,
                                "candidates": sorted(_GROUP_ENUMS),
                            })
                        return group_map[tok_val]
                    else:
                        if tok_val not in field_map:
                            idx = len(field_map) + 1
                            ph = f"$X{idx}"
                            field_map[tok_val] = ph
                            field_slots.append({
                                "name": ph,
                                "literal": tok_val,
                                "semantic_hint": "",
                            })
                        return field_map[tok_val]

                if tok_type in ("NUMBER", "SIGNED_NUMBER"):
                    val_key = tok_val
                    if val_key not in param_map:
                        idx = len(param_map) + 1
                        ph = f"$W{idx}"
                        param_map[val_key] = ph
                        ptype = _classify_number(tok_val)
                        seen_val = _parse_num(tok_val)
                        param_slots.append({
                            "name": ph,
                            "literal": tok_val,
                            "type": ptype,
                            "range": list(_WINDOW_RANGES.get(ptype, (0, 252))),
                            "seen": [seen_val] if seen_val is not None else [],
                        })
                    return param_map[val_key]

                if tok_type == "STRING":
                    inner = tok_val.strip("'\"")
                    if inner in _GROUP_ENUMS:
                        if inner not in group_map:
                            idx = len(group_map) + 1
                            ph = f"$G{idx}"
                            group_map[inner] = ph
                            group_slots.append({
                                "name": ph,
                                "literal": inner,
                                "candidates": sorted(_GROUP_ENUMS),
                            })
                        return f"'{group_map[inner]}'"
                    # non-enum string (e.g. range spec) — keep as-is
                    return tok_val

                return tok_val

            return str(node)

        template_str = walk(tree)
        ast_json = _tree_to_json(tree)

        return SkeletonTemplate(
            template_str=template_str,
            template_ast_json=json.dumps(ast_json),
            operators_used=list(dict.fromkeys(operators)),
            field_slots=field_slots,
            param_slots=param_slots,
            group_slots=group_slots,
        )

    # ── LLM hint annotation ───────────────────────────────────────────────────

    async def _annotate_field_hints(
        self,
        template: SkeletonTemplate,
        expression: str,
        hypothesis: str,
    ) -> None:
        if not template.field_slots:
            return
        slots_text = "\n".join(
            f"  {s['name']} (was: '{s['literal']}')"
            for s in template.field_slots
        )
        prompt = _LLM_HINT_PROMPT.format(
            expression=expression[:200],
            hypothesis=hypothesis[:200] if hypothesis else "(not provided)",
            slots_text=slots_text,
        )
        try:
            import litellm  # noqa: PLC0415
            completion_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            if supports_json_response_format(self._model):
                completion_kwargs["response_format"] = {"type": "json_object"}
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: litellm.completion(**completion_kwargs),
            )
            hints = json.loads(resp.choices[0].message.content or "{}")
            for slot in template.field_slots:
                slot["semantic_hint"] = hints.get(slot["name"], "")
        except Exception:
            pass

    # ── instantiation ─────────────────────────────────────────────────────────

    @staticmethod
    def instantiate(
        template_str: str,
        field_mapping: dict[str, str],
        param_mapping: dict[str, Any],
        group_mapping: dict[str, str] | None = None,
    ) -> str:
        """Replace placeholders in a template_str with concrete values.

        Args:
            template_str:   Skeleton string containing $X1, $W1, $G1, etc.
            field_mapping:  {"$X1": "close", "$X2": "volume", ...}
            param_mapping:  {"$W1": 20, "$W2": 30, ...}
            group_mapping:  {"$G1": "subindustry", ...}

        Returns:
            Concrete FASTEXPR expression string.
        """
        result = template_str
        # Apply field substitutions
        for ph, field_name in field_mapping.items():
            result = result.replace(ph, str(field_name))
        # Apply param substitutions
        for ph, value in param_mapping.items():
            result = result.replace(ph, str(value))
        # Apply group substitutions
        if group_mapping:
            for ph, group_name in group_mapping.items():
                # strip the quotes from template string if present
                result = result.replace(f"'{ph}'", group_name)
                result = result.replace(ph, group_name)
        return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _classify_number(val: str) -> str:
    """Guess numeric constant type from its string representation."""
    try:
        f = float(val)
        if f == int(f) and int(f) >= 2:
            return "int_window"
        if 0.0 <= f <= 1.0:
            return "quantile"
        return "float_threshold"
    except ValueError:
        return "int_window"


def _parse_num(val: str) -> float | int | None:
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except ValueError:
        return None


def _tree_to_json(node: Any) -> Any:
    """Serialize a lark Tree to a plain JSON-serializable dict."""
    if isinstance(node, Tree):
        return {"_t": node.data, "c": [_tree_to_json(c) for c in node.children]}
    if isinstance(node, Token):
        return {"_tok": node.type, "v": str(node)}
    return str(node)


def _json_to_tree(obj: Any) -> Any:
    """Deserialize a JSON dict back to a lark Tree (for display only)."""
    if isinstance(obj, dict):
        if "_t" in obj:
            children = [_json_to_tree(c) for c in obj.get("c", [])]
            return Tree(obj["_t"], children)
        if "_tok" in obj:
            return Token(obj["_tok"], obj["v"])
    return obj
