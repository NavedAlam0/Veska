"""
Structured output support.

Handles forcing AI to return valid Pydantic models:
  1. Extract JSON schema from a Pydantic model
  2. Build prompt instructions telling AI to return that schema
  3. Extract JSON from AI response text (handles markdown wrapping)
  4. Validate against the Pydantic model
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError


def dict_to_model(output_format: dict[str, type]) -> Type[BaseModel]:
    """Convert a simple dictionary of {field_name: type} into a Pydantic model.

    Example:
        {"title": str, "rating": float} becomes a Pydantic model with those fields.
    """
    TYPE_MAP = {str: str, int: int, float: float, bool: bool, list: list, dict: dict}
    fields = {}
    for field_name, field_type in output_format.items():
        resolved = TYPE_MAP.get(field_type, field_type)
        fields[field_name] = (resolved, ...)

    return type("OutputModel", (BaseModel,), {"__annotations__": {k: v[0] for k, v in fields.items()}})


def build_schema_instructions(model_class: Type[BaseModel]) -> str:
    """Build prompt instructions from a Pydantic model class."""
    schema = model_class.model_json_schema()

    # Clean up schema for readability (remove $defs, title, etc.)
    clean = _simplify_schema(schema)

    return (
        "\n\n---\n"
        "IMPORTANT: You MUST respond with ONLY valid JSON (no markdown, no explanation, no extra text).\n"
        f"Your response must match this exact schema:\n{json.dumps(clean, indent=2)}\n"
        "Return ONLY the JSON object. Nothing else."
    )


def extract_and_validate(
    model_class: Type[BaseModel],
    text: str,
) -> tuple[Optional[BaseModel], Optional[str]]:
    """
    Try to parse and validate AI response text into a Pydantic model.

    Returns:
        (parsed_model, None) on success.
        (None, error_message) on failure.
    """
    json_str = _extract_json(text)
    if json_str is None:
        return None, f"No valid JSON found in response. Raw text:\n{text[:500]}"

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}\nExtracted text:\n{json_str[:500]}"

    try:
        parsed = model_class.model_validate(data)
        return parsed, None
    except ValidationError as e:
        return None, f"JSON doesn't match schema: {e}"


def build_retry_message(error: str) -> str:
    """Build the retry message to send back to the AI."""
    return (
        f"Your previous response didn't match the required schema.\n"
        f"Error: {error}\n\n"
        f"Please try again. Return ONLY valid JSON matching the schema. Nothing else."
    )


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON from AI response text.

    Handles:
      - Raw JSON
      - JSON wrapped in ```json ... ``` blocks
      - JSON wrapped in ``` ... ``` blocks
      - JSON buried in surrounding text
    """
    text = text.strip()

    # Try raw parse first (best case: AI returned just JSON)
    if _is_json(text):
        return text

    # Try extracting from markdown code blocks
    patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",  # ```json ... ```
        r"```\s*\n?(.*?)\n?\s*```",        # ``` ... ```
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match and _is_json(match.group(1).strip()):
            return match.group(1).strip()

    # Try finding a JSON object in the text
    # Look for the outermost { ... } or [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching closing bracket
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    if _is_json(candidate):
                        return candidate
                    break

    return None


def _is_json(text: str) -> bool:
    """Check if a string is valid JSON."""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Simplify a Pydantic JSON schema for prompt injection.

    Removes internal Pydantic metadata that would confuse the AI.
    Resolves $defs references inline.
    """
    defs = schema.pop("$defs", {})

    def resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            # Resolve $ref
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    return resolve(defs[ref_name])
                return obj

            result = {}
            for key, value in obj.items():
                if key in ("title",):
                    continue
                result[key] = resolve(value)
            return result
        if isinstance(obj, list):
            return [resolve(item) for item in obj]
        return obj

    return resolve(schema)
