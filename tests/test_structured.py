"""Tests for structured output — dict_to_model, extract, validate."""

from veska.core.structured import dict_to_model, extract_and_validate, build_schema_instructions


def test_dict_to_model_creates_valid_model():
    """dict_to_model should create a Pydantic model from a dict."""
    Model = dict_to_model({"title": str, "rating": float, "ok": bool})
    obj = Model(title="Test", rating=9.0, ok=True)

    assert obj.title == "Test"
    assert obj.rating == 9.0
    assert obj.ok is True


def test_extract_and_validate_valid_json():
    """Valid JSON matching the schema should parse successfully."""
    Model = dict_to_model({"rating": float, "ok": bool})
    result, err = extract_and_validate(Model, '{"rating": 9.0, "ok": true}')

    assert err is None
    assert result.rating == 9.0
    assert result.ok is True


def test_extract_and_validate_wrong_type():
    """JSON with wrong types should return an error."""
    Model = dict_to_model({"rating": float, "ok": bool})
    result, err = extract_and_validate(Model, '{"rating": "nine", "ok": true}')

    assert result is None
    assert err is not None


def test_extract_and_validate_missing_field():
    """JSON missing a required field should return an error."""
    Model = dict_to_model({"title": str, "rating": float})
    result, err = extract_and_validate(Model, '{"title": "Test"}')

    assert result is None
    assert err is not None


def test_extract_json_from_markdown_block():
    """Should extract JSON from ```json ... ``` blocks."""
    Model = dict_to_model({"name": str})
    text = 'Here is the result:\n```json\n{"name": "Veska"}\n```\nDone!'
    result, err = extract_and_validate(Model, text)

    assert err is None
    assert result.name == "Veska"


def test_extract_json_from_plain_block():
    """Should extract JSON from ``` ... ``` blocks without json tag."""
    Model = dict_to_model({"count": int})
    text = 'Result:\n```\n{"count": 42}\n```'
    result, err = extract_and_validate(Model, text)

    assert err is None
    assert result.count == 42


def test_extract_json_from_surrounding_text():
    """Should find JSON buried in surrounding text."""
    Model = dict_to_model({"value": int})
    text = 'The answer is {"value": 7} and that is it.'
    result, err = extract_and_validate(Model, text)

    assert err is None
    assert result.value == 7


def test_no_json_returns_error():
    """Plain text with no JSON should return an error."""
    Model = dict_to_model({"name": str})
    result, err = extract_and_validate(Model, "No JSON here at all.")

    assert result is None
    assert err is not None


def test_build_schema_instructions():
    """Schema instructions should contain the field names."""
    Model = dict_to_model({"title": str, "score": float})
    instructions = build_schema_instructions(Model)

    assert "title" in instructions
    assert "score" in instructions
    assert "JSON" in instructions
