"""Profile loading (rag/config.py) — the seam that keeps the core domain-agnostic."""

import json
from pathlib import Path

import pytest

from rag import config

# Every profile folder in the project — including ones you add later, so a typo in a
# new profile.json (or a malformed eval.json) fails here, and in CI, not at startup.
PROJECT_PROFILES = sorted(
    p.name for p in config.PROFILES_DIR.iterdir() if (p / "profile.json").is_file()
)


def _write_profile(profiles_dir, name, data):
    folder = profiles_dir / name
    folder.mkdir(parents=True)
    (folder / "profile.json").write_text(json.dumps(data), encoding="utf-8")
    return folder


# ---------- The profiles in the project ----------


def test_reference_profiles_are_present():
    assert {"general", "air_india"} <= set(PROJECT_PROFILES)


@pytest.mark.parametrize("name", PROJECT_PROFILES)
def test_every_profile_loads_and_validates(name):
    profile = config.load_profile(name)
    assert profile["name"] == name
    assert profile["app_name"]
    if profile["eval_dataset"]:
        items = json.loads(Path(profile["eval_dataset"]).read_text(encoding="utf-8"))
        assert isinstance(items, list) and items, "eval.json must be a non-empty list"
        for item in items:
            assert isinstance(item.get("question"), str) and item["question"].strip()
            assert isinstance(item.get("reference_answer"), str) and item["reference_answer"].strip()


def test_general_profile_is_free_of_any_specific_domain():
    raw = (config.PROFILES_DIR / "general" / "profile.json").read_text(encoding="utf-8").lower()
    assert "air india" not in raw and "flight" not in raw
    assert "flight" not in config.load_profile("general")["enabled_tools"]


def test_air_india_profile_reproduces_the_original_assistant():
    profile = config.load_profile("air_india")
    assert profile["app_name"] == "Air India Assistant"
    assert profile["default_knowledge_base"] == "Air India"
    assert set(profile["enabled_tools"]) == {"flight", "weather", "currency"}
    assert profile["eval_dataset"].endswith("eval.json")
    assert profile["docs_folder"].replace("\\", "/").endswith("profiles/air_india/docs")


# ---------- Defaults, paths, validation ----------


def test_minimal_profile_gets_defaults_and_conventional_paths(tmp_path):
    folder = _write_profile(tmp_path, "hr", {"app_name": "HR Helper", "domain_description": "our HR policies"})

    profile = config.load_profile("hr", profiles_dir=tmp_path)

    assert profile["app_name"] == "HR Helper"
    assert profile["enabled_tools"] == []                       # tools are opt-in
    assert profile["default_knowledge_base"] == config.PROFILE_DEFAULTS["default_knowledge_base"]
    assert profile["docs_folder"] == str(folder / "docs")
    assert profile["eval_dataset"] is None


def test_eval_dataset_is_picked_up_when_present(tmp_path):
    folder = _write_profile(tmp_path, "hr", {})
    (folder / "eval.json").write_text("[]", encoding="utf-8")
    assert config.load_profile("hr", profiles_dir=tmp_path)["eval_dataset"] == str(folder / "eval.json")


def test_docs_folder_can_point_anywhere(tmp_path):
    _write_profile(tmp_path, "rel", {"docs_folder": "shared/manuals"})
    _write_profile(tmp_path, "abs", {"docs_folder": str(tmp_path / "elsewhere")})

    assert config.load_profile("rel", profiles_dir=tmp_path)["docs_folder"] == str(config.PROJECT_ROOT / "shared/manuals")
    assert config.load_profile("abs", profiles_dir=tmp_path)["docs_folder"] == str(tmp_path / "elsewhere")


def test_underscore_keys_are_comments(tmp_path):
    _write_profile(tmp_path, "c", {"_comment": "notes for humans", "app_name": "X"})
    assert "_comment" not in config.load_profile("c", profiles_dir=tmp_path)


def test_typo_in_a_key_fails_loudly(tmp_path):
    _write_profile(tmp_path, "typo", {"enabled_tool": ["weather"]})
    with pytest.raises(ValueError, match="enabled_tool"):
        config.load_profile("typo", profiles_dir=tmp_path)


def test_unknown_tool_fails_loudly(tmp_path):
    _write_profile(tmp_path, "t", {"enabled_tools": ["weather", "stock_prices"]})
    with pytest.raises(ValueError, match="stock_prices"):
        config.load_profile("t", profiles_dir=tmp_path)


def test_list_fields_must_be_lists_of_strings(tmp_path):
    _write_profile(tmp_path, "l", {"suggested_questions": "just one string"})
    with pytest.raises(ValueError, match="suggested_questions"):
        config.load_profile("l", profiles_dir=tmp_path)


def test_missing_profile_lists_the_available_ones(tmp_path):
    _write_profile(tmp_path, "alpha", {})
    with pytest.raises(FileNotFoundError, match="alpha"):
        config.load_profile("does_not_exist", profiles_dir=tmp_path)
