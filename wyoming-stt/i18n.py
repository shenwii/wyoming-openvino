"""
Internationalization (i18n) module for Wyoming OpenVINO STT.
Loads structured translation catalogs from YAML files in the 'lang' directory.
Language selection is governed by standard environment variables:
WYOMING_LANG, LC_ALL, LC_MESSAGES, or LANG.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_current_locale() -> str:
    """
    Detect the active locale code from standard environment variables.
    Returns 'en' (default), 'zh', 'ja', or 'ko'.
    """
    raw = (
        os.environ.get("WYOMING_LANG")
        or os.environ.get("LC_ALL")
        or os.environ.get("LC_MESSAGES")
        or os.environ.get("LANG")
        or "en"
    ).strip().lower()

    raw = raw.split(".")[0].split("@")[0]
    if raw.startswith("zh"):
        return "zh"
    elif raw.startswith("ja"):
        return "ja"
    elif raw.startswith("ko"):
        return "ko"
    return "en"


CURRENT_LOCALE = get_current_locale()
LANG_DIR = Path(__file__).resolve().parent / "lang"

_CATALOG_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_yaml(locale: str) -> Dict[str, Any]:
    """Load translation YAML for a given locale."""
    if locale in _CATALOG_CACHE:
        return _CATALOG_CACHE[locale]

    file_path = LANG_DIR / f"{locale}.yaml"
    if not file_path.is_file():
        file_path = LANG_DIR / f"{locale}.yml"

    if file_path.is_file():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                _CATALOG_CACHE[locale] = data
                return data
        except Exception:
            pass

    _CATALOG_CACHE[locale] = {}
    return {}


_ACTIVE_CATALOG = _load_yaml(CURRENT_LOCALE)
_FALLBACK_CATALOG = _load_yaml("en") if CURRENT_LOCALE != "en" else _ACTIVE_CATALOG


def set_locale(locale: str) -> None:
    """Dynamically switch active locale."""
    global CURRENT_LOCALE, _ACTIVE_CATALOG, _FALLBACK_CATALOG
    CURRENT_LOCALE = locale
    _ACTIVE_CATALOG = _load_yaml(locale)
    _FALLBACK_CATALOG = _load_yaml("en") if locale != "en" else _ACTIVE_CATALOG


def _lookup_key(catalog: Dict[str, Any], key: str) -> Optional[str]:
    """Traverse dot-separated key in catalog."""
    curr = catalog
    for part in key.split("."):
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return None
    return curr if isinstance(curr, str) else None


def t(key: str, **kwargs: Any) -> str:
    """
    Translate structured key (e.g. 'cli.options.model_dir') to the active locale.
    Falls back to English if the key is missing in the active locale.
    Formats with kwargs if provided.
    """
    val = _lookup_key(_ACTIVE_CATALOG, key)
    if val is None and CURRENT_LOCALE != "en":
        val = _lookup_key(_FALLBACK_CATALOG, key)
    if val is None:
        val = key

    if kwargs:
        try:
            return val.format(**kwargs)
        except Exception:
            return val
    return val


_ = t
