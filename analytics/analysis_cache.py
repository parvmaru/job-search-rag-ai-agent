"""
Analysis Cache - Store and retrieve analysis results for fast repeated runs
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


CACHE_DIR = Path("analytics")
CACHE_FILE = CACHE_DIR / "cache.json"


def get_cache_key(jd_name: str, resume_name: str) -> str:
    """Generate a cache key from JD and resume names."""
    key_string = f"{jd_name}_{resume_name}"
    return hashlib.md5(key_string.encode()).hexdigest()


def load_cache() -> Dict[str, Any]:
    """Load cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}


def save_cache(cache: Dict[str, Any]):
    """Save cache to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_cached_analysis(jd_name: str, resume_name: str) -> Optional[Dict[str, Any]]:
    """Get cached analysis if available."""
    cache = load_cache()
    cache_key = get_cache_key(jd_name, resume_name)
    return cache.get(cache_key)


def save_analysis(jd_name: str, resume_name: str, analysis: Dict[str, Any]):
    """Save analysis to cache."""
    cache = load_cache()
    cache_key = get_cache_key(jd_name, resume_name)
    cache[cache_key] = {
        'jd_name': jd_name,
        'resume_name': resume_name,
        'analysis': analysis
    }
    save_cache(cache)


def clear_cache():
    """Clear all cached analyses."""
    if CACHE_FILE.exists():
        os.remove(CACHE_FILE)
    
    # Also clear specific cache entry
    return True

def clear_specific_cache(jd_name: str, resume_name: str):
    """Clear cache for a specific JD-Resume combination."""
    cache = load_cache()
    cache_key = get_cache_key(jd_name, resume_name)
    if cache_key in cache:
        del cache[cache_key]
        save_cache(cache)
        return True
    return False

