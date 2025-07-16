"""
Asset Manager for CrackSeg GUI Application.

This module provides centralized asset management with optimization,
caching, and dynamic loading capabilities.
"""

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import streamlit as st


@dataclass
class AssetMetadata:
    """Metadata for asset tracking and optimization."""

    path: str
    size: int
    hash: str
    mime_type: str
    last_modified: float
    optimized: bool = False
    cached: bool = False
    load_priority: str = "normal"  # critical, high, normal, low, lazy


@dataclass
class AssetRegistry:
    """Central registry for all application assets."""

    assets: dict[str, AssetMetadata] = field(default_factory=dict)
    version: str = "1.0.0"
    last_updated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "assets": {
                name: {
                    "path": meta.path,
                    "size": meta.size,
                    "hash": meta.hash,
                    "mime_type": meta.mime_type,
                    "last_modified": meta.last_modified,
                    "optimized": meta.optimized,
                    "cached": meta.cached,
                    "load_priority": meta.load_priority,
                }
                for name, meta in self.assets.items()
            },
            "version": self.version,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssetRegistry":
        """Create from dictionary."""
        registry = cls(
            version=data.get("version", "1.0.0"),
            last_updated=data.get("last_updated", 0.0),
        )

        for name, meta_data in data.get("assets", {}).items():
            registry.assets[name] = AssetMetadata(**meta_data)

        return registry


class AssetManager:
    """Centralized asset management system."""

    _instance: Optional["AssetManager"] = None
    _registry: AssetRegistry | None = None
    _cache: dict[str, Any] = {}
    _base_path: Path | None = None

    def __new__(cls) -> "AssetManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize asset manager."""
        if self._base_path is None:
            self._base_path = Path(__file__).parent
            self._load_registry()

    def _load_registry(self) -> None:
        """Load asset registry from file."""
        if self._base_path is None:
            st.error("Asset manager base path not initialized")
            return

        registry_path = self._base_path / "manifest" / "asset_registry.json"

        if registry_path.exists():
            try:
                with open(registry_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._registry = AssetRegistry.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                st.warning(f"Failed to load asset registry: {e}")
                self._registry = AssetRegistry()
        else:
            self._registry = AssetRegistry()
            self._save_registry()

    def _save_registry(self) -> None:
        """Save asset registry to file."""
        if self._registry is None or self._base_path is None:
            return

        registry_path = self._base_path / "manifest" / "asset_registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(self._registry.to_dict(), f, indent=2)
        except Exception as e:
            st.error(f"Failed to save asset registry: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for caching."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for file."""
        suffix = file_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
            ".ttf": "font/ttf",
            ".otf": "font/otf",
        }
        return mime_types.get(suffix, "application/octet-stream")

    def register_asset(
        self, name: str, relative_path: str, load_priority: str = "normal"
    ) -> bool:
        """Register an asset in the system."""
        if self._base_path is None:
            st.error("Asset manager base path not initialized")
            return False

        file_path = self._base_path / relative_path

        if not file_path.exists():
            st.warning(f"Asset not found: {file_path}")
            return False

        stat = file_path.stat()
        file_hash = self._get_file_hash(file_path)

        metadata = AssetMetadata(
            path=relative_path,
            size=stat.st_size,
            hash=file_hash,
            mime_type=self._get_mime_type(file_path),
            last_modified=stat.st_mtime,
            load_priority=load_priority,
        )

        if self._registry is None:
            self._registry = AssetRegistry()

        self._registry.assets[name] = metadata
        self._save_registry()
        return True

    def get_asset_path(self, name: str) -> Path | None:
        """Get full path to asset."""
        if (
            self._registry is None
            or name not in self._registry.assets
            or self._base_path is None
        ):
            return None

        metadata = self._registry.assets[name]
        return self._base_path / metadata.path

    def get_asset_url(self, name: str) -> str | None:
        """Get asset URL for use in Streamlit."""
        asset_path = self.get_asset_path(name)
        if asset_path is None or not asset_path.exists():
            return None

        # For Streamlit, we need to encode assets as data URLs
        return self._get_data_url(asset_path)

    def _get_data_url(self, file_path: Path) -> str:
        """Create data URL for asset."""
        cache_key = f"data_url_{file_path.as_posix()}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            mime_type = self._get_mime_type(file_path)
            encoded = base64.b64encode(content).decode("utf-8")
            data_url = f"data:{mime_type};base64,{encoded}"

            self._cache[cache_key] = data_url
            return data_url
        except Exception as e:
            st.error(f"Failed to create data URL for {file_path}: {e}")
            return ""

    def get_css_content(self, name: str) -> str | None:
        """Get CSS content as string."""
        asset_path = self.get_asset_path(name)
        if asset_path is None or not asset_path.exists():
            return None

        cache_key = f"css_{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with open(asset_path, encoding="utf-8") as f:
                content = f.read()

            self._cache[cache_key] = content
            return content
        except Exception as e:
            st.error(f"Failed to load CSS {name}: {e}")
            return None

    def inject_css(self, name: str) -> bool:
        """Inject CSS asset into Streamlit."""
        content = self.get_css_content(name)
        if content is None:
            return False

        st.markdown(f"<style>{content}</style>", unsafe_allow_html=True)
        return True

    def get_critical_assets(self) -> list[str]:
        """Get list of critical assets that should be loaded first."""
        if self._registry is None:
            return []

        return [
            name
            for name, meta in self._registry.assets.items()
            if meta.load_priority == "critical"
        ]

    def preload_critical_assets(self) -> None:
        """Preload critical assets into cache."""
        critical_assets = self.get_critical_assets()

        for asset_name in critical_assets:
            # Preload into cache
            self.get_asset_url(asset_name)
            if asset_name.endswith((".css",)):
                self.get_css_content(asset_name)

    def get_theme_assets(self, theme_name: str) -> dict[str, list[str]]:
        """Get assets specific to a theme."""
        if self._registry is None:
            return {"css": [], "images": [], "fonts": []}

        theme_assets = {"css": [], "images": [], "fonts": []}

        for name, metadata in self._registry.assets.items():
            if f"themes/{theme_name}" in metadata.path:
                if metadata.path.endswith(".css"):
                    theme_assets["css"].append(name)
                elif metadata.path.startswith("images/"):
                    theme_assets["images"].append(name)
                elif metadata.path.startswith("fonts/"):
                    theme_assets["fonts"].append(name)

        return theme_assets

    def cleanup_cache(self) -> None:
        """Clear asset cache."""
        self._cache.clear()

    def get_registry_stats(self) -> dict[str, Any]:
        """Get statistics about the asset registry."""
        if self._registry is None:
            return {}

        total_size = sum(meta.size for meta in self._registry.assets.values())
        priority_counts = {}

        for meta in self._registry.assets.values():
            priority = meta.load_priority
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        return {
            "total_assets": len(self._registry.assets),
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "priority_distribution": priority_counts,
            "cached_items": len(self._cache),
            "version": self._registry.version,
        }


# Global instance
asset_manager = AssetManager()


# Convenience functions
def get_asset_path(name: str) -> Path | None:
    """Get asset path."""
    return asset_manager.get_asset_path(name)


def get_asset_url(name: str) -> str | None:
    """Get asset URL."""
    return asset_manager.get_asset_url(name)


def inject_css(name: str) -> bool:
    """Inject CSS asset."""
    return asset_manager.inject_css(name)


def register_asset(name: str, path: str, priority: str = "normal") -> bool:
    """Register asset."""
    return asset_manager.register_asset(name, path, priority)
