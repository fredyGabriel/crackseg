# CrackSeg GUI Asset Management System

This directory contains the comprehensive asset management system for the CrackSeg Streamlit application.

## 🎯 Overview

The asset management system provides:

- **Centralized asset registry** with metadata tracking
- **Optimized loading** with caching and compression
- **Theme-aware assets** that adapt to current theme
- **Professional organization** with clear structure
- **Performance monitoring** and optimization

## 📁 Directory Structure

```txt
scripts/gui/assets/
├── README.md              # This documentation
├── structure.md           # Detailed structure guide
├── manager.py             # Asset management system
├── init_assets.py         # Initialization script
├── css/                   # Stylesheets
│   ├── global/            # Global application styles
│   ├── components/        # Component-specific styles
│   └── themes/            # Theme overrides
├── images/                # Image assets
│   ├── logos/             # Application branding
│   ├── icons/             # UI icons and symbols
│   ├── backgrounds/       # Background patterns
│   └── samples/           # Demo/test images
├── fonts/                 # Typography assets
│   ├── primary/           # Main fonts
│   └── fallback/          # System fallbacks
├── js/                    # JavaScript enhancements
│   ├── components/        # Component scripts
│   └── utils/             # Utility functions
└── manifest/              # Asset metadata
    ├── asset_registry.json    # Central registry
    └── optimization_config.json  # Optimization settings
```

## 🚀 Quick Start

### Initialize the Asset System

```bash
python scripts/gui/assets/init_assets.py
```

### Use in Components

```python
from scripts.gui.assets.manager import asset_manager

# Inject CSS
asset_manager.inject_css("base_css")
asset_manager.inject_css("navigation_css")

# Get asset URLs
logo_url = asset_manager.get_asset_url("primary_logo")
icon_url = asset_manager.get_asset_url("nav_icon")

# Register new assets
asset_manager.register_asset("custom_css", "css/components/custom.css", "high")
```

## 🎨 Theme Integration

Assets automatically adapt to the current theme:

```python
# Get theme-specific assets
theme_assets = asset_manager.get_theme_assets("dark")
for css_file in theme_assets["css"]:
    asset_manager.inject_css(css_file)
```

## 📊 Asset Categories

### 1. **CSS Assets**

- **Global**: `css/global/base.css` - Core styles and CSS variables
- **Components**: `css/components/navigation.css` - Component-specific styling
- **Themes**: Theme-specific overrides and customizations

### 2. **Image Assets**

- **Logos**: Primary branding with fallback generation
- **Icons**: UI elements and status indicators
- **Backgrounds**: Patterns and textures
- **Samples**: Demo images for testing

### 3. **Performance Features**

- **Caching**: In-memory caching for repeated access
- **Compression**: Optimized file sizes
- **Lazy loading**: Assets loaded as needed
- **CDN-ready**: Base64 encoding for Streamlit

## 🔧 Asset Registry

The registry tracks all assets with metadata:

```json
{
  "assets": {
    "base_css": {
      "path": "css/global/base.css",
      "size": 15234,
      "hash": "a1b2c3...",
      "mime_type": "text/css",
      "load_priority": "critical"
    }
  }
}
```

## 📈 Performance Monitoring

Check asset statistics:

```python
stats = asset_manager.get_registry_stats()
print(f"Total assets: {stats['total_assets']}")
print(f"Total size: {stats['total_size_mb']} MB")
print(f"Cached items: {stats['cached_items']}")
```

## 🛠️ Development Workflow

1. **Add Assets**: Place files in appropriate directories
2. **Register**: Use `register_asset()` or run init script
3. **Test**: Verify loading in application
4. **Optimize**: Run optimization for production

## 🔍 Troubleshooting

### Common Issues

- **Asset not found**: Check path and registry
- **Caching issues**: Use `asset_manager.cleanup_cache()`
- **Theme problems**: Verify theme-specific assets exist

### Debug Mode

```python
# Enable detailed logging
import logging
logging.getLogger('asset_manager').setLevel(logging.DEBUG)
```

## 📚 Documentation

- **`structure.md`**: Detailed structure and naming conventions
- **`css/` READMEs**: CSS organization and best practices
- **`images/` READMEs**: Image asset guidelines
- **Component docs**: Integration examples

## 🔄 Migration from Legacy

The asset system maintains compatibility with existing components while providing enhanced
functionality through the centralized manager.

## ⚡ Performance Tips

1. **Use critical priority** for essential assets
2. **Enable caching** for repeated access
3. **Optimize images** before adding to registry
4. **Monitor asset budgets** using performance metrics

## 🎨 Logo System (Legacy Compatible)

The application maintains the sophisticated `LogoComponent` system:

### Fallback Chain

1. **Asset Manager**: Centralized primary logo
2. **Primary Path**: User-specified logo path
3. **Default Locations**: docs/designs/, assets/, scripts/gui/assets/
4. **Auto-generation**: Programmatic logo creation

### Generated Logo Features

- Asphalt surface with realistic crack patterns
- Semi-transparent segmentation overlays
- Professional typography with shadows
- Multiple style variants (default, light, minimal)
- PNG optimization and caching

### Integration Example

```python
from scripts.gui.components.logo_component import LogoComponent

# Uses Asset Manager first, then fallbacks
LogoComponent.render(
    style="default",
    width=150,
    alt_text="CrackSeg Logo"
)
```
