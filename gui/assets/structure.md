# Asset Organization Structure

## Directory Structure

```txt
gui/assets/
├── README.md              # This documentation
├── structure.md           # Asset organization guide
├── css/                   # Stylesheets and CSS files
│   ├── components/        # Component-specific styles
│   ├── themes/            # Theme-specific overrides
│   └── global/            # Global styles and utilities
├── images/                # Image assets
│   ├── logos/             # Application logos and branding
│   ├── icons/             # UI icons and symbols
│   ├── backgrounds/       # Background images
│   └── samples/           # Sample images for demo/testing
├── fonts/                 # Font files
│   ├── primary/           # Primary application fonts
│   └── fallback/          # Fallback font options
├── js/                    # JavaScript files for custom functionality
│   ├── components/        # Component-specific scripts
│   └── utils/             # Utility functions
└── manifest/              # Asset manifests and metadata
    ├── asset_registry.json    # Central asset registry
    └── optimization_config.json  # Asset optimization settings
```

## Asset Categories

### 1. **CSS Assets**

- **Global styles**: Base styles, reset, utilities
- **Component styles**: Specific component styling
- **Theme overrides**: Theme-specific customizations
- **Responsive layouts**: Mobile and tablet adaptations

### 2. **Image Assets**

- **Logos**: Primary logo, variations, favicon
- **Icons**: Navigation, status, action icons
- **Backgrounds**: Patterns, textures, gradients
- **Sample images**: Demo data for testing

### 3. **Font Assets**

- **Primary fonts**: Main application typography
- **Fallback fonts**: System font alternatives
- **Icon fonts**: Font-based icons if used

### 4. **JavaScript Assets**

- **Component scripts**: Enhanced interactivity
- **Utility functions**: Reusable helper functions
- **Third-party integrations**: External library customizations

## Naming Conventions

### Files

- Use kebab-case: `primary-logo.png`, `nav-styles.css`
- Include size indicators: `logo-small.png`, `logo-large.png`
- Version indicators: `styles-v1.css`, `icon-set-v2.svg`

### Directories

- Use lowercase with hyphens for multi-word names
- Group by function, not file type when logical
- Keep directory names short but descriptive

## Asset Loading Strategy

### 1. **Critical Assets** (Load first)

- Primary CSS
- Application logo
- Essential fonts

### 2. **Progressive Assets** (Load as needed)

- Theme-specific styles
- Non-critical images
- Enhanced functionality scripts

### 3. **Lazy-loaded Assets**

- Sample images
- Background images
- Optional enhancements

## Performance Optimization

### Image Optimization

- Use appropriate formats (WebP > PNG > JPG)
- Multiple sizes for responsive design
- Compression without quality loss

### CSS Optimization

- Minification for production
- Critical CSS inlining
- Unused CSS removal

### Caching Strategy

- Long-term caching for static assets
- Cache busting for updated assets
- Progressive web app considerations

## Integration with Components

### Asset Reference Pattern

```python
from scripts.gui.assets.manager import AssetManager

# Get optimized asset path
logo_path = AssetManager.get_asset('images/logos/primary-logo.png')
css_content = AssetManager.get_css('components/theme-switcher.css')
```

### Dynamic Asset Loading

```python
# Load assets based on current theme
theme_assets = AssetManager.get_theme_assets(current_theme)
AssetManager.inject_css(theme_assets['css'])
```

## Maintenance Guidelines

### Asset Auditing

- Regular cleanup of unused assets
- Size optimization checks
- Accessibility compliance verification

### Documentation Updates

- Keep asset registry updated
- Document new asset categories
- Update optimization configurations

### Version Control

- Avoid large binary files in main repo
- Use Git LFS for large assets when necessary
- Document asset sources and licensing
