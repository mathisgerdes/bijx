#!/bin/bash

# Script to regenerate favicon files from SVG sources
# Run this whenever you update bijx.svg or bijx-favicon.svg

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICONS_DIR="$SCRIPT_DIR/source/_static/icons"

echo "🎨 Updating bijx icons..."

# Check if ImageMagick is available
if ! command -v magick &> /dev/null; then
    echo "❌ Error: ImageMagick is required but not installed."
    echo "   Install with: brew install imagemagick (macOS) or apt-get install imagemagick (Ubuntu)"
    exit 1
fi

# Check if source SVG files exist
if [[ ! -f "$ICONS_DIR/bijx-favicon.svg" ]]; then
    echo "❌ Error: bijx-favicon.svg not found in $ICONS_DIR"
    exit 1
fi

if [[ ! -f "$ICONS_DIR/bijx.svg" ]]; then
    echo "❌ Error: bijx.svg not found in $ICONS_DIR"
    exit 1
fi

cd "$ICONS_DIR"

echo "📱 Generating favicon PNG files..."

# Generate favicon PNG files at different sizes with sharp rendering
# Use higher sampling density and better resampling for small sizes
magick bijx-favicon.svg -density 384 -resize 16x16 -filter Triangle -define filter:support=2 -unsharp 0.25x0.25+8+0.065 -dither None -posterize 136 -quality 95 -define png:compression-filter=5 -define png:compression-level=9 -define png:compression-strategy=1 -define png:exclude-chunk=all -interlace none -colorspace sRGB favicon-16x16.png

magick bijx-favicon.svg -density 384 -resize 32x32 -filter Triangle -define filter:support=2 -unsharp 0.25x0.25+8+0.065 -dither None -posterize 136 -quality 95 -define png:compression-filter=5 -define png:compression-level=9 -define png:compression-strategy=1 -define png:exclude-chunk=all -interlace none -colorspace sRGB favicon-32x32.png

magick bijx-favicon.svg -density 288 -resize 48x48 -filter Triangle -define filter:support=2 -quality 95 -define png:compression-filter=5 -define png:compression-level=9 -define png:compression-strategy=1 -define png:exclude-chunk=all -interlace none -colorspace sRGB favicon-48x48.png

magick bijx-favicon.svg -resize 192x192 -background none -quality 95 -define png:compression-filter=5 -define png:compression-level=9 -define png:compression-strategy=1 -define png:exclude-chunk=all -interlace none -colorspace sRGB favicon-192x192.png

echo "🏷️  Creating favicon.ico..."

# Create ICO file with multiple sizes embedded using the sharp PNG files
magick favicon-16x16.png favicon-32x32.png favicon-48x48.png favicon.ico

echo "📋 Updating web manifest..."

# Update the web manifest with current theme colors
# Extract the main color from the SVG (you can adjust these manually if needed)
cat > ../site.webmanifest << EOF
{
    "name": "bijx",
    "short_name": "bijx",
    "icons": [
        {
            "src": "icons/favicon-16x16.png",
            "sizes": "16x16",
            "type": "image/png"
        },
        {
            "src": "icons/favicon-32x32.png",
            "sizes": "32x32",
            "type": "image/png"
        },
        {
            "src": "icons/favicon-192x192.png",
            "sizes": "192x192",
            "type": "image/png"
        }
    ],
    "theme_color": "#112136",
    "background_color": "#ffffff",
    "display": "standalone"
}
EOF

echo "✅ Icon update complete!"
echo ""
echo "Generated files:"
echo "  - favicon-16x16.png"
echo "  - favicon-32x32.png"
echo "  - favicon-48x48.png"
echo "  - favicon-192x192.png"
echo "  - favicon.ico"
echo "  - site.webmanifest"
echo ""
echo "💡 Tip: Run 'make livehtml' to see the updated icons in your documentation."
