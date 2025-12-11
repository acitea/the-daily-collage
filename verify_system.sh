#!/bin/bash
echo "═══════════════════════════════════════════════════════════"
echo "The Daily Collage - System Verification"
echo "═══════════════════════════════════════════════════════════"
echo ""

VENV_PATH="backend/server/.venv/bin/activate"

if [ ! -f "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found"
    exit 1
fi

source "$VENV_PATH"

echo "1️⃣ Checking imports..."
python -c "from visualization.composition import TemplateComposer; print('   ✓ Visualization module')" 2>/dev/null || echo "   ❌ Visualization module"
python -c "from utils.classification import classify_articles; print('   ✓ Classification module')" 2>/dev/null || echo "   ❌ Classification module"
python -c "from ingestion.script import get_news_for_location; print('   ✓ Ingestion module')" 2>/dev/null || echo "   ❌ Ingestion module"

echo ""
echo "2️⃣ Testing image generation..."
cd backend
python -c "
from visualization.composition import TemplateComposer, SignalIntensity
composer = TemplateComposer()
signals = [SignalIntensity('traffic', 50), SignalIntensity('weather', 75)]
img = composer.compose(signals, 'Test')
print(f'   ✓ Generated {len(img)} bytes')
" 2>/dev/null || echo "   ❌ Image generation failed"

echo ""
echo "3️⃣ Testing classification..."
cd ..
python -c "
import sys
sys.path.insert(0, 'backend')
import polars as pl
from utils.classification import classify_articles
df = pl.DataFrame({'title': ['Traffic jam'], 'url': ['test'], 'source': ['test'], 'date': ['2025-12-11'], 'tone': [0.5]})
result = classify_articles(df)
print(f'   ✓ Classified {len(result)} articles')
" 2>/dev/null || echo "   ❌ Classification failed"

echo ""
echo "4️⃣ Checking dependencies..."
python -c "import fastapi, uvicorn, polars, gdeltdoc, PIL; print('   ✓ All dependencies installed')" 2>/dev/null || echo "   ❌ Missing dependencies"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Verification complete!"
echo "═══════════════════════════════════════════════════════════"
