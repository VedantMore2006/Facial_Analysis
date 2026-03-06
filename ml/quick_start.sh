#!/bin/bash
# Quick Start Script - Mental Health ML Pipeline
# Run this to train your first model!

echo "======================================================================"
echo "                   MENTAL HEALTH ML PIPELINE                         "
echo "                        QUICK START                                   "
echo "======================================================================"
echo ""
echo "This will:"
echo "  ✓ Process your baseline data (7,101 frames)"
echo "  ✓ Generate 12,000 synthetic training samples"
echo "  ✓ Train XGBoost classifier"
echo "  ✓ Create evaluation reports & visualizations"
echo ""
echo "Time: ~2-5 minutes"
echo "======================================================================"
echo ""

# Check if conda environment is active
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  Warning: No conda environment active"
    echo "   Run: conda activate face_env"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import xgboost, sklearn, matplotlib, seaborn" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "✗ Missing dependencies!"
    echo "  Install: pip install -r ml/requirements.txt"
    exit 1
fi

echo "✓ All dependencies installed"
echo ""

# Run pipeline
echo "Starting ML pipeline..."
echo ""

cd ml/
python run_full_pipeline.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "                         SUCCESS! 🎉                                 "
    echo "======================================================================"
    echo ""
    echo "📊 Check your results:"
    echo "   ml/evaluation/         - Plots & performance metrics"
    echo "   ml/training_dataset.csv - 12,000 training samples"
    echo "   ml/mental_health_model.pkl - Trained classifier"
    echo ""
    echo "🎯 Optional Next Steps:"
    echo "   1. Review evaluation plots"
    echo "   2. Record acted sessions (2 min each):"
    echo "      python ml/record_acted_session.py depression"
    echo "   3. Extract real patterns & retrain"
    echo ""
else
    echo ""
    echo "✗ Pipeline failed. Check errors above."
    exit 1
fi
