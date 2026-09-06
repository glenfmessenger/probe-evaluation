#!/bin/bash
# Fix numpy/pandas/sklearn binary incompatibility

echo "Fixing numpy/pandas binary incompatibility..."

# Option 1: Upgrade numpy (usually works)
pip install --upgrade numpy

# Option 2: If that doesn't work, pin specific versions
# pip install numpy==1.26.4 pandas==2.0.3 scikit-learn==1.3.2

# Option 3: Reinstall pandas to match numpy
# pip install --force-reinstall pandas

# Verify
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

echo "Now try running the test again"
