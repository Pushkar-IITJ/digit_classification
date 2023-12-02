#!/bin/bash
# Run the plot_digit_classification.py
echo "Running plot_digit_classification.py"
python plot_digits_classification.py

# Run the test_models.py script
echo "Running test_models.py"
python -m unittest test_models.py
