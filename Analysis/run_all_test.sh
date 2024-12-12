#!/bin/bash

# Define the list of fold numbers
FOLDS=(1 2 3 4 5)

# Define the list of thresholds
THRESHOLDS=(0.45 0.5 0.55 0.6 0.65)

# Loop through each fold and threshold
for FOLD in "${FOLDS[@]}"
do
    for THRESHOLD in "${THRESHOLDS[@]}"
    do
        echo "Running script with fold number $FOLD and threshold $THRESHOLD"
        python test_nonOverlap.py --fold_num $FOLD --prediction_threshold $THRESHOLD
    done
done