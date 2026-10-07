#!/bin/bash

set -e

PROJECT_DIR="$HOME/user-engagement-platform"

cd "$PROJECT_DIR"

echo "======================================"
echo "USER ENGAGEMENT RETRAINING PIPELINE"
echo "======================================"

echo ""
echo "[1/3] Training TensorFlow model..."

docker run --rm --platform=linux/amd64 \
  -v "$PWD/src:/app/src" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  user-engagement-tfx \
  python src/models/train_tensorflow.py


echo ""
echo "[2/3] Evaluating candidate model..."

docker run --rm --platform=linux/amd64 \
  -v "$PWD/src:/app/src" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/artifacts:/app/artifacts" \
  user-engagement-tfx \
  python src/evaluation/evaluate_model.py


echo ""
echo "Checking validation gate..."

PASSED=$(python3 -c "
import json

with open('artifacts/evaluation/evaluation_results.json') as f:
    result = json.load(f)

print(str(result['validation_passed']).lower())
")


if [ "$PASSED" != "true" ]; then
    echo ""
    echo "======================================"
    echo "MODEL VALIDATION FAILED"
    echo "======================================"
    echo ""
    echo "Candidate model did not meet required thresholds."
    echo "Existing serving model will remain unchanged."
    exit 1
fi


echo ""
echo "Validation passed."


echo ""
echo "[3/3] Exporting model for serving..."

docker run --rm --platform=linux/amd64 \
  -v "$PWD/src:/app/src" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/serving:/app/serving" \
  user-engagement-tfx \
  python src/serving/export_model.py


echo ""
echo "======================================"
echo "RETRAINING COMPLETE"
echo "======================================"
echo ""
echo "Candidate model passed validation and was exported."