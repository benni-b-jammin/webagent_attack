#!/bin/bash
set -e
set -x

# Move to repo root (script is stored in src/scripts/)
cd "$(dirname "$0")/../.."

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_file="./results/generate_triggers_log_$timestamp.txt"

# Log everything to both terminal and file
exec > >(tee -a "$log_file") 2>&1

echo "Running dataset capture..."
python3 -m src.attacks.capture_dataset

echo "Running trigger generation..."

for file in src/config/narrow_triggers/*.yaml src/config/narrow_triggers/*.yml src/config/narrow_triggers/*.json; do
    if [ -f "$file" ] && [[ "$file" != *default* ]]; then
        echo "Using config: $file"

        python3 -m src.attacks.make_trigger --algo gcg --dataset "src/data/datasets/demo_dataset" --config "$file"
        python3 -m src.attacks.make_trigger --algo gcg --dataset "src/data/datasets/demo_dataset" --config "$file"
        python3 -m src.attacks.make_trigger --algo gcg --dataset "src/data/datasets/demo_dataset" --config "$file"
    fi
done

echo "Done."