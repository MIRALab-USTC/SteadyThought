#!/usr/bin/env bash
set -e

script_dir="$(cd "$(dirname "$0")" && pwd)"

bash "$script_dir/01_identify_thoughts.sh"
bash "$script_dir/02_synthesize_trajectories.sh"
bash "$script_dir/03_build_stpo_dataset.sh"
