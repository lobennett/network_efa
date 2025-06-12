#!/bin/bash

# Usage: run_efa.sh [email] [atlas_dimension] [additional args for run_efa.py]
# Example: run_efa.sh user@email.com 1024 --n-factors 7

# Get script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
EMAIL="${1:-logben@stanford.edu}"
ATLAS_DIMENSION="${2:-1024}"
shift 2  # Remove email and atlas_dimension from args, pass rest to run_efa.py

# Create log directory
mkdir -p "${SCRIPT_DIR}/log"

# Create results directory structure
mkdir -p "${SCRIPT_DIR}/results"

# Check if virtual environment exists
VENV_PATH="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Warning: Virtual environment not found at $VENV_PATH"
    echo "Make sure your Python environment is set up correctly."
fi

echo "Using Python environment with module loads and virtual environment"
echo "Looking for subject files with atlas dimension: $ATLAS_DIMENSION"

# Find all subject-specific TSV files 
DATA_DIR="$SCRIPT_DIR/data"
SUBJECT_FILES=($(find "$DATA_DIR" -name "parcellated_difumo_${ATLAS_DIMENSION}_s*.tsv" | sort -V))

if [ ${#SUBJECT_FILES[@]} -eq 0 ]; then
    echo "Error: No subject files found matching pattern: parcellated_difumo_${ATLAS_DIMENSION}_s*.tsv"
    echo "Please run the prep script first to generate subject-specific data files."
    echo "Expected files in: $DATA_DIR"
    echo "Available files:"
    ls -la "$DATA_DIR"/ | head -10
    exit 1
fi

echo "Found ${#SUBJECT_FILES[@]} subject files:"
for file in "${SUBJECT_FILES[@]}"; do
    echo "  $(basename "$file")"
done

echo ""
echo "Submitting ${#SUBJECT_FILES[@]} EFA jobs to SLURM..."

# Submit separate EFA job for each subject
for subject_file in "${SUBJECT_FILES[@]}"; do
    # Extract subject ID from filename (e.g., parcellated_difumo_1024_s03.tsv -> s03)
    filename=$(basename "$subject_file")
    subject_id=$(echo "$filename" | sed -n "s/parcellated_difumo_${ATLAS_DIMENSION}_\(s[0-9]\+\)\.tsv/\1/p")
    
    if [ -z "$subject_id" ]; then
        echo "Warning: Could not extract subject ID from $filename, skipping..."
        continue
    fi
    
    # Create output directory for this subject
    output_dir="$SCRIPT_DIR/results/efa_${subject_id}"
    
    echo "Submitting job for subject: $subject_id"
    echo "  Input file: $subject_file"
    echo "  Output dir: $output_dir"
    
    # Submit SLURM job for this subject
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=efa_${subject_id}
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=russpold,hns,normal
#SBATCH --output=${SCRIPT_DIR}/log/efa_${subject_id}-%j.out
#SBATCH --error=${SCRIPT_DIR}/log/efa_${subject_id}-%j.err
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=END

echo "Starting EFA analysis for subject: $subject_id"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Working directory: \$PWD"
echo "Input file: $subject_file"
echo "Output directory: $output_dir"
echo "Atlas dimension: $ATLAS_DIMENSION"

# Change to script directory
cd "$SCRIPT_DIR"

# Create output directory
mkdir -p "$output_dir"

# Load required modules and activate virtual environment
echo "Loading required modules and activating virtual environment..."
module load py-rpy2/2.9.2_py36
module load python/3.6.1
source .venv/bin/activate

# Run EFA analysis
echo "Running EFA analysis with the following command:"
echo "python3 run_efa.py --data-path $subject_file --output-dir $output_dir --atlas-dimension $ATLAS_DIMENSION --subject-id $subject_id $@"

python3 run_efa.py \\
    --data-path "$subject_file" \\
    --output-dir "$output_dir" \\
    --atlas-dimension $ATLAS_DIMENSION \\
    --subject-id "$subject_id" \\
    $@

echo "Completed EFA analysis for subject: $subject_id"
echo "Results saved in: $output_dir"
EOF

done

echo ""
echo "Submitted ${#SUBJECT_FILES[@]} EFA jobs successfully!"
echo "Monitor job progress with: squeue -u \$USER"
echo "Check individual job logs in: $SCRIPT_DIR/log/efa_sXX-JOBID.{out,err}"
echo "Results will be saved in: $SCRIPT_DIR/results/efa_sXX/"