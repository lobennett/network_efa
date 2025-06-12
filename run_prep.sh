#!/bin/bash

# Batch submission script for run_prep.py using SLURM array jobs
# Usage: run_prep.sh [email] [input_dirs...] [--additional-args]
# Example: run_prep.sh user@email.com ../discovery_wm/output_lev1_mni ../discovery_wm/output_lev1_mni_validation --atlas-dimension 128

# Configuration
MAX_CONCURRENT_JOBS=10

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 [email] [input_dirs...] [--additional-args]"
    echo "Example: $0 user@email.com ../discovery_wm/output_lev1_mni ../discovery_wm/output_lev1_mni_validation --atlas-dimension 128"
    exit 1
fi

EMAIL="$1"
shift

# Separate input directories from additional arguments
INPUT_DIRS=()
ADDITIONAL_ARGS=()
parsing_dirs=true

for arg in "$@"; do
    if [[ "$arg" == --* ]] && [ "$parsing_dirs" = true ]; then
        parsing_dirs=false
    fi
    
    if [ "$parsing_dirs" = true ]; then
        INPUT_DIRS+=("$arg")
    else
        ADDITIONAL_ARGS+=("$arg")
    fi
done

if [ ${#INPUT_DIRS[@]} -eq 0 ]; then
    echo "Error: No input directories provided"
    exit 1
fi

echo "Email: $EMAIL"
echo "Input directories: ${INPUT_DIRS[*]}"
echo "Additional arguments: ${ADDITIONAL_ARGS[*]}"

# Create log directory
mkdir -p "${SCRIPT_DIR}/log"

# Check if apptainer image exists
APPTAINER_IMAGE="$SCRIPT_DIR/apptainer/fmri_env_latest.sif"
if [ ! -f "$APPTAINER_IMAGE" ]; then
    echo "Error: Apptainer image not found: $APPTAINER_IMAGE"
    echo "Please run the pull_image.sh script first to download the required image:"
    echo "  $SCRIPT_DIR/pull_image.sh"
    exit 1
fi

echo "Using apptainer image: $APPTAINER_IMAGE"

# Function to discover subjects from input directories
discover_subjects() {
    local subjects=()
    
    for input_dir in "${INPUT_DIRS[@]}"; do
        if [ ! -d "$input_dir" ]; then
            echo "Warning: Input directory does not exist: $input_dir" >&2
            continue
        fi
        
        echo "Discovering subjects in: $input_dir" >&2
        
        # Find all sub-* directories and extract subject IDs
        for subject_dir in "$input_dir"/sub-*; do
            if [ -d "$subject_dir" ]; then
                subject_id=$(basename "$subject_dir" | sed 's/^sub-//')
                subjects+=("$subject_id")
            fi
        done
    done
    
    # Remove duplicates and sort, output only to stdout
    printf '%s\n' "${subjects[@]}" | sort -u
}

# Discover subjects and create subject list file
echo "Discovering subjects from input directories..."
mapfile -t SUBJECTS < <(discover_subjects)

if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "Error: No subjects found in input directories"
    exit 1
fi

echo "Found ${#SUBJECTS[@]} subjects: ${SUBJECTS[*]}"

# Create temporary subject list file
SUBJECT_LIST_FILE="${SCRIPT_DIR}/log/subjects_$(date +%Y%m%d_%H%M%S).txt"
printf '%s\n' "${SUBJECTS[@]}" > "$SUBJECT_LIST_FILE"
echo "Created subject list file: $SUBJECT_LIST_FILE"

# Calculate array indices (0-based)
LAST_INDEX=$((${#SUBJECTS[@]} - 1))

echo "Submitting SLURM array job with ${#SUBJECTS[@]} subjects (indices 0-$LAST_INDEX)"
echo "Maximum $MAX_CONCURRENT_JOBS concurrent jobs"

# Build input directories argument string
INPUT_DIRS_STR="${INPUT_DIRS[*]}"

# Submit SLURM array job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=run_prep_array
#SBATCH --array=0-${LAST_INDEX}%${MAX_CONCURRENT_JOBS}
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=russpold,hns,normal
#SBATCH --output=${SCRIPT_DIR}/log/run_prep_%A_%a.out
#SBATCH --error=${SCRIPT_DIR}/log/run_prep_%A_%a.err
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=END

# Get subject ID from the list file based on array task ID
SUBJECT_ID=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECT_LIST_FILE")

echo "Starting data preparation for subject: \$SUBJECT_ID"
echo "Job ID: \$SLURM_JOB_ID"
echo "Array Task ID: \$SLURM_ARRAY_TASK_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Working directory: \$PWD"

# Change to script directory
cd "$SCRIPT_DIR"

# Run prep analysis using apptainer for single subject
apptainer exec "$APPTAINER_IMAGE" python3 run_prep.py $INPUT_DIRS_STR --subject-id \$SUBJECT_ID ${ADDITIONAL_ARGS[*]}

echo "Completed data preparation for subject: \$SUBJECT_ID"
EOF

echo
echo "Array job submitted successfully!"
echo "Monitor job progress with: squeue -u $USER -n 'run_prep_array'"
echo "Check logs in: $SCRIPT_DIR/log/"
echo "Subject list file: $SUBJECT_LIST_FILE"