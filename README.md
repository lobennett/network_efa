# Network EFA Analysis

An exploratory factor analysis (EFA) pipeline for neuroimaging data using Horn's parallel analysis. This package performs EFA on parcellated brain data to identify latent factor structures in task contrasts across brain regions.

**Important**: This pipeline requires a two-step workflow:
1. **Data Preparation** (`run_prep.sh`) - Parcellates raw fMRI data into TSV files
2. **EFA Analysis** (`run_efa.sh`) - Performs factor analysis on the prepared data

## Features

- Object-oriented design 
- Horn's parallel analysis to determine number of latent factors 
- Data visualization (correlation matrices, factor loadings, bartlett transforms)
- Support for multiple rotation methods (promax, varimax, oblimin)
- SLURM cluster integration with apptainer containers
- Detailed logging and error handling

## Prerequisites

- SLURM cluster environment
- Apptainer/Singularity container runtime
- Parcellated neuroimaging data in TSV format

## Installation

1. Clone this repository:
```bash
git clone https://github.com/lobennett/network_efa.git
cd network_efa
```

2. **Pull the required apptainer image** (this step is required before running the pipeline):

   **Option A - Direct execution:**
   ```bash
   ./pull_image.sh
   ```
   
   **Option B - SLURM batch submission:**
   ```bash
   sbatch pull_image.sh
   ```
   
   This will download the fMRI processing environment container to `./apptainer/fmri_env_latest.sif`.

3. Ensure scripts are executable:
```bash
chmod +x run_efa.sh run_prep.sh pull_image.sh
```

## Usage

This pipeline consists of two main steps that must be run in order:

### Step 1: Data Preparation (Required First)

Generate parcellated data files from raw neuroimaging data:

```bash
# Prepare EFA datafiles - MUST BE RUN FIRST
./run_prep.sh [email] [input_dirs...] [--additional-args]

# Example:
./run_prep.sh user@stanford.edu ../discovery_wm/output_lev1_mni --atlas-dimension 1024
```

This step:
- Parcellates fMRI contrast images using functional atlases (DiFuMo)
- Creates TSV files in the `data/` directory (e.g., `parcellated_difumo_1024_s03.tsv`)
- Must be completed before running EFA analysis

### Step 2: EFA Analysis

Run exploratory factor analysis on the prepared data:

```bash
# Run EFA analysis - after run_prep.sh completes
# By default, processes ALL parcellated TSV files automatically
./run_efa.sh [email] [atlas_dimension] [--additional-args]

# Example - processes all parcellated_difumo_1024_s*.tsv files:
./run_efa.sh user@stanford.edu 1024

# Example with custom parameters:
./run_efa.sh user@stanford.edu 1024 --n-iterations 500 --rotation varimax
```

**Note**: `run_efa.sh` automatically discovers and processes ALL subject files matching the pattern `parcellated_difumo_[atlas_dimension]_s*.tsv` in the `data/` directory, creating separate EFA analyses for each subject in `results/efa_s*/`.

### Data Preparation Arguments

The data preparation script (`run_prep.sh`) supports:

- `[email]`: Email address for SLURM job notifications
- `[input_dirs...]`: One or more directories containing subject fMRI data
- `--atlas-dimension`: Atlas dimension (64, 128, 256, 512, 1024) - default: 1024
- `--subject-id`: Process specific subject (used internally by SLURM array jobs)

### EFA Analysis Arguments

The EFA wrapper script (`run_efa.sh`) supports:

- `[email]`: Email address for SLURM job notifications
- `[atlas_dimension]`: Atlas dimension to process (64, 128, 256, 512, 1024) - default: 1024
- Additional arguments passed to `run_efa.py`:
  - `--n-iterations`: Number of iterations for parallel analysis (default: 1000)
  - `--rotation`: Rotation method - promax, varimax, oblimin, none (default: promax)
  - `--force-factors`: Force specific number of factors (overrides parallel analysis)

### Examples

```bash
# Basic usage - processes all parcellated_difumo_1024_s*.tsv files
./run_efa.sh user@stanford.edu 1024

# Process different atlas dimension
./run_efa.sh user@stanford.edu 64

# Custom parameters for all subjects
./run_efa.sh user@stanford.edu 1024 --n-iterations 500 --rotation varimax

# Force specific number of factors for all subjects
./run_efa.sh user@stanford.edu 1024 --force-factors 5

# Run single subject locally (without SLURM submission)
python run_efa.py --data-path data/parcellated_difumo_1024_s03.tsv --output-dir results/efa_s03 --subject-id s03
```

## Input Data Format

The input data should be a tab-separated values (TSV) file with:
- **Rows**: Brain parcels/regions (observations in factor analysis)
- **Columns**: Task contrasts (variables in factor analysis)
- **First column**: Parcel/region names (used as index)
- **First row**: Task contrast names (column headers)

Example structure:
```
	contrast1	contrast2	contrast3	...
region1	value1	value2	value3	...
region2	value4	value5	value6	...
...
```

The data is used as-is (no transposition), with parcels as observations and contrasts as variables. This identifies factors that represent patterns of task contrasts within brain regions.

## Output Structure

The analysis generates the following outputs in the specified output directory:

```
efa_results/
├── correlation_matrix.png             # Correlation matrix heatmap (contrasts × contrasts)
├── parallel_analysis.png              # Horn's parallel analysis scree plot
├── efa_loadings_Nfactors.tsv          # Factor loadings matrix (contrasts × factors)
├── efa_loadings_Nfactors.png          # Factor loadings heatmap
├── factor_correlations_Nfactors.tsv   # Factor correlation matrix (oblique rotations)
└── factor_correlations_Nfactors.png   # Factor correlation plot (oblique rotations)
```

Where `N` is the number of factors determined by parallel analysis or forced by flag.

## Class Structure

The pipeline uses an object-oriented design with four main classes:

### `EFAProcessor`
Main orchestrator class that coordinates the entire workflow:
- Data loading and preprocessing
- Output directory management
- Workflow coordination

### `ParallelAnalysis`
Handles Horn's parallel analysis:
- Generates random datasets for comparison
- Calculates eigenvalues for factor determination
- Suggests optimal number of factors

### `PlotGenerator`
Manages all visualization functions:
- Correlation matrix heatmaps
- Parallel analysis scree plots
- Factor loadings heatmaps
- Factor correlation matrices

### `FactorAnalysisModel`
Handles EFA model fitting and validation:
- Data adequacy testing (KMO, Bartlett's)
- Factor analysis model fitting
- Variance explained calculations
- Model fit statistics

## SLURM Configuration

The script submits jobs with the following default settings:
- **Time limit**: 2 days 
- **CPUs per task**: 4
- **Memory**: 16GB
- **Partitions**: russpold, hns, normal
- **Single job**: (not array-based since EFA processes entire dataset)

## Logs

Job logs are saved in the `log/` directory:
- `log/run_efa-{job_id}.out`
- `log/run_efa-{job_id}.err`

## Requirements

### Python Dependencies

All dependencies are managed within the apptainer container, including:
- pandas
- numpy
- matplotlib
- seaborn
- factor-analyzer
- scikit-learn
- All other required packages

### Data Requirements

- Parcellated neuroimaging data in TSV format
- Sufficient observations (task contrasts) for factor analysis
- Numeric data with no missing values (automatically imputed with column means)
- Non-zero variance across all variables

## Methodology

### Horn's Parallel Analysis
The pipeline uses Horn's parallel analysis to determine the optimal number of factors:
1. Calculates eigenvalues from real data
2. Generates random datasets with same dimensions
3. Compares real vs. random eigenvalues
4. Suggests factors where real eigenvalues exceed random eigenvalues

### Factor Analysis
- Uses minimum residual (MinRes) extraction method
- Supports multiple rotation methods (promax, varimax, oblimin)
- Validates data adequacy with KMO and Bartlett's tests
- Reports variance explained and model fit statistics

## Troubleshooting

### Checking Job Status

```bash
# Check job status
squeue -u $USER

# View job output
cat log/run_efa-{job_id}.out

# View job errors
cat log/run_efa-{job_id}.err
```

### Advanced Customization

To modify analysis parameters, you can:

1. **Edit default parameters** in `run_efa.py`
2. **Modify SLURM settings** in `run_efa.sh`
3. **Adjust class behavior** by modifying the respective class methods

## Example Workflow

```bash
# 1. Ensure container is available
./pull_image.sh

# 2. STEP 1: Prepare parcellated data (REQUIRED FIRST)
./run_prep.sh user@stanford.edu /path/to/fmri/data --atlas-dimension 1024

# 3. Wait for data preparation to complete, then check generated data files
ls data/parcellated_difumo_*

# 4. STEP 2: Run EFA analysis on ALL prepared data files
./run_efa.sh user@stanford.edu 1024

# 5. Check EFA results for all subjects
ls results/
# Shows: efa_s03/ efa_s10/ efa_s19/ etc.

# 6. Run EFA with custom parameters on all subjects
./run_efa.sh user@stanford.edu 1024 --n-iterations 500 --rotation varimax
```

## TODO

- [ ] Containerize the EFA environment

## License

MIT License