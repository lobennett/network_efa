#!/usr/bin/env python
"""
Exploratory Factor Analysis script for network analysis.

This script performs EFA on parcellated fMRI contrast data,
generating factor loadings, correlations, and brain visualizations.
Structured using object-oriented design for better maintainability.
"""

import argparse
import logging
import sys
from pathlib import Path
import re
from typing import List, Dict, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from nilearn.datasets import load_mni152_template
from nilearn.datasets import fetch_atlas_difumo
from nilearn.maskers import NiftiMapsMasker
from nilearn import plotting

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def plot_fa_parallel(out, title_name, out_img_path):
    """
    Plots the results from the fa.parallel function from the psych package.

    Parameters:
        out (list): The output from the fa.parallel function containing the eigenvalues
                    and simulated values for factor analysis and principal components.

    Returns:
        None
    """

    # FA data eigen + sim eigen
    plt.plot(
        range(1, len(out[0]) + 1),
        out[0],
        label="FA EVs",
        color="black",
        marker="o",
        linestyle="-",
        markersize=3,
    )
    plt.plot(
        range(1, len(out[0]) + 1),
        out[4],
        label="FA Sim EVs",
        color="black",
        marker="x",
        linestyle="--",
        markersize=3,
    )

    # PC data eigen + sim eigen
    plt.plot(
        range(1, len(out[0]) + 1),
        out[1],
        label="PC EVs",
        color="red",
        marker="o",
        linestyle="-",
        markersize=3,
    )
    plt.plot(
        range(1, len(out[0]) + 1),
        out[2],
        label="PC Sim EVs",
        color="red",
        marker="x",
        linestyle="--",
        markersize=3,
    )

    # factors/components
    recommended_factors = int(list(out.rx2("nfact"))[0])
    recommended_components = int(list(out.rx2("ncomp"))[0])

    plt.axvline(
        x=recommended_factors,
        color="blue",
        linestyle=":",
        label=f"Rec Facts: ({recommended_factors})",
    )
    plt.axvline(
        x=recommended_components,
        color="green",
        linestyle=":",
        label=f"Rec Comp: ({recommended_components})",
    )

    plt.xlabel("")
    plt.ylabel("Eigenvalue")
    plt.title(f"{title_name}: Parallel Analysis Eigenvalues")
    plt.legend(loc="best")
    plt.grid(True)
    plt.figtext(
        0.95,
        0.02,
        "Note: PC EVs is used in Scree plots eigen > 1",
        ha="right",
        fontsize=8,
        color="black",
    )
    plt.savefig(out_img_path, bbox_inches="tight")
    plt.show()


def plot_loading_phi(
    fa_object,
    title_name,
    out_img_path,
    out_type="phi",
    phi_plot_diag=True,
    row_labs=None,
):
    # Get number of factors
    n_factors = np.array(fa_object.rx2("loadings")).shape[-1]

    if out_type == "phi":
        # Convert R object to a NumPy array
        phi_matrix = np.array(fa_object.rx2("Phi"))

        # Print shape to see if we have a square matrix
        print("Phi matrix shape:", phi_matrix.shape)

        n_factors = phi_matrix.shape[0]  # Assuming it's n_factors x n_factors
        plt.figure(figsize=(10, 10))  # Adjust as needed

        sns.heatmap(
            phi_matrix,
            annot=True,
            cmap="coolwarm",
            mask=np.triu(
                np.ones_like(phi_matrix, dtype=bool), k=1 if phi_plot_diag else 0
            ),
            vmin=-1,
            vmax=1,
            fmt=".2f",
            linewidths=0.25,
            xticklabels=[f"{i + 1}" for i in range(n_factors)],
            yticklabels=[f"{i + 1}" for i in range(n_factors)],
        )

        plt.title(f"{title_name}: Phi Matrix", fontsize=12)
        plt.xlabel("Factors", fontsize=10)
        plt.ylabel("Factors", fontsize=10)
        plt.tight_layout()

    elif out_type == "loadings":
        # Convert and filter loadings as before
        factor_loadings = np.array(fa_object.rx2("loadings"))
        factor_loadings[factor_loadings < 0.30] = np.nan
        no_id_labels = [label.replace(f"{title_name}_", "") for label in row_labs]
        df_loadings = pd.DataFrame(
            factor_loadings,
            columns=[f"Factor {i + 1}" for i in range(factor_loadings.shape[1])],
            index=no_id_labels,
        )
        plt.figure(figsize=(12, 20))
        sns.heatmap(
            df_loadings,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            fmt=".2f",
            linewidths=0.25,
        )
        plt.title(f"{title_name}: EFA Loadings", fontsize=12)
        plt.xlabel("Factors", fontsize=10)
        plt.ylabel("Contrasts", fontsize=10)

    plt.savefig(out_img_path, bbox_inches="tight")
    plt.show()


def print_fit_stats(label, model_obj):
    """
    Summarizes key metrics (RMSEA, CFI, BIC) from a factor analysis object.

    Args:
        label: Subject or Group label (str)
        model_obj: An R object returned by psych::fa accessed via rpy2.

    Returns:
        str: print values
    """
    # extract/round to 2 dec
    rmsea = model_obj.rx2("RMSEA")
    rmsea_values = (
        f"RMSEA: {rmsea[0]:.2f} (lower: {rmsea[1]:.2f}, upper: {rmsea[2]:.2f})"
    )
    cfi = model_obj.rx2("CFI")
    cfi_value = f"CFI: {cfi[0]:.2f}"
    bic = model_obj.rx2("BIC")
    bic_value = f"BIC: {bic[0]:.2f}"

    print(
        f"Subject: * {label} * \n\t Global Fit. \n \t {rmsea_values} \n \t {cfi_value} & {bic_value}"
    )


class EFAProcessor:
    """Main processor for EFA workflow."""

    def __init__(
        self,
        data_path: Path,
        output_dir: Path,
        atlas_dimension: int,
        n_factors: Optional[int] = None,
        subject_id: Optional[str] = None,
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.atlas_dimension = atlas_dimension
        self.n_factors = n_factors
        self.subject_id = subject_id or self._extract_subject_id()
        self.logger = self._setup_logging()

        # Set template path
        # TODO: Do not use fullpath
        self.template = Path(
            "/scratch/users/logben/network_efa/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz"
        )

        # Initialize R environment
        self._setup_r_environment()

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure logging for the processor."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def _setup_r_environment(self):
        """Initialize R environment and libraries."""
        self.logger.info("Setting up R environment...")
        pandas2ri.activate()
        self.utils = importr("utils")
        self.psych = importr("psych")

    def _extract_subject_id(self) -> str:
        """Extract subject ID from data file path."""
        filename = self.data_path.name
        match = re.search(r"_(s\d+)\.tsv$", filename)
        return match.group(1) if match else "unknown"

    def validate_inputs(self) -> None:
        """Validate input files and parameters."""
        self.logger.info("Validating input parameters...")

        if not self.data_path.exists():
            self.logger.error(f"Data file does not exist: {self.data_path}")
            sys.exit(1)

        if not self.template.exists():
            self.logger.error(f"Template file does not exist: {self.template}")
            sys.exit(1)

    def process(self) -> None:
        """Main processing pipeline."""
        self.logger.info(f"Starting EFA analysis for subject: {self.subject_id}")
        self.logger.info(f"Data file: {self.data_path}")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Validate inputs
        self.validate_inputs()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare data
        self.logger.info("Loading data...")
        df = pd.read_csv(self.data_path, sep="\t")

        # NOTE: We decided to include only a subset of the contrasts, 
        # motivated in part by the massive overlap we'd expect in lots
        # of contrasts that aren't interesting, like shapeMatching's
        # DNN, DDD, SDD, etc. 
        contrasts_to_include = {
            "directedForgetting_neg-con",
            "directedForgetting_response_time",
            "directedForgetting_task-baseline",
            "flanker_incongruent-congruent",
            "flanker_response_time",
            "flanker_task-baseline",
            "nBack_twoBack-oneBack",
            "nBack_match-mismatch",
            "nBack_response_time",
            "nBack_task-baseline",
            "stopSignal_stop_success-go",
            "stopSignal_stop_failure-go",
            "stopSignal_response_time",
            "stopSignal_task-baseline",
            "goNogo_nogo_success-go",
            "goNogo_response_time",
            "goNogo_task-baseline",
            "cuedTS_task_switch_cost",
            "cuedTS_cue_switch_cost",
            "cuedTS_response_time",
            "cuedTS_task-baseline",
            "shapeMatching_main_vars",
            "shapeMatching_response_time",
            "shapeMatching_task-baseline",
            "spatialTS_cue_switch_cost",
            "spatialTS_task_switch_cost",
            "spatialTS_response_time",
            "spatialTS_task-baseline",
        }

        # Filter for only the selected contrasts
        prefix = f"sub-{self.subject_id}_"
        filtered_columns = [prefix + name for name in contrasts_to_include]
        existing_columns = [col for col in filtered_columns if col in df.columns]
        df = df[existing_columns]

        numeric_df = df.select_dtypes(include=["number"])

        self.logger.info("Sorting data columns alphabetically...")
        sorted_columns = sorted(numeric_df.columns)
        numeric_df = numeric_df[sorted_columns]
        self.logger.info(f"Data shape: {numeric_df.shape}")
        # Generate correlation matrix
        self._generate_correlation_plot(numeric_df)

        # Create R dataframe for parallel analysis
        r_df = pandas2ri.py2ri(numeric_df)

        # Parallel analysis
        self._run_parallel_analysis(r_df)

        # Run EFA
        fa_model = self._run_efa(numeric_df)

        # Generate brain visualizations
        self._generate_brain_visualizations(fa_model)

        self.logger.info(f"EFA analysis completed for subject: {self.subject_id}")

    def _generate_correlation_plot(self, numeric_df: pd.DataFrame) -> None:
        """Generate and save correlation matrix plot."""
        self.logger.info("Generating correlation matrix plot...")

        corr_matrix = numeric_df.corr()

        plt.figure(figsize=(16, 12))
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            xticklabels=corr_matrix.columns,
            yticklabels=corr_matrix.columns,
        )
        plt.title(f"Correlation Matrix - {self.subject_id}")
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()

        output_path = self.output_dir / "correlation_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Correlation matrix saved to: {output_path}")

    def _run_parallel_analysis(self, r_df) -> int:
        """Run parallel analysis and return recommended number of factors."""
        self.logger.info("Running parallel analysis...")

        out_parallel = self.psych.fa_parallel(
            r_df, fm="ml", fa="both", n_iter=1000, show_legend=True, plot=False
        )

        recommended_factors = int(list(out_parallel.rx2("nfact"))[0])

        output_path = self.output_dir / "parallel_analysis.png"
        plot_fa_parallel(
            out_parallel, title_name=self.subject_id, out_img_path=str(output_path)
        )

        self.logger.info(
            f"Parallel analysis completed. Recommended factors: {recommended_factors}"
        )
        self.logger.info(f"Parallel analysis plot saved to: {output_path}")

        # Use provided n_factors or recommended from parallel analysis
        self.n_factors = self.n_factors or recommended_factors
        return self.n_factors

    def _run_efa(self, numeric_df: pd.DataFrame):
        """Run exploratory factor analysis."""
        self.logger.info(f"Running EFA with {self.n_factors} factors...")

        fa = self.psych.fa(
            numeric_df,
            fm="ml",
            rotate="promax",
            nfactors=self.n_factors,
            scores="Bartlett",
        )

        print_fit_stats(label=self.subject_id, model_obj=fa)

        # Generate loadings plot
        loadings_path = self.output_dir / f"efa_loadings_{self.n_factors}factors.png"
        plot_loading_phi(
            fa_object=fa,
            title_name=self.subject_id,
            out_type="loadings",
            row_labs=numeric_df.columns,
            out_img_path=str(loadings_path),
        )

        # Generate factor correlations plot
        phi_path = self.output_dir / f"factor_correlations_{self.n_factors}factors.png"
        plot_loading_phi(
            fa_object=fa,
            title_name=self.subject_id,
            out_type="phi",
            phi_plot_diag=True,
            out_img_path=str(phi_path),
        )

        # Save loadings and correlations as TSV
        self._save_efa_results(fa)

        self.logger.info("EFA analysis completed")
        return fa

    def _save_efa_results(self, fa_model) -> None:
        """Save EFA results as TSV files."""
        # Save loadings
        loadings = np.array(fa_model.rx2("loadings"))
        loadings_df = pd.DataFrame(
            loadings,
            columns=[f"Factor_{i + 1}" for i in range(loadings.shape[1])],
            index=[f"Parcel_{i + 1:03d}" for i in range(loadings.shape[0])],
        )
        loadings_path = self.output_dir / f"efa_loadings_{self.n_factors}factors.tsv"
        loadings_df.to_csv(loadings_path, sep="\t")
        self.logger.info(f"Loadings saved to: {loadings_path}")

        # Save factor correlations (Phi matrix)
        phi_matrix = np.array(fa_model.rx2("Phi"))
        phi_df = pd.DataFrame(
            phi_matrix,
            columns=[f"Factor_{i + 1}" for i in range(phi_matrix.shape[1])],
            index=[f"Factor_{i + 1}" for i in range(phi_matrix.shape[0])],
        )
        phi_path = self.output_dir / f"factor_correlations_{self.n_factors}factors.tsv"
        phi_df.to_csv(phi_path, sep="\t")
        self.logger.info(f"Factor correlations saved to: {phi_path}")

    def _generate_brain_visualizations(self, fa_model) -> None:
        """Generate brain visualizations for each factor."""
        self.logger.info("Generating brain visualizations...")

        # Get Bartlett scores
        bartlett_scores = np.array(fa_model.rx2("scores"))

        # Setup DiFuMo atlas masker
        difumo = fetch_atlas_difumo(
            dimension=self.atlas_dimension, resolution_mm=2, legacy_format=False
        )
        # TODO: Do not use fullpath
        gm_mask = "/scratch/users/logben/network_efa/tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz"
        masker = NiftiMapsMasker(
            maps_img=difumo.maps,
            # mask_img=gm_mask,
            allow_overlap=True,
            standardize=False,
            resampling_target="data",
            memory="nilearn_cache",
            memory_level=1,
            verbose=0,
        ).fit()

        # Store all factor scores for combined visualization
        all_factor_scores = []

        # Process each factor
        for factor_index in range(bartlett_scores.shape[1]):
            factor_num = factor_index + 1
            self.logger.info(
                f"Processing factor {factor_num}/{bartlett_scores.shape[1]}"
            )

            # Get factor scores
            factor_scores = bartlett_scores[:, factor_index]
            all_factor_scores.append(factor_scores)

            # Transform back to brain space
            brain_scores = masker.inverse_transform(factor_scores.reshape(1, -1))

            # Save NIfTI file
            nifti_path = self.output_dir / f"factor_{factor_num}_brain_map.nii.gz"
            brain_scores.to_filename(str(nifti_path))

            # Save factor scores as TSV
            scores_df = pd.DataFrame(
                factor_scores.reshape(-1, 1),
                columns=[f"Factor_{factor_num}"],
                index=[f"Parcel_{i + 1:03d}" for i in range(len(factor_scores))],
            )
            scores_path = self.output_dir / f"factor_{factor_num}_parcel_scores.tsv"
            scores_df.to_csv(scores_path, sep="\t")

            # Create brain visualization
            plt.figure(figsize=(12, 4))
            display = plotting.plot_stat_map(
                stat_map_img=brain_scores,
                display_mode="tiled",
                cut_coords=(0, -28, 15),
                title=f"{self.subject_id} - Factor {factor_num}",
                draw_cross=False,
                colorbar=True,
                vmax=0.01,
                bg_img=str(self.template),
            )

            viz_path = self.output_dir / f"factor_{factor_num}_brain_visualization.png"
            plt.savefig(str(viz_path), bbox_inches="tight", dpi=300)
            plt.close()

        # Save all factor scores combined
        all_scores_df = pd.DataFrame(
            np.column_stack(all_factor_scores),
            columns=[f"Factor_{i + 1}" for i in range(len(all_factor_scores))],
            index=[f"Parcel_{i + 1:03d}" for i in range(len(all_factor_scores[0]))],
        )
        all_scores_path = self.output_dir / "all_factor_parcel_scores.tsv"
        all_scores_df.to_csv(all_scores_path, sep="\t")

        # Create combined visualization of all factors
        self._create_combined_factor_visualization(all_factor_scores, masker)

        self.logger.info("Brain visualizations completed")

    def _create_combined_factor_visualization(
        self, all_factor_scores: List, masker
    ) -> None:
        """Create a combined visualization of all factors."""
        self.logger.info("Creating combined factor visualization...")

        n_factors = len(all_factor_scores)
        n_cols = min(3, n_factors)
        n_rows = (n_factors + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_factors == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, factor_scores in enumerate(all_factor_scores):
            row = i // n_cols
            col = i % n_cols

            brain_scores = masker.inverse_transform(factor_scores.reshape(1, -1))

            ax = axes[row, col] if n_rows > 1 else axes[col]
            display = plotting.plot_stat_map(
                stat_map_img=brain_scores,
                display_mode="z",
                cut_coords=1,
                title=f"Factor {i + 1}",
                axes=ax,
                colorbar=False,
                vmax=0.01,
                bg_img=str(self.template),
            )

        # Hide unused subplots
        for i in range(n_factors, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.axis("off")

        plt.suptitle(f"{self.subject_id} - All Factors", fontsize=16)
        plt.tight_layout()

        combined_path = self.output_dir / "all_factors_brain_visualization.png"
        plt.savefig(str(combined_path), bbox_inches="tight", dpi=300)
        plt.close()

        self.logger.info(f"Combined visualization saved to: {combined_path}")


def get_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Exploratory Factor Analysis on parcellated brain data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to parcellated TSV data file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for EFA results",
    )

    parser.add_argument(
        "--atlas-dimension",
        type=int,
        default=1024,
        choices=[64, 128, 256, 512, 1024],
        help="DiFuMo atlas dimension used in parcellation",
    )

    parser.add_argument(
        "--n-factors",
        type=int,
        help="Number of factors to extract (if not provided, uses parallel analysis recommendation)",
    )

    parser.add_argument(
        "--subject-id",
        type=str,
        help="Subject ID (if not provided, extracts from filename)",
    )

    return parser


def main():
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args()

    # Initialize processor
    processor = EFAProcessor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        atlas_dimension=args.atlas_dimension,
        n_factors=args.n_factors,
        subject_id=args.subject_id,
    )

    # Run EFA analysis
    try:
        processor.process()

    except Exception as e:
        logging.error(f"EFA processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
