#!/usr/bin/env python
"""
Data preparation script for network EFA analysis.

This script parcellates fMRI contrast images using functional atlases
to create TSV files suitable for exploratory factor analysis.
Based on the DiFuMo atlas parcellation workflow but implemented
using object-oriented design for better maintainability. The current
version also passes in a grey matter mask to the masker.
"""

import argparse
import logging
import sys
from pathlib import Path
import re
from typing import List, Dict, Optional
import warnings
import pandas as pd
import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class DataProcessor:
    """Main processor for parcellation workflow."""

    def __init__(
        self,
        input_dirs: List[Path],
        output_dir: Path,
        atlas_dimension: int,
        atlas_resolution: int,
        n_jobs: int,
        subject_id: Optional[str] = None,
    ):
        self.input_dirs = input_dirs
        self.output_dir = output_dir
        self.atlas_dimension = atlas_dimension
        self.atlas_resolution = atlas_resolution
        self.n_jobs = n_jobs
        self.subject_id = subject_id
        self.logger = self._setup_logging()

        # Initialize component classes
        self.file_manager = FileManager(self.logger)
        self.atlas_manager = AtlasManager(self.logger)
        self.parcellation_processor = ParcellationProcessor(self.logger, n_jobs=n_jobs)

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure logging for the processor."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def validate_inputs(self) -> None:
        """Validate input directories and parameters."""
        self.logger.info("Validating input parameters...")

        # Check input directories
        for input_dir in self.input_dirs:
            if not input_dir.exists():
                self.logger.error(f"Input directory does not exist: {input_dir}")
                sys.exit(1)
            if not input_dir.is_dir():
                self.logger.error(f"Input path is not a directory: {input_dir}")
                sys.exit(1)

        # Validate atlas parameters
        valid_dimensions = [64, 128, 256, 512, 1024]
        if self.atlas_dimension not in valid_dimensions:
            self.logger.warning(
                f"Atlas dimension {self.atlas_dimension} may not be available. "
                f"Valid options are typically: {valid_dimensions}"
            )

        if self.atlas_resolution not in [2, 3]:
            self.logger.warning(
                f"Atlas resolution {self.atlas_resolution}mm may not be available. "
                f"Valid options are typically: 2mm, 3mm"
            )

    def process(self) -> None:
        """Main processing pipeline."""
        self.logger.info("Starting parcellation processing pipeline")
        self.logger.info("Using the following configuration...")
        for attr, value in self.__dict__.items():
            if attr != "logger" and not attr.startswith("_"):
                if attr == "input_dirs":
                    self.logger.info(f"  {attr}: {[str(d) for d in value]}")
                else:
                    self.logger.info(f"  {attr}: {value}")

        # Validate inputs
        self.validate_inputs()

        # Create output directory
        self.logger.info(f"Creating output directory: {self.output_dir.absolute()}")
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"Output directory created/verified: {self.output_dir.absolute()}"
            )
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise

        # Get NIfTI files
        try:
            nifti_files = self.file_manager.get_contrast_files(
                self.input_dirs, self.subject_id
            )
            if not nifti_files:
                subject_msg = (
                    f" for subject {self.subject_id}" if self.subject_id else ""
                )
                self.logger.error(
                    f"No contrast files found in input directories{subject_msg}"
                )
                sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error getting contrast files: {e}")
            sys.exit(1)

        # Fetch and setup atlas
        try:
            atlas = self.atlas_manager.fetch_difumo_atlas(
                dimension=self.atlas_dimension, resolution_mm=self.atlas_resolution
            )
            masker = self.atlas_manager.create_masker(atlas)
        except Exception as e:
            self.logger.error(f"Error setting up atlas: {e}")
            sys.exit(1)

        # Create parcellated matrix
        try:
            matrix = self.parcellation_processor.create_contrast_matrix(
                nifti_files, masker
            )

            if self.subject_id:
                # Single subject mode - save individual file directly
                output_file = (
                    self.output_dir
                    / f"parcellated_difumo_{self.atlas_dimension}_{self.subject_id}.tsv"
                )
                matrix.to_csv(output_file, sep="\t")
                self.logger.info(
                    f"Saved parcellated matrix for {self.subject_id} to: {output_file}"
                )
            else:
                # Multi-subject mode - save combined matrix only
                output_file = (
                    self.output_dir / f"parcellated_difumo_{self.atlas_dimension}.tsv"
                )
                matrix.to_csv(output_file, sep="\t")
                self.logger.info(f"Saved parcellated matrix to: {output_file}")
                self.logger.info(
                    "Note: Use --subject-id to process individual subjects for EFA compatibility"
                )

            # Log matrix info
            self.logger.info(f"Final matrix shape: {matrix.shape}")
            self.logger.info(
                f"Matrix dimensions: {matrix.shape[0]} parcels × {matrix.shape[1]} contrasts"
            )

            self.logger.info("Parcellation processing completed successfully!")

        except Exception as e:
            self.logger.error(f"Error in parcellation processing: {e}")
            sys.exit(1)


class FileManager:
    """Handles files and filename parsing."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def get_contrast_files(
        self, input_dirs: List[Path], subject_id: Optional[str] = None
    ) -> List[Path]:
        """
        get all fixed-effects contrast NIfTI files in the specified directories.

        Args:
            input_dirs: List of base directories containing subject outputs
            subject_id: Optional subject ID to filter files (e.g., 's03', 's10')

        Returns:
            List of Path objects pointing to contrast files
        """
        all_files = []
        tag = "fixed-effects.nii.gz"

        if subject_id:
            # Single subject mode - use specific pattern
            pattern = f"sub-{subject_id}/*/fixed_effects/*{tag}"
            search_msg = f"contrast files for subject {subject_id}"
        else:
            # Multi-subject mode - use wildcard pattern
            pattern = f"sub-*/*/fixed_effects/*{tag}"
            search_msg = "contrast files"

        for input_dir in input_dirs:
            self.logger.info(f"Searching for {search_msg} in: {input_dir}")
            files = list(input_dir.glob(pattern))
            all_files.extend(files)
            self.logger.info(f"Found {len(files)} contrast files in {input_dir}")

        # Sort files by subject ID numerically for consistent ordering
        all_files.sort(key=self._extract_subject_number)

        total_files = len(all_files)
        subject_msg = f" for subject {subject_id}" if subject_id else ""
        self.logger.info(
            f"Found {total_files} total contrast files{subject_msg} matching pattern '{pattern}'"
        )

        if total_files == 0:
            self.logger.warning(
                f"No contrast files found{subject_msg}. Please check your input directories and file patterns."
            )

        return all_files

    def _extract_subject_number(self, filepath: Path) -> int:
        """Extract numeric subject ID for sorting."""
        match = re.search(r"sub-s?(\d+)", str(filepath))
        return int(match.group(1)) if match else float("inf")

    def parse_filename_info(self, filepath: Path) -> Dict[str, str]:
        """
        Parse filename to extract subject, task, and contrast information.

        Expected format:
        .../sub-sXX/task/fixed_effects/sub-sXX_task-YYY_contrast-ZZZ_..._stat-fixed-effects.nii.gz

        Args:
            filepath: Path object for the NIfTI file

        Returns:
            Dictionary containing parsed information
        """
        filename = filepath.name
        info = {}

        # Extract subject (sub-sXX)
        sub_match = re.search(r"(sub-[a-zA-Z0-9]+)", filename)
        info["subject"] = sub_match.group(1) if sub_match else "unknown_subject"

        # Extract task (task-YYY)
        task_match = re.search(r"task-([a-zA-Z0-9]+)", filename)
        info["task"] = task_match.group(1) if task_match else "unknown_task"

        # Extract contrast (contrast-ZZZ)
        contrast_match = re.search(r"contrast-(.+?)_(?:rtmodel|stat)", filename)
        info["contrast"] = (
            contrast_match.group(1) if contrast_match else "unknown_contrast"
        )

        # Create standardized column name
        info["column_name"] = f"{info['subject']}_{info['task']}_{info['contrast']}"

        return info


class AtlasManager:
    """Handles atlas fetching and masker creation."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def fetch_difumo_atlas(self, dimension: int, resolution_mm: int):
        """
        Fetch the DiFuMo functional atlas.

        Args:
            dimension: Atlas dimensionality (e.g., 64, 128, 256, 512, 1024)
            resolution_mm: Resolution in millimeters (2 or 3)

        Returns:
            The fetched atlas object
        """
        self.logger.info(
            f"Fetching DiFuMo atlas (dimension={dimension}, resolution={resolution_mm}mm)"
        )

        try:
            atlas = datasets.fetch_atlas_difumo(
                dimension=dimension, resolution_mm=resolution_mm
            )
            self.logger.info(f"Successfully fetched DiFuMo-{dimension} atlas")
            return atlas
        except Exception as e:
            self.logger.error(f"Failed to fetch DiFuMo atlas: {e}")
            raise

    def create_masker(self, atlas, memory_level: int = 1) -> NiftiMapsMasker:
        """
        Create a NiftiMapsMasker from the atlas.

        Args:
            atlas: The fetched atlas object
            memory_level: Caching level for nilearn

        Returns:
            Fitted NiftiMapsMasker object
        """
        self.logger.info("Creating and fitting atlas masker...")

        try:
            # TODO: Do not use fullpath
            gm_mask = "/scratch/users/logben/network_efa/tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz"
            masker = NiftiMapsMasker(
                maps_img=atlas["maps"],
                mask_img=gm_mask,
                allow_overlap=True,
                standardize=False,
                resampling_target="data",
                memory="nilearn_cache",
                memory_level=memory_level,
                verbose=0,
            )
            masker.fit()
            self.logger.info("Atlas masker fitted successfully")
            return masker
        except Exception as e:
            self.logger.error(f"Failed to create/fit masker: {e}")
            raise


class ParcellationProcessor:
    """Handles parcellation and matrix creation."""

    def __init__(self, logger: logging.Logger, n_jobs: int = 1):
        self.logger = logger
        self.n_jobs = n_jobs

    def create_contrast_matrix(
        self, nifti_files: List[Path], masker: NiftiMapsMasker
    ) -> pd.DataFrame:
        """
        Create a contrast matrix from NIfTI files using the provided masker.

        Args:
            nifti_files: List of contrast file paths
            masker: Fitted NiftiMapsMasker object

        Returns:
            DataFrame with parcels as rows and contrasts as columns
        """
        self.logger.info(f"Creating contrast matrix from {len(nifti_files)} files...")

        # Parse all filenames to get column names
        file_manager = FileManager(self.logger)
        column_info = []
        for filepath in nifti_files:
            info = file_manager.parse_filename_info(filepath)
            column_info.append(info)

        column_names = [info["column_name"] for info in column_info]

        # Check for duplicate column names
        if len(column_names) != len(set(column_names)):
            duplicates = [name for name in column_names if column_names.count(name) > 1]
            self.logger.warning(f"Found duplicate column names: {set(duplicates)}")

        # Process each file and collect data
        data_dict = {}

        for i, (filepath, info) in enumerate(zip(nifti_files, column_info)):
            self.logger.info(f"Processing {i + 1}/{len(nifti_files)}: {filepath.name}")

            try:
                # Transform the NIfTI file using the masker
                parcellated_data = masker.transform(filepath)
                data_dict[info["column_name"]] = parcellated_data.flatten()

            except Exception as e:
                self.logger.error(f"Failed to process {filepath}: {e}")
                # Skip this file and continue
                continue

        if not data_dict:
            raise ValueError("No files were successfully processed")

        # Create DataFrame
        matrix = pd.DataFrame(data_dict)

        # Set index names (parcel names if available)
        n_parcels = matrix.shape[0]
        matrix.index = [f"parcel_{i + 1:03d}" for i in range(n_parcels)]
        matrix.index.name = "parcel"

        self.logger.info(f"Created contrast matrix with shape: {matrix.shape}")
        self.logger.info(
            f"Matrix contains {matrix.shape[0]} parcels and {matrix.shape[1]} contrasts"
        )

        # Log any missing data
        missing_data = matrix.isnull().sum().sum()
        if missing_data > 0:
            self.logger.warning(f"Matrix contains {missing_data} missing values")

        return matrix


def get_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Create parcellated contrast matrix for EFA analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_dirs",
        type=Path,
        nargs="+",
        help="Input directories containing contrast files",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for parcellated matrix",
    )

    parser.add_argument(
        "--atlas-dimension",
        type=int,
        default=64,
        choices=[64, 128, 256, 512, 1024],
        help="DiFuMo atlas dimension",
    )

    parser.add_argument(
        "--atlas-resolution",
        type=int,
        default=2,
        choices=[2, 3],
        help="Atlas resolution in millimeters",
    )

    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Number of parallel jobs for processing"
    )

    parser.add_argument(
        "--subject-id",
        type=str,
        help="Process only this subject ID (e.g., 's03', 's10'). If not provided, process all subjects.",
    )

    return parser


def main():
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args()

    # Initialize processor
    processor = DataProcessor(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        atlas_dimension=args.atlas_dimension,
        atlas_resolution=args.atlas_resolution,
        n_jobs=args.n_jobs,
        subject_id=args.subject_id,
    )

    # Run parcellation
    try:
        processor.process()

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
