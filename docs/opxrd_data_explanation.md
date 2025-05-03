# opXRD Dataset: Explanation of Extractable Data

This document provides an overview of the data available within the opXRD (Open Experimental Powder X-ray Diffraction) dataset, focusing on the information that can be extracted for use in machine learning and data analysis projects.

## Dataset Overview

The opXRD dataset is a collection of experimental Powder X-ray Diffraction (pXRD) patterns. Its primary goal is to provide a resource for training and evaluating machine learning models for automated pXRD analysis, particularly addressing the challenges of working with real-world experimental data which often contains noise and background effects not present in ideal simulated data.

## File Format and Structure

- **Archive:** The dataset is distributed as a zip archive (`opxrd.zip`).
- **Unzipped Structure:** Upon unzipping, the data is organized into subdirectories within `./opxrd_data/`. These top-level directories correspond to the contributing institutions (e.g., `CNRS`, `HKUST`, `IKFT`, `INT`, `USC`).
- **Data Files:** Individual pXRD patterns are stored as JSON files (`.json`) within these institutional subdirectories. Some institutions may further subdivide their data into project-specific folders.

## Extractable Data Fields

Each JSON file represents a single experimental pXRD pattern and contains several key fields. The primary data and associated metadata are stored under specific keys.

### Core pXRD Data

- **`two_theta_values`**: This key holds a list of floating-point numbers representing the 2-theta angles at which the diffraction intensity was measured. This is one of the two essential components of a pXRD pattern.
- **`intensities`**: This key holds a list of floating-point numbers representing the measured intensities corresponding to each 2-theta value in the `two_theta_values` list. This is the second essential component of a pXRD pattern.

These two lists are of equal length and together define the experimental diffractogram.

### Metadata

Several keys contain metadata providing context about the experiment and the sample. Note that the presence and structure of some of these keys, particularly `label` and `metadata`, can vary between files and contributing institutions.

- **`label`**: This field, when present and populated (especially for labeled data), contains detailed structural information about the sample. This can include:
    - `phases`: A list describing the different crystalline phases present. Each phase entry can be a string representation of a dictionary containing:
        - `chemical_composition`: The chemical formula of the phase.
        - `spacegroup`: The space group number.
        - `lengths`, `angles`: Unit cell parameters (a, b, c, alpha, beta, gamma).
        - `phase_fraction`: The relative fraction of this phase in the sample.
        - `base`: Atomic positions within the unit cell.
        - `volume_uc`: Unit cell volume.
        - `atomic_volume`: Atomic volume.
        - `wyckoff_symbols`: Wyckoff symbols for atomic positions.
        - `crystal_system`: The crystal system (e.g., cubic, tetragonal, monoclinic).
    - `xray_info`: Information about the X-ray source and optics, potentially including:
        - `primary_wavelength`, `secondary_wavelength`: Wavelengths of the X-rays used.
    - `is_simulated`: A boolean or integer indicating if the pattern is simulated (should be false/0 for this dataset, but the key might exist).
    - `crystallite_size`: Information about the average crystallite size.
    - `temp_in_celcius`: The temperature during the experiment in Celsius.

    The structure within the `label` field can be nested and may require careful parsing, as seen in the example files where phase information is stored as string representations of dictionaries within a list.

- **`metadata`**: This field often contains general information about the data collection process and origin, such as:
    - `filename`: Original filename.
    - `institution`: The contributing institution.
    - `contributor_name`: Name of the person who contributed the data.
    - `original_file_format`: The format of the original data file before conversion to JSON.
    - `measurement_date`: Date of the experiment.
    - `tags`: Any relevant tags associated with the data.
    - `xrdpattern_version`: Version information related to the data format.

    Similar to the `label` field, the content and structure of the `metadata` field can vary.

- **Other Potential Keys:** Based on the exploration and the paper, other keys that might be present and useful include:
    - `material_name`: A common name for the material.
    - `formula`: The overall chemical formula of the sample.

## Considerations for AI Engineers

- **Data Variability:** Be prepared for variations in the structure and completeness of the `label` and `metadata` fields across different files and institutions. Robust parsing and data cleaning steps will be necessary.
- **Nested Structures:** The `label` field, in particular, can contain nested JSON-like structures represented as strings, requiring additional parsing (e.g., using `json.loads` on the string values).
- **Missing Data:** Some metadata fields may be missing for certain patterns. The parsing logic should gracefully handle `None` values or missing keys.
- **Data Scale:** The dataset contains over 90,000 diffractograms, providing a substantial amount of experimental data for training and evaluation.
- **Labeled vs. Unlabeled Data:** The dataset includes both labeled and unlabeled data. The labeled data is crucial for supervised learning tasks like phase identification, while unlabeled data can be valuable for unsupervised learning, data augmentation, or transfer learning techniques.

## Parsing Script

The provided Python script `parse_opxrd.py` serves as a starting point for extracting data from the JSON files. It includes functions to load files, extract core pXRD data, and extract common metadata fields. This script can be extended to handle the specific variations and nested structures encountered in the `label` and `metadata` fields as needed for your specific application.

By leveraging the data fields described above, AI engineers can develop and evaluate machine learning models for various pXRD analysis tasks, contributing to the automation of materials characterization.