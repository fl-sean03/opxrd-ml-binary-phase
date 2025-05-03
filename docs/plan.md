# opXRD Dataset Analysis and Parsing Guide Plan

## Project Goal
Analyze the structure and content of the `opxrd.zip` dataset and generate a comprehensive Python guide for parsing the data within it. This guide will be used later for data extraction in a machine learning project focused on analyzing experimental Powder X-ray Diffraction (pXRD) patterns.

## Input Files
- `data/opxrd.zip`: The main dataset archive.
- `data/2503.05577v2.pdf`: The research paper describing the opXRD database.
- `data/2503.05577v2.txt`: Text version of the research paper.

## Completed Tasks

1.  **Unzip and Explore:**
    - Unzipped `opxrd.zip` into the `./opxrd_data/` directory.
    - Explored the directory structure and identified the top-level folders.

2.  **Identify File Format and Structure:**
    - Confirmed that individual data files are in JSON format by inspecting example files (`./opxrd_data/CNRS/pattern_0.json` and `./opxrd_data/USC/pattern_0.json`).
    - Analyzed the structure of typical JSON files and identified main top-level keys.

3.  **Determine Data Content:**
    - Examined the values associated with the main keys.
    - Identified keys for core pXRD data (`two_theta_values` and `intensities`).
    - Identified common metadata keys (`label`, `metadata`, `xray_info`, `is_simulated`, `crystallite_size`, `temp_in_celcius`). Noted that the structure and presence of keys within `label` and `metadata` can vary.

## Findings

- The dataset is organized into subdirectories within `./opxrd_data/` based on contributing institutions: `CNRS`, `HKUST`, `IKFT`, `INT`, and `USC`.
- Data files are in JSON format.
- Core pXRD data is stored under the keys `two_theta_values` (list of floats for 2-theta angles) and `intensities` (list of floats for corresponding intensities).
- Metadata is available under keys such as `label`, `metadata`, `xray_info`, `is_simulated`, `crystallite_size`, and `temp_in_celcius`. The content and structure of `label` and `metadata` are not strictly uniform across all files.

## Remaining Task

1.  **Create Python Parsing Guide:**
    - Generate a well-commented Python script (`parse_opxrd.py`) that includes functions to:
        - Iterate through the unzipped directory structure.
        - Load a single JSON file.
        - Extract core pXRD data (angles and intensities).
        - Extract relevant metadata fields, handling variations and missing data.
        - Include example usage.

## Next Step

Complete the implementation of the `parse_opxrd.py` script based on the identified file structure, format, and data content.