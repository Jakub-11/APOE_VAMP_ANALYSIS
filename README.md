# APOE Dimer Simulation Analysis

This repository contains the code used to reproduce the results of the analysis of **APOE dimer molecular dynamics (MD) simulations**, as presented in ***[MANUSCRIPT TITLE]***. The methodology used in the analysis is described in detail in the manuscript.

## Overview

The analysis focuses on characterizing APOE dimer conformations using inter-chain distance metrics, dimensionality reduction (VAMP), clustering, and structural comparison to known reference states.

## Data

The MD trajectory files (`.xtc`) referenced in the scripts are provided in ***[LINK TO DATASET]***. Please update the paths in the scripts according to the location where you downloaded the data.

## Usage

The scripts should be run in the following order:

1. **`data_processing.py`**  
   Calculates the inter-residue, inter-chain distances between APOE chains.  
   Outputs are saved to `data/input/`.

2. **`vamp_projection.py`**  
   Computes the VAMP projection of the preprocessed distance data.  
   Results are saved to `output/projected/vamp/`.

3. **`clustering.py`**  
   Applies K-means clustering to the projected data.  
   - Cluster assignments are saved to `output/projected/assignments/`.  
   - A clustering figure is saved to the `output/` directory.

4. **`sampling.py`**  
   Samples representative frames from each cluster, including the filtered "monomeric" state (label `-1`).  
   Outputs are saved to `output/representative_structures/`.

5. **`reference_structure_analysis.py`**  
   Performs RMSD analysis against reference APOE dimer structures (T-shape, V-shape, parallel, anti-T-shape).  
   A summary pie chart is saved to the `output/` directory.

## Requirements

Tested with Python 3.8. Required packages:

- `deeptime`
- `pyemma`
- `mdtraj`

Install with:

```bash
pip install deeptime pyemma mdtraj
```

Or using a virtual environment:

```bash
python3.8 -m venv apoe-env
source apoe-env/bin/activate
pip install deeptime pyemma mdtraj
```

## Issues

Please use the [GitHub Issues](../../issues) section for reporting bugs or requesting help.
