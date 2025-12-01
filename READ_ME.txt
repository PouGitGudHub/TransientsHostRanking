==============================================================================
PROJECT: AstroHostFinder
==============================================================================

DESCRIPTION:
Machine Learning pipeline for identifying host galaxies of astronomical
transients (e.g., Supernovae). The system utilizes a ranking approach based
on Gradient Boosting to evaluate host candidates within the context of their
local environment.

TECH STACK:
- Language: Python 3.9+
- ML Framework: Scikit-Learn (HistGradientBoostingClassifier)
- Data Processing: Pandas, NumPy
- Astronomy: Astropy (Coordinate calculations)

==============================================================================
1. PROJECT STRUCTURE
==============================================================================

AstroHostFinder/
│
├── data/
│   ├── training_data/       # Storage for training datasets (CSV)
│   │                        # Requires: Candidate file & Ground Truth file
│   ├── input/               # Input directory for new data (Inference)
│   │                        # User places CSVs here for analysis
│   └── output/              # Output directory for results
│                            # Contains processed CSVs with probabilities
│
├── models/                  # Storage for serialized models (.pkl)
│   ├── host_finder_model.pkl  # The trained model artifact
│   └── model_features.pkl     # List of features for validation
│
├── src/                     # Source Code Modules (Backend Logic)
│   ├── __init__.py          # Initialization file
│   ├── features.py          # Central feature engineering module
│   └── training.py          # Script for model creation (Training)
│
├── main.py                  # Main application (CLI) for prediction
├── results_analysis.ipynb   # Jupyter Notebook for statistical evaluation
├── README.txt               # Project documentation
└── requirements.txt         # List of Python dependencies

==============================================================================
2. MODULE DOCUMENTATION: src/features.py
==============================================================================

This module ensures consistency between training and inference phases. It
transforms raw catalog data into the model's feature space.

FUNCTIONS:

A) calculate_separation_vect(ra1, dec1, ra2, dec2)
   - Purpose: Computes spherical distance between two coordinate points.
   - Method: Vectorized calculation using the Spherical Law of Cosines.
   - Output: Angular separation in arcseconds (arcsec).

B) prepare_features(df)
   - Purpose: Performs feature engineering on a Pandas DataFrame.
   - Process Steps:
     1. Calculation of absolute separation (angular distance).
     2. Sorting of candidates per transient (SN_ID) by d_DLR.
     3. Calculation of relative context features (see below).
     4. One-Hot Encoding of categorical variables (Morphological Type).

GENERATED CONTEXT FEATURES:
The model evaluates candidates relative to their local neighbors.

1. d_DLR_diff
   - Difference between the candidate's d_DLR and the group minimum.
   - Formula: d_DLR(i) - min(d_DLR across all candidates of the SN)

2. sep_diff
   - Difference between the candidate's angular separation and the separation
     of the geometrically nearest neighbor.
   - Formula: separation(i) - min(separation across all candidates of the SN)

3. mag_diff
   - Magnitude difference (r-band) relative to the brightest galaxy in the group.
   - Formula: mag_r(i) - min(mag_r across all candidates of the SN)

4. z_phot_diff
   - Absolute deviation of photometric redshift from the group median.
   - Formula: |z_phot(i) - median(z_phot across all candidates of the SN)|

5. n_candidates
   - Total count of candidates available for the specific transient.

==============================================================================
3. MODULE DOCUMENTATION: src/training.py
==============================================================================

Script for model creation. Executes the following pipeline:
1. Loads data from 'data/training_data/'.
2. Labeling: Matches candidate coordinates against Ground Truth.
   - Criterion: Distance < 1.5 arcsec = True Host (Label 1).
   - All others = False (Label 0).
3. Applies 'src/features.py' for transformation.
4. Trains a HistGradientBoostingClassifier.
5. Serializes (saves) the model and feature list to 'models/'.

==============================================================================
4. MODULE DOCUMENTATION: main.py
==============================================================================

The interface for the end-user (Inference).
1. Scans 'data/input/' for CSV files.
2. Loads the model artifact from 'models/'.
3. Computes features for new data via 'src/features.py'.
4. Performs prediction (Inference).
5. Post-Processing:
   - Normalization: Probabilities per SN sum to 1.0.
   - Ranking: Sorts by probability (Rank 1 = Top Candidate).
6. Saves results as 'SCORED_[Filename].csv' to 'data/output/'.

==============================================================================
5. WORKFLOW
==============================================================================

A. INITIALIZATION / TRAINING
   (Only required at project start or when updating training data)
   1. Place candidate CSV and Ground Truth CSV in 'data/training_data/'.
   2. Execute 'src/training.py'.

B. APPLICATION (Standard Process)
   1. Place new transient list files (CSV) in 'data/input/'.
   2. Execute 'main.py'.
   3. Retrieve results from 'data/output/'.

C. ANALYSIS
   1. Open 'results_analysis.ipynb'.
   2. Execute cells to visualize distributions and confidence metrics.

==============================================================================