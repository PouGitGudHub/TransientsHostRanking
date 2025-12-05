import pandas as pd
import numpy as np
import joblib

MODEL_IN = "/home/pouria/PycharmProjects/TransientsHostFinder/models/hgb_model.pkl"
FEATURES_IN = "/home/pouria/PycharmProjects/TransientsHostFinder/models/feature_list.pkl"
INPUT_FILE = "/data/input/CF_limit_hosts_combined_dr4.csv"
OUTPUT_FILE = "/data/output/predicted_hosts_dr4.csv"

# ===============================
# 1. Load model + feature list
# ===============================
clf = joblib.load(MODEL_IN)
feature_list = joblib.load(FEATURES_IN)

# ===============================
# 2. Load input data
# ===============================
df = pd.read_csv(INPUT_FILE)

if 'z_phot_mean_i' in df.columns:
    df.rename(columns={'z_phot_mean_i': 'z_phot'}, inplace=True)

# ===============================
# 3. Feature Engineering
# ===============================
def calculate_separation_vect(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_theta = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))*3600

def add_features(df):
    df['separation'] = calculate_separation_vect(
        df['SN_RA'], df['SN_DEC'], df['Host_RA'], df['Host_DEC']
    )

    df = df.sort_values(['SN_ID', 'd_DLR'])

    min_dlr = df.groupby('SN_ID')['d_DLR'].transform('min')
    df['d_DLR_diff'] = df['d_DLR'] - min_dlr

    min_sep = df.groupby('SN_ID')['separation'].transform('min')
    df['sep_diff'] = df['separation'] - min_sep

    min_mag = df.groupby('SN_ID')['mag_r'].transform('min')
    df['mag_diff'] = df['mag_r'] - min_mag

    median_z = df.groupby('SN_ID')['z_phot'].transform('median')
    df['z_phot_diff'] = (df['z_phot'] - median_z).abs()

    df['n_candidates'] = df.groupby('SN_ID')['SN_ID'].transform('count')
    return df

df_feats = add_features(df)

# One-hot encoding
df_feats['type'] = df_feats['type'].fillna('Unknown')
df_feats = pd.get_dummies(df_feats, columns=['type'], prefix='type')

# Ensure all expected model columns exist
for col in feature_list:
    if col not in df_feats:
        df_feats[col] = 0  # add missing columns

X = df_feats[feature_list]

# ===============================
# 4. Prediction
# ===============================
raw_probs = clf.predict_proba(X)[:, 1]
df_feats['raw_prob'] = raw_probs

# Normalize probabilities per SN
def normalize(x):
    s = x.sum()
    return x if s == 0 else x / s

df_feats['normalized_prob'] = df_feats.groupby('SN_ID')['raw_prob'].transform(normalize)

df_feats['rank'] = df_feats.groupby('SN_ID')['normalized_prob'] \
                          .rank(ascending=False, method='first')

# ===============================
# 5. Save output
# ===============================

OUTPUT_FILE = "/home/pouria/PycharmProjects/TransientsHostFinder/data/output/predicted_hosts_dr4_full.csv"

# Keep ALL original columns + added scores
df_output = df.copy()
df_output['separation'] = df_feats['separation']
df_output['raw_prob'] = df_feats['raw_prob']
df_output['normalized_prob'] = df_feats['normalized_prob']
df_output['rank'] = df_feats['rank']

df_output = df_output.sort_values(['SN_ID', 'rank'])

df_output.to_csv(OUTPUT_FILE, index=False)

print("Full prediction output saved to:", OUTPUT_FILE)

