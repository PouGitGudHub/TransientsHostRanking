import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

# ===============================
# 1. Load data
# ===============================
TRAIN_DATA = "/home/pouria/PycharmProjects/TransientsHostFinder/data/training_data/CF_limit_hosts_combined.csv"
GROUND_TRUTH = "/home/pouria/PycharmProjects/TransientsHostFinder/data/training_data/ztfdr2_potential_hosts.csv"
MODEL_OUT = "/home/pouria/PycharmProjects/TransientsHostFinder/models/hgb_model.pkl"
FEATURES_OUT = "/home/pouria/PycharmProjects/TransientsHostFinder/models/feature_list.pkl"

df_candidates = pd.read_csv(TRAIN_DATA)
df_ground_truth = pd.read_csv(GROUND_TRUTH)

# ===============================
# 2. Labeling
# ===============================
true_hosts = df_ground_truth[df_ground_truth['is_host'] == 1][['ztfname', 'ra', 'dec']].copy()
true_hosts.rename(columns={'ztfname': 'SN_ID', 'ra': 'True_RA', 'dec': 'True_DEC'}, inplace=True)

df_train_merged = pd.merge(df_candidates, true_hosts, on='SN_ID', how='left')

def calculate_separation_vect(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_theta = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))*3600

df_train_merged['dist_to_true'] = calculate_separation_vect(
    df_train_merged['Host_RA'], df_train_merged['Host_DEC'],
    df_train_merged['True_RA'], df_train_merged['True_DEC']
)

df_train_merged['label'] = (df_train_merged['dist_to_true'] < 1.5).astype(int)

# ===============================
# 3. Feature Engineering
# ===============================
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

df_feats = add_features(df_train_merged)

# Remove almost-empty columns
cols_to_check = ['sersic', 'shape_r', 'shape_e1', 'shape_e2', 'mag_i']
cols_to_drop = [c for c in cols_to_check if df_feats[c].isna().mean() > 0.9]

base_features = ['d_DLR', 'mag_g', 'mag_r', 'mag_z', 'z_phot', 'separation']
base_features = [f for f in base_features if f not in cols_to_drop]

context_features = ['d_DLR_diff', 'sep_diff', 'mag_diff', 'z_phot_diff', 'n_candidates']

features = base_features + context_features

# One-hot encoding
df_feats['type'] = df_feats['type'].fillna('Unknown')
df_feats = pd.get_dummies(df_feats, columns=['type'], prefix='type')

type_features = [c for c in df_feats.columns if c.startswith('type_')]
final_features = features + type_features

# ===============================
# 4. Model Training
# ===============================
X_train = df_feats[final_features]
y_train = df_feats['label']

clf = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=200,
    max_depth=8,
    random_state=42
)
clf.fit(X_train, y_train)

# ===============================
# 5. Save model and feature list
# ===============================
joblib.dump(clf, MODEL_OUT)
joblib.dump(final_features, FEATURES_OUT)

print("Model and feature list saved successfully.")
