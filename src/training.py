import pandas as pd
import numpy as np
import joblib  # Zum Speichern des Modells
import os
from sklearn.ensemble import HistGradientBoostingClassifier

# ==========================================
# 1. DATEN LADEN
# ==========================================
# Passe die Pfade an, falls nötig (relativ zum Projektordner)
base_dir = os.path.dirname(os.path.dirname(__file__)) # Geht eins hoch
data_dir = os.path.join(base_dir, 'Data', 'training_data')
model_dir = os.path.join(base_dir, 'models')

print("🚀 Starte Training (All-in-One Skript)...")

try:
    df_candidates = pd.read_csv(os.path.join(data_dir, 'CF_limit_hosts_combined.csv'))
    df_ground_truth = pd.read_csv(os.path.join(data_dir, 'ztfdr2_potential_hosts.csv'))
except Exception as e:
    print(f"❌ Fehler beim Laden der Daten: {e}")
    exit(1)

# ==========================================
# 2. LABELING & MERGE
# ==========================================
# Wir holen die echten Hosts aus dem Ground Truth
true_hosts = df_ground_truth[df_ground_truth['is_host'] == 1][['ztfname', 'ra', 'dec']].copy()
true_hosts.rename(columns={'ztfname': 'SN_ID', 'ra': 'True_RA', 'dec': 'True_DEC'}, inplace=True)

# Merge mit Kandidaten
df_train_merged = pd.merge(df_candidates, true_hosts, on='SN_ID', how='left')

# Vektorisierte Winkeldistanz (in Bogensekunden) - direkt hier im Skript
def calculate_separation_vect(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_theta = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta)) * 3600

# Distanz zum WAHREN Host berechnen
df_train_merged['dist_to_true'] = calculate_separation_vect(
    df_train_merged['Host_RA'], df_train_merged['Host_DEC'],
    df_train_merged['True_RA'], df_train_merged['True_DEC']
)

# Regel: < 1.5 arcsec = Host (1), sonst 0
df_train_merged['label'] = 0
df_train_merged.loc[df_train_merged['dist_to_true'] < 1.5, 'label'] = 1

print(f"✅ Labeling fertig: {df_train_merged['label'].sum()} echte Hosts gefunden.")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def add_features(df):
    # Basis Separation zur SN
    df['separation'] = calculate_separation_vect(
        df['SN_RA'], df['SN_DEC'], df['Host_RA'], df['Host_DEC']
    )

    # Sortieren für Gruppenoperationen
    df = df.sort_values(['SN_ID', 'd_DLR'])

    # --- Kontext Features ---
    # 1. d_DLR Differenz zum Minimum
    min_dlr = df.groupby('SN_ID')['d_DLR'].transform('min')
    df['d_DLR_diff'] = df['d_DLR'] - min_dlr

    # 2. Separation Differenz zum Minimum
    min_sep = df.groupby('SN_ID')['separation'].transform('min')
    df['sep_diff'] = df['separation'] - min_sep

    # 3. Magnitude Differenz zum Hellsten
    min_mag = df.groupby('SN_ID')['mag_r'].transform('min')
    df['mag_diff'] = df['mag_r'] - min_mag

    # 4. Redshift Differenz zum Median
    # WICHTIG: Wenn die Spalte 'z_phot' heißt (nicht z_phot_mean_i), nimm die!
    # Im alten Code war es in candidates schon 'z_phot' (aus csv), in dr4 'z_phot_mean_i'.
    # CF_limit_hosts_combined.csv hat normalerweise 'z_phot'.
    z_col = 'z_phot' if 'z_phot' in df.columns else 'z_phot_mean_i'
    median_z = df.groupby('SN_ID')[z_col].transform('median')
    df['z_phot_diff'] = (df[z_col] - median_z).abs()

    # 5. Anzahl der Kandidaten
    df['n_candidates'] = df.groupby('SN_ID')['SN_ID'].transform('count')

    return df

# Features berechnen
df_train_feats = add_features(df_train_merged)

# Leere Spalten entfernen (EXAKT wie im alten Code)
cols_to_check = ['sersic', 'shape_r', 'shape_e1', 'shape_e2', 'mag_i']
drop_threshold = 0.9
cols_to_drop = [c for c in cols_to_check if c in df_train_feats.columns and df_train_feats[c].isna().mean() > drop_threshold]

print(f"🗑️ Entferne leere Spalten: {cols_to_drop}")

# Features zusammenstellen
base_features = ['d_DLR', 'mag_g', 'mag_r', 'mag_z', 'z_phot', 'separation']
# Falls z_phot anders heißt
if 'z_phot' not in df_train_feats.columns and 'z_phot_mean_i' in df_train_feats.columns:
    base_features = [b if b != 'z_phot' else 'z_phot_mean_i' for b in base_features]

base_features = [f for f in base_features if f not in cols_to_drop]
context_features = ['d_DLR_diff', 'sep_diff', 'mag_diff', 'z_phot_diff', 'n_candidates']
features = base_features + context_features

# One-Hot Encoding für 'type'
# Wir setzen is_train=1, um es später zu erkennen (Legacy Logik)
df_train_feats['is_train'] = 1
# Wir haben kein Target mehr zum Mergen, also machen wir get_dummies direkt
combined = df_train_feats.copy()
combined['type'] = combined['type'].fillna('Unknown')
combined = pd.get_dummies(combined, columns=['type'], prefix='type')

type_features = [c for c in combined.columns if c.startswith('type_')]
final_features = features + type_features

print(f"🔢 Trainiere mit {len(final_features)} Features: {final_features}")

# ==========================================
# 4. TRAINING
# ==========================================
X_train = combined[final_features]
y_train = combined['label']

print("🏋️ Starte Training (HistGradientBoosting)...")
clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=8, random_state=42)
clf.fit(X_train, y_train)

# ==========================================
# 5. SPEICHERN
# ==========================================
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Modell speichern
joblib.dump(clf, os.path.join(model_dir, 'host_finder_model.pkl'))

# Feature-Liste speichern (DAMIT wir später wissen, was rein muss)
joblib.dump(final_features, os.path.join(model_dir, 'model_features.pkl'))

print(f"✅ FERTIG! Modell gespeichert in: {model_dir}")