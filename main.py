import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.units as u
import seaborn as sns
import re
import os
import joblib
import sys

from tqdm import tqdm
from dl import authClient as ac
from getpass import getpass
from dl import queryClient as qc
from astropy.coordinates import SkyCoord

USERNAME = "pousadel"
PASSWORD = "Dl9781408835005!"

# Pfade (angepasst an deine Struktur)
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Data', 'output')


# =============================================================
#  LOGIN
# =============================================================
def login_datalab() -> str:
    global USERNAME, PASSWORD
    if not USERNAME:
        USERNAME = input("Enter your Datalab username: ")
    if not PASSWORD:
        PASSWORD = getpass("Enter your Datalab password: ")
    try:
        token = ac.login(USERNAME, PASSWORD)
        print("✅ Logged in as", USERNAME)
        return token
    except Exception as e:
        print("❌ Login failed:", e)
        return ""


# =============================================================
#  LOAD + UTILS
# =============================================================
def load(filename: str, cols: list[str]) -> np.ndarray:
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"The following columns are missing in the file: {missing_cols}")
    return df[cols].to_numpy()


# =============================================================
#  FIND HOST CANDIDATES (SQL QUERY)
# =============================================================
def find_host_candidates(common_matrix: np.ndarray, radius_arcsec=30):
    def nmgy_to_mag(x):
        try:
            x = float(x)
            if x > 0: return 22.5 - 2.5 * np.log10(x)
            return np.nan
        except Exception:
            return np.nan

    global HOSTCANDS_COLS_SQL
    HOSTCANDS_COLS_SQL = [
        "SN_ID", "ra", "dec", "type", "sersic", "shape_r", "shape_e1", "shape_e2",
        "ls_id", "mag_g", "mag_r", "mag_i", "mag_z", "z_phot_mean_i"
    ]

    hosts = []
    noHost_yet = []

    for sn_id, ra, dec in tqdm(common_matrix, desc="🔎 Finding hosts", unit="SN"):
        ra = float(ra)
        dec = float(dec)
        r_deg = radius_arcsec / 3600.0
        delta_ra = r_deg / np.cos(np.radians(dec))

        min_ra, max_ra = ra - delta_ra, ra + delta_ra
        min_dec, max_dec = dec - r_deg, dec + r_deg

        query = f"""
            SELECT TOP 10000
                L.ra, L.dec, L.type, L.sersic,
                L.shape_r, L.shape_e1, L.shape_e2, L.ls_id,
                L.flux_g, L.flux_r, L.flux_i, L.flux_z,
                Z.z_phot_mean_i
            FROM ls_dr10.tractor AS L
            LEFT JOIN ls_dr10.photo_z AS Z ON L.ls_id = Z.ls_id
            WHERE L.ra BETWEEN {min_ra} AND {max_ra}
              AND L.dec BETWEEN {min_dec} AND {max_dec}
              AND L.ra_ivar > 0
              AND L.type IN ('REX', 'EXP', 'DEV', 'COMP', 'SER')
              AND L.brick_primary = 1
        """

        try:
            result = qc.query(sql=query, fmt="pandas")
        except Exception:
            continue

        if len(result) > 0:
            result["mag_g"] = result["flux_g"].apply(nmgy_to_mag)
            result["mag_r"] = result["flux_r"].apply(nmgy_to_mag)
            result["mag_i"] = result["flux_i"].apply(nmgy_to_mag)
            result["mag_z"] = result["flux_z"].apply(nmgy_to_mag)

            for _, host in result.iterrows():
                hosts.append([
                    sn_id,
                    host["ra"], host["dec"], host["type"], host["sersic"],
                    host["shape_r"], host["shape_e1"], host["shape_e2"],
                    host["ls_id"],
                    host["mag_g"], host["mag_r"], host["mag_i"], host["mag_z"],
                    host["z_phot_mean_i"]
                ])
        else:
            noHost_yet.append([sn_id, ra, dec])

    return np.array(hosts), np.array(noHost_yet)


# =============================================================
#  DLR CALCULATION
# =============================================================
def calculate_dlr(sn_ra, sn_dec, host_ra, host_dec, shape_r, e1, e2, host_type):
    sn_coord = SkyCoord(sn_ra, sn_dec, unit="deg")
    host_coord = SkyCoord(host_ra, host_dec, unit="deg")
    d_arcsec = sn_coord.separation(host_coord).arcsec

    if host_type == "REX":
        DLR = abs(shape_r)
    else:
        ellip = np.abs(np.sqrt(e1 ** 2 + e2 ** 2))
        a_b = (1 + ellip) / (1 - ellip) if ellip < 1 else 0.1
        theta = 0.5 * np.arctan2(e2, e1)
        theta_deg = 90 - theta * u.rad.to(u.deg)
        theta_rad = (theta_deg * u.deg).to(u.rad).value

        gamma = np.arctan2(
            (sn_coord.dec.value - host_coord.dec.value),
            (np.cos(sn_coord.dec.to(u.rad).value) * (sn_coord.ra.value - host_coord.ra.value))
        )

        phi = theta_rad - gamma
        DLR = abs(shape_r) / np.sqrt((a_b * np.sin(phi)) ** 2 + (np.cos(phi)) ** 2)

    if DLR <= 0: return np.inf
    return d_arcsec / DLR


# =============================================================
#  ASSIGN HOSTS (Ohne Filterung)
# =============================================================
def assign_hosts(common_matrix: np.ndarray, hosts: np.ndarray, limit=9999):
    global ALL_HOSTS_COLS_DLR
    ALL_HOSTS_COLS_DLR = [
        "SN_ID", "SN_RA", "SN_DEC", "Host_RA", "Host_DEC", "d_DLR", "type",
        "ls_id", "mag_g", "mag_r", "mag_i", "mag_z", "z_phot_mean_i"
    ]

    all_hosts = []

    for sn in tqdm(common_matrix, desc="Assigning hosts", unit="SN"):
        sn_id, sn_ra, sn_dec = sn[0], float(sn[1]), float(sn[2])
        candidates = hosts[hosts[:, 0] == sn_id]
        if len(candidates) == 0: continue

        sn_results = []
        for cand in candidates:
            host_ra, host_dec = float(cand[1]), float(cand[2])
            host_type = cand[3]
            shape_r, e1, e2 = float(cand[5]), float(cand[6]), float(cand[7])
            ls_id = cand[8]
            mag_g, mag_r, mag_i, mag_z = cand[9], cand[10], cand[11], cand[12]
            z = cand[13]

            dlr_val = calculate_dlr(sn_ra, sn_dec, host_ra, host_dec, shape_r, e1, e2, host_type)
            if np.isinf(dlr_val): continue

            record = [sn_id, sn_ra, sn_dec, host_ra, host_dec,
                      dlr_val, host_type, ls_id, mag_g, mag_r, mag_i, mag_z, z]
            sn_results.append(record)

        if len(sn_results) == 0: continue

        # Sort by d_DLR just for structure, but keep ALL
        sn_results = np.array(sn_results, dtype=object)
        sn_results = sn_results[np.argsort(sn_results[:, 5].astype(float))]

        all_hosts.extend(sn_results.tolist())

    # Wir geben nur all_hosts zurück, da wir nichts filtern
    return np.array(all_hosts, dtype=object)


# =============================================================
#  ML FEATURE LOGIC (Lokal, wie im Training)
# =============================================================
def calculate_separation_vect(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_theta = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta)) * 3600


def add_features_local(df):
    df = df.copy()

    # 1. Spaltenumbenennung (Wichtig! DataLab -> Training Schema)
    if 'z_phot_mean_i' in df.columns and 'z_phot' not in df.columns:
        df.rename(columns={'z_phot_mean_i': 'z_phot'}, inplace=True)

    # 2. Separation
    df['separation'] = calculate_separation_vect(
        df['SN_RA'], df['SN_DEC'], df['Host_RA'], df['Host_DEC']
    )

    df = df.sort_values(['SN_ID', 'd_DLR'])

    # 3. Kontext Features (1:1 aus training.py)
    min_dlr = df.groupby('SN_ID')['d_DLR'].transform('min')
    df['d_DLR_diff'] = df['d_DLR'] - min_dlr

    min_sep = df.groupby('SN_ID')['separation'].transform('min')
    df['sep_diff'] = df['separation'] - min_sep

    min_mag = df.groupby('SN_ID')['mag_r'].transform('min')
    df['mag_diff'] = df['mag_r'] - min_mag

    z_col = 'z_phot' if 'z_phot' in df.columns else 'z_phot_mean_i'
    median_z = df.groupby('SN_ID')[z_col].transform('median')
    df['z_phot_diff'] = (df[z_col] - median_z).abs()

    df['n_candidates'] = df.groupby('SN_ID')['SN_ID'].transform('count')

    # 4. Encoding
    df = pd.get_dummies(df, columns=['type'], prefix='type')
    return df


# =============================================================
#  MAIN
# =============================================================
if __name__ == "__main__":
    token = login_datalab()
    if not token: exit(1)

    # 1. INPUT DATEI LADEN
    # Hier kannst du den Pfad hardcoden oder input() nutzen
    # Wir nehmen an: Data/input/supernovae.csv
    input_file = "Data/input/supernovae.csv"
    if not os.path.exists(input_file):
        print(f"❌ Input file missing: {input_file}")
        exit(1)

    sn_matrix = load(input_file, ['ztfname', 'ra', 'dec'])

    # Deduplizieren
    df_sn = pd.DataFrame(sn_matrix, columns=['ztfname', 'ra', 'dec'])
    df_unique = df_sn.drop_duplicates(subset='ztfname', keep='first')
    common_matrix = df_unique.to_numpy()

    # 2. KANDIDATEN FINDEN (SQL)
    hostcands, noHost_yet = find_host_candidates(common_matrix, 30)

    # 3. d_DLR BERECHNEN (Kein Limit, wir behalten alle)
    # limit=9999 sorgt dafür, dass nichts in 'd_dlr_larger_limit' fällt was wir nicht wollen
    all_hosts_array = assign_hosts(common_matrix, hostcands, limit=9999)

    # DataFrame erstellen
    df_cands = pd.DataFrame(all_hosts_array, columns=ALL_HOSTS_COLS_DLR)

    # Sicherstellen, dass Zahlen auch Zahlen sind
    cols_num = ["SN_RA", "SN_DEC", "Host_RA", "Host_DEC", "d_DLR", "mag_g", "mag_r", "mag_i", "mag_z", "z_phot_mean_i"]
    for c in cols_num:
        df_cands[c] = pd.to_numeric(df_cands[c], errors='coerce')

    # 4. ML MODELL ANWENDEN
    print("\n🤖 Lade Modell und bewerte Kandidaten...")

    model_path = os.path.join(MODEL_DIR, 'host_finder_model.pkl')
    feats_path = os.path.join(MODEL_DIR, 'model_features.pkl')

    if os.path.exists(model_path):
        clf = joblib.load(model_path)
        model_features = joblib.load(feats_path)

        # Features berechnen
        df_proc = add_features_local(df_cands)

        # Spalten auffüllen
        for col in model_features:
            if col not in df_proc.columns: df_proc[col] = 0

        X = df_proc[model_features]
        raw_probs = clf.predict_proba(X)[:, 1]

        df_cands['normalized_prob'] = raw_probs


        # Normalisierung (Summe=1)
        def normalize(g):
            s = g.sum()
            return g / s if s > 0 else g


        df_cands['normalized_prob'] = df_cands.groupby('SN_ID')['normalized_prob'].transform(normalize)
        df_cands['rank'] = df_cands.groupby('SN_ID')['normalized_prob'].rank(ascending=False, method='first')

    else:
        print("⚠️ WARNUNG: Kein Modell gefunden! Speichere ohne ML-Score.")

    # 5. SPEICHERN
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Aufräumen für finalen Output
    out_cols = [
        'SN_ID', 'Host_RA', 'Host_DEC', 'd_DLR',
        'mag_r', 'mag_g', 'mag_i', 'mag_z', 'z_phot_mean_i',
        'normalized_prob', 'rank'
    ]
    # Falls ML fehlte, gibt es diese Spalten nicht
    valid_cols = [c for c in out_cols if c in df_cands.columns]

    df_final = df_cands[valid_cols].sort_values(['SN_ID', 'rank'])

    out_path = os.path.join(OUTPUT_DIR, "CF_limit_hosts_combined_dr4_SCORED.csv")
    df_final.to_csv(out_path, index=False)

    print(f"✅ Fertig! Datei gespeichert: {out_path}")
    print(f"   Anzahl SNe: {df_final['SN_ID'].nunique()}")