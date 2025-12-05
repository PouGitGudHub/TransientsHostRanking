import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import astropy.units as u
import seaborn as sns
import re

from tqdm import tqdm
from dl import authClient as ac
from getpass import getpass
from dl import queryClient as qc
from astropy.coordinates import SkyCoord

USERNAME = "your_username"
PASSWORD = "your_password"


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
        print("Token:", token[:10] + "...")
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
            if x > 0:
                return 22.5 - 2.5 * np.log10(x)
            return np.nan
        except Exception:
            return np.nan

    # NEW: Define the column structure for the output
    global HOSTCANDS_COLS_SQL
    # CORRECTED: Changed z_phot to z_phot_mean_i
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

        min_ra = ra - delta_ra
        max_ra = ra + delta_ra
        min_dec = dec - r_deg
        max_dec = dec + r_deg

        # NEW: Updated query to JOIN photo_z and get all fluxes (g, r, i, z)
        # CORRECTED: Changed Z.z_phot to Z.z_phot_mean_i
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
        except Exception as e:
            print(f"\n❌ Query failed for SN {sn_id}: {e}")
            continue

        if len(result) > 0:
            # NEW: Calculate magnitudes for all bands
            result["mag_g"] = result["flux_g"].apply(nmgy_to_mag)
            result["mag_r"] = result["flux_r"].apply(nmgy_to_mag)
            result["mag_i"] = result["flux_i"].apply(nmgy_to_mag)
            result["mag_z"] = result["flux_z"].apply(nmgy_to_mag)

            for _, host in result.iterrows():
                # NEW: Append all new data in the correct order
                # CORRECTED: Appending host["z_phot_mean_i"]
                hosts.append([
                    sn_id,
                    host["ra"], host["dec"], host["type"], host["sersic"],
                    host["shape_r"], host["shape_e1"], host["shape_e2"],
                    host["ls_id"],
                    host["mag_g"], host["mag_r"], host["mag_i"], host["mag_z"],
                    host["z_phot_mean_i"] # This is the redshift
                ])
        else:
            noHost_yet.append([sn_id, ra, dec])

    return np.array(hosts), np.array(noHost_yet)


# =S============================================================
#  DLR CALCULATION
# =============================================================
def calculate_dlr(sn_ra, sn_dec, host_ra, host_dec, shape_r, e1, e2, host_type):
    sn_coord = SkyCoord(sn_ra, sn_dec, unit="deg")
    host_coord = SkyCoord(host_ra, host_dec, unit="deg")
    d_arcsec = sn_coord.separation(host_coord).arcsec

    # KEIN Filter für d_arcsec < 0.5"
    # Diese Version geht davon aus, dass die d_DLR-Logik
    # "Plops" korrekt als Hosts mit kleinem d_DLR behandelt.

    # Normale DLR-Berechnung
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

    # Wir behalten diesen Check, falls DLR (shape_r) 0 ist
    if DLR <= 0:
        return np.inf

    return d_arcsec / DLR


# =============================================================
#  ASSIGN HOSTS
# =============================================================
def assign_hosts(common_matrix: np.ndarray, hosts: np.ndarray, limit=4):
    # NEW: Define column structure for the output of this function
    global ALL_HOSTS_COLS_DLR
    # CORRECTED: Changed z_phot to z_phot_mean_i
    ALL_HOSTS_COLS_DLR = [
        "SN_ID", "SN_RA", "SN_DEC", "Host_RA", "Host_DEC", "d_DLR", "type",
        "ls_id", "mag_g", "mag_r", "mag_i", "mag_z", "z_phot_mean_i"
    ]

    d_dlr_smaller_limit = []
    d_dlr_larger_limit = []
    best_hosts = []
    all_hosts = []

    for sn in tqdm(common_matrix, desc="Assigning hosts", unit="SN"):
        sn_id, sn_ra, sn_dec = sn[0], float(sn[1]), float(sn[2])
        candidates = hosts[hosts[:, 0] == sn_id]
        if len(candidates) == 0:
            continue

        sn_results = []
        for cand in candidates:
            # NEW: Updated indices to match new hostcands structure
            host_ra, host_dec = float(cand[1]), float(cand[2])
            host_type = cand[3]
            # sersic = cand[4] # Not needed for DLR
            shape_r, e1, e2 = float(cand[5]), float(cand[6]), float(cand[7])
            ls_id = cand[8]
            mag_g, mag_r, mag_i, mag_z = cand[9], cand[10], cand[11], cand[12]
            z = cand[13]  # This is now z_phot_mean_i

            # Wir rufen die "einfache" Version von calculate_dlr auf
            dlr_val = calculate_dlr(sn_ra, sn_dec, host_ra, host_dec,
                                    shape_r, e1, e2, host_type)

            # Dieser Check fängt alle 'inf' ab (ungültige DLR)
            if np.isinf(dlr_val):
                continue

            # NEW: Record now includes all magnitudes and redshift
            # CORRECTED: 'z' variable now correctly holds z_phot_mean_i
            record = [sn_id, sn_ra, sn_dec, host_ra, host_dec,
                      dlr_val, host_type, ls_id, mag_g, mag_r, mag_i, mag_z, z]
            sn_results.append(record)

        if len(sn_results) == 0:
            continue

        sn_results = np.array(sn_results, dtype=object)
        sn_results = sn_results[np.argsort(sn_results[:, 5].astype(float))]

        all_hosts.extend(sn_results.tolist())
        best_hosts.append(sn_results[0].tolist())

        mask_small = sn_results[:, 5].astype(float) <= limit
        if np.any(mask_small):
            d_dlr_smaller_limit.extend(sn_results[mask_small].tolist())
        else:
            d_dlr_larger_limit.extend(sn_results.tolist())

    return (
        np.array(d_dlr_smaller_limit, dtype=object),
        np.array(d_dlr_larger_limit, dtype=object),
        np.array(best_hosts, dtype=object),
        np.array(all_hosts, dtype=object)
    )

# =============================================================
#  CF-Test
# =============================================================
def purity_efficiency_test_and_plot(best_hosts, best_hosts_sim_cut):
    """
    Berechnet Efficiency & Purity vs. d_DLR, plottet sie und gibt den optimalen Cut zurück.
    Nichts an der Logik verändert – 1:1 aus main übernommen.
    """

    # ---------------------------------------------------------------
    # Inputs: d_DLR arrays
    # ---------------------------------------------------------------
    d_dlr_real = best_hosts[:, 5]
    d_dlr_random = best_hosts_sim_cut[:, 5]

    # Sicherheits-Check: konvertiere alles zu float und entferne NaNs
    d_dlr_real = np.array(d_dlr_real, dtype=float)
    d_dlr_real = d_dlr_real[np.isfinite(d_dlr_real)]

    d_dlr_random = np.array(d_dlr_random, dtype=float)
    d_dlr_random = d_dlr_random[np.isfinite(d_dlr_random)]

    # ---------------------------------------------------------------
    # Berechnung von Efficiency und Purity über verschiedene Cuts
    # ---------------------------------------------------------------
    cuts = np.linspace(0, 30, 300)  # DLR-Bereich anpassen, falls nötig
    eff = []
    pur = []

    for c in cuts:
        n_real = np.sum(d_dlr_real < c)
        n_rand = np.sum(d_dlr_random < c)
        eff.append(n_real / len(d_dlr_real))
        pur.append(n_real / (n_real + n_rand) if (n_real + n_rand) > 0 else 0)

    eff = np.array(eff)
    pur = np.array(pur)

    # ---------------------------------------------------------------
    # Finde den Punkt für 98% Efficiency
    # ---------------------------------------------------------------
    cut_98 = np.interp(0.98, eff, cuts)
    pur_98 = np.interp(cut_98, cuts, pur)

    print(f"DLR cut for 98% efficiency: {cut_98:.2f}")
    print(f"Purity at that cut: {pur_98 * 100:.1f}%")

    # ---------------------------------------------------------------
    # Plot: Efficiency und Purity
    # ---------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(cuts, eff, label="Efficiency (real)", lw=2, color="#1f77b4")
    plt.plot(cuts, pur, label="Purity", lw=2, color="#ff7f0e")
    plt.axvline(cut_98, color="gray", ls="--", lw=1.5)
    plt.text(cut_98 + 0.3, 0.2, f"Cut = {cut_98:.2f}", rotation=90, color="gray")

    plt.xlabel("d_DLR cutoff")
    plt.ylabel("Fraction")
    plt.title("Purity–Efficiency Tradeoff vs. d_DLR")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return cut_98


# =============================================================
#  SAVE TO CSV
# =============================================================
def save_array(array: np.ndarray, columns: list[str], filename: str, folder: str = "Data"):
    path = f"{folder}/{filename}"
    pd.DataFrame(array, columns=columns).to_csv(path, index=False)
    print(f"✅ Results saved: {path}")


# =============================================================
#  MAIN
# =============================================================
if __name__ == "__main__":
    token = login_datalab()
    if not token:
        exit(1)

    sn_matrix = load(
        "/home/pouria/PycharmProjects/TransientsHostFinder/data/input/supernovae.csv",
        ['ztfname', 'ra', 'dec']
    )

    df_sn = pd.DataFrame(sn_matrix, columns=['ztfname', 'ra', 'dec'])
    df_unique = df_sn.drop_duplicates(subset='ztfname', keep='first')
    common_matrix = df_unique.to_numpy()

    hostcands, noHost_yet = find_host_candidates(common_matrix, 30)

    found_hosts_cand, d_dlr_larger_limit, best_hosts, all_hosts = assign_hosts(common_matrix, hostcands, limit=7)

    # Load simulated sample
    best_hosts_sim_cut = load(
        "/home/pouria/PycharmProjects/TransientsHostFinder/data/sim/best_hosts_sim_cut.csv",
        ["SN_ID", "SN_RA", "SN_DEC", "Host_RA", "Host_DEC", "d_DLR", "type", "gal_id"]
    )

    # Compute optimal cut (CF-test)
    cut_98 = purity_efficiency_test_and_plot(best_hosts, best_hosts_sim_cut)
    print(f"✅ Using CF-Test cut limit = {cut_98:.2f}")

    # =========================================================
    # 🧩 FINAL OUTPUT: ONE COMBINED CSV
    # =========================================================
    print("\n🛠 Erzeuge zusammengefasste Ergebnisdatei ...")

    limit = cut_98
    final_rows = []

    # NEW: Use the globally defined column lists
    df_dlr = pd.DataFrame(all_hosts, columns=ALL_HOSTS_COLS_DLR)

    # ----- CORRECTED SECTION -----
    # 1. Define the supplemental columns we want to add
    HOSTCANDS_SUPPLEMENTAL_COLS = ["SN_ID", "ls_id", "ra", "dec", "sersic", "shape_r", "shape_e1", "shape_e2"]

    # 2. Create the full df_hosts, then subset it to *only* supplemental data
    df_hosts_full = pd.DataFrame(hostcands, columns=HOSTCANDS_COLS_SQL)
    df_hosts = df_hosts_full[HOSTCANDS_SUPPLEMENTAL_COLS]
    # ----- END CORRECTED SECTION -----

    # NEW: Merge DLR data with host properties.
    # This merge will no longer create _x and _y columns for magnitudes/redshift
    df_merged = pd.merge(df_dlr, df_hosts, how="left",
                         left_on=["Host_RA", "Host_DEC", "ls_id", "SN_ID"],
                         right_on=["ra", "dec", "ls_id", "SN_ID"])

    # NEW: This entire loop implements the new logic
    for sn_id, group in df_merged.groupby("SN_ID"):
        group = group.copy()
        # Ensure d_DLR and mag_i are numeric for comparisons
        # This line will now work, as 'mag_i' is not duplicated
        group["d_DLR"] = pd.to_numeric(group["d_DLR"], errors='coerce')
        group["mag_i"] = pd.to_numeric(group["mag_i"], errors='coerce')

        # 1. Filter out all candidates above the CF test limit
        candidates_under_limit = group[group["d_DLR"] <= limit].sort_values("d_DLR")

        if len(candidates_under_limit) == 0:
            # No candidates below limit. Take best overall and mark as non-host.
            best_overall = group.sort_values("d_DLR").iloc[0:1].copy()
            if not best_overall.empty:
                best_overall["is_host"] = 0
                final_rows.append(best_overall)
            # (If group is empty, nothing is appended, will be caught later)

        elif len(candidates_under_limit) == 1:
            # Only one candidate below the limit. This is our host.
            candidates_under_limit = candidates_under_limit.copy()
            candidates_under_limit["is_host"] = 1
            final_rows.append(candidates_under_limit)

        elif len(candidates_under_limit) >= 2:
            # Two or more candidates. Apply the new tie-breaker logic.
            candidates_under_limit = candidates_under_limit.copy()

            rank1 = candidates_under_limit.iloc[0]
            rank2 = candidates_under_limit.iloc[1]
            diff = rank2["d_DLR"] - rank1["d_DLR"]

            # Set all to 0 first, we will assign 1 to the winner
            candidates_under_limit["is_host"] = 0

            if diff > 0.5:
                # d_DLR is decisive. Rank 1 wins.
                candidates_under_limit.iloc[0, candidates_under_limit.columns.get_loc("is_host")] = 1
            else:
                # Tie-breaker: compare mag_i of Rank 1 and Rank 2
                rank1_mag = rank1["mag_i"]
                rank2_mag = rank2["mag_i"]

                # Handle NaN magnitudes (fainter by default)
                if pd.isna(rank1_mag) and pd.isna(rank2_mag):
                    # Both NaN, default to d_DLR winner
                    candidates_under_limit.iloc[0, candidates_under_limit.columns.get_loc("is_host")] = 1
                elif pd.isna(rank1_mag):
                    # Rank 1 mag is NaN, so Rank 2 is "brighter"
                    candidates_under_limit.iloc[1, candidates_under_limit.columns.get_loc("is_host")] = 1
                elif pd.isna(rank2_mag):
                    # Rank 2 mag is NaN, so Rank 1 is "brighter"
                    candidates_under_limit.iloc[0, candidates_under_limit.columns.get_loc("is_host")] = 1
                elif rank1_mag <= rank2_mag:
                    # Rank 1 is brighter (or equal). Rank 1 wins.
                    candidates_under_limit.iloc[0, candidates_under_limit.columns.get_loc("is_host")] = 1
                else:
                    # Rank 2 is brighter. Rank 2 wins.
                    candidates_under_limit.iloc[1, candidates_under_limit.columns.get_loc("is_host")] = 1

            final_rows.append(candidates_under_limit)

    df_final = pd.concat(final_rows, ignore_index=True)

    # NEW: Add SNe that had no candidates in the query at all
    sn_with_hosts = df_final['SN_ID'].unique()
    sn_no_cands = df_unique[~df_unique['ztfname'].isin(sn_with_hosts)]

    null_rows_for_no_cands = []
    for idx, row in sn_no_cands.iterrows():
        null_row = {col: np.nan for col in df_final.columns}
        null_row["SN_ID"] = row['ztfname']
        null_row["SN_RA"] = row['ra']
        null_row["SN_DEC"] = row['dec']
        null_row["is_host"] = 0
        null_rows_for_no_cands.append(null_row)

    if null_rows_for_no_cands:
        df_final = pd.concat([df_final, pd.DataFrame(null_rows_for_no_cands)], ignore_index=True)

    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # Clean up merge-related columns ('ra', 'dec')
    if 'ra' in df_final.columns:
        df_final = df_final.drop(columns=['ra'])
    if 'dec' in df_final.columns:
        df_final = df_final.drop(columns=['dec'])


    # =========================================================
    #  SAVE FINAL CSV INTO INPUT DIRECTORY FOR ML PIPELINE
    # =========================================================

    output_path = "/home/pouria/PycharmProjects/TransientsHostFinder/data/input/CF_limit_hosts_combined_dr4.csv"
    df_final.to_csv(output_path, index=False)

    print(f"✅ Combined host-candidate file saved to:")
    print(f"   {output_path}")

    print(f"\n✅ Total SNe: {df_final['SN_ID'].nunique()}")
    print(f"✅ Total rows: {len(df_final)}")
    print("🟢 Done!")
