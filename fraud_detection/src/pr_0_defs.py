 # This .py program is created to define all the necessary functions/methods
 #    for this python project.
 #    This file is already part of the pr_0_common_imports.py file and needs only to
 #    be imported as follows: from pr_0_common_imports import pr_0_defs.
 #    The functions/methods can be called as follows: pr_0_defs.<def name (relevant parameters)>

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import chi2_contingency, kruskal
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, log_loss

# Distance between customer and merchant
def distance(df: pd.DataFrame) -> pd.DataFrame:
    res=df.copy()
    res['distance_to_merchant'] = np.sqrt(
        (res['lat'] - res['merch_lat']) ** 2 +
        (res['long'] - res['merch_long']) ** 2
    ) * 111  # Approximate km conversion
    return res


def generate_time_features_full(df: pd.DataFrame, trans_col:str =None, trans_time_col:str =None)->pd.DataFrame:
    df = df.copy()
    if trans_col:
        df['datetime'] = pd.to_datetime(df[trans_col])
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df['is_weekend'] = df['day'].isin([5, 6]).astype(int)

        df['time'] = df[trans_time_col].apply(
            lambda x: datetime.strptime(x, '%H:%M:%S').time()
        )
        df['hour'] = df['time'].apply(lambda x: x.hour).astype('int64')
        # df["minute"] = df['time'].apply(lambda x: x.minute)
        # df["second"] = df['time'].apply(lambda x: x.second)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)

    return df



def get_credit_card_company(card_number :int)->str:
    # Convert to string and remove spaces, dashes
    card_str = str(card_number).replace(' ', '').replace('-', '')

    # Remove non-digit characters
    # card_str = ''.join(filter(str.isdigit, card_str))
    card_str = ''.join(filter(lambda x: x.isdigit(), card_str))

    if not card_str:
        return "Invalid"

    # Get first digit and first two digits
    first_digit = card_str[0]
    first_two = card_str[:2] if len(card_str) >= 2 else card_str
    first_four = card_str[:4] if len(card_str) >= 4 else card_str

    # Visa: starts with 4
    if first_digit == '4':
        return "Visa"

    # Mastercard: starts with 51-55 or 2221-2720
    if first_two in ['51', '52', '53', '54', '55']:
        return "Mastercard"
    if len(card_str) >= 4:
        first_four_int = int(first_four)
        if 2221 <= first_four_int <= 2720:
            return "Mastercard"

    # American Express: starts with 34 or 37
    if first_two in ['34', '37']:
        return "americanexpress"

    # Discover: starts with 6011, 622126-622925, 644-649, or 65
    if first_four == '6011' or first_two == '65':
        return "discover"
    if len(card_str) >= 6:
        first_six_int = int(card_str[:6])
        if 622126 <= first_six_int <= 622925:
            return "discover"
    if len(card_str) >= 3:
        first_three_int = int(card_str[:3])
        if 644 <= first_three_int <= 649:
            return "discover"

    # Diners Club: starts with 300-305, 36, or 38
    if first_two in ['36', '38']:
        return "dinersclub"
    if len(card_str) >= 3:
        first_three_int = int(card_str[:3])
        if 300 <= first_three_int <= 305:
            return "dinersclub"

    # JCB: starts with 3528-3589 (expanded to include 3524-3527)
    if len(card_str) >= 4:
        first_four_int = int(first_four)
        if 3524 <= first_four_int <= 3589:
            return "jcb"

    # UnionPay: starts with 62
    if first_two == '62':
        return "unionpay"

    # Maestro: starts with 50, 56-69
    if first_two == '50':
        return "maestro"
    if len(first_two) == 2:
        first_two_int = int(first_two)
        if 56 <= first_two_int <= 69:
            # Could be Maestro or Discover, check more specific patterns
            if first_two not in ['65'] and first_four != '6011':
                return "maestro"

    # If no specific issuer found, return ISO/IEC 7812 industry category
    industry_mapping = {
        '1': 'airlines',
        '2': 'airlinesfinancialindustry',
        '3': 'travelentertainment',
        '4': 'bankingfinancial',
        '5': 'bankingfinancial',
        '6': 'merchandisingbankingfinancing',
        '7': 'petroleumfutureindustry',
        '8': 'healthcarecommunication',
        '9': 'nationalassignment'
    }
    return industry_mapping.get(first_digit, 'Unknown')



def association_tests(df :pd.DataFrame, col, col_):
    print(f"\ncol={col}, col_={col_}")
    # Chi square test of independence
    # This will tell you whether the distribution of fraud (0/1) differs significantly between categories.
    cont_table = pd.crosstab(df[col], df[col_])
    chi2, p, dof, expected = chi2_contingency(cont_table)

    print(f"Chi-square statistic = {chi2:.3f}")
    print(f"Degrees of freedom = {dof}")
    print(f"P-value = {p:.5f}")

    if p < 0.05:
        print(f"→ Significant difference: {col_} depends on {col}.")
    else:
        print(f"→ No significant difference: {col_} is similar across {col} values.")

    # Even if the Chi-square test is significant, you might want to know how strong the association is
    n = cont_table.sum().sum()
    cramers_v_ = np.sqrt(chi2 / (n * (min(cont_table.shape) - 1)))
    print(f"Cramér’s V = {cramers_v_:.3f}")

    groups = [df.loc[df[col]==cat, col_] for cat in df[col].unique()]
    stat, p = kruskal(*groups)
    print(f"Kruskal-Wallis statistic={stat:.3f}, p-value={p:.5f}")
    sp.posthoc_dunn(df, val_col=col_, group_col=col, p_adjust='bonferroni')

def cramers_v(x, y):
    """Cramér's V for categorical-categorical association"""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion.shape) - 1)))

def correlation_ratio(categories, values):
    """Correlation Ratio (η²) for numeric-categorical association"""
    cats = pd.Categorical(categories)
    groups = [values[cats == cat] for cat in cats.categories]
    # n = len(values)
    grand_mean = np.mean(values)
    ss_between = sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups])
    ss_total = sum((values - grand_mean)**2)
    return ss_between / ss_total if ss_total != 0 else 0

# ---------- Main dependency matrix function ----------
def dependency_matrix(rhs):
    """Compute association/dependency matrix for mixed-type DataFrame"""
    df = rhs.copy()
    # Detect column types
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    num_cols = df.select_dtypes(include=[np.number]).columns

    cols = df.columns
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
                continue

            # Numeric-Numeric → Pearson
            if col1 in num_cols and col2 in num_cols:
                matrix.loc[col1, col2] = df[[col1, col2]].corr(method='pearson').iloc[0,1]

            # Categorical-Categorical → Cramér’s V
            elif col1 in cat_cols and col2 in cat_cols:
                matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

            # Numeric-Categorical → Correlation ratio (η²)
            else:
                num_var = col1 if col1 in num_cols else col2
                cat_var = col2 if col1 in num_cols else col1
                matrix.loc[col1, col2] = correlation_ratio(df[cat_var], df[num_var])

    return matrix.astype(float)


def ssn_stratified_split(
        df: pd.DataFrame,
        group_col: str = "ssn",
        label_col: str = "is_fraud",
        train_frac: float = 0.8,
        dev_frac: float = 0.1,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Balanced split that:
    1. Keeps SSNs intact (no SSN appears in multiple splits)
    2. Maintains similar fraud rates across splits
    3. Does NOT consider time at all
    """

    df = df.copy()

    # Compute per-user fraud statistics
    user_stats = df.groupby(group_col).agg({
        label_col: ['count', 'sum', 'mean']
    }).reset_index()

    user_stats.columns = [group_col, 'n_txn', 'n_fraud', 'fraud_rate']

    # Create fraud rate strata for balancing
    user_stats['fraud_stratum'] = pd.qcut(
        user_stats['fraud_rate'].rank(method="first"),
        q=20,
        labels=False,
        duplicates='drop'
    )

    # Prepare split column
    user_stats['split'] = None

    rng = np.random.default_rng(random_state)

    # Split inside each stratum (randomly)
    for stratum in sorted(user_stats['fraud_stratum'].unique()):
        stratum_idx = user_stats[user_stats['fraud_stratum'] == stratum].index

        # Shuffle SSNs inside stratum
        shuffled = rng.permutation(stratum_idx)

        n = len(shuffled)
        train_end = int(n * train_frac)
        dev_end = train_end + int(n * dev_frac)

        user_stats.loc[shuffled[:train_end], 'split'] = 'train'
        user_stats.loc[shuffled[train_end:dev_end], 'split'] = 'dev'
        user_stats.loc[shuffled[dev_end:], 'split'] = 'test'

    # Merge splits back into the data
    df = df.merge(user_stats[[group_col, 'split']], on=group_col, how='left')

    X_train = df[df['split'] == 'train'].drop(columns=['split']).reset_index(drop=True)
    X_dev = df[df['split'] == 'dev'].drop(columns=['split']).reset_index(drop=True)
    X_test = df[df['split'] == 'test'].drop(columns=['split']).reset_index(drop=True)

    # Summary
    print("=" * 80)
    print("Balanced SSN Split Summary (No Time Involved)")
    print("=" * 80)

    for name, part in [("Train", X_train), ("Dev", X_dev), ("Test", X_test)]:
        fraud_rate = 100 * part[label_col].mean()
        n_users = part[group_col].nunique()
        print(f"{name:6}: {len(part):>8,} rows | {n_users:>6} users | Fraud: {fraud_rate:6.3f}%")

    # Leakage check
    print("\n" + "=" * 80)
    print("Data Integrity Checks")
    print("=" * 80)

    train_ssns = set(X_train[group_col])
    dev_ssns = set(X_dev[group_col])
    test_ssns = set(X_test[group_col])

    leak_td = len(train_ssns & dev_ssns)
    leak_tt = len(train_ssns & test_ssns)
    leak_dt = len(dev_ssns & test_ssns)

    print(f"Train-Dev overlap:  {leak_td} SSNs")
    print(f"Train-Test overlap: {leak_tt} SSNs (SHOULD BE 0)")
    print(f"Dev-Test overlap:   {leak_dt} SSNs (SHOULD BE 0)")

    if leak_tt == 0 and leak_dt == 0:
        print("✓ NO DATA LEAKAGE (SSN-level)")
    else:
        print("✗ DATA LEAKAGE DETECTED")

    return X_train, X_dev, X_test


def temporal_split_balanced(
        df: pd.DataFrame,
        group_col: str = "ssn",
        time_col: str = "unix_time",
        label_col: str = "is_fraud",
        train_frac: float = 0.8,
        dev_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split that:
    1. Keeps SSNs intact (no SSN appears in multiple splits)
    2. Maintains similar fraud rates across splits
    3. Preserves temporal order (old users -> train, new users -> test)
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    df = df.dropna(subset=[time_col])

    # Get user statistics
    user_stats = df.groupby(group_col).agg({
        time_col: ['min', 'median', 'max'],
        label_col: ['count', 'sum', 'mean']
    }).reset_index()

    user_stats.columns = [group_col, 'first_txn', 'median_txn', 'last_txn',
                          'n_txn', 'n_fraud', 'fraud_rate']

    # Sort users by MEDIAN time (when they were active)
    user_stats = user_stats.sort_values('median_txn').reset_index(drop=True)

    # Stratify by fraud rate to ensure balanced splits
    user_stats['fraud_stratum'] = pd.qcut(
        user_stats['fraud_rate'].rank(method='first'),
        q=20,  # 20 strata
        labels=False,
        duplicates='drop'
    )

    # Assign users to splits within each stratum
    user_stats['split'] = None

    for stratum in sorted(user_stats['fraud_stratum'].unique()):
        stratum_mask = user_stats['fraud_stratum'] == stratum
        stratum_users = user_stats[stratum_mask].index

        n = len(stratum_users)
        train_end = int(n * train_frac)
        dev_end = train_end + int(n * dev_frac)

        # Assign sequentially by time (oldest -> train, newest -> test)
        user_stats.loc[stratum_users[:train_end], 'split'] = 'train'
        user_stats.loc[stratum_users[train_end:dev_end], 'split'] = 'dev'
        user_stats.loc[stratum_users[dev_end:], 'split'] = 'test'

    # Map splits back to original data
    df = df.merge(user_stats[[group_col, 'split']], on=group_col, how='left')

    X_train = df[df['split'] == 'train'].drop(columns=['split']).reset_index(drop=True)
    X_dev = df[df['split'] == 'dev'].drop(columns=['split']).reset_index(drop=True)
    X_test = df[df['split'] == 'test'].drop(columns=['split']).reset_index(drop=True)

    # Summary
    print("=" * 80)
    print("Temporal Split Summary")
    print("=" * 80)

    for name, part in [("Train", X_train), ("Dev", X_dev), ("Test", X_test)]:
        fraud_rate = 100 * part[label_col].mean()
        n_users = part[group_col].nunique()
        print(f"{name:6}: {len(part):>8,} rows | {n_users:>6} users | Fraud: {fraud_rate:6.3f}%")

    # Leakage check
    print("\n" + "=" * 80)
    print("Data Integrity Checks")
    print("=" * 80)

    train_ssns = set(X_train[group_col])
    dev_ssns = set(X_dev[group_col])
    test_ssns = set(X_test[group_col])

    leak_td = len(train_ssns & dev_ssns)
    leak_tt = len(train_ssns & test_ssns)
    leak_dt = len(dev_ssns & test_ssns)

    print(f"Train-Dev overlap:  {leak_td} SSNs")
    print(f"Train-Test overlap: {leak_tt} SSNs (SHOULD BE 0)")
    print(f"Dev-Test overlap:   {leak_dt} SSNs (SHOULD BE 0)")

    if leak_tt == 0 and leak_dt == 0:
        print("✓ NO DATA LEAKAGE")
    else:
        print("✗ DATA LEAKAGE DETECTED")

    # Temporal check
    print("\n" + "=" * 80)
    print("Temporal Ordering (User Median Times)")
    print("=" * 80)
    print(f"Train: {X_train[time_col].min()} to {X_train[time_col].max()}")
    print(f"Dev:   {X_dev[time_col].min()} to {X_dev[time_col].max()}")
    print(f"Test:  {X_test[time_col].min()} to {X_test[time_col].max()}")

    return X_train, X_dev, X_test


def rolling_features(user_df :pd.DataFrame) -> pd.DataFrame:
    user_df = user_df.copy()
    user_df['unix_time'] = pd.to_datetime(user_df['unix_time'], unit='s', errors='coerce')

    # user_df = user_df.set_index("unix_time").sort_index()
    user_df = user_df.set_index("unix_time").sort_index()
    user_df['amt_over_user_avg'] = user_df['amt'] / (user_df['amt'].mean() + 1e-6)
    # user_df['days_since_last_txn'] = (user_df.index.to_series().diff().fillna(0) / (24 * 3600)).astype(float)
    user_df['days_since_last_txn'] = user_df.index.to_series().diff().dt.total_seconds().fillna(0) / (24 * 3600)
    # Transaction frequency (shifted to exclude current transaction)
    user_df["txn_count_last_7d"] = user_df["amt"].rolling("7D").count().shift(1)
    user_df["txn_count_last_30d"] = user_df["amt"].rolling("30D").count().shift(1)
    # Average spend (shifted)
    user_df["avg_amt_last_7d"] = user_df["amt"].rolling("7D").mean().shift(1)
    user_df["avg_amt_last_30d"] = user_df["amt"].rolling("30D").mean().shift(1)
    # Volatility (shifted)
    user_df["amt_std_last_7d"] = user_df["amt"].rolling("7D").std().shift(1)
    user_df["amt_std_last_30d"] = user_df["amt"].rolling("30D").std().shift(1)
    return user_df.reset_index()

def feat_eng_rolling(df :pd.DataFrame, merchant_stats:float =None, job_target_mean :float =None) -> pd.DataFrame:
    # --- Apply per-user rolling windows ---
    # df = df.groupby("ssn", group_keys=False,observed=True).apply(rolling_features)
    df = df.groupby("ssn", group_keys=False).apply(rolling_features)
    # --- Fill missing values with 0 ---
    cols = [
        "txn_count_last_7d", "txn_count_last_30d",
        "avg_amt_last_7d", "avg_amt_last_30d",
        "amt_std_last_7d", "amt_std_last_30d",
        "amt_over_user_avg", "days_since_last_txn"
    ]
    df[cols] = df[cols].fillna(0)
    print('rolling features done')
    # --- Derived ratios for anomaly detection ---
    df["txn_count_ratio_7d_30d"] = df["txn_count_last_7d"] / (df["txn_count_last_30d"] + 1e-6)
    df["avg_amt_ratio_7d_30d"] = df["avg_amt_last_7d"] / (df["avg_amt_last_30d"] + 1e-6)
    df["amt_std_ratio_7d_30d"] = df["amt_std_last_7d"] / (df["amt_std_last_30d"] + 1e-6)

    # --- Z-score of amount per user ---
    # For each SSN, compute (amt - mean) / std
    user_mean = df.groupby("ssn")["amt"].transform("mean")
    user_std = df.groupby("ssn")["amt"].transform("std")
    df["zscore_amt_per_user"] = (df["amt"] - user_mean) / (user_std + 1e-6)

    # --- Merchant-level aggregated behavior (use pre-computed or calculate) ---
    if merchant_stats is None:
        merchant_stats = (
            df.groupby("merchant",observed=True)
            .agg(
                merchant_avg_amt=("amt", "mean"),
                merchant_fraud_rate=("is_fraud", "mean"),
                merchant_txn_count=("amt", "count")
            )
            .reset_index()
        )
    df = df.merge(merchant_stats, on="merchant", how="left")
    # --- Amount relative to merchant average ---
    df["amt_over_merchant_avg"] = df["amt"] / (df["merchant_avg_amt"] + 1e-6)
    # --- User–merchant frequency ---
    df["user_merchant_frequency"] = (
        df.groupby(["ssn", "merchant"],observed=True)["merchant"]
        .transform("count")
    )
    df["user_total_txn"] = df.groupby("ssn")["merchant"].transform("count")
    df["user_merchant_freq_ratio"] = df["user_merchant_frequency"] / (df["user_total_txn"] + 1e-6)
    # --- Target encoding (use pre-computed or calculate) ---
    if job_target_mean is None:
        job_target_mean = df.groupby("job",observed=True)["is_fraud"].mean()

    df["job_te"] = df["job"].map(job_target_mean)
    job_avg = df.groupby("job", observed=True)["amt"].transform("mean")
    df["amt_over_job_avg"] = df["amt"] / (job_avg + 1e-6)
    df.drop(columns=["user_total_txn"], inplace=True)
    print('fraud detection features done')
    return df


def classification_metrics(y, yhat):
    prf1 = precision_recall_fscore_support(y,yhat)
    res = {'Accuracy': accuracy_score(y,yhat),
           'Precision':prf1[0][1],
           'Recall': prf1[1][1],
           'f1-score': prf1[2][1],
           'Log-loss': log_loss(y,yhat),
           'AUC': roc_auc_score(y,yhat)
          }
    return res