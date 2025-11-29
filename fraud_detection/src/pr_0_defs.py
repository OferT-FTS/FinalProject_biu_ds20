''' This .py program is created to define all the necessary functions/methods
    for this python project.
    This file is already part of the pr_0_common_imports.py file and needs only to
    be imported as follows: from pr_0_common_imports import pr_0_defs.
    The functions/methods can be called as follows: pr_0_defs.<def name (relevant parameters)>
'''

from pr_0_common_imports import (
    pd, np, sns, plt, pickle, LabelEncoder, os, great_circle, train_test_split,OrdinalEncoder, Tuple, datetime, RandomForestClassifier, roc_auc_score,
    ProfileReport, webbrowser, chi2_contingency, kruskal, sp, precision_recall_fscore_support, accuracy_score, log_loss
)

def save_csv(df :pd.DataFrame, path :str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"DataFrame saved to {path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")

def write_df_to_csv(df :pd.DataFrame, csv_file :str) -> None:
    if os.path.exists(csv_file) and os.path.isfile(csv_file):
        os.remove(csv_file)
        print(f"The file '{csv_file}' has been deleted.")
    else:
        print(f"The file '{csv_file}' does not exist or is not a file.")
    save_csv(df, csv_file)

def import_csv(file_path :str) -> pd.DataFrame:
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"CSV file loaded successfully. Shape: {df.shape}")
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty.")
            df = None
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file: {e}")
            df = None
        except UnicodeDecodeError as e:
            print(f"Encoding error: {e}")
            df = None
        except Exception as e:
            print(f"Unexpected error: {e}")
            df = None
    else:
        print(f"File not found: {file_path}")
        df = None
    return df

def save_pickle(df: pd.DataFrame, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_pickle(path)
        print(f"DataFrame saved to {path}")
    except Exception as e:
        print(f"Error saving DataFrame to pickle: {e}")

def write_df_to_pickle(df :pd.DataFrame, pickle_file :str) -> None:
    if os.path.exists(pickle_file):
        try:
            os.remove(pickle_file)
            print(f"File '{pickle_file}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting file '{pickle_file}': {e}")
    else:
        print(f"File '{pickle_file}' does not exist.")

    save_pickle(df, pickle_file)

def import_pickle(file_path :str) -> pd.DataFrame:
    try:
        df = pd.read_pickle(file_path)
        print("Pickle file loaded successfully.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        df = None
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Error loading pickle: {e}")
        df = None
    except Exception as e:
        print(f"Unexpected error: {e}")
        df = None
    return df

# Distance between customer and merchant
def distance(df: pd.DataFrame) -> pd.DataFrame:
    res=df.copy()
    res['distance_to_merchant'] = np.sqrt(
        (res['lat'] - res['merch_lat']) ** 2 +
        (res['long'] - res['merch_long']) ** 2
    ) * 111  # Approximate km conversion
    return res

def split_on_month(df: pd.DataFrame, mnth :int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = df[df['first_tx_month']< mnth]
    X_test = df[df['first_tx_month']>= mnth]
    return X_train, X_test

def time_based_split_with_drift_plot(df: pd.DataFrame, date_col: str, target_col: str, cutoff_date: str = None, test_size: float = 0.3,
    stratify: bool = False, freq: str = "M",  # 'M' for monthly, 'W' for weekly
    random_state: int = 42,show_plot: bool = True):
    """
    Perform a time-based train/test split with optional stratified sampling,
    and visualize fraud rate drift over time.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with date and target columns.
    date_col : str
        Name of datetime column.
    target_col : str
        Name of target (e.g., 'is_fraud').
    cutoff_date : str, optional
        Date to split on (YYYY-MM-DD). If None, uses test_size proportion.
    test_size : float, default=0.3
        Proportion for test set if no cutoff_date.
    stratify : bool, default=False
        Whether to apply stratified sampling within the test split.
    freq : str, default='M'
        Frequency for fraud rate trend ('M' = monthly, 'W' = weekly).
    random_state : int, default=42
        For reproducibility.
    show_plot : bool, default=True
        Whether to display the fraud drift chart.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # --- Time-based split ---
    if cutoff_date:
        cutoff_date = pd.to_datetime(cutoff_date)
        train_df = df[df[date_col] <= cutoff_date]
        test_df  = df[df[date_col] >  cutoff_date]
    else:
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df  = df.iloc[split_idx:]

    # --- Separate features/target ---
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col])
    y_test  = test_df[target_col]

    # --- Optional stratified sampling (mainly for test) ---
    if stratify and len(y_test.unique()) > 1:
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test,
            stratify=y_test,
            test_size=0.0,
            random_state=random_state
        )

    # --- Drift plot ---
    if show_plot:
        df["period"] = df[date_col].dt.to_period(freq).astype(str)
        fraud_rate = df.groupby("period")[target_col].mean()

        cutoff = cutoff_date if cutoff_date else df[date_col].iloc[int(len(df)*(1-test_size))]
        cutoff_str = pd.to_datetime(cutoff).strftime("%Y-%m-%d")

        plt.figure(figsize=(10, 5))
        plt.plot(fraud_rate.index, fraud_rate.values, marker="o", linewidth=2)
        plt.axvline(cutoff_str, color="red", linestyle="--", label=f"Split @ {cutoff_str}")
        plt.title(f"Fraud Rate Drift Over Time ({freq}-level)")
        plt.xlabel("Period")
        plt.ylabel("Fraud Rate")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    # --- Summary ---
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Train fraud rate: {y_train.mean():.4f}")
    print(f"Test fraud rate:  {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test

def delete_old_nonfraud(df, pct_to_drop=0.25, date_col="trans_date", id_col="ssn"):
    # Sort by ID and date
    df = df.sort_values([id_col, date_col])

    # Define inner function for group filtering
    def drop_old_nonfraud(group):
        nonfraud = group[group["is_fraud"] == 0]
        n_drop = int(len(nonfraud) * pct_to_drop)
        drop_idx = nonfraud.head(n_drop).index
        return group.drop(index=drop_idx)

    # Compute before/after stats
    orig_nonfraud = (df["is_fraud"] == 0).sum()

    # Apply per ID group
    df_filtered = df.groupby(id_col, group_keys=False)[df.columns].apply(drop_old_nonfraud)

    # Compute stats
    filtered_nonfraud = (df_filtered["is_fraud"] == 0).sum()
    pct_remaining = filtered_nonfraud / orig_nonfraud * 100

    print(f"Original rows: {df.shape[0]:,}")
    print(f"Filtered rows: {df_filtered.shape[0]:,}")
    print(f"Remaining non-fraud rows: {filtered_nonfraud:,} ({pct_remaining:.1f}% of original non-fraud)")

    return df_filtered

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

def encode_age_bins(df: pd.DataFrame, col:str="age", intv:int =5, method: str ="ordinal", drop_original: bool =False):
    # Define bins (start at 14, up to 100)
    bins = list(range(14, 105, intv))
    labels = [f"{i + 1}-{i + intv}" for i in bins[:-1]]

    # Cut into bins
    df["age_bins"] = pd.cut(df[col], bins=bins, labels=labels, right=True)

    if method == "ordinal":
        encoder = OrdinalEncoder(categories=[labels])
        df["age_bins_ord"] = encoder.fit_transform(df[["age_bins"]])

    elif method == "ohe":
        age_ohe = pd.get_dummies(df["age_bins"], prefix="age_bin")
        df = pd.concat([df, age_ohe], axis=1)

    else:
        raise ValueError("method must be 'ordinal' or 'ohe'")

    if drop_original:
        df = df.drop(columns=[col])

    return df

def get_city_to_region(df: pd.DataFrame, new_col_nm: str)->pd.DataFrame:
    # Create a mapping dictionary of cities to California regions
    city_region_mapping = {
        # San Diego County
        'San Diego': 'San Diego', 'Encinitas': 'San Diego', 'Oceanside': 'San Diego',
        'Escondido': 'San Diego', 'Carlsbad': 'San Diego', 'El Cajon': 'San Diego',
        'Chula Vista': 'San Diego', 'La Mesa': 'San Diego', 'San Ysidro': 'San Diego',
        'La Jolla': 'San Diego', 'Lakeside': 'San Diego', 'San Marcos': 'San Diego',
        'Coronado': 'San Diego', 'Poway': 'San Diego', 'Ramona': 'San Diego',
        'Valley Center': 'San Diego', 'Dulzura': 'San Diego', 'Camp Pendleton': 'San Diego',
        'Fallbrook': 'San Diego', 'Santee': 'San Diego', 'Spring Valley': 'San Diego',
        'Lemon Grove': 'San Diego', 'San Juan Capistrano': 'San Diego',

        # Orange County
        'Lake Forest': 'Orange County', 'Orange Cove': 'Orange County', 'Laguna Niguel': 'Orange County',
        'Mission Viejo': 'Orange County', 'San Juan Capistrano': 'Orange County',
        'Fountain Valley': 'Orange County', 'Tustin': 'Orange County', 'Corona Del Mar': 'Orange County',
        'Brea': 'Orange County', 'Newport Beach': 'Orange County', 'Placentia': 'Orange County',
        'Laguna Beach': 'Orange County', 'Yorba Linda': 'Orange County', 'Los Alamitos': 'Orange County',
        'Laguna Hills': 'Orange County', 'Dana Point': 'Orange County', 'Aliso Viejo': 'Orange County',
        'Midway City': 'Orange County', 'Orange': 'Orange County', 'Ladera Ranch': 'Orange County',
        'San Clemente': 'Orange County', 'Trabuco Canyon': 'Orange County',
        'Rancho Santa Margarita': 'Orange County',

        # Los Angeles County
        'Bell': 'Los Angeles', 'Monrovia': 'Los Angeles', 'Gardena': 'Los Angeles',
        'Santa Monica': 'Los Angeles', 'Stanton': 'Los Angeles', 'Valley Village': 'Los Angeles',
        'Redondo Beach': 'Los Angeles', 'Sherman Oaks': 'Los Angeles', 'Pacoima': 'Los Angeles',
        'Tujunga': 'Los Angeles', 'Sylmar': 'Los Angeles', 'Duarte': 'Los Angeles',
        'Walnut': 'Los Angeles', 'Marina Del Rey': 'Los Angeles', 'North Hollywood': 'Los Angeles',
        'Cerritos': 'Los Angeles', 'Northridge': 'Los Angeles', 'Van Nuys': 'Los Angeles',
        'La Crescenta': 'Los Angeles', 'Canyon Country': 'Los Angeles', 'Canoga Park': 'Los Angeles',
        'Wilmington': 'Los Angeles', 'Maywood': 'Los Angeles', 'Venice': 'Los Angeles',
        'Woodland Hills': 'Los Angeles', 'Panorama City': 'Los Angeles', 'Reseda': 'Los Angeles',
        'Rancho Palos Verdes': 'Los Angeles', 'Carson': 'Los Angeles',
        'La Canada Flintridge': 'Los Angeles', 'Valencia': 'Los Angeles', 'Winnetka': 'Los Angeles',
        'Chatsworth': 'Los Angeles', 'Hacienda Heights': 'Los Angeles', 'Lomita': 'Los Angeles',
        'San Pedro': 'Los Angeles', 'Encino': 'Los Angeles', 'Granada Hills': 'Los Angeles',
        'Studio City': 'Los Angeles', 'South El Monte': 'Los Angeles', 'North Hills': 'Los Angeles',
        'Beverly Hills': 'Los Angeles', 'Hermosa Beach': 'Los Angeles', 'South Pasadena': 'Los Angeles',
        'Alta Loma': 'Los Angeles', 'Manhattan Beach': 'Los Angeles', 'Artesia': 'Los Angeles',
        'San Fernando': 'Los Angeles', 'Altadena': 'Los Angeles', 'Huntington Park': 'Los Angeles',
        'Lawndale': 'Los Angeles', 'Sun Valley': 'Los Angeles', 'Newhall': 'Los Angeles',
        'Rowland Heights': 'Los Angeles', 'Santa Clarita': 'Los Angeles',
        'Stevenson Ranch': 'Los Angeles',

        # Inland Empire (Riverside & San Bernardino Counties)
        'Palm Desert': 'Inland Empire', 'Coachella': 'Inland Empire', 'Running Springs': 'Inland Empire',
        'La Quinta': 'Inland Empire', 'Victorville': 'Inland Empire', 'Winchester': 'Inland Empire',
        'Cathedral City': 'Inland Empire', 'Yucaipa': 'Inland Empire', 'Indio': 'Inland Empire',
        'Yucca Valley': 'Inland Empire', 'Mecca': 'Inland Empire', 'Desert Hot Springs': 'Inland Empire',
        'Palm Springs': 'Inland Empire', 'Phelan': 'Inland Empire', 'Thermal': 'Inland Empire',
        'Hesperia': 'Inland Empire', 'Thousand Palms': 'Inland Empire', 'Barstow': 'Inland Empire',
        'Wrightwood': 'Inland Empire', 'San Jacinto': 'Inland Empire', 'Bloomington': 'Inland Empire',
        'Mira Loma': 'Inland Empire', 'Twentynine Palms': 'Inland Empire', 'Big Bear City': 'Inland Empire',
        'Big Bear Lake': 'Inland Empire', 'Crestline': 'Inland Empire', 'Rancho Mirage': 'Inland Empire',

        # Central Valley
        'Parlier': 'Central Valley', 'Sanger': 'Central Valley', 'Lodi': 'Central Valley',
        'Nuevo': 'Central Valley', 'Winton': 'Central Valley', 'El Nido': 'Central Valley',
        'Hughson': 'Central Valley', 'Escalon': 'Central Valley', 'Armona': 'Central Valley',
        'Dos Palos': 'Central Valley', 'Manteca': 'Central Valley', 'Livingston': 'Central Valley',
        'La Grange': 'Central Valley', 'Lindsay': 'Central Valley', 'Selma': 'Central Valley',
        'Butte City': 'Central Valley', 'Mc Farland': 'Central Valley', 'Kerman': 'Central Valley',
        'Riverdale': 'Central Valley', 'Reedley': 'Central Valley', 'Dinuba': 'Central Valley',
        'Salida': 'Central Valley', 'Denair': 'Central Valley', 'Catheys Valley': 'Central Valley',
        'Crows Landing': 'Central Valley', 'Woodlake': 'Central Valley', 'Fowler': 'Central Valley',
        'Pixley': 'Central Valley', 'Newman': 'Central Valley', 'Lathrop': 'Central Valley',
        'Sloughhouse': 'Central Valley', 'Coulterville': 'Central Valley', 'Bradley': 'Central Valley',
        'Clements': 'Central Valley', 'Galt': 'Central Valley', 'Huron': 'Central Valley',
        'Gridley': 'Central Valley', 'Prather': 'Central Valley', 'Auberry': 'Central Valley',
        'Shafter': 'Central Valley', 'Avenal': 'Central Valley', 'Arvin': 'Central Valley',

        # Sacramento/Gold Country
        'Sacramento': 'Gold Country', 'Felton': 'Gold Country', 'Foresthill': 'Gold Country',
        'Lincoln': 'Gold Country', 'Citrus Heights': 'Gold Country', 'Antelope': 'Gold Country',
        'Folsom': 'Gold Country', 'Dixon': 'Gold Country', 'Elk Grove': 'Gold Country',
        'Roseville': 'Gold Country', 'Davis': 'Gold Country', 'Napa': 'Gold Country',
        'Placerville': 'Gold Country', 'Knights Landing': 'Gold Country', 'El Macero': 'Gold Country',
        'Auburn': 'Gold Country', 'Pittsburg': 'Gold Country', 'Grass Valley': 'Gold Country',
        'El Dorado Hills': 'Gold Country', 'Plymouth': 'Gold Country', 'Arnold': 'Gold Country',
        'Shingle Springs': 'Gold Country', 'Rio Linda': 'Gold Country', 'Rocklin': 'Gold Country',
        'Rancho Cordova': 'Gold Country', 'Dublin': 'Gold Country', 'Burson': 'Gold Country',
        'Carmichael': 'Gold Country', 'Georgetown': 'Gold Country', 'Sutter Creek': 'Gold Country',
        'Granite Bay': 'Gold Country', 'Pine Grove': 'Gold Country', 'Alamo': 'Gold Country',
        'Olivehurst': 'Gold Country', 'Woodland': 'Gold Country', 'Orangevale': 'Gold Country',
        'Pioneer': 'Gold Country', 'Elverta': 'Gold Country', 'Newcastle': 'Gold Country',
        'Magalia': 'Gold Country', 'Fair Oaks': 'Gold Country', 'Nevada City': 'Gold Country',
        'Acton': 'Gold Country', 'Valley Springs': 'Gold Country', 'Live Oak': 'Gold Country',
        'Acampo': 'Gold Country', 'Sonora': 'Gold Country', 'Fulton': 'Gold Country',
        'West Sacramento': 'Gold Country', 'Yuba City': 'Gold Country', 'Hercules': 'Gold Country',
        'North Highlands': 'Gold Country', 'Hamilton City': 'Gold Country', 'Marysville': 'Gold Country',
        'Oroville': 'Gold Country', 'Tuolumne': 'Gold Country',

        # Bay Area
        'Danville': 'Bay Area', 'Hollister': 'Bay Area', 'San Anselmo': 'Bay Area',
        'San Rafael': 'Bay Area', 'San Carlos': 'Bay Area', 'Pacifica': 'Bay Area',
        'Millbrae': 'Bay Area', 'Clayton': 'Bay Area', 'Burlingame': 'Bay Area',
        'Saratoga': 'Bay Area', 'Greenbrae': 'Bay Area', 'Moraga': 'Bay Area',
        'Emeryville': 'Bay Area', 'Corte Madera': 'Bay Area', 'El Cerrito': 'Bay Area',
        'Mill Valley': 'Bay Area', 'San Lorenzo': 'Bay Area', 'San Juan Bautista': 'Bay Area',
        'Newark': 'Bay Area', 'Morgan Hill': 'Bay Area', 'Gilroy': 'Bay Area',
        'San Martin': 'Bay Area', 'Half Moon Bay': 'Bay Area',

        # Central Coast
        'Ventura': 'Central Coast', 'Moorpark': 'Central Coast', 'Santa Paula': 'Central Coast',
        'Goleta': 'Central Coast', 'Santa Cruz': 'Central Coast', 'Brooks': 'Central Coast',
        'La Honda': 'Central Coast', 'Woodbridge': 'Central Coast', 'Pacific Grove': 'Central Coast',
        'Freedom': 'Central Coast', 'Carpinteria': 'Central Coast', 'Cloverdale': 'Central Coast',
        'South San Francisco': 'Central Coast', 'Rohnert Park': 'Central Coast', 'Vista': 'Central Coast',
        'Solvang': 'Central Coast', 'Sebastopol': 'Central Coast', 'Windsor': 'Central Coast',
        'Arroyo Grande': 'Central Coast', 'Pearblossom': 'Central Coast', 'American Canyon': 'Central Coast',
        'Greenfield': 'Central Coast', 'Ojai': 'Central Coast', 'Aptos': 'Central Coast',
        'Healdsburg': 'Central Coast', 'Guadalupe': 'Central Coast', 'Marina': 'Central Coast',

        # North Coast
        'Kneeland': 'North Coast', 'Mckinleyville': 'North Coast', 'Redway': 'North Coast',
        'Carlotta': 'North Coast', 'Blue Lake': 'North Coast', 'Ferndale': 'North Coast',
        'Whitethorn': 'North Coast', 'Jenner': 'North Coast', 'Redwood Valley': 'North Coast',
        'Montgomery Creek': 'North Coast', 'Guerneville': 'North Coast', 'Albion': 'North Coast',
        'Greenwood': 'North Coast', 'Duncans Mills': 'North Coast',

        # High Sierra
        'Lake Arrowhead': 'High Sierra', 'South Lake Tahoe': 'High Sierra',
        'Mountain Center': 'High Sierra', 'Mammoth Lakes': 'High Sierra', 'Bishop': 'High Sierra',

        # Shasta Cascade
        'Stirling City': 'Shasta Cascade', 'Cobb': 'Shasta Cascade', 'Alturas': 'Shasta Cascade',
        'Etna': 'Shasta Cascade', 'Corning': 'Shasta Cascade',

        # Imperial Valley (Desert)
        'Holtville': 'Imperial Valley', 'Seeley': 'Imperial Valley', 'El Centro': 'Imperial Valley',
        'Imperial': 'Imperial Valley', 'Calipatria': 'Imperial Valley', 'Calexico': 'Imperial Valley',
        'Heber': 'Imperial Valley', 'Brawley': 'Imperial Valley',

        # Ventura/Santa Barbara (sometimes grouped with Central Coast or separate)
        'Thousand Oaks': 'Ventura County', 'Oxnard': 'Ventura County', 'Lompoc': 'Ventura County',
        'Port Hueneme': 'Ventura County', 'Westlake Village': 'Ventura County',
        'Oak Park': 'Ventura County', 'Simi Valley': 'Ventura County', 'Agoura Hills': 'Ventura County',
        'Camarillo': 'Ventura County', 'Port Hueneme Cbc Base': 'Ventura County',

        # Lancaster/Palmdale (Antelope Valley)
        'Palmdale': 'Antelope Valley', 'Lancaster': 'Antelope Valley', 'Rosamond': 'Antelope Valley',
        'Edwards': 'Antelope Valley',

        # Misc/Special Cases
        'Vacaville': 'Bay Area', 'Apple Valley': 'Inland Empire', 'National City': 'San Diego',
        'Santa Rosa': 'Bay Area', 'Jackson': 'Gold Country', 'Calabasas': 'Los Angeles',
        'Rio Vista': 'Gold Country', 'Atascadero': 'Central Coast', 'San Quentin': 'Bay Area',
        'Palos Verdes Peninsula': 'Los Angeles', 'Mission Hills': 'Los Angeles',
        'Glenhaven': 'Gold Country', 'Hidden Valley Lake': 'Gold Country',

        # Additional Major Cities - Bay Area
        'Fremont': 'Bay Area', 'San Jose': 'Bay Area', 'Oakland': 'Bay Area',
        'Alameda': 'Bay Area', 'Antioch': 'Bay Area', 'San Ramon': 'Bay Area',
        'San Francisco': 'Bay Area', 'Sunnyvale': 'Bay Area', 'San Mateo': 'Bay Area',
        'Los Altos': 'Bay Area', 'Milpitas': 'Bay Area', 'Hayward': 'Bay Area',
        'Concord': 'Bay Area', 'Palo Alto': 'Bay Area', 'Daly City': 'Bay Area',
        'Union City': 'Bay Area', 'Fairfield': 'Bay Area', 'Berkeley': 'Bay Area',
        'Campbell': 'Bay Area', 'Redwood City': 'Bay Area', 'Petaluma': 'Bay Area',
        'Pleasant Hill': 'Bay Area', 'Castro Valley': 'Bay Area', 'Martinez': 'Bay Area',
        'Cupertino': 'Bay Area', 'Walnut Creek': 'Bay Area', 'Santa Clara': 'Bay Area',
        'Livermore': 'Bay Area', 'Brentwood': 'Bay Area', 'San Leandro': 'Bay Area',
        'Mountain View': 'Bay Area', 'Lafayette': 'Bay Area', 'Novato': 'Bay Area',
        'Menlo Park': 'Bay Area', 'Benicia': 'Bay Area', 'Vallejo': 'Bay Area',
        'Pleasanton': 'Bay Area', 'San Pablo': 'Bay Area', 'Richmond': 'Bay Area',

        # Additional Major Cities - Los Angeles
        'Los Angeles': 'Los Angeles', 'Whittier': 'Los Angeles', 'Norwalk': 'Los Angeles',
        'Compton': 'Los Angeles', 'Long Beach': 'Los Angeles', 'Glendale': 'Los Angeles',
        'Pasadena': 'Los Angeles', 'Lakewood': 'Los Angeles', 'Culver City': 'Los Angeles',
        'Inglewood': 'Los Angeles', 'Torrance': 'Los Angeles', 'Arcadia': 'Los Angeles',
        'West Covina': 'Los Angeles', 'La Puente': 'Los Angeles', 'Hawthorne': 'Los Angeles',
        'Baldwin Park': 'Los Angeles', 'El Monte': 'Los Angeles', 'Paramount': 'Los Angeles',
        'Downey': 'Los Angeles', 'Burbank': 'Los Angeles', 'Monterey Park': 'Los Angeles',
        'Lynwood': 'Los Angeles', 'Pico Rivera': 'Los Angeles', 'Montebello': 'Los Angeles',
        'Alhambra': 'Los Angeles', 'La Verne': 'Los Angeles', 'San Dimas': 'Los Angeles',
        'Temple City': 'Los Angeles', 'Azusa': 'Los Angeles', 'Covina': 'Los Angeles',
        'Rosemead': 'Los Angeles', 'South Gate': 'Los Angeles', 'Glendora': 'Los Angeles',
        'Pomona': 'Los Angeles', 'San Gabriel': 'Los Angeles', 'Claremont': 'Los Angeles',

        # Additional Major Cities - Orange County
        'Santa Ana': 'Orange County', 'Fullerton': 'Orange County', 'Anaheim': 'Orange County',
        'Buena Park': 'Orange County', 'Huntington Beach': 'Orange County',
        'Costa Mesa': 'Orange County', 'Garden Grove': 'Orange County', 'Irvine': 'Orange County',
        'Cypress': 'Orange County', 'La Palma': 'Orange County', 'Westminster': 'Orange County',
        'La Habra': 'Orange County', 'La Mirada': 'Orange County',

        # Additional Major Cities - Inland Empire
        'San Bernardino': 'Inland Empire', 'Norco': 'Inland Empire', 'Riverside': 'Inland Empire',
        'Lake Elsinore': 'Inland Empire', 'Chino Hills': 'Inland Empire', 'Banning': 'Inland Empire',
        'Rancho Cucamonga': 'Inland Empire', 'Diamond Bar': 'Inland Empire', 'Redlands': 'Inland Empire',
        'Ontario': 'Inland Empire', 'Upland': 'Inland Empire', 'Temecula': 'Inland Empire',
        'Corona': 'Inland Empire', 'Perris': 'Inland Empire', 'Rialto': 'Inland Empire',
        'Colton': 'Inland Empire', 'Fontana': 'Inland Empire', 'Moreno Valley': 'Inland Empire',
        'Beaumont': 'Inland Empire', 'Hemet': 'Inland Empire', 'Chino': 'Inland Empire',
        'Menifee': 'Inland Empire', 'Murrieta': 'Inland Empire', 'Highland': 'Inland Empire',
        'Montclair': 'Inland Empire',

        # Additional Major Cities - Central Valley
        'Tracy': 'Central Valley', 'Stockton': 'Central Valley', 'Visalia': 'Central Valley',
        'Clovis': 'Central Valley', 'Modesto': 'Central Valley', 'Bakersfield': 'Central Valley',
        'Merced': 'Central Valley', 'Riverbank': 'Central Valley', 'Soledad': 'Central Valley',
        'Delano': 'Central Valley', 'Fresno': 'Central Valley', 'Los Banos': 'Central Valley',
        'Tulare': 'Central Valley', 'Madera': 'Central Valley', 'Porterville': 'Central Valley',
        'Turlock': 'Central Valley', 'Atwater': 'Central Valley', 'Hanford': 'Central Valley',
        'Oakdale': 'Central Valley', 'Lemoore': 'Central Valley', 'Ceres': 'Central Valley',
        'Wasco': 'Central Valley', 'Paradise': 'Central Valley',

        # Additional Major Cities - Central Coast
        'Salinas': 'Central Coast', 'San Luis Obispo': 'Central Coast', 'Santa Barbara': 'Central Coast',
        'Carmel': 'Central Coast', 'Santa Maria': 'Central Coast', 'Paso Robles': 'Central Coast',
        'Watsonville': 'Central Coast', 'Seaside': 'Central Coast', 'Morro Bay': 'Central Coast',
        'Capitola': 'Central Coast',

        # Additional Major Cities - North Coast
        'Crescent City': 'North Coast', 'Eureka': 'North Coast', 'Fort Bragg': 'North Coast',
        'Ukiah': 'North Coast',

        # Additional Major Cities - Shasta Cascade
        'Chico': 'Shasta Cascade', 'Anderson': 'Shasta Cascade', 'Redding': 'Shasta Cascade',

        # Additional Major Cities - Gold Country
        'Clearlake': 'Gold Country', 'Bellflower': 'Gold Country'
    }

    print(f"Total cities mapped: {len(city_region_mapping)}")
    df[new_col_nm] = df['city'].map(city_region_mapping)
    return df

def get_credit_card_company(card_number :int)->str:
    # Convert to string and remove spaces, dashes
    card_str = str(card_number).replace(' ', '').replace('-', '')

    # Remove non-digit characters
    card_str = ''.join(filter(str.isdigit, card_str))

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

def eda_report_profile_rep(df : pd.DataFrame) -> None:
    report = ProfileReport(df, title="Fraud Detection EDA", explorative=True)
    report.to_file("fraud_report.html")
    print('eda profile report created')
    webbrowser.open('file://' + os.path.realpath("fraud_report.html"))

# def eda_report_autoviz(csv_file : str, target_col_ :str) -> None:
#     # eda_report_profile_rep(df)
#
#     # Initialize AutoViz
#     AV = AutoViz_Class()
#
#     # ====== CONFIGURATION ======
#     csv_file = csv_file   # 👈 your dataset
#     target_column = target_col_           # or "" if unknown
#     chart_format = "html"                # 'svg', 'png', or 'html'
#     max_rows = 150000
#     max_cols = 30
#
#     # ====== RUN AUTOVIZ ======
#     print("🔍 Running AutoViz... this may take a minute depending on data size.\n")
#
#     df_cleaned = AV.AutoViz(
#         filename=csv_file,
#         sep=",",
#         depVar=target_column,
#         dfte=None,
#         header=0,
#         verbose=1,
#         lowess=False,
#         chart_format=chart_format,
#         max_rows_analyzed=max_rows,
#         max_cols_analyzed=max_cols
#     )
#
#     # ====== SHOW RESULTS ======
#     # AutoViz saves HTML reports in a folder like: AutoViz_Plots/
#     output_folder = os.path.join(os.getcwd(), "AutoViz_Plots")
#
#     if os.path.exists(output_folder):
#         print(f"\n✅ AutoViz completed! Reports saved in:\n{output_folder}")
#
#         # Try to find and open one of the HTML files automatically
#         html_files = [f for f in os.listdir(output_folder) if f.endswith(".html")]
#         if html_files:
#             report_path = os.path.join(output_folder, html_files[0])
#             print(f"📂 Opening report: {report_path}")
#             webbrowser.open(f"file:///{report_path}")
#         else:
#             print("⚠️ No HTML files found in AutoViz_Plots/")
#     else:
#         print("⚠️ No output folder found. Did AutoViz complete successfully?")
#
#     # ====== OPTIONAL: Inspect cleaned DataFrame ======
#     print("\n🧾 Cleaned DataFrame preview:")
#     print(df_cleaned.head())

def auto_feature_selection_train(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    corr_threshold: float = 0.9,
    top_n_list: list = [10, 20, 30, 40, 50],
    feature_csv_path: str = "best_features.csv",
    model_path: str = "fraud_model.pkl",
):
    """
    Automatically selects optimal top-N features, retrains final model,
    and saves both the best feature list and model.
    """

    df = df.copy()
    df = df.dropna(subset=[target_col])

    # --- 1. Encode categoricals ---
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # --- 2. Train/test split ---
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- 3. Baseline model ---
    base_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, n_jobs=-1,
        random_state=42, class_weight="balanced_subsample"
    )
    base_model.fit(X_train, y_train)
    base_auc = roc_auc_score(y_test, base_model.predict_proba(X_test)[:, 1])
    print(f"\n📊 Baseline AUC (all features): {base_auc:.4f}")

    # --- 4. Feature importances ---
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": base_model.feature_importances_
    }).sort_values("importance", ascending=False)

    # --- 5. Drop correlated features ---
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    print(f"🧹 Dropped {len(to_drop)} correlated features (> {corr_threshold})")

    # --- 6. Test several Top-N thresholds ---
    auc_scores = []
    for n in top_n_list:
        top_features = importances.head(n)["feature"].tolist()
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]

        model = RandomForestClassifier(
            n_estimators=300, max_depth=12, n_jobs=-1,
            random_state=42, class_weight="balanced_subsample"
        )
        model.fit(X_train_top, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_top)[:, 1])
        auc_scores.append(auc)
        print(f"Top {n:>3} features → AUC: {auc:.4f}")

    # --- 7. Plot AUC vs Top-N features ---
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=top_n_list, y=auc_scores, marker="o", linewidth=2)
    plt.title("AUC vs Number of Top Features")
    plt.xlabel("Number of Features (Top N)")
    plt.ylabel("ROC-AUC")
    plt.grid(True)
    plt.show()

    # --- 8. Identify best N ---
    best_idx = int(np.argmax(auc_scores))
    best_n = top_n_list[best_idx]
    best_auc = auc_scores[best_idx]
    best_features = importances.head(best_n)["feature"].tolist()

    print(f"\n🏆 Optimal number of features: {best_n}")
    print(f"✅ Best AUC: {best_auc:.4f}")
    print(f"Top {best_n} features:\n{best_features}")

    # --- 9. Save best features ---
    pd.Series(best_features, name="best_features").to_csv(feature_csv_path, index=False)
    print(f"\n💾 Saved best features to: {feature_csv_path}")

    # --- 10. Train final model on best features ---
    final_model = RandomForestClassifier(
        n_estimators=400, max_depth=14, n_jobs=-1,
        random_state=42, class_weight="balanced_subsample"
    )
    final_model.fit(X[best_features], y)

    # --- 11. Save model to disk ---
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"💾 Final model saved to: {model_path}")

    # --- 12. Return summary ---
    return {
        "base_auc": base_auc,
        "best_auc": best_auc,
        "best_n": best_n,
        "best_features": best_features,
        "to_drop": to_drop,
        "importances": importances,
        "auc_scores": auc_scores,
        "top_n_list": top_n_list
    }

def build_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- 1. Basic cleaning ---
    df = df.copy()
    df["trans_date"] = pd.to_datetime(df["trans_date"])
    df = df.sort_values(["ssn", "trans_date"]).reset_index(drop=True)

    # --- 3. Transaction amount statistics ---
    df["txn_amt_zscore"] = df.groupby("ssn")["amt"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-6)
    )
    df["avg_txn_amt"] = df.groupby("ssn")["amt"].transform("mean")
    df["std_txn_amt"] = df.groupby("ssn")["amt"].transform("std")

    # --- 6. Location ---
    df["is_new_city"] = df.groupby("ssn")["city"].transform(lambda x: ~x.duplicated()).astype(int)

    if {"latitude", "longitude"}.issubset(df.columns):
        def calc_distance(sub):
            coords = list(zip(sub["latitude"], sub["longitude"]))
            dist = [0]
            for i in range(1, len(coords)):
                dist.append(great_circle(coords[i-1], coords[i]).kilometers)
            return pd.Series(dist, index=sub.index)
        # df["distance_from_last_txn"] = df.groupby("ssn", group_keys=False).apply(calc_distance)
        df["distance_from_last_txn"] = (
            df.groupby("ssn", group_keys=False)
            .apply(calc_distance)
            .reset_index(level=0, drop=True)  # <--- drops the group index for alignment
        )
    else:
        df["distance_from_last_txn"] = np.nan

    # --- 7. Merchant/category-level global features ---
    df["is_new_merchant"] = df.groupby("ssn")["merchant"].transform(lambda x: ~x.duplicated()).astype(int)
    merchant_fraud_rate = df.groupby("merchant")["is_fraud"].mean()
    category_fraud_rate = df.groupby("category")["is_fraud"].mean()
    avg_cat_amt = df.groupby("category")["amt"].mean()

    df["merchant_fraud_rate"] = df["merchant"].map(merchant_fraud_rate)
    df["category_fraud_rate"] = df["category"].map(category_fraud_rate)
    df["avg_txn_amt_category"] = df["category"].map(avg_cat_amt)
    df["avg_txn_amt_category"] = df["avg_txn_amt_category"].astype(float)
    print(df["avg_txn_amt_category"])
    # --- 8. SSN-level historical aggregates ---
    ssn_stats = df.groupby("ssn").agg(
        total_txn_count=("is_fraud", "count"),
        total_fraud_count=("is_fraud", "sum"),
        avg_time_between_txn=("time_diff", "mean"),
        unique_merchants=("merchant", "nunique"),
    ).reset_index()
    ssn_stats["fraud_ratio"] = ssn_stats["total_fraud_count"] / ssn_stats["total_txn_count"]

    df = df.merge(ssn_stats, on="ssn", how="left")

    # --- 9. Personalized (ssn + category / ssn + merchant) aggregates ---
    ssn_cat_stats = df.groupby(["ssn", "category"]).agg(
        ssn_cat_avg_amt=("amt", "mean"),
        ssn_cat_txn_count=("amt", "count"),
        ssn_cat_fraud_rate=("is_fraud", "mean"),
    ).reset_index()

    ssn_merch_stats = df.groupby(["ssn", "merchant"]).agg(
        ssn_merch_avg_amt=("amt", "mean"),
        ssn_merch_txn_count=("amt", "count"),
        ssn_merch_fraud_rate=("is_fraud", "mean"),
    ).reset_index()

    df = df.merge(ssn_cat_stats, on=["ssn", "category"], how="left")
    df = df.merge(ssn_merch_stats, on=["ssn", "merchant"], how="left")

    # --- 10. Anomaly & derived flags ---
    df["amt_zscore_global"] = (df["amt"] - df["amt"].mean()) / (df["amt"].std() + 1e-6)
    df["is_amt_outlier"] = (df["amt_zscore_global"].abs() > 3).astype(int)
    df["is_large_txn"] = (df["amt"] > df.groupby("ssn")["amt"].transform(lambda x: x.quantile(0.95))).astype(int)
    df["is_first_txn"] = df.groupby("ssn").cumcount().eq(0).astype(int)
    df["has_recent_fraud"] = df.groupby("ssn")["is_fraud"].shift(1).fillna(0).astype(int)

    # --- 11. Interaction features ---
    df["amt_x_new_city"] = df["amt"] * df["is_new_city"]
    df["txn_speed_kmh"] = df["distance_from_last_txn"] / (df["time_diff"] / 3600 + 1e-6)
    df["amt_over_avg"] = df["amt"] / (df["avg_txn_amt"] + 1e-6)
    df["amt_over_cat_avg"] = df["amt"] / (df["avg_txn_amt_category"] + 1e-6)
    df["amt_over_ssn_cat_avg"] = df["amt"] / (df["ssn_cat_avg_amt"] + 1e-6)

    # --- 12. Fill missing values ---
    fill_defaults = {
        "rolling_7D_fraud_rate": 0,
        "rolling_30D_fraud_rate": 0,
        "rolling_7D_avg_amt": df["amt"].mean(),
        "rolling_30D_avg_amt": df["amt"].mean(),
        "ssn_cat_fraud_rate": 0,
        "ssn_merch_fraud_rate": 0,
    }
    df.fillna(fill_defaults, inplace=True)

    return df

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
    cramers_v = np.sqrt(chi2 / (n * (min(cont_table.shape) - 1)))
    print(f"Cramér’s V = {cramers_v:.3f}")

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
    n = len(values)
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

def balanced_group_time_split(
    df: pd.DataFrame,
    group_col: str = "ssn",
    time_col: str = "unix_time",
    label_col: str = "is_fraud",
    train_frac: float = 0.8,
    dev_frac: float = 0.10,
    n_time_bins: int = 5,
    random_state: int = 42,
)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(random_state)
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], unit='s', errors='coerce')

    grp = (
        df.groupby(group_col)
        .agg(
            first_txn=(time_col, "min"),
            last_txn=(time_col, "max"),
            n_txn=(label_col, "count"),
            n_fraud=(label_col, "sum"),
        )
        .reset_index()
    )
    grp["fraud_rate"] = grp["n_fraud"] / grp["n_txn"]

    # Sort by time and cut into time bins
    grp = grp.sort_values("first_txn").reset_index(drop=True)
    grp["time_bin"] = pd.qcut(np.arange(len(grp)), q=n_time_bins, labels=False)

    grp["split"] = None
    for _, tdf in grp.groupby("time_bin"):
        # Stratify by fraud rate inside each time bin
        tdf["fraud_bin"] = pd.qcut(tdf["fraud_rate"].rank(method="first"), q=5, labels=False)
        for _, fdf in tdf.groupby("fraud_bin"):
            n = len(fdf)
            n_train = int(np.floor(train_frac * n))
            n_dev = int(np.floor(dev_frac * n))
            idx = fdf.sample(frac=1, random_state=random_state).index
            grp.loc[idx[:n_train], "split"] = "train"
            grp.loc[idx[n_train:n_train+n_dev], "split"] = "dev"
            grp.loc[idx[n_train+n_dev:], "split"] = "test"

    df = df.merge(grp[[group_col, "split"]], on=group_col, how="left")

    df_train = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    df_dev   = df[df["split"] == "dev"].drop(columns=["split"]).reset_index(drop=True)
    df_test  = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    def summarize(name :str, d :pd.DataFrame):
        total = len(d)
        frauds = d[label_col].sum()
        pct = 100 * frauds / total if total > 0 else 0
        return f"{name:>6}: {len(d):>8,} rows | {d[group_col].nunique():>6} groups | Fraud rate = {pct:6.3f}%"

    print("\n Split summary:")
    print(summarize("Train", df_train))
    print(summarize(" Dev ", df_dev))
    print(summarize(" Test", df_test))
    print("Disjoint groups:",
          set(df_train[group_col]).isdisjoint(set(df_dev[group_col])) and
          set(df_train[group_col]).isdisjoint(set(df_test[group_col])) and
          set(df_dev[group_col]).isdisjoint(set(df_test[group_col])))

    # Check distributions across key columns
    print("\n=== Distribution Check ===")
    return df_train, df_dev, df_test

def add_fraud_per_capita(X_train :pd.DataFrame, X_dev :pd.DataFrame, X_test :pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Calculate on training data only
    fraud_stats = X_train.groupby('city').agg({
        'amt': 'sum',
        'city_pop': 'first',
        'is_fraud': 'sum'  # Count of frauds
    })
    fraud_stats['fraud_per_capita'] = fraud_stats['amt'] / fraud_stats['city_pop']
    fraud_stats['fraud_count_per_capita'] = fraud_stats['is_fraud'] / fraud_stats['city_pop']

    # Apply to all three sets
    for df in [X_train, X_dev, X_test]:
        df['fraud_per_capita'] = df['city'].map(fraud_stats['fraud_per_capita'])
        df['fraud_count_per_capita'] = df['city'].map(fraud_stats['fraud_count_per_capita'])

    return X_train, X_dev, X_test

def rolling_features(user_df :pd.DataFrame) -> pd.DataFrame:
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
            df.groupby("merchant")
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
        df.groupby(["ssn", "merchant"])["merchant"]
        .transform("count")
    )
    df["user_total_txn"] = df.groupby("ssn")["merchant"].transform("count")
    df["user_merchant_freq_ratio"] = df["user_merchant_frequency"] / (df["user_total_txn"] + 1e-6)
    # --- Target encoding (use pre-computed or calculate) ---
    if job_target_mean is None:
        job_target_mean = df.groupby("job")["is_fraud"].mean()

    df["job_te"] = df["job"].map(job_target_mean)
    job_avg = df.groupby("job")["amt"].transform("mean")
    df["amt_over_job_avg"] = df["amt"] / (job_avg + 1e-6)
    df.drop(columns=["user_total_txn"], inplace=True)
    print('fraud detection features done')
    return df


def strict_time_group_split(
    df: pd.DataFrame,
    group_col: str = "ssn",
    time_col: str = "unix_time",
    label_col: str = "is_fraud",
    train_frac: float = 0.8,
    dev_frac: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train/dev/test sets with:
      - Disjoint groups (no group leakage)
      - Strict chronological order (no time overlap)
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    df = df.dropna(subset=[time_col])  # remove invalid timestamps
    # Sort by time
    df = df.sort_values(time_col)

    # Compute time cutoffs
    t1 = df[time_col].quantile(train_frac)
    t2 = df[time_col].quantile(train_frac + dev_frac)

    # Split by actual timestamps
    df_train = df[df[time_col] <= t1]
    df_dev   = df[(df[time_col] > t1) & (df[time_col] <= t2)]
    df_test  = df[df[time_col] > t2]

    # Remove any group that appears in multiple sets
    def drop_overlaps(primary, *others):
        overlap_groups = set(primary[group_col]) & set().union(*[set(o[group_col]) for o in others])
        return primary[~primary[group_col].isin(overlap_groups)]

    df_train = drop_overlaps(df_train, df_dev, df_test)
    df_dev   = drop_overlaps(df_dev, df_train, df_test)
    df_test  = drop_overlaps(df_test, df_train, df_dev)

    # Print summary
    print("=== Split Summary ===")
    for name, part in [("Train", df_train), ("Dev", df_dev), ("Test", df_test)]:
        fraud_rate = 100 * part[label_col].mean()
        print(f" {name:5}: {len(part):,} rows | {part[group_col].nunique():5} groups | Fraud rate = {fraud_rate:6.3f}%")

    print("\n=== Temporal Ordering ===")
    for name, part in [("Train", df_train), ("Dev", df_dev), ("Test", df_test)]:
        print(f"{name:5}: {pd.to_datetime(part[time_col], unit='s').min()} → {pd.to_datetime(part[time_col], unit='s').max()}")

    print("\nStrict chronological order:")
    print("  Train max < Dev min:", df_train[time_col].max() < df_dev[time_col].min())
    print("  Dev max < Test min:", df_dev[time_col].max() < df_test[time_col].min())

    return df_train, df_dev, df_test


def temporal_group_split(
    df: pd.DataFrame,
    group_col: str = "ssn",
    time_col: str = "unix_time",
    label_col: str = "is_fraud",
    train_frac: float = 0.8,
    dev_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally while keeping user groups intact.

    Rationale:
    - Fraud patterns evolve over time (temporal ordering matters)
    - Prevent information leakage from future → past
    - Keep each user/group in exactly one split
    - Stratify by fraud rate to maintain class balance across splits
    """

    # ============================================================
    # Prepare data
    # ============================================================
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    df = df.dropna(subset=[time_col])  # remove invalid timestamps

    # ============================================================
    # Aggregate group-level stats
    # ============================================================
    group_stats = df.groupby(group_col).agg({
        time_col: ["min", "median", "max"],
        label_col: ["count", "sum"]
    }).reset_index()

    group_stats.columns = [
        group_col, "first_txn", "median_txn", "last_txn", "n_txn", "n_fraud"
    ]
    group_stats["fraud_rate"] = group_stats["n_fraud"] / group_stats["n_txn"]

    # Sort groups by median transaction time (temporal order)
    # group_stats = group_stats.sort_values("median_txn").reset_index(drop=True)
    group_stats = group_stats.sort_values("last_txn").reset_index(drop=True)
    # ============================================================
    # Stratify by fraud rate
    # ============================================================
    group_stats["fraud_stratum"] = pd.qcut(
        group_stats["fraud_rate"].rank(method="first"),
        q=10, labels=False, duplicates="drop"
    )

    # ============================================================
    # Assign splits within each stratum (chronologically)
    # ============================================================
    group_stats["split"] = None

    for stratum in group_stats["fraud_stratum"].dropna().unique():
        stratum_mask = group_stats["fraud_stratum"] == stratum
        stratum_groups = group_stats[stratum_mask].index.to_list()
        n = len(stratum_groups)

        # Handle small strata
        if n < 3:
            group_stats.loc[stratum_groups, "split"] = (
                "train" if n == 1 else "test"
            )
            continue

        n_train = int(np.floor(train_frac * n))
        n_dev = int(np.floor(dev_frac * n))
        n_test = n - n_train - n_dev

        group_stats.loc[stratum_groups[:n_train], "split"] = "train"
        group_stats.loc[stratum_groups[n_train:n_train + n_dev], "split"] = "dev"
        group_stats.loc[stratum_groups[n_train + n_dev:], "split"] = "test"

    # ============================================================
    # Merge split assignments back to the main DataFrame
    # ============================================================
    df = df.merge(group_stats[[group_col, "split"]], on=group_col, how="left")

    df_train = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    df_dev = df[df["split"] == "dev"].drop(columns=["split"]).reset_index(drop=True)
    df_test = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)


    # Diagnostic summaries

    def summarize(name: str, d: pd.DataFrame):
        total = len(d)
        frauds = d[label_col].sum()
        pct = 100 * frauds / total if total > 0 else 0
        return f"{name:>6}: {total:>8,} rows | {d[group_col].nunique():>6} groups | Fraud rate = {pct:6.3f}%"

    print("\n=== Split Summary ===")
    print(summarize("Train", df_train))
    print(summarize(" Dev ", df_dev))
    print(summarize(" Test", df_test))

    print("\n=== Data Integrity ===")
    disjoint = (
        set(df_train[group_col]).isdisjoint(df_dev[group_col])
        and set(df_train[group_col]).isdisjoint(df_test[group_col])
        and set(df_dev[group_col]).isdisjoint(df_test[group_col])
    )
    print("Groups are disjoint:", disjoint)

    print("\n=== Temporal Ordering ===")
    print(f"Train: {df_train[time_col].min()} → {df_train[time_col].max()}")
    print(f"Dev:   {df_dev[time_col].min()} → {df_dev[time_col].max()}")
    print(f"Test:  {df_test[time_col].min()} → {df_test[time_col].max()}")

    train_max = df_train[time_col].max()
    dev_min, dev_max = df_dev[time_col].min(), df_dev[time_col].max()
    test_min = df_test[time_col].min()

    print("\nStrict chronological order:")
    print(f"  Train max < Dev min: {train_max < dev_min}")
    print(f"  Dev max < Test min:  {dev_max < test_min}")


    # Return splits and metadata
    return df_train, df_dev, df_test, group_stats




def time_stratified_split_summary(
        df: pd.DataFrame,
        time_col: str = 'unix_time',
        label_col: str = "is_fraud",
        train_size: float = 0.8,
        dev_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42
):
    """
    Splits a DataFrame into train/dev/test sets by time while roughly stratifying by label
    and prints a summary of splits.

    Returns:
        df_train, df_dev, df_test
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    df = df.dropna(subset=[time_col])  # remove invalid timestamps
    assert np.isclose(train_size + dev_size + test_size, 1.0), "Splits must sum to 1"

    df_sorted = df.sort_values(time_col).reset_index(drop=True)

    df_train_list = []
    df_dev_list = []
    df_test_list = []

    for label in df_sorted[label_col].unique():
        df_label = df_sorted[df_sorted[label_col] == label]
        n = len(df_label)

        train_end = int(n * train_size)
        dev_end = train_end + int(n * dev_size)

        df_train_list.append(df_label.iloc[:train_end])
        df_dev_list.append(df_label.iloc[train_end:dev_end])
        df_test_list.append(df_label.iloc[dev_end:])

    df_train = pd.concat(df_train_list).sort_values(time_col).reset_index(drop=True)
    df_dev = pd.concat(df_dev_list).sort_values(time_col).reset_index(drop=True)
    df_test = pd.concat(df_test_list).sort_values(time_col).reset_index(drop=True)

    # Summary
    def summary(df_split, name):
        n_total = len(df_split)
        n_fraud = df_split[label_col].sum()
        fraud_rate = n_fraud / n_total if n_total > 0 else 0
        time_min = df_split[time_col].min()
        time_max = df_split[time_col].max()
        print(f"{name} | Samples: {n_total} | Fraud rate: {fraud_rate:.4f} | Time: {time_min} -> {time_max}")

    print("===== Split Summary =====")
    summary(df_train, "Train")
    summary(df_dev, "Dev")
    summary(df_test, "Test")

    return df_train, df_dev, df_test

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

