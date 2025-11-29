''' In this file the credit_card_fraud.csv is imported and filtered for
further use. The credit_card_fraud.csv is a large file and needs to
be imported once, that is the reason for the elapsed time computation.
At the end of the file the results are saved as csv and pickle files.
The csv and pickle files are of reasonable size.
'''
from pathlib import Path
from src.common.base_component import BaseComponent
from pr_0_common_imports import (pd, time, script_directory,ProfileReport, webbrowser,IsolationForest,StratifiedKFold,
      AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,SVC,clone,XGBClassifier,average_precision_score,SMOTETomek,
                                 RandomUnderSampler,RandomOverSampler, SMOTE,accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score,os, clone,RandomOverSampler, SMOTE,RandomUnderSampler)
import pr_0_defs

from pr_0_common_imports import (
    pd, np, script_directory, XGBClassifier, RandomOverSampler, SMOTE, RandomUnderSampler, SMOTETomek,
    accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, Lasso, LinearSVC, Ridge,
    GradientBoostingClassifier,
    RandomForestClassifier, time, RFE, classification_report, GridSearchCV, roc_auc_score, shap, plt,
    LogisticRegression,
    AdaBoostClassifier, SVC
)
import pr_0_defs

class ReadFraudData(BaseComponent):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.logger.info("Fraud initialized")
        self.output_dir: Path = config.fr_data_process_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def handle_data_load(self) -> None:
        start = time.perf_counter()
        df = pd.read_csv('credit_card_fraud.csv', index_col=0)
        end = time.perf_counter()
        self.logger.info(f"\nElapsed time reading credit card fraud CSV file: {end - start:.4f} seconds")

        # retrieve year used to filter dataset on 2020
        df['trans_date'] = pd.to_datetime(df['trans_date'])
        df['year'] = df['trans_date'].dt.year.astype('Int64')
        # filter on year 2020 and California State (CA)
        # self.logger.info(df.state.unique())
        df_2020_ca = df[(df['year'] == 2020) & (df['state'] == 'CA')]
        df_2020_ma = df[(df['year'] == 2020) & (df['state'] == 'MA')]

        # # save resulting dataset as pickle and csv file
        pr_0_defs.write_df_to_csv(df_2020_ca, script_directory + "/pr_1_post_2020_ca.csv") # type: ignore
        pr_0_defs.write_df_to_pickle(df_2020_ca, script_directory + "/pr_1_post_2020_ca.pkl") # type: ignore

        pr_0_defs.write_df_to_csv(df_2020_ma, script_directory + "/pr_1_post_2020_ma.csv") # type: ignore
        pr_0_defs.write_df_to_pickle(df_2020_ma, script_directory + "/pr_1_post_2020_ma.pkl") # type: ignore


    def data_preparation(self) -> None:


        # script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
        # self.logger.info(f"Directory of the executing script: {script_directory}")

        # df = pr_0_defs.import_pickle(script_directory + "/pr_1_post_2020_ca.pkl") # type: ignore
        df = pr_0_defs.import_pickle(script_directory + "/pr_1_post_2020_ma.pkl")  # type: ignore

        df['ssn'] = df['ssn'].astype(str).str.strip().str.replace('-', '', regex=False).astype(int)
        self.logger.info('ssn done')

        df['gender'] = df['gender'].map({'F': 0, 'M': 1})
        self.logger.info(df.gender.value_counts())
        self.logger.info('gender done')

        df["profile"] = df["profile"].str.replace(".json", "", regex=False)
        df[["age_group", "gender_", "area"]] = df["profile"].str.rsplit("_", n=2, expand=True)
        df.drop(columns=["gender_"], axis=1, inplace=True)
        df[['profile', 'category', 'age_group', 'area', 'merchant']] = df[
            ['profile', 'category', 'age_group', 'area', 'merchant']].astype('category')
        self.logger.info('profile done')

        df["merchant"] = (df["merchant"].str.replace(r"^fraud_", "", regex=True).str.translate(
            str.maketrans("", "", "-,. "))).str.lower().astype(str)
        self.logger.info('merchant done')

        df['job'] = df['job'].str.replace(r'[,\[\]\s-]', '', regex=True).str.lower()
        self.logger.info('job cleaning done')

        today = pd.Timestamp.today()
        df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
        df["age"] = ((today.year - 1) - pd.to_datetime(df["dob"]).dt.year
                     - ((pd.to_datetime(df["dob"]).dt.month > today.month) |
                        ((pd.to_datetime(df["dob"]).dt.month == today.month) &
                         (pd.to_datetime(df["dob"]).dt.day > today.day))))
        self.logger.info('age done')

        df.drop(['first', 'last', 'street', 'acct_num', 'trans_num', 'state', 'year', 'age_group', 'dob'], axis=1,
                inplace=True)
        self.logger.info('drop columns done')

        cols_to_convert = df.select_dtypes(include='object').columns
        df[cols_to_convert] = df[cols_to_convert].apply(lambda x: x.astype('category'))
        self.logger.info("object to category done")

        df['city'] = df['city'].str.replace(r'[,\[\]\s-]', '', regex=True).str.lower().astype(str)
        self.logger.info("city done")

        df['city_pop'] = df['city_pop'].astype(float)
        self.logger.info('city pop done')

        # save results to csv and pickle
        pr_0_defs.write_df_to_csv(df, script_directory + "/pr_2_post_data_prep_ma.csv")  # type: ignore
        pr_0_defs.write_df_to_pickle(df, script_directory + "/pr_2_post_data_prep_ma.pkl")  # type: ignore

        # pr_0_defs.write_df_to_csv(df, script_directory + "/pr_2_post_data_prep.csv") # type: ignore
        # pr_0_defs.write_df_to_pickle(df, script_directory + "/pr_2_post_data_prep.pkl") # type: ignore


    def tests_data_preparation(self) -> None:
        df = pr_0_defs.import_pickle(script_directory + "/pr_2_post_data_prep.pkl")  # type: ignore

        self.logger.info(df.info())

        # Load your data
        df = pd.read_csv('pr2_df_2020_ca_data_prep.csv')

        # Generate report
        def prof_report(df):
            report = ProfileReport(df, title="Fraud Detection EDA", explorative=True)
            # Save to HTML
            report.to_file("fraud_report.html")
            webbrowser.open('file://' + os.path.realpath("fraud_report.html"))

    def eda_tests(self) -> None:
        ''' EDA AND ASSOCIATION TESTS ON ORIGINAL COLUMNS '''
        df = pr_0_defs.import_pickle(script_directory + "/pr_2_post_data_prep.pkl")
        self.logger.info(df.info())
        # Generate report
        # pr_0_defs.eda_report_profile_rep(df)
        #
        # csv_file_autoviz = script_directory + "/pr_2_post_data_prep.csv"
        # # target_col = "is_fraud"
        # target_col = ""
        # pr_0_defs.eda_report_autoviz(csv_file_autoviz, target_col)

        ''' --- Outliers detection with IsolationForest --- '''
        y = df["is_fraud"]
        X = df['amt']
        # ============================================
        # 2. Isolation Forest for Outlier Detection
        # ============================================

        self.logger.info("\n" + "=" * 50)
        self.logger.info("Isolation Forest - Outlier Detection")
        self.logger.info("=" * 50)

        # Create Isolation Forest
        # contamination = expected proportion of outliers (0.05 = 5%)
        iso_forest = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )

        # Fit and predict
        self.logger.info("Detecting outliers...")
        outlier_predictions = iso_forest.fit_predict(X)
        outlier_scores = iso_forest.score_samples(X)

        # -1 = outlier, 1 = normal
        n_outliers = (outlier_predictions == -1).sum()
        n_normal = (outlier_predictions == 1).sum()

        self.logger.info(f"\nOutliers detected: {n_outliers}")
        self.logger.info(f"Normal points: {n_normal}")
        self.logger.info(f"Outlier percentage: {n_outliers / len(X) * 100:.2f}%")

        # Add outlier predictions to dataframe
        X_with_outliers = X.copy()
        X_with_outliers['outlier'] = outlier_predictions
        X_with_outliers['outlier_score'] = outlier_scores

        # View outliers
        outliers_df = X_with_outliers[X_with_outliers['outlier'] == -1]
        self.logger.info(f"\nFirst few outliers:")
        self.logger.info(outliers_df.head())

    def feature_engineering(self) -> None:
        # df = pr_0_defs.import_pickle(script_directory + "/pr_2_post_data_prep.pkl") # type: ignore
        df = pr_0_defs.import_pickle(script_directory + "/pr_2_post_data_prep_ma.pkl")  # type: ignore

        ''' --- Feature Engineering --- '''
        df = df.sort_values(by=["ssn", "unix_time"]).reset_index(drop=True)

        df = pr_0_defs.generate_time_features_full(df, 'trans_date', 'trans_time')
        self.logger.info("times done")

        df = pr_0_defs.distance(df)
        self.logger.info("distance done")

        df['cc_com'] = df['cc_num'].map(pr_0_defs.get_credit_card_company)
        self.logger.info('cc company assignment done')

        ''' --- Encoding --- '''

        df['area_encoded'] = df['area'].map({'urban': 0, 'rural': 1}).astype(int)
        self.logger.info('area_encoded done')

        df_onehot = pd.get_dummies(df["category"], prefix="cat")
        df = pd.concat([df, df_onehot], axis=1)
        self.logger.info("category encoding done ")

        cc_com_dummies = pd.get_dummies(
            df['cc_com'],
            prefix='cc_com',
            dtype=int
        )
        df = pd.concat([df, cc_com_dummies], axis=1)
        self.logger.info('cc_com encoded done')

        df.drop(columns=['lat', 'long', 'merch_lat', 'merch_long', 'zip', 'trans_time', 'age', 'time',
                         'profile', 'category', 'cc_com', 'datetime', 'trans_date', 'trans_time', 'year'], axis=1,
                inplace=True)
        # df.drop(columns=['city', 'lat', 'long', 'merch_lat', 'merch_long', 'zip', 'city_pop', 'ssn', 'trans_time', 'age',
        #                  'city_bin', 'profile', 'category', 'cc_com', 'datetime', 'day', 'year', 'hour','job','merchant', 'trans_date', 'trans_time'], axis=1, inplace=True)
        self.logger.info('columns drop done')
        self.logger.info(df.info())

        missing_summary = (
            df.isna().sum()
            .to_frame("missing_count")
            .assign(missing_pct=lambda x: 100 * x["missing_count"] / len(df))
            .query("missing_count > 0")
        )
        self.logger.info(missing_summary)

        # pr_0_defs.write_df_to_csv(df, script_directory + "/pr_4_interim_feat_engin_enc.csv") # type: ignore
        # pr_0_defs.write_df_to_pickle(df, script_directory + "/pr_4_interim_post_feat_engin_enc.pkl") # type: ignore

        pr_0_defs.write_df_to_csv(df, script_directory + "/pr_4_interim_feat_engin_enc_ma.csv")  # type: ignore
        pr_0_defs.write_df_to_pickle(df, script_directory + "/pr_4_interim_post_feat_engin_enc_ma.pkl")  # type: ignore

        ''' EDA ON ENCODED DATA '''
        # # df = pr_0_defs.import_pickle(script_directory + "/pr_2_post_data_prep.pkl")
        # # self.logger.info(df.info())
        # # Generate report
        # pr_0_defs.eda_report_profile_rep(df)
        #
        # csv_file_autoviz = script_directory + "/pr_4_interim_feat_engin_enc.csv"
        # target_col = "is_fraud"
        # # target_col = ""
        # # pr_0_defs.eda_report_autoviz(csv_file_autoviz, target_col)
        #
        # df = df.copy()
        # bool_cols = df.select_dtypes(include=['bool']).columns
        # df[bool_cols] = df[bool_cols].astype(int)

    def roll_stats_selection_models_fit(self):
        # df = pr_0_defs.import_pickle(script_directory + "/pr_4_interim_post_feat_engin_enc.pkl") # type: ignore
        df = pr_0_defs.import_pickle(script_directory + "/pr_4_interim_post_feat_engin_enc_ma.pkl")  # type: ignore
        ''' X data Split balanced, grouped '''
        X_train, X_dev, X_test = pr_0_defs.temporal_split_balanced(df)

        self.logger.info("Before feature engineering:")
        self.logger.info(f"X_train: {X_train.shape}")
        self.logger.info(f"X_dev:   {X_dev.shape}")
        self.logger.info(f"X_test:  {X_test.shape}")
        self.logger.info(f"Train SSNs in Test: {len(set(X_train['ssn']).intersection(set(X_test['ssn'])))}")

        # ===== Feature Engineering  =====

        X_train = X_train.sort_values(by=["ssn", "unix_time"]).reset_index(drop=True)
        X_dev = X_dev.sort_values(by=["ssn", "unix_time"]).reset_index(drop=True)
        X_test = X_test.sort_values(by=["ssn", "unix_time"]).reset_index(drop=True)

        # # Calculate merchant statistics on training data only
        # merchant_stats = X_train.groupby("merchant").agg({
        #     "amt": "mean",
        #     "is_fraud": "mean",
        # }).reset_index()
        # merchant_stats.columns = ["merchant", "merchant_avg_amt", "merchant_fraud_rate"]
        # merchant_stats["merchant_txn_count"] = X_train.groupby("merchant").size().values

        # Apply rolling features to each split (rolling is per-user, no leakage)
        X_train = pr_0_defs.feat_eng_rolling(X_train)
        X_dev = pr_0_defs.feat_eng_rolling(X_dev)
        X_test = pr_0_defs.feat_eng_rolling(X_test)

        # Calculate fraud per capita on training data only
        fraud_stats = X_train.groupby('city').agg({
            'amt': 'sum',
            'city_pop': 'first',
            'is_fraud': 'sum'
        })
        fraud_stats['fraud_per_capita'] = fraud_stats['amt'] / fraud_stats['city_pop']
        fraud_stats['fraud_count_per_capita'] = fraud_stats['is_fraud'] / fraud_stats['city_pop']

        # Apply to all splits
        for dataset in [X_train, X_dev, X_test]:
            dataset['fraud_per_capita'] = dataset['city'].map(fraud_stats['fraud_per_capita'])
            dataset['fraud_count_per_capita'] = dataset['city'].map(fraud_stats['fraud_count_per_capita'])

        # y split
        y_train, y_dev, y_test = X_train['is_fraud'], X_dev['is_fraud'], X_test['is_fraud']

        self.logger.info("\nAfter feature engineering:")
        self.logger.info(f"X_train: {X_train.shape}")
        self.logger.info(f"X_dev:   {X_dev.shape}")
        self.logger.info(f"X_test:  {X_test.shape}")
        self.logger.info(f"Train SSNs in Test: {len(set(X_train['ssn']).intersection(set(X_test['ssn'])))}")

        self.logger.info("\nFraud rates:")
        self.logger.info(f"Train: {y_train.mean():.4f}")
        self.logger.info(f"Dev:   {y_dev.mean():.4f}")
        self.logger.info(f"Test:  {y_test.mean():.4f}")
        '''---------------------------------------------------------------------------------'''

        # drop columns
        for df_ in [X_train, X_dev, X_test]: df_.drop(
            columns=['ssn', 'is_fraud', 'amt', 'unix_time', 'merchant', 'job', 'city', 'cc_num', 'city_pop', 'area'],
            inplace=True)

        self.logger.info(X_train.shape, X_dev.shape, X_test.shape)
        self.logger.info(X_train.columns)
        data = list(X_train.columns)
        import csv
        with open('output.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(data)

        ''' --- Feature Selection --- '''
        # Feature selection on 80% of the data:
        y = y_train
        X = X_train

        ''' Grid Search Parameter Selection '''

        # Lasso
        start = time.perf_counter()
        lasso_params = {'alpha': [0.001, 0.1]}
        lasso_cv = GridSearchCV(Lasso(), lasso_params, cv=5)
        lasso_cv.fit(X, y)
        best_lasso = lasso_cv.best_estimator_
        lasso_selected = (np.abs(best_lasso.coef_) > 0).astype(int)
        end = time.perf_counter()
        self.logger.info(f"\nElapsed time Lasso Grid Search: {(end - start) / 60.:.4f} minutes")
        self.logger.info(f"Best alpha for Lasso: {lasso_cv.best_params_['alpha']}")

        # Ridge
        start = time.perf_counter()
        ridge_params = {'alpha': [0.4, 0.5, 0.6]}
        ridge_cv = GridSearchCV(Ridge(), ridge_params, cv=5)
        ridge_cv.fit(X, y)
        best_ridge = ridge_cv.best_estimator_
        ridge_selected = (np.abs(best_ridge.coef_) > 0).astype(int)
        end = time.perf_counter()
        self.logger.info(f"\nElapsed time Ridge Grid Search: {(end - start) / 60.:.4f} minutes")
        self.logger.info(f"Best alpha for Ridge: {ridge_cv.best_params_['alpha']}")

        # SVM
        # start = time.perf_counter()
        # svm_params = {'C': [4,5,6]}
        # svm_cv = GridSearchCV(LinearSVC(penalty="l1", dual=False, max_iter=2000), svm_params, cv=5)
        # svm_cv.fit(X, y)
        # best_svm = svm_cv.best_estimator_
        # svm_selected = (np.abs(best_svm.coef_[0]) > 0).astype(int)
        # end = time.perf_counter()
        # self.logger.info(f"\nElapsed time SVM Grid Search: {(end - start)/60.:.4f} minutes")
        # self.logger.info(f"Best C for SVM: {svm_cv.best_params_['C']}")

        # XGBoost Feature Importance
        start = time.perf_counter()
        self.logger.info("=" * 50)
        self.logger.info("XGBoost Feature Importance")
        self.logger.info("=" * 50)

        # Create and fit XGBoost classifier
        xgb_model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        self.logger.info("Fitting XGBoost model...")
        xgb_model.fit(X, y)

        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.logger.info("\nFeature Importance:")
        self.logger.info(feature_importance)

        xgb_selected = (xgb_model.feature_importances_ > 0).astype(int)

        end = time.perf_counter()
        self.logger.info(f"\nElapsed time XGBoost feature importance: {(end - start):.4f} seconds")

        # RFE with XGBoost (Optional - commented out as it's slow)

        start = time.perf_counter()
        self.logger.info("=" * 50)
        self.logger.info("RFE Feature Selection with XGBoost")
        self.logger.info("=" * 50)

        xgb_rfe = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        rfe = RFE(estimator=xgb_rfe, n_features_to_select=30, step=5)
        self.logger.info("Fitting RFE... (this may take a moment)")
        rfe.fit(X, y)

        xgb_rfe_selected = rfe.support_.astype(int)
        end = time.perf_counter()
        self.logger.info(f"\nElapsed time XGBoost RFE: {(end - start) / 60.:.4f} minutes")

        # GradientBoost
        start = time.perf_counter()
        self.logger.info("=" * 50)
        self.logger.info("GradientBoost Feature Selection")
        self.logger.info("=" * 50)

        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X, y)
        gb_selected = (gb.feature_importances_ > 0).astype(int)

        end = time.perf_counter()
        self.logger.info(f"Elapsed time GradientBoost: {(end - start):.4f} seconds")

        # RandomForest
        start = time.perf_counter()
        self.logger.info("=" * 50)
        self.logger.info("RandomForest Feature Selection")
        self.logger.info("=" * 50)

        rf = RandomForestClassifier(n_estimators=200,
                                    class_weight='balanced',
                                    min_samples_leaf=5,
                                    max_depth=15,
                                    random_state=42,
                                    n_jobs=-1
                                    )
        rf.fit(X, y)
        rf_selected = (rf.feature_importances_ > 0).astype(int)

        end = time.perf_counter()
        self.logger.info(f"Elapsed time RandomForest: {(end - start):.4f} seconds")

        # Create Results DataFrame

        selection_df = pd.DataFrame({
            'Feature': X.columns,
            'XGBoost': xgb_selected,
            'Lasso': lasso_selected,
            'Ridge': ridge_selected,
            # 'SVM': svm_selected,
            'GradientBoost': gb_selected,
            'RandomForest': rf_selected
            # ,'XGBoost_RFE': xgb_rfe_selected  # Uncomment if using RFE
        })

        self.logger.info("\n" + "=" * 50)
        self.logger.info("Feature Selection Results Summary")
        self.logger.info("=" * 50)
        self.logger.info(selection_df)
        self.logger.info(f"\nTotal features: {len(selection_df)}")
        self.logger.info(f"\nFeatures selected by each method:")
        # self.logger.info(selection_df[['XGBoost', 'Lasso', 'Ridge', 'SVM', 'GradientBoost', 'RandomForest']].sum())
        self.logger.info(selection_df[['XGBoost', 'Lasso', 'Ridge', 'RandomForest', 'GradientBoost']].sum())

        # Sum the number of selections for each feature
        # selection_df['Sum'] = selection_df[['Lasso', 'SVM', 'GradientBoost', 'RandomForest','Ridge','XGBoost']].sum(axis=1)
        selection_df['Sum'] = selection_df[['Lasso', 'RandomForest', 'Ridge', 'XGBoost']].sum(axis=1)

        # Output the results
        self.logger.info('Sum the number of selections for each feature:')
        self.logger.info(selection_df)

        # Selecting variables with a sum of selections >= 4
        final_var = selection_df[selection_df['Sum'] >= 4]['Feature'].tolist()
        self.logger.info(final_var)
        X_model = X_train[final_var].copy()
        # df_model['is_fraud'] = df['is_fraud'].copy()

        # Output the result to verify
        self.logger.info('Selecting variables with a sum of selections >= 4:')
        self.logger.info(X_model.info())

        self.logger.info(f"\n\n" + "=" * 50)
        self.logger.info(f"FINAL SELECTED FEATURES: {len(final_var)} features")
        self.logger.info("=" * 50)
        for i, feat in enumerate(final_var, 1):
            self.logger.info(f"{i:2d}. {feat}")


        X_train = X_train[final_var]
        X_dev = X_dev[final_var]
        X_test = X_test[final_var]

        # # Fill NaN values
        X_train = X_train.fillna(X_train.mean())  # Fill with training mean
        X_dev = X_dev.fillna(X_train.mean())  # Fill with training mean (not dev mean!)
        X_test = X_test.fillna(X_train.mean())  # Fill with training mean


        models_list = pd.DataFrame()

        # Logistic Regression

        logi = LogisticRegression(random_state=42, max_iter=1000)
        logi.fit(X_train, y_train)

        pred_logi = logi.predict(X_dev)
        # Calculate confusion matrix
        cm = confusion_matrix(y_dev, pred_logi)
        model_dict = {'model': "Logistic Regression"}
        new_row = pd.DataFrame([{**model_dict, **pr_0_defs.classification_metrics(y_dev, pred_logi)}])
        models_list = pd.concat([models_list, new_row], ignore_index=True)

        # ADA Boost
        ada = AdaBoostClassifier()
        ada.fit(X_train, y_train)
        pred_ada = ada.predict(X_dev)
        self.logger.info("Confusion Matrix:")
        self.logger.info(confusion_matrix(y_dev, pred_ada))
        self.logger.info(classification_report(y_dev, pred_ada))
        model_dict = {'model': "Ada Boost"}
        new_row = pd.DataFrame([{**model_dict, **pr_0_defs.classification_metrics(y_dev, pred_ada)}])
        models_list = pd.concat([models_list, new_row], ignore_index=True)

        # GBM Gradient Boost Classifier
        gbm = GradientBoostingClassifier()
        gbm.fit(X_train, y_train)
        pred_gbm = gbm.predict(X_dev)
        self.logger.info("Confusion Matrix:")
        self.logger.info(confusion_matrix(y_dev, pred_gbm))
        self.logger.info(classification_report(y_dev, pred_gbm))
        model_dict = {'model': "GBM Boost"}
        new_row = pd.DataFrame([{**model_dict, **pr_0_defs.classification_metrics(y_dev, pred_gbm)}])
        models_list = pd.concat([models_list, new_row], ignore_index=True)

        # Random Forest
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        pred_rf = rfc.predict(X_dev)
        self.logger.info("Confusion Matrix:")
        self.logger.info(confusion_matrix(y_dev, pred_rf))
        self.logger.info(classification_report(y_dev, pred_rf))
        model_dict = {'model': "Random Forest"}
        new_row = pd.DataFrame([{**model_dict, **pr_0_defs.classification_metrics(y_dev, pred_rf)}])
        models_list = pd.concat([models_list, new_row], ignore_index=True)

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=461,
            max_depth=13,
            learning_rate=0.15703206747252366,
            subsample=0.7467659251079928,
            colsample_bytree=0.6966231041161306,
            gamma=0.11568411276837443,
            min_child_weight=6)
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_dev)
        self.logger.info("Confusion Matrix:")
        self.logger.info(confusion_matrix(y_dev, pred_xgb))
        self.logger.info(classification_report(y_dev, pred_xgb))
        model_dict = {'model': "XGBoost"}
        new_row = pd.DataFrame([{**model_dict, **pr_0_defs.classification_metrics(y_dev, pred_xgb)}])
        models_list = pd.concat([models_list, new_row], ignore_index=True)
        #
        # # SVM
        # svm = SVC(probability=True)
        # svm.fit(X_train,y_train)
        # pred_svm = svm.predict(X_dev)
        # self.logger.info("Confusion Matrix:")
        # self.logger.info(confusion_matrix(y_dev,pred_svm))
        # self.logger.info(classification_report(y_dev,pred_svm))
        # model_dict = {'model': "SVM"}
        # new_row = pd.DataFrame([{**model_dict, **pr_0_defs.classification_metrics(y_dev, pred_svm)}])
        # models_list = pd.concat([models_list, new_row], ignore_index=True)

        self.logger.info(models_list.sort_values('Accuracy', ascending=False))
        self.logger.info(models_list.sort_values('AUC', ascending=False))
        #
        # #
        # # ''' --------------------------------------------------------------------------------------------------------------- '''
        # # #
        import optuna
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        import numpy as np

        # ============================================================
        # XGBOOST OPTUNA OPTIMIZATION
        # ============================================================

        def objective(trial):
            """
            Objective function for XGBoost hyperparameter optimization
            """
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss',
                'verbosity': 0
            }

            clf = XGBClassifier(**params)

            # Use F1-score for imbalanced fraud detection
            scores = cross_val_score(
                clf, X_train, y_train,
                cv=3,
                scoring='f1',
                n_jobs=-1
            )

            return scores.mean()

        # ============================================================
        # RUN OPTIMIZATION
        # ============================================================

        self.logger.info("=" * 100)
        self.logger.info("XGBOOST HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
        self.logger.info("=" * 100)

        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)

        # ============================================================
        # RESULTS ANALYSIS
        # ============================================================

        self.logger.info("\n" + "=" * 100)
        self.logger.info("OPTIMIZATION RESULTS")
        self.logger.info("=" * 100)
        self.logger.info(f"\nBest F1-Score: {study.best_value:.4f}")
        self.logger.info(f"\nBest Parameters:")
        for key, value in study.best_params.items():
            self.logger.info(f"  {key}: {value}")

        # ============================================================
        # TRAIN FINAL MODEL WITH BEST PARAMETERS
        # ============================================================

        best_params = study.best_params
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

        final_xgb = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        self.logger.info("\n" + "=" * 100)
        self.logger.info("TRAINING FINAL XGBOOST MODEL")
        self.logger.info("=" * 100)

        # Train with early stopping on dev set
        final_xgb.fit(
            X_train, y_train,
            eval_set=[(X_dev, y_dev)],
            early_stopping_rounds=10,
            verbose=50
        )

        self.logger.info(f"\n✓ Model trained. Best iteration: {final_xgb.best_iteration}")

        # ============================================================
        # EVALUATION ON DEV SET
        # ============================================================

        y_dev_pred = final_xgb.predict(X_dev)
        y_dev_proba = final_xgb.predict_proba(X_dev)[:, 1]

        self.logger.info("\n" + "=" * 100)
        self.logger.info("DEV SET PERFORMANCE")
        self.logger.info("=" * 100)

        tn, fp, fn, tp = confusion_matrix(y_dev, y_dev_pred).ravel()
        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"  TP: {tp}, FP: {fp}")
        self.logger.info(f"  FN: {fn}, TN: {tn}")

        self.logger.info(f"\nClassification Report:")
        self.logger.info(classification_report(y_dev, y_dev_pred))

        self.logger.info(f"\nMetrics:")
        self.logger.info(f"  Precision: {precision_score(y_dev, y_dev_pred):.4f}")
        self.logger.info(f"  Recall: {recall_score(y_dev, y_dev_pred):.4f}")
        self.logger.info(f"  F1-Score: {f1_score(y_dev, y_dev_pred):.4f}")
        self.logger.info(f"  AUC-ROC: {roc_auc_score(y_dev, y_dev_proba):.4f}")

        # ============================================================
        # EVALUATION ON TEST SET
        # ============================================================

        y_test_pred = final_xgb.predict(X_test)
        y_test_proba = final_xgb.predict_proba(X_test)[:, 1]

        self.logger.info("\n" + "=" * 100)
        self.logger.info("TEST SET PERFORMANCE")
        self.logger.info("=" * 100)

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"  TP: {tp}, FP: {fp}")
        self.logger.info(f"  FN: {fn}, TN: {tn}")

        self.logger.info(f"\nClassification Report:")
        self.logger.info(classification_report(y_test, y_test_pred))

        self.logger.info(f"\nMetrics:")
        self.logger.info(f"  Precision: {precision_score(y_test, y_test_pred):.4f}")
        self.logger.info(f"  Recall: {recall_score(y_test, y_test_pred):.4f}")
        self.logger.info(f"  F1-Score: {f1_score(y_test, y_test_pred):.4f}")
        self.logger.info(f"  AUC-ROC: {roc_auc_score(y_test, y_test_proba):.4f}")


        # ============================================================
        # CONFIGURATION
        # ============================================================

        N_SPLITS = 5
        RANDOM_STATE = 42

        # ============================================================
        # DEFINE IMBALANCED DATA HANDLING TECHNIQUES
        # ============================================================

        techniques = {
            "ROS": RandomOverSampler(random_state=47),
            "RUS": RandomUnderSampler(random_state=47),
            "SMOTE": SMOTE(random_state=47),
            "SMOTETomek": SMOTETomek(random_state=47),
            "None": None  # Baseline: no resampling
        }

        # ============================================================
        # DEFINE MODELS
        # ============================================================

        models_config = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Ada Boost": AdaBoostClassifier(random_state=42),
            "GBM Boost": GradientBoostingClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(
                n_estimators=461,
                max_depth=13,
                learning_rate=0.15703206747252366,
                subsample=0.7467659251079928,
                colsample_bytree=0.6966231041161306,
                gamma=0.11568411276837443,
                min_child_weight=6,
                random_state=42,
                verbosity=0
            )
            #, "SVM": SVC(probability=True, random_state=42)
        }

        # ============================================================
        # TRAIN AND EVALUATE ALL COMBINATIONS
        # ============================================================

        results_list = []

        for technique_name, technique in techniques.items():
            self.logger.info(f"\n{'=' * 100}")
            self.logger.info(f"Processing: {technique_name}")
            self.logger.info(f"{'=' * 100}")

            # Apply resampling
            if technique is None:
                X_train_resampled, y_train_resampled = X_train, y_train
                self.logger.info("No resampling applied (baseline)")
            else:
                X_train_resampled, y_train_resampled = technique.fit_resample(X_train, y_train)
                self.logger.info(f"Resampled data shape: {X_train_resampled.shape}")
                self.logger.info(f"Class distribution: {np.bincount(y_train_resampled)}")

            # Train each model
            for model_name, model in models_config.items():
                self.logger.info(f"\n  Training {model_name}...", end=" ")

                # Clone model to avoid refitting issues
                if model_name == "Logistic Regression":
                    current_model = LogisticRegression(random_state=42, max_iter=1000)
                elif model_name == "Ada Boost":
                    current_model = AdaBoostClassifier(random_state=42)
                elif model_name == "GBM Boost":
                    current_model = GradientBoostingClassifier(random_state=42)
                elif model_name == "Random Forest":
                    current_model = RandomForestClassifier(random_state=42, n_jobs=-1)
                else:  # model_name == "XGBoost":
                    current_model = XGBClassifier(
                        n_estimators=461,
                        max_depth=13,
                        learning_rate=0.15703206747252366,
                        subsample=0.7467659251079928,
                        colsample_bytree=0.6966231041161306,
                        gamma=0.11568411276837443,
                        min_child_weight=6,
                        random_state=42,
                        verbosity=0
                    )
                # else:  # SVM
                #     current_model = SVC(probability=True, random_state=42)

                # Train model
                current_model.fit(X_train_resampled, y_train_resampled)

                # Make predictions
                y_dev_pred = current_model.predict(X_dev)
                y_dev_proba = current_model.predict_proba(X_dev)[:, 1] if hasattr(current_model, 'predict_proba') else None

                # Calculate metrics
                accuracy = accuracy_score(y_dev, y_dev_pred)
                precision = precision_score(y_dev, y_dev_pred, zero_division=0)
                recall = recall_score(y_dev, y_dev_pred, zero_division=0)
                f1 = f1_score(y_dev, y_dev_pred, zero_division=0)
                auc = roc_auc_score(y_dev, y_dev_proba) if y_dev_proba is not None else None

                # Get confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_dev, y_dev_pred).ravel()

                # Store results
                results_list.append({
                    'Technique': technique_name,
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'AUC': auc,
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'TN': tn
                })

                self.logger.info(f"✓ (Acc: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f})")

        # ============================================================
        # COMPILE RESULTS
        # ============================================================

        results_df = pd.DataFrame(results_list)

        self.logger.info("\n" + "=" * 100)
        self.logger.info("COMPREHENSIVE RESULTS: ALL TECHNIQUES × ALL MODELS")
        self.logger.info("=" * 100)
        self.logger.info(results_df.to_string(index=False))

        # ============================================================
        # ANALYSIS 1: Best Models by Technique
        # ============================================================

        self.logger.info("\n" + "=" * 100)
        self.logger.info("BEST MODELS BY TECHNIQUE (Sorted by F1-Score)")
        self.logger.info("=" * 100)

        for technique in techniques.keys():
            technique_results = results_df[results_df['Technique'] == technique].sort_values('F1-Score', ascending=False)
            self.logger.info(f"\n{technique}:")
            self.logger.info(technique_results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].head(3).to_string(
                index=False))

        # ============================================================
        # ANALYSIS 2: Best Techniques by Model
        # ============================================================

        self.logger.info("\n" + "=" * 100)
        self.logger.info("BEST TECHNIQUES BY MODEL (Sorted by F1-Score)")
        self.logger.info("=" * 100)

        for model in models_config.keys():
            model_results = results_df[results_df['Model'] == model].sort_values('F1-Score', ascending=False)
            self.logger.info(f"\n{model}:")
            self.logger.info(model_results[['Technique', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].head(3).to_string(
                index=False))

        # ============================================================
        # ANALYSIS 3: Overall Rankings
        # ============================================================

        self.logger.info("\n" + "=" * 100)
        self.logger.info("TOP 10 CONFIGURATIONS (Sorted by F1-Score)")
        self.logger.info("=" * 100)
        top_10_f1 = results_df.nlargest(10, 'F1-Score')[
            ['Technique', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
        self.logger.info(top_10_f1.to_string(index=False))

        self.logger.info("\n" + "=" * 100)
        self.logger.info("TOP 10 CONFIGURATIONS (Sorted by AUC)")
        self.logger.info("=" * 100)
        top_10_auc = results_df.nlargest(10, 'AUC')[
            ['Technique', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
        self.logger.info(top_10_auc.to_string(index=False))

        self.logger.info("\n" + "=" * 100)
        self.logger.info("TOP 10 CONFIGURATIONS (Sorted by Recall)")
        self.logger.info("=" * 100)
        top_10_recall = results_df.nlargest(10, 'Recall')[
            ['Technique', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
        self.logger.info(top_10_recall.to_string(index=False))

        # ============================================================
        # ANALYSIS 4: Technique Impact Summary
        # ============================================================

        self.logger.info("\n" + "=" * 100)
        self.logger.info("TECHNIQUE IMPACT SUMMARY (Average Metrics Across All Models)")
        self.logger.info("=" * 100)

        technique_summary = results_df.groupby('Technique')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].mean()
        self.logger.info(technique_summary.sort_values('F1-Score', ascending=False).to_string())

        # ============================================================
        # ANALYSIS 5: Model Impact Summary
        # ============================================================

        self.logger.info("\n" + "=" * 100)
        self.logger.info("MODEL PERFORMANCE SUMMARY (Average Metrics Across All Techniques)")
        self.logger.info("=" * 100)

        model_summary = results_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].mean()
        self.logger.info(model_summary.sort_values('F1-Score', ascending=False).to_string())

        # ============================================================
        # SAVE RESULTS
        # ============================================================

        results_df.to_csv('imbalanced_data_techniques_results.csv', index=False)
        self.logger.info("\n✓ Results saved to 'imbalanced_data_techniques_results.csv'")


