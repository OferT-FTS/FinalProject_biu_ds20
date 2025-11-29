''' This pr_0_common_imports.py file is used for imports relevant for the project .py files.
Use in the relevant .py files for as follows:  from pr_0_common_imports import <package/method>
For example: from pr_0_common_imports import pd, np, sns, plt

Important Note: The warnings are imported implicitly
'''

import time
import webbrowser
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import scikit_posthocs as sp
from datetime import datetime
import os
import re
import warnings
import shap


# from autoviz.AutoViz_Class import AutoViz_Class
from geopy.distance import great_circle
from datetime import datetime
from ydata_profiling import ProfileReport

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.svm import (
    LinearSVC, SVC
)
from sklearn.model_selection import (
    train_test_split, GridSearchCV,StratifiedKFold
)
from sklearn.linear_model import (
    Lasso, Ridge, LogisticRegression
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, IsolationForest,AdaBoostClassifier
)
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support, log_loss,average_precision_score
)
from sklearn.feature_selection import (
    SelectFromModel, RFE
)
from typing import (
    Tuple, List, Dict, BinaryIO
)
from scipy.stats import (
    chisquare, chi2_contingency, kruskal
)
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
warnings.filterwarnings("ignore") # imported implicitly by .py programs

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", message=".*deprecated.*")
# warnings.filterwarnings("ignore", message=".*Chart elements should only be supplied a single kdim.*")
# warnings.filterwarnings("ignore", category=UserWarning, module="bokeh")

try:
    script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\','/')
except NameError:
    script_directory = os.getcwd()

# script_directory = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
# print(f"\nDirectory of the executing script: {script_directory}")