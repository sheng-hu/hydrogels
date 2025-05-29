import importlib
import random

from icecream import ic
from lovely_numpy import lo

import hu_utils

hu_utils = importlib.reload(hu_utils)

plt.style.use("classic")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# sns.set_style('whitegrid')
import plt_configs

importlib.reload(plt_configs)


# 180 points.
df = pd.read_excel("Original Data_ML_20220829.xlsx", engine="openpyxl")

x_columns = [
    "Nucleophilic-HEA",
    "Hydrophobic-BA",
    "Acidic-CBEA",
    "Cationic-ATAC",
    "Aromatic-PEA",
    "Amide-AAm",
]
y_column = "Glass (kPa)"

X = df.loc[:, x_columns]
y = df.loc[:, y_column]

np.random.seed(0)
random.seed(0)


gp_cv = hu_utils.setup_gridsearch_model("GP")
gp_cv.fit(X, y)
ic(gp_cv.best_params_)
best_gp_cv = hu_utils.BestEstimatorCV(estimator=gp_cv.best_estimator_, X=X, y=y, cv=10)

hu_rf_model = gp_cv.best_estimator_
y_neg = np.negative(y)


def wrapper(ratio_arr):
    """Wrapper function (real) used in gp_minimize().

    Args:
        ratio_arr ([float]): formula ratio array.

    Returns:
        float: predicted value.
    """
    normalized_arr = np.array(ratio_arr) / np.sum(ratio_arr)
    df_arr = normalized_arr
    return hu_rf_model.predict(df_arr)[0]


import GPy
import GPyOpt

bounds = [
    {"name": "var_1", "type": "continuous", "domain": (0, 1)},
    {"name": "var_2", "type": "continuous", "domain": (0, 1)},
    {"name": "var_3", "type": "continuous", "domain": (0, 1)},
    {"name": "var_4", "type": "continuous", "domain": (0, 1)},
    {"name": "var_5", "type": "continuous", "domain": (0, 1)},
    {"name": "var_6", "type": "continuous", "domain": (0, 1)},
]

y_reform = y_neg.to_numpy().reshape(180, 1)
myProblem = GPyOpt.methods.BayesianOptimization(
    wrapper,
    bounds,
    model_type="GP",
    X=X.to_numpy(),
    Y=y_reform,
    acquisition_type="EI",
    initial_design_numdata=0,
    normalize_Y=False,
    evaluator_type="local_penalization",
    batch_size=10,
)
myProblem.run_optimization(1)  # Only 1 iteration

"""_summary_
run_optimization(max_iter=0, max_time=inf, eps=1e-08, context=None, verbosity=False, save_models_parameters=True,
report_file=None, evaluations_file=None, models_file=None)
"""

gp_df = hu_utils.pred_ei_x_to_df_LP(myProblem, 180)
gp_df = gp_df[gp_df["point_idx"] > 179]

gp_df = gp_df.fillna(0)
print(gp_df)
