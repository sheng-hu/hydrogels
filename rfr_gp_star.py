import importlib
import random

from icecream import ic
from lovely_numpy import lo

import hu_utils

hu_utils = importlib.reload(hu_utils)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from skopt.learning import RandomForestRegressor as opt_RF

plt.style.use("classic")
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
rs = 929


rf_cv = hu_utils.setup_gridsearch_model("RFRsk")
rf_cv.fit(X, y)
ic(rf_cv.best_params_)


hu_rf_model = rf_cv.best_estimator_
y_neg = np.negative(y)

hu_rf_model.fit(X, y_neg)


np.random.seed(237)

from skopt import forest_minimize, gbrt_minimize, gp_minimize


def wrapper(ratio_arr):
    """Wrapper function (real) used in gp_minimize().

    Args:
        ratio_arr ([float]): formula ratio array.

    Returns:
        float: predicted value.
    """
    normalized_arr = ratio_arr / np.sum(ratio_arr)
    df_arr = [normalized_arr]
    return hu_rf_model.predict(df_arr)[0]


# hu_rf_model = opt_RF(n_estimators=200, max_features=1.0, max_leaf_nodes=70, min_samples_leaf=1, random_state=1126, n_jobs=-1)
# hu_rf_model.fit(X, y_neg)
n_calls = 50
res = gp_minimize(
    wrapper,  # the function to minimize
    [
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ],  # the bounds on each dimension of x
    acq_func="EI",  # the acquisition function, EI, LCB, PI, gp_hedge (choose one from three each iter)
    n_calls=n_calls,  # the number of evaluations of f
    random_state=0,
    n_initial_points=10,
    acq_optimizer="lbfgs",
    x0=X.to_numpy().tolist(),
    y0=y_neg,
    verbose=True,
    n_jobs=-1,
)


# res.models
ic(len(res.x_iters))
ic(len(res.func_vals))
ic(np.argsort(res.func_vals))

all_index = np.argsort(res.func_vals)
for index in all_index:
    print(
        index,
        list(map(lambda x: round(x, 2), hu_utils.normalize(res.x_iters[index]))),
        "ML predicted:",
        -res.func_vals[index],
    )
