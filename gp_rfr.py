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

plt.style.use("classic")
import seaborn as sns

# sns.set_style('whitegrid')
import plt_configs

importlib.reload(plt_configs)


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

rf_cv = hu_utils.setup_gridsearch_model("RFR")
rf_cv.fit(X, y)
ic(rf_cv.best_params_)


gp_cv = hu_utils.setup_gridsearch_model("GP")
gp_cv.fit(X, y)
ic(gp_cv.best_params_)
best_gp_cv = hu_utils.BestEstimatorCV(estimator=gp_cv.best_estimator_, X=X, y=y, cv=10)


hu_rf_model = gp_cv.best_estimator_
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
    # return -model.predict(df_arr)[0]


n_calls = 40
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
    base_estimator=rf_cv.best_estimator_,
    n_points=100000,
    acq_func="EI",  # the acquisition function, EI, LCB, PI, gp_hedge (choose one from three each iter)
    n_calls=n_calls,  # the number of evaluations of f
    random_state=0,
    n_initial_points=0,
    x0=X.to_numpy().tolist(),
    y0=y_neg,
    n_jobs=1,
)  # the random seed

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
