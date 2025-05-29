import importlib
import random

from icecream import ic
from lovely_numpy import lo

import hu_utils

hu_utils = importlib.reload(hu_utils)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

### Do GridSearchCV for top 8 models
np.random.seed(0)
random.seed(0)
rs = 929

rmse_dict = dict()

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
    Seems it also works.

    Args:
        onedarr (_type_): _description_

    Returns:
        _type_: _description_
    """
    normalized_arr = ratio_arr / np.sum(ratio_arr)
    df_arr = [normalized_arr]
    return hu_rf_model.predict(df_arr)[0]


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
    acq_func="EI",  # the acquisition function, EI, LCB, PI, gp_hedge (choose one from three each iter)
    n_calls=n_calls,  # the number of evaluations of f
    random_state=0,
    n_initial_points=0,
    acq_optimizer="lbfgs",
    x0=X.to_numpy().tolist(),
    y0=y_neg,
    n_jobs=-1,
)

all_index = np.argsort(res.func_vals)

ic(len(res.x_iters))
ic(len(res.func_vals))
ic(np.argsort(res.func_vals))

for index in all_index:
    print(
        index,
        list(map(lambda x: round(x, 2), hu_utils.normalize(res.x_iters[index]))),
        "ML predicted:",
        -res.func_vals[index],
    )
