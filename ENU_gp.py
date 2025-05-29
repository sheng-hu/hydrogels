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

gp_cv = hu_utils.setup_gridsearch_model("GP")
gp_cv.fit(X, y)
# ic(gp_cv.best_params_)
best_gp_cv = hu_utils.BestEstimatorCV(estimator=gp_cv.best_estimator_, X=X, y=y, cv=10)

best_gp_cv.output_stats()
best_gp_cv.plot_hold_out(f"gp.pdf", save=False)

model = gp_cv.best_estimator_

np.random.seed(0)

X_enumerate_dirty = np.random.random((10000000, 6))

ic(X_enumerate_dirty[0])

from sklearn.preprocessing import normalize

X_enumerate = normalize(X_enumerate_dirty, axis=1, norm="l1")

y_enumerate_pred = model.predict(X_enumerate)
# sorted(y_enumerate_pred, reverse=True)
minfirst = np.argsort(y_enumerate_pred)
maxfirst = minfirst[::-1]

X_1000 = X_enumerate[maxfirst[:1000]]
y_1000 = y_enumerate_pred[maxfirst[:1000]]

df_1000 = pd.DataFrame(X_1000)
df_1000.columns = ["HEA", "BA", "CBEA", "ATAC", "PEA", "AAm"]


df_1000["F_a"] = y_1000
# df_1000['cluster_num'] = km_model.labels_
print(df_1000.head(20))
