"""
Cleaned Utility functions for regression analysis and plotting.
"""
import ast
import math
import random
import warnings
from collections import Counter

import numpy as np
# Imports
import pandas as pd
import seaborn as sns
import shap
import umap
from adjustText import adjust_text
from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
from scipy.stats import norm
from sklearn.dummy import DummyRegressor
# from lightgbm import LGBMRegressor
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, IsolationForest,
                              RandomForestRegressor)
# from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as sk_GP
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     StratifiedShuffleSplit, cross_val_predict,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.neural_network import MLPRegressor  # Change here to skorch.
from sklearn.svm import SVR
from sklearn.utils import shuffle
from skopt.acquisition import gaussian_ei
from skopt.learning import ExtraTreesRegressor as opt_ETR
from skopt.learning import GaussianProcessRegressor
from skopt.learning import RandomForestRegressor as opt_RFR
from skopt.utils import cook_estimator
from xgboost import XGBRegressor

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
__all__ = [
    "rmse_calc",
    "BestEstimatorCV",
    "setup_gridsearch_model",
    "plot_compared_methods_rmse",
    # additional cleaned function names can be listed here
]


def print_umap(X_tsne, y_tsne, min_dist=0, df=None):
    # The shape mostly depends on (n_neighbors, min_dist)
    reducer = umap.UMAP(n_neighbors=10, min_dist=min_dist, random_state=42)
    umap_result = reducer.fit_transform(X_tsne)
    # umap_result.shape
    umap_df = pd.DataFrame(
        {"umap_1": umap_result[:, 0], "umap_2": umap_result[:, 1], "label": y_tsne}
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.scatterplot(x="umap_1", y="umap_2", hue="label", data=umap_df, ax=ax)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    to_adjusted = []
    # If need to add tags.
    if df is not None:
        df_all = df
        for i in range(len(umap_result)):
            if df.loc[i, "Glass (kPa)_max"] > 250:
                to_adjusted.append(
                    plt.text(
                        umap_result[i][0],
                        umap_result[i][1],
                        round(df_all.loc[i, "Glass (kPa)_max"]),
                        fontsize=10,
                    )
                )
        adjust_text(to_adjusted)

    # # plt.savefig('pca_figs/90_2_umap_scatter.pdf',bbox_inches='tight')
    umap_df["Fa"] = df["Glass (kPa)_max"]
    umap_df.to_csv(f"pca_figs/umap_df_341_min_dist{min_dist}.csv")
    plt.savefig(
        f"pca_figs/umap_scatter_train_341_min_dist{min_dist}.pdf", bbox_inches="tight"
    )
    plt.clf()


def rmse_calc(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


class BestEstimatorCV:
    """For containing the best_estimator and related stats, plots."""

    def __init__(self, cv, X, y, estimator, redc=False, random_state=929):
        """Prepare train, test data.

        Args:
            cv (int): cv folds
            X (df): Train data
            y (df): Test data
            estimator (Regressor): best_estimator_
            redc (bool, optional): If sort. Defaults to False.
            random_state (int, optional): rs. Defaults to 929.
        """
        self.cv = cv
        self.X = X
        self.y = y
        self.estimator = estimator
        self.random_state = random_state
        self.redc_model = redc

    def output_stats(self):
        """Collect the RMSE stats.
        Conduct a 10-fold CV with shuffle for the current object. Collect the statistics.
        Want to shuffle. If not shuffle, then cross_val_score (cross_validate) is enough. (But can pass a Kfold(shuffle=True) object)
        Actually this func can be replaced by cross_validate(model, X, y, scoring=['neg_root_mean_squared_error', 'r2'], cv=cvf, return_train_score=True)
        scores['test_score'] = cross_validate()
        scores['test_r2'] = cross_validate()
        scores['train_score'] = cross_validate()

        Args:
            cvf = KFold(n_splits=10 ,random_state=1126, shuffle=True)

        Returns:
            _type_: _description_
        """
        cvf = KFold(n_splits=self.cv, random_state=1126, shuffle=True)
        model = self.estimator
        err_trn = []
        err_tes = []
        r_2_tes = []
        r_2_trn = []
        mae_trn = []
        mae_tes = []
        test_point = pd.DataFrame(columns=self.X.columns)
        for train_index, test_index in cvf.split(self.X):
            x_trn = pd.DataFrame(np.array(self.X)[train_index], columns=self.X.columns)
            x_tes = pd.DataFrame(np.array(self.X)[test_index], columns=self.X.columns)
            y_trn = np.array(self.y)[train_index]
            y_tes = np.array(self.y)[test_index]
            if model.__class__.__name__.startswith("Tab"):
                model.fit(
                    x_trn.to_numpy(),
                    y_trn.reshape(-1, 1),
                    batch_size=32,
                    virtual_batch_size=32,
                )
                x_trn_pred = model.predict(x_trn.to_numpy())
                x_tes_pred = model.predict(x_tes.to_numpy())
            else:
                model.fit(x_trn, y_trn)
                x_trn_pred = model.predict(x_trn)
                x_tes_pred = model.predict(x_tes)

            point = pd.concat([x_tes, pd.Series(y_tes), pd.Series(x_tes_pred)], axis=1)
            test_point = pd.concat([test_point, point])  # Not used.

            err_tes.append(mean_squared_error(x_tes_pred, y_tes))
            err_trn.append(mean_squared_error(x_trn_pred, y_trn))
            mae_tes.append(mean_absolute_error(x_tes_pred, y_tes))
            mae_trn.append(mean_absolute_error(x_trn_pred, y_trn))
            r_2_tes.append(r2_score(y_tes, x_tes_pred))
            r_2_trn.append(r2_score(y_trn, x_trn_pred))
        v_tes = np.sqrt(np.array(err_tes))  # 5 fold
        v_trn = np.sqrt(np.array(err_trn))
        print(
            "RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... train"
            % (
                v_trn.mean(),
                v_trn.std(),
                v_trn.min(),
                v_trn.max(),
                np.array(r_2_trn).mean(),
            )
        )  # 5 fold.
        print(
            "RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... test"
            % (
                v_tes.mean(),
                v_tes.std(),
                v_tes.min(),
                v_tes.max(),
                np.array(r_2_tes).mean(),
            )
        )
        ret = {}
        ret[
            "trn_mean"
        ] = v_trn.mean()  # 5 fold mean. It's different from the overal rmse.
        ret["trn_std"] = v_trn.std()
        ret["trn_r2"] = np.array(r_2_trn).mean()
        ret["tes_mean"] = v_tes.mean()
        ret["tes_std"] = v_tes.std()
        ret["tes_r2"] = np.array(r_2_tes).mean()
        ret["tes_mae"] = np.array(mae_tes).mean()
        ret["trn_mae"] = np.array(mae_trn).mean()
        return ret, v_tes.mean()

    def output_stats_difference(self):
        """Collect the RMSE stats.
        Conduct a 10-fold CV with shuffle for the current object. Collect the statistics.
        Want to shuffle. If not shuffle, then cross_val_score (cross_validate) is enough. (But can pass a Kfold(shuffle=True) object)
        Actually this func can be replaced by cross_validate(model, X, y, scoring=['neg_root_mean_squared_error', 'r2'], cv=cvf, return_train_score=True)
        scores['test_score'] = cross_validate()
        scores['test_r2'] = cross_validate()
        scores['train_score'] = cross_validate()

        Args:
            cvf = KFold(n_splits=10 ,random_state=1126, shuffle=True)

        Returns:
            _type_: _description_
        """
        cvf = KFold(n_splits=self.cv, random_state=1126, shuffle=True)
        model = self.estimator
        err_trn = []
        err_tes = []
        r_2_tes = []
        r_2_trn = []
        mae_trn = []
        mae_tes = []
        diff_collector = []
        y_test_collector = []
        y_pred_collector = []
        test_point = pd.DataFrame(columns=self.X.columns)
        for train_index, test_index in cvf.split(self.X):
            x_trn = pd.DataFrame(np.array(self.X)[train_index], columns=self.X.columns)
            x_tes = pd.DataFrame(np.array(self.X)[test_index], columns=self.X.columns)
            y_trn = np.array(self.y)[train_index]
            y_tes = np.array(self.y)[test_index]
            if model.__class__.__name__.startswith("Tab"):
                model.fit(
                    x_trn.to_numpy(),
                    y_trn.reshape(-1, 1),
                    batch_size=32,
                    virtual_batch_size=32,
                )
                x_trn_pred = model.predict(x_trn.to_numpy())
                x_tes_pred = model.predict(x_tes.to_numpy())
            else:
                model.fit(x_trn, y_trn)
                x_trn_pred = model.predict(x_trn)
                x_tes_pred = model.predict(x_tes)

            point = pd.concat([x_tes, pd.Series(y_tes), pd.Series(x_tes_pred)], axis=1)
            test_point = pd.concat([test_point, point])  # Not used.

            diff_collector.extend(list(y_tes - x_tes_pred))
            y_test_collector.extend(list(y_tes))
            y_pred_collector.extend(list(x_tes_pred))
            err_tes.append(mean_squared_error(x_tes_pred, y_tes))
            err_trn.append(mean_squared_error(x_trn_pred, y_trn))
            mae_tes.append(mean_absolute_error(x_tes_pred, y_tes))
            mae_trn.append(mean_absolute_error(x_trn_pred, y_trn))
            r_2_tes.append(r2_score(y_tes, x_tes_pred))
            r_2_trn.append(r2_score(y_trn, x_trn_pred))
        v_tes = np.sqrt(np.array(err_tes))  # 5 fold
        v_trn = np.sqrt(np.array(err_trn))
        print(
            "RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... train"
            % (
                v_trn.mean(),
                v_trn.std(),
                v_trn.min(),
                v_trn.max(),
                np.array(r_2_trn).mean(),
            )
        )  # 5 fold.
        print(
            "RMSE %1.3f (sd: %1.3f, min:%1.3f, max:%1.3f, det:%1.3f) ... test"
            % (
                v_tes.mean(),
                v_tes.std(),
                v_tes.min(),
                v_tes.max(),
                np.array(r_2_tes).mean(),
            )
        )
        diff_np = np.array(diff_collector)
        min_idx = diff_np.argmin()
        max_idx = diff_np.argmax()
        print(
            "Too optimistic case:", y_test_collector[min_idx], y_test_collector[max_idx]
        )
        print(
            "Too pessimistic case", y_pred_collector[min_idx], y_pred_collector[max_idx]
        )
        plt.hist(diff_collector)

    def plot_hold_out(self, fname, lim_range=[0, 150], save=True, random_state=1126):
        """For a BestEsitimatorCV object, plot a hold-out Truth-Pred fig.
        This only plot hold-out (not cv) Pred-Truth plot. (y_pred vs y_truth plot).

        Args:
            title (str, optional): Use obj.__class__.__name__ now.
            filename (str, optional): Store filename. Defaults to ''.
            save (bool, optional): save the one-shot or not. Defaults to True. False.
            lim_range (list, optional): range for current fig. Defaults to [0, 150].
            with_value (bool, optional): write the value on fig. Defaults to False.

        Return: None

        optional: with_value: whether to write out RMSE for train and test data on the plot.
        """
        feat = self.X
        target = self.y
        model = self.estimator

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.set_aspect("equal")

        x_train, x_test, y_train, y_test = train_test_split(
            feat, target, test_size=0.1, random_state=55, shuffle=True
        )
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        ax.plot(y_test, y_test_pred, "ro", alpha=0.5, label="test")
        ax.plot(y_train, y_train_pred, "bo", alpha=0.5, label="train")
        # df_save_train = pd.DataFrame({'y_train_truth': y_train, 'y_train_pred': y_train_pred})
        # df_save_train.to_csv(f"output/{fname}_train.csv")
        # df_save_test= pd.DataFrame({'y_test_truth':y_test, 'y_test_pred':y_test_pred})
        # df_save_test.to_csv(f"output/{fname}_test.csv")

        ax.plot(lim_range, lim_range, "--")
        ax.set_xlim(lim_range)
        ax.set_ylim(lim_range)
        ax.set_xticks([0, 30, 60, 90, 120, 150])
        ax.set_yticks([0, 30, 60, 90, 120, 150])
        # ax.set_title(f"{model.__class__.__name__}")
        ax.set_xlabel("$F_a$ (kPa), Truth")
        ax.set_ylabel("$F_a$ (kPa), Predicted")
        if save:
            plt.savefig(f"output/{fname}.pdf", bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()

    def plot_importance(self, ylabels, topk, save=True, fname="sample"):
        """Like the importance variable used in fitted model.feature_importances_.
        plot_importance(X.columns, topk=6, fname='rf_feature_importance)

        Args:
            ylabels (_type_): yaxis ticks labels.
            topk (_type_): _description_
            save (bool, optional): _description_. Defaults to True.
            fname (str, optional): _description_. Defaults to "sample".
        """
        model = self.estimator
        plt.figure(figsize=(4, 4))
        importances = model.feature_importances_
        indices = np.argsort(importances)
        topk_idx = indices[-topk:]
        plt.barh(
            range(len(topk_idx)), importances[topk_idx], color="blue", align="center"
        )
        plt.yticks(range(len(topk_idx)), ylabels[topk_idx])
        plt.ylim([-1, len(topk_idx)])
        plt.xlabel("Feature Importance")
        # plt.xlim([0, 0.5])
        if save:
            plt.savefig(f"output/{fname}.pdf", bbox_inches="tight")
        plt.close()


def plot_compared_methods_rmse(rmse_dict, type="rmse", updated=False):
    """plt.plot the comparison of rmse and std

    Args:
        rmse_dict (_type_): Contains all the rmse stats for each method.
        rmse_dict['LASSO'] = cvobj.outoput_stats()[0]
    """
    methods = ["Lasso", "Ridge", "KNN", "KRR", "SVR", "ETR", "XGB", "RFR", "GP"]

    tes_mean = [rmse_dict[m]["tes_mean"] for m in methods]
    tes_sd = [rmse_dict[m]["tes_std"] for m in methods]
    trn_mean = [rmse_dict[m]["trn_mean"] for m in methods]
    trn_sd = [rmse_dict[m]["trn_std"] for m in methods]
    trn_r2 = [rmse_dict[m]["trn_r2"] for m in methods]
    tes_r2 = [rmse_dict[m]["tes_r2"] for m in methods]
    trn_mae = [rmse_dict[m]["trn_mae"] for m in methods]
    tes_mae = [rmse_dict[m]["tes_mae"] for m in methods]

    if type == "rmse":
        df_baselines = pd.DataFrame(
            {"Training": trn_mean, "Test": tes_mean}, index=methods
        )
        # 2*sigma = 95% CI
        tes_ci = 2 * np.array(tes_sd)
        trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mean) + 10),
        )
        ax.set_ylabel("RMSE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_216.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons.pdf", bbox_inches="tight")

    elif type == "mae":
        df_baselines = pd.DataFrame(
            {"Training": trn_mae, "Test": tes_mae}, index=methods
        )
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mae) + 10),
        )
        ax.set_ylabel("MAE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig(
                "sns_figs/baseline_comparisons_mae_216.pdf", bbox_inches="tight"
            )
        else:
            plt.savefig("sns_figs/baseline_comparisons_mae.pdf", bbox_inches="tight")

    elif type == "r2":
        df_baselines = pd.DataFrame({"Training": trn_r2, "Test": tes_r2}, index=methods)
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_r2) + 0.5),
        )
        ax.set_ylabel("r2")
        ind = np.arange(len(methods))

        for x, y in zip(ind - 0.2, trn_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_r2_216.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons_r2.pdf", bbox_inches="tight")
    else:
        print("Wrong plot type.")


def plot_compared_methods_rmse_180(rmse_dict, type="rmse", updated=True):
    """plt.plot the comparison of rmse and std

    Args:
        rmse_dict (_type_): Contains all the rmse stats for each method.
        rmse_dict['LASSO'] = cvobj.outoput_stats()[0]
    """
    methods = ["Lasso", "Ridge", "KNN", "KRR", "SVR", "ETR", "XGB", "RFR", "GP"]

    tes_mean = [rmse_dict[m]["tes_mean"] for m in methods]
    tes_sd = [rmse_dict[m]["tes_std"] for m in methods]
    trn_mean = [rmse_dict[m]["trn_mean"] for m in methods]
    trn_sd = [rmse_dict[m]["trn_std"] for m in methods]
    trn_r2 = [rmse_dict[m]["trn_r2"] for m in methods]
    tes_r2 = [rmse_dict[m]["tes_r2"] for m in methods]
    trn_mae = [rmse_dict[m]["trn_mae"] for m in methods]
    tes_mae = [rmse_dict[m]["tes_mae"] for m in methods]

    if type == "rmse":
        df_baselines = pd.DataFrame(
            {"Training": trn_mean, "Test": tes_mean}, index=methods
        )
        # 2*sigma = 95% CI
        tes_ci = 2 * np.array(tes_sd)
        trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mean) + 10),
        )
        ax.set_ylabel("RMSE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_180.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons.pdf", bbox_inches="tight")

    elif type == "mae":
        df_baselines = pd.DataFrame(
            {"Training": trn_mae, "Test": tes_mae}, index=methods
        )
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mae) + 10),
        )
        ax.set_ylabel("MAE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig(
                "sns_figs/baseline_comparisons_mae_180.pdf", bbox_inches="tight"
            )
        else:
            plt.savefig("sns_figs/baseline_comparisons_mae.pdf", bbox_inches="tight")

    elif type == "r2":
        df_baselines = pd.DataFrame({"Training": trn_r2, "Test": tes_r2}, index=methods)
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_r2) + 0.5),
        )
        ax.set_ylabel("r2")
        ind = np.arange(len(methods))

        for x, y in zip(ind - 0.2, trn_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_r2_180.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons_r2.pdf", bbox_inches="tight")
    else:
        print("Wrong plot type.")


def plot_compared_methods_rmse_289(rmse_dict, type="rmse", updated=True):
    """plt.plot the comparison of rmse and std

    Args:
        rmse_dict (_type_): Contains all the rmse stats for each method.
        rmse_dict['LASSO'] = cvobj.outoput_stats()[0]
    """
    methods = ["Lasso", "Ridge", "KNN", "KRR", "SVR", "ETR", "XGB", "RFR", "GP"]

    tes_mean = [rmse_dict[m]["tes_mean"] for m in methods]
    tes_sd = [rmse_dict[m]["tes_std"] for m in methods]
    trn_mean = [rmse_dict[m]["trn_mean"] for m in methods]
    trn_sd = [rmse_dict[m]["trn_std"] for m in methods]
    trn_r2 = [rmse_dict[m]["trn_r2"] for m in methods]
    tes_r2 = [rmse_dict[m]["tes_r2"] for m in methods]
    trn_mae = [rmse_dict[m]["trn_mae"] for m in methods]
    tes_mae = [rmse_dict[m]["tes_mae"] for m in methods]

    if type == "rmse":
        df_baselines = pd.DataFrame(
            {"Training": trn_mean, "Test": tes_mean}, index=methods
        )
        # 2*sigma = 95% CI
        tes_ci = 2 * np.array(tes_sd)
        trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mean) + 10),
        )
        ax.set_ylabel("RMSE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_289.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons.pdf", bbox_inches="tight")

    elif type == "mae":
        df_baselines = pd.DataFrame(
            {"Training": trn_mae, "Test": tes_mae}, index=methods
        )
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mae) + 10),
        )
        ax.set_ylabel("MAE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig(
                "sns_figs/baseline_comparisons_mae_289.pdf", bbox_inches="tight"
            )
        else:
            plt.savefig("sns_figs/baseline_comparisons_mae.pdf", bbox_inches="tight")

    elif type == "r2":
        df_baselines = pd.DataFrame({"Training": trn_r2, "Test": tes_r2}, index=methods)
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_r2) + 0.5),
        )
        ax.set_ylabel("r2")
        ind = np.arange(len(methods))

        for x, y in zip(ind - 0.2, trn_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_r2_289.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons_r2.pdf", bbox_inches="tight")
    else:
        print("Wrong plot type.")


def plot_compared_methods_rmse_316(rmse_dict, type="rmse", updated=True):
    """plt.plot the comparison of rmse and std

    Args:
        rmse_dict (_type_): Contains all the rmse stats for each method.
        rmse_dict['LASSO'] = cvobj.outoput_stats()[0]
    """
    methods = ["Lasso", "Ridge", "KNN", "KRR", "SVR", "ETR", "XGB", "RFR", "GP"]

    tes_mean = [rmse_dict[m]["tes_mean"] for m in methods]
    tes_sd = [rmse_dict[m]["tes_std"] for m in methods]
    trn_mean = [rmse_dict[m]["trn_mean"] for m in methods]
    trn_sd = [rmse_dict[m]["trn_std"] for m in methods]
    trn_r2 = [rmse_dict[m]["trn_r2"] for m in methods]
    tes_r2 = [rmse_dict[m]["tes_r2"] for m in methods]
    trn_mae = [rmse_dict[m]["trn_mae"] for m in methods]
    tes_mae = [rmse_dict[m]["tes_mae"] for m in methods]

    if type == "rmse":
        df_baselines = pd.DataFrame(
            {"Training": trn_mean, "Test": tes_mean}, index=methods
        )
        # 2*sigma = 95% CI
        tes_ci = 2 * np.array(tes_sd)
        trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mean) + 10),
        )
        ax.set_ylabel("RMSE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_316.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons.pdf", bbox_inches="tight")

    elif type == "mae":
        df_baselines = pd.DataFrame(
            {"Training": trn_mae, "Test": tes_mae}, index=methods
        )
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mae) + 10),
        )
        ax.set_ylabel("MAE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig(
                "sns_figs/baseline_comparisons_mae_316.pdf", bbox_inches="tight"
            )
        else:
            plt.savefig("sns_figs/baseline_comparisons_mae.pdf", bbox_inches="tight")

    elif type == "r2":
        df_baselines = pd.DataFrame({"Training": trn_r2, "Test": tes_r2}, index=methods)
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_r2) + 0.5),
        )
        ax.set_ylabel("r2")
        ind = np.arange(len(methods))

        for x, y in zip(ind - 0.2, trn_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_r2_316.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons_r2.pdf", bbox_inches="tight")
    else:
        print("Wrong plot type.")


"""1 repo, for shap figure output."""


def plot_compared_methods_rmse_341(rmse_dict, type="rmse", updated=True):
    """plt.plot the comparison of rmse and std

    Args:
        rmse_dict (_type_): Contains all the rmse stats for each method.
        rmse_dict['LASSO'] = cvobj.outoput_stats()[0]
    """
    methods = ["Lasso", "Ridge", "KNN", "KRR", "SVR", "ETR", "XGB", "RFR", "GP"]

    tes_mean = [rmse_dict[m]["tes_mean"] for m in methods]
    tes_sd = [rmse_dict[m]["tes_std"] for m in methods]
    trn_mean = [rmse_dict[m]["trn_mean"] for m in methods]
    trn_sd = [rmse_dict[m]["trn_std"] for m in methods]
    trn_r2 = [rmse_dict[m]["trn_r2"] for m in methods]
    tes_r2 = [rmse_dict[m]["tes_r2"] for m in methods]
    trn_mae = [rmse_dict[m]["trn_mae"] for m in methods]
    tes_mae = [rmse_dict[m]["tes_mae"] for m in methods]

    if type == "rmse":
        df_baselines = pd.DataFrame(
            {"Training": trn_mean, "Test": tes_mean}, index=methods
        )
        # 2*sigma = 95% CI
        tes_ci = 2 * np.array(tes_sd)
        trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mean) + 10),
        )
        ax.set_ylabel("RMSE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mean):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_341.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons.pdf", bbox_inches="tight")

    elif type == "mae":
        df_baselines = pd.DataFrame(
            {"Training": trn_mae, "Test": tes_mae}, index=methods
        )
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_mae) + 10),
        )
        ax.set_ylabel("MAE (kPa)")
        ind = np.arange(len(methods))
        # plt.ylabel("RMSE (kPa)")
        # plt.xticks(ind + width / 2, methods, rotation=45, fontsize=12)
        # plt.xlim(-1, len(methods))
        # plt.legend(("Training Error", "Test Error"), loc="upper right", prop={"size": 12})
        # plt.ylim(0, np.max(tes_mean) + 5)
        for x, y in zip(ind - 0.2, trn_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_mae):
            plt.text(
                x,
                y + 0.5,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig(
                "sns_figs/baseline_comparisons_mae_341.pdf", bbox_inches="tight"
            )
        else:
            plt.savefig("sns_figs/baseline_comparisons_mae.pdf", bbox_inches="tight")

    elif type == "r2":
        df_baselines = pd.DataFrame({"Training": trn_r2, "Test": tes_r2}, index=methods)
        # 2*sigma = 95% CI
        # tes_ci = 2 * np.array(tes_sd)
        # trn_ci = 2 * np.array(trn_sd)
        width = 0.8
        ax = df_baselines.plot(
            y=["Training", "Test"],
            kind="bar",
            color=["blue", "red"],
            rot=45,
            alpha=0.4,
            figsize=(6, 4),
            lw=2,
            edgecolor=["black"],
            width=width,
            ylim=(0, np.max(tes_r2) + 0.5),
        )
        ax.set_ylabel("r2")
        ind = np.arange(len(methods))

        for x, y in zip(ind - 0.2, trn_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        for x, y in zip(ind + 0.2, tes_r2):
            plt.text(
                x,
                y + 0.001,
                "%.2f" % y,
                ha="center",
                va="bottom",
                rotation="90",
                fontsize=10,
            )
        if updated:
            plt.savefig("sns_figs/baseline_comparisons_r2_341.pdf", bbox_inches="tight")
        else:
            plt.savefig("sns_figs/baseline_comparisons_r2.pdf", bbox_inches="tight")
    else:
        print("Wrong plot type.")


def plot_shap_waterfall(
    model,
    feat,
    target=None,
    base_val=None,
    extended=False,
    figsize=None,
    save=None,
    title_name=None,
):
    # model = ExtraTreesRegressor(n_estimators=100, random_state=1107, n_jobs=4)
    # model.fit(feat, target)

    explainer = shap.Explainer(model, feat)
    shap_values = explainer(feat)
    shap_values.values = np.round(shap_values.values, 2)
    shap_values.base_values = np.round(shap_values.base_values, 2)
    shap_values.data = np.round(shap_values.data, 2)

    idx_range = [41, 106, 179, 85] if not extended else [181, 183, 182, 187]

    for i in idx_range:
        plt.figure(facecolor="white", figsize=(4, 4))
        shap.plots.waterfall(shap_values[i], max_display=9, show=False)
        plt.gcf().set_size_inches(4, 4)
        save_file_name = f"shap/shap_water_{i}.pdf"
        plt.savefig(save_file_name, bbox_inches="tight")
        plt.clf()


def crossvalid_plot(xx, yy, model, cvf, xylim=[0, 35]):
    """Plot all cv folds(tests, different colors) on one scatter plots.
    Used only in revised comments. (not published)
    crossvalid_plot(feat, target, model, cvf)

    Args:
        xx (_type_): _description_
        yy (_type_): _description_
        model (_type_): _description_
        cvf (_type_): _description_
        xylim (list, optional): _description_. Defaults to [0, 35].
    """
    err_trn = []
    err_tes = []
    r_2_tes = []
    r_2_trn = []
    count = 0
    for train_index, test_index in cvf.split(xx):
        count += 1
        x_trn = np.array(xx)[train_index]
        x_tes = np.array(xx)[test_index]
        y_trn = np.array(yy)[train_index]
        y_tes = np.array(yy)[test_index]
        model.fit(x_trn, y_trn)
        x_trn_pred = model.predict(x_trn)
        x_tes_pred = model.predict(x_tes)
        plt.scatter(y_tes, x_tes_pred, s=3, label=f"Fold {count}")

    plt.plot([-100, 200], [-100, 200], c="0", ls="-", lw=1.0)
    plt.xlim(xylim)
    plt.ylim(xylim)
    plt.xlabel("Experimental Yield [%]")
    plt.ylabel("Predicted Yield [%]")
    # plt.legend()
    return


def normalize(ratio_arr):
    return [float(r) / sum(ratio_arr) for r in ratio_arr]


def pred_ei_x_to_df_local_pred(res, num, y_max=-146.64):
    """Already remove the bug of EI computation."""
    round2 = lambda x: round(x, 2)
    point_idx = list()
    pred_list = list()
    pred_local = list()
    pred_mid = list()
    ei_list = list()
    x_list = list()
    df_dict = {
        "point_idx": point_idx,
        "pred_value": pred_list,
        "ei_value": ei_list,
        "x": x_list,
        "pred_local": pred_local,
        "pred_mid": pred_mid,
    }
    for i, v in enumerate(res.func_vals):
        if i < num:
            ei_value = gaussian_ei([res.x_iters[i]], res.models[0], y_opt=y_max)
        else:
            ei_value = gaussian_ei([res.x_iters[i]], res.models[i - num], y_opt=y_max)
        point_idx.append(i)
        if v > 0:
            pred_list.append(v)
        else:
            pred_list.append(-v)
        ei_list.append(ei_value[0])
        x_list.append(list(map(round2, normalize(res.x_iters[i]))))
        if i < num:
            pred_local.append(-res.models[0].predict([(res.x_iters[i])])[0])
        else:
            pred_local.append(-res.models[i - num].predict([(res.x_iters[i])])[0])
        if i < num:
            pred_mid.append(-res.models[0].predict([(res.x_iters[i])])[0])
        else:
            pred_mid.append(-res.models[-1].predict([(res.x_iters[i])])[0])

        # ic(i, v, ei_value)
    df = pd.DataFrame(df_dict)
    return df


def pred_ei_x_to_df(res, num, y_max=-146.64):
    """Already remove the bug of EI computation."""
    round2 = lambda x: round(x, 2)
    point_idx = list()
    pred_list = list()
    ei_list = list()
    x_list = list()
    df_dict = {
        "point_idx": point_idx,
        "pred_value": pred_list,
        "ei_value": ei_list,
        "x": x_list,
    }
    for i, v in enumerate(res.func_vals):
        if i < num:
            ei_value = gaussian_ei([res.x_iters[i]], res.models[0], y_opt=y_max)
        else:
            ei_value = gaussian_ei([res.x_iters[i]], res.models[i - num], y_opt=y_max)
        point_idx.append(i)
        if v > 0:
            pred_list.append(v)
        else:
            pred_list.append(-v)
        ei_list.append(ei_value[0])
        x_list.append(list(map(round2, normalize(res.x_iters[i]))))
        # ic(i, v, ei_value)
    df = pd.DataFrame(df_dict)
    return df


def pred_ei_x_to_df_LP(res, num):
    round2 = lambda x: round(x, 2)
    point_idx = list()
    pred_list = list()
    ei_list = list()
    x_list = list()
    df_dict = {
        "point_idx": point_idx,
        "pred_value": pred_list,
        "ei_value": ei_list,
        "x": x_list,
    }
    for i, v in enumerate(res.Y):
        if i < num:
            # ei_value = gaussian_ei([res.x_iters[i]], res.models[0], y_opt=-146.64)
            ei_value = 0
        else:
            # ei_value = gaussian_ei([res.x_iters[i]], res.models[i-num], y_opt=-146.64)
            ei_value = 0
        point_idx.append(i)
        if v > 0:
            pred_list.append(v)
        else:
            pred_list.append(-v)
        ei_list.append(ei_value)
        x_list.append(list(map(round2, normalize(res.X[i]))))
        # ic(i, v, ei_value)
    df = pd.DataFrame(df_dict)
    return df


def pred_ei_x_to_df_base(res, num):
    round2 = lambda x: round(x, 2)
    point_idx = list()
    pred_list = list()
    ei_list = list()
    x_list = list()
    df_dict = {
        "point_idx": point_idx,
        "pred_value": pred_list,
        "ei_value": ei_list,
        "x": x_list,
    }
    for i, v in enumerate(res.func_vals):
        if i < num:
            ei_value = gaussian_ei([res.x_iters[i]], res.models[0], y_opt=-146.64)
        else:
            ei_value = gaussian_ei([res.x_iters[i]], res.models[0], y_opt=-146.64)
        point_idx.append(i)
        if v > 0:
            pred_list.append(v)
        else:
            pred_list.append(-v)
        ei_list.append(ei_value[0])
        x_list.append(list(map(round2, normalize(res.x_iters[i]))))
        # ic(i, v, ei_value)
    df = pd.DataFrame(df_dict)
    return df


def pred_ei_x_to_df216(res):
    round2 = lambda x: round(x, 2)
    point_idx = list()
    pred_list = list()
    ei_list = list()
    x_list = list()
    df_dict = {
        "point_idx": point_idx,
        "pred_value": pred_list,
        "ei_value": ei_list,
        "x": x_list,
    }
    for i, v in enumerate(res.func_vals):
        if i < 226:
            # ic(i, res.x_iters[i])
            ei_value = gaussian_ei([res.x_iters[i]], res.models[0], y_opt=-308.90)
        else:
            ei_value = gaussian_ei([res.x_iters[i]], res.models[i - 225], y_opt=-308.90)
        point_idx.append(i)
        if v > 0:
            pred_list.append(v)
        else:
            pred_list.append(-v)
        ei_list.append(ei_value[0])
        x_list.append(list(map(round2, normalize(res.x_iters[i]))))
        # ic(i, v, ei_value)
    df = pd.DataFrame(df_dict)
    return df


def load_logs(file):
    log_func_vals = []
    log_x_iters = []
    log_idx = []
    with open(file, "r") as f:
        line_list = f.readlines()
        for i, v in enumerate(line_list):
            if i < 13:
                continue
            v = v.strip()
            v = v.split()
            vec = "".join(v[1:7])
            ratio_list = ast.literal_eval(vec)
            p_idx = int(v[0])
            log_idx.append(p_idx)
            p_pred = float(v[9])
            log_func_vals.append(p_pred)
            log_x_iters.append(ratio_list)

    recover_order = np.argsort(log_idx)
    x = [log_x_iters[i] for i in recover_order]
    y = [log_func_vals[i] for i in recover_order]
    return x, y


def load_logs216(file):
    log_func_vals = []
    log_x_iters = []
    log_idx = []
    with open(file, "r") as f:
        line_list = f.readlines()
        for i, v in enumerate(line_list):
            if i < 15:
                continue
            v = v.strip()
            v = v.split()
            vec = "".join(v[1:7])
            ratio_list = ast.literal_eval(vec)
            p_idx = int(v[0])
            log_idx.append(p_idx)
            p_pred = float(v[9])
            log_func_vals.append(p_pred)
            log_x_iters.append(ratio_list)

    recover_order = np.argsort(log_idx)
    x = [log_x_iters[i] for i in recover_order]
    y = [log_func_vals[i] for i in recover_order]
    return x, y


def load_logs_auto(file):
    log_func_vals = []
    log_x_iters = []
    log_idx = []
    with open(file, "r") as f:
        line_list = f.readlines()
        for i, v in enumerate(line_list):
            v = v.strip()
            v = v.split()
            if len(str(v[-1])) < 7:  # last entry < 7
                continue
            vec = "".join(v[1:7])
            ratio_list = ast.literal_eval(vec)
            p_idx = int(v[0])
            log_idx.append(p_idx)
            p_pred = float(v[9])
            log_func_vals.append(p_pred)
            log_x_iters.append(ratio_list)

    recover_order = np.argsort(log_idx)
    x = [log_x_iters[i] for i in recover_order]
    y = [log_func_vals[i] for i in recover_order]
    return x, y


def output_diff_exp_pred(input_file):
    gp_df = pd.read_csv(input_file)
    exp_np = gp_df["exp_value"].to_numpy()
    gp_df = gp_df[gp_df["point_idx"] > 189].sort_values("pred_value", ascending=False)
    # ic(gp_df)
    # ax = gp_df.plot(x=0, kind='bar', title='Promising candidates sorted by predicted values', figsize=(20,10), secondary_y=['ei_value'])
    # plt.savefig('ranked_by_predicted.pdf', bbox_inches='tight')
    gp_df["exp_value"] = exp_np
    gp_df = gp_df[gp_df["exp_value"] > 0]
    gp_df["exp_pred_diff"] = gp_df["exp_value"] - gp_df["pred_value"]
    gp_df.plot(
        kind="bar",
        x=0,
        y="exp_pred_diff",
        figsize=(4, 1),
        xlabel=f"{input_file}-difference",
        legend=False,
    )
    plt.savefig(f"clustering/{input_file}_diff.pdf", bbox_inches="tight")
    gp_df.to_csv(f"clustering/{input_file}_diff.csv")


def return_sorted_df(input_file):
    gp_df = pd.read_csv(input_file)
    exp_np = gp_df["exp_value"].to_numpy()
    gp_df = gp_df[gp_df["point_idx"] > 189].sort_values("pred_value", ascending=False)
    gp_df["exp_value"] = exp_np
    gp_df = gp_df[gp_df["point_idx"] > 189].sort_values("point_idx", ascending=True)
    return gp_df
    # ic(gp_df)
    # ax = gp_df.plot(x=0, kind='bar', title='Promising candidates sorted by predicted values', figsize=(20,10), secondary_y=['ei_value'])
    # plt.savefig('ranked_by_predicted.pdf', bbox_inches='tight')


def output_format_df(input_file, label):
    gp_df = pd.read_csv(input_file)
    exp_np = gp_df["exp_value"].to_numpy()
    gp_df = gp_df[gp_df["point_idx"] > 189].sort_values("pred_value", ascending=False)
    # ic(gp_df)
    # ax = gp_df.plot(x=0, kind='bar', title='Promising candidates sorted by predicted values', figsize=(20,10), secondary_y=['ei_value'])
    # plt.savefig('ranked_by_predicted.pdf', bbox_inches='tight')
    x_columns = [
        "Nucleophilic-HEA",
        "Hydrophobic-BA",
        "Acidic-CBEA",
        "Cationic-ATAC",
        "Aromatic-PEA",
        "Amide-AAm",
    ]
    y_column = "Glass (kPa)_10s"
    gp_df["exp_value"] = exp_np
    coords_list = gp_df["x"].to_list()
    c_lists = [ast.literal_eval(coord) for coord in coords_list]
    new_df = pd.DataFrame(c_lists, columns=x_columns)
    new_df["Glass (kPa)_10s"] = exp_np
    new_df["label"] = label

    # gp_df.rename(columns={'exp_value':'Glass (kPa)_10s'}, inplace=True))
    # gp_df = gp_df[gp_df['exp_value']>0]
    # gp_df['exp_pred_diff'] = gp_df['exp_value'] - gp_df['pred_value']
    # gp_df.plot(kind='bar', x=0, y='exp_pred_diff', figsize=(4,1), xlabel=f'{input_file}-difference', legend=False)
    # plt.savefig(f'clustering/{input_file}_diff.pdf', bbox_inches='tight')
    # gp_df.to_csv(f'clustering/{input_file}_diff.csv')
    return new_df


def setup_gridsearch_model(model, rs=1126):
    """Setup the GridSearchCV and params ranges for a estimator.
        - DummyRegressor
        - LassoCV, RidgeCV
        - KneighborsRegressor, KernelRidge, SVR
        - RandomForest, ExtraTrees, XGB (LGBM)
        - MLPRegressor (from skorch)

    # xgb ranges
    ntree = 500
    range_depth = [6, 7, 8]
    range_subsample = [0.8, 0.9, 1]
    range_colsample = [0.8, 0.9, 1]
    range_lr = [0.1, 0.05]

    Args:
        model (string): str that tells the model.
        rs (int, optional): random state. Defaults to 1126.

    Returns:
        GridSearchCV: object.
    """
    cv_object = None
    if model == "RFR":
        cv_object = GridSearchCV(
            opt_RFR(n_jobs=-1, n_estimators=200, random_state=rs),
            param_grid={
                "max_features": ["sqrt", 1.0],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_leaf_nodes": ["None", 70],
            },
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
    elif model == "RFRsk":
        cv_object = GridSearchCV(
            RandomForestRegressor(n_jobs=-1, n_estimators=200, random_state=rs),
            param_grid={
                "max_features": ["sqrt", 1.0],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_leaf_nodes": ["None", 70],
            },
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
    elif model == "ETR":
        cv_object = GridSearchCV(
            ExtraTreesRegressor(n_jobs=-1, n_estimators=200, random_state=rs),
            param_grid={
                "max_features": ["sqrt", 1.0],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_leaf_nodes": ["None", 70],
                "bootstrap": [True, False],
            },
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
    elif model == "XGB":
        cv_object = GridSearchCV(
            XGBRegressor(n_jobs=-1, importance_type="total_gain", random_state=rs),
            param_grid={
                "n_estimators": [200, 500, 1000],
                "max_depth": [6, 7, 8],
                "learning_rate": [0.1, 0.05],
                "subsample": [0.8, 0.9, 1],
                "colsample_bytree": [0.8, 0.9, 1],
            },
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
    elif model == "LASSO":
        cv_object = GridSearchCV(
            Lasso(random_state=rs),
            param_grid={"alpha": [1e-2, 1e-1, 1e0, 1e1, 1e2]},
            cv=5,
            n_jobs=-1,
        )
    elif model == "RIDGE":
        cv_object = GridSearchCV(
            Ridge(random_state=rs),
            param_grid={"alpha": [1e-2, 1e-1, 1e0, 1e1, 1e2]},
            cv=5,
            n_jobs=-1,
        )
    elif model == "KRR":
        cv_object = GridSearchCV(
            KernelRidge(kernel="rbf"),
            param_grid={
                "alpha": [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                "gamma": [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            },
            cv=5,
            n_jobs=-1,
        )
    elif model == "SVR":
        cv_object = GridSearchCV(
            SVR(kernel="rbf"),
            param_grid={
                "C": [1.0, 1e1, 1e2, 1e3, 1e4, 1e5],
                "gamma": [
                    1.0,
                    1e-1,
                    1e-2,
                    1e-3,
                    1e-4,
                    1e-5,
                    1e-6,
                    1e-7,
                    1e-8,
                    1e-9,
                    1e-10,
                ],
                "epsilon": [1e-2, 1e-1, 1e0, 1e1, 1e2],
            },
            cv=5,
            n_jobs=-1,
        )
    elif model == "KNN":
        cv_object = GridSearchCV(
            KNeighborsRegressor(),
            param_grid={"n_neighbors": [2, 4, 8, 16], "p": [2, 3]},
            cv=5,
            n_jobs=-1,
        )
    elif model == "MLP":
        cv_object = GridSearchCV(
            MLPRegressor(random_state=rs),
            param_grid={
                "hidden_layer_sizes": [(100,), (50,)],
            },
            cv=5,
            n_jobs=-1,
        )
    elif model == "TABNET":
        cv_object = GridSearchCV(
            DummyRegressor(),
            # param_grid={"n_d": [8, 16, 32], "n_a": [8, 16, 32], 'n_steps': [1, 4, 7], 'gamma': [1.0, 1.5, 2.0]},
            param_grid={},
            cv=5,
            n_jobs=-1,
            scoring="r2",
        )
    elif model == "GP":
        base_estimator = cook_estimator(
            "GP",
            space=[
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
            ],
            random_state=rs,
            noise="gaussian",
        )
        cv_object = GridSearchCV(
            base_estimator,
            param_grid={"alpha": [1e-10]},
            cv=5,
            n_jobs=-1,
        )
    elif model == "DUM":
        cv_object = GridSearchCV(
            DummyRegressor(),
            param_grid={
                "strategy": ["mean", "median"],
            },
            cv=5,
            n_jobs=-1,
        )
    else:
        warnings.warn("Machine Learning model cannot defined")
    return cv_object
