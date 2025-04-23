from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import numpy as np
import matplotlib.pyplot as plt

from ..models.load import Load

class DataDrift:
    def __init__(self, train_data: str, current_data: str):
        self.df_train = Load(train_data).load_dataset()
        self.df_current = Load(current_data).load_dataset()

        self.df_train = self.df_train.drop(columns=["Unnamed: 0"], errors="ignore")
        self.df_current = self.df_current.drop(columns=["Unnamed: 0"], errors="ignore")

        common_cols = self.df_train.columns.intersection(self.df_current.columns)
        self.df_train = self.df_train[common_cols]
        self.df_current = self.df_current[common_cols]

    def kolmogorov_smirnov(self, feature):       

        stat, p_value = ks_2samp(self.df_train[feature], self.df_current[feature])
        return stat, p_value

    def jensen_shannon(self, feature):      

        try:
            p = self.df_train[feature].value_counts(normalize=True).sort_index()
            q = self.df_current[feature].value_counts(normalize=True).sort_index()

            all_idx = sorted(set(p.index).union(set(q.index)))
            p = p.reindex(all_idx, fill_value=0)
            q = q.reindex(all_idx, fill_value=0)

            distance = jensenshannon(p, q)
            return distance
        except Exception as e:
            return None

    def generate_histogram(self, feature):    

        fig, ax = plt.subplots()
        ax.hist(
            self.df_train[feature],
            bins=50,
            alpha=0.5,
            label="Train Data"
        )
        ax.hist(
            self.df_current[feature],
            bins=50,
            alpha=0.5,
            label="Current Data"
        )
        ax.legend()
        ax.set_title(f"Distribution of {feature}")
        return fig