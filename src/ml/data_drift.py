import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from ..models.load import Load


class DataDrift:
    def __init__(self, train_data: str, current_data: str):
        self.df_train = Load(train_data).load_dataset()
        self.df_current_raw = Load(current_data).load_dataset()

        self.df_train = self.clean_columns(self.df_train)
        self.df_current_raw = self.clean_columns(self.df_current_raw)

        self.df_current = self.preprocess_current_data(self.df_current_raw)

    def clean_columns(self, df):
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
        df.columns = df.columns.str.strip()
        return df

    def preprocess_current_data(self, df):
        df = df.copy()

        if "start_session" not in df.columns or "end_session" not in df.columns:
            return df

        df["start_session"] = pd.to_datetime(df["start_session"], format="mixed")
        df["end_session"] = pd.to_datetime(df["end_session"], format="mixed")

        df["duration"] = (df["end_session"] - df["start_session"]).dt.total_seconds()
        df["total_packets_used"] = df["packets"]
        df["bytes_flow"] = df["bytes"]
        df["bytes_per_packet"] = df["bytes"] / df["packets"]
        df["packets_per_seconds"] = df["packets"] / df["duration"].replace(0, 0.001)
        df["hour_of_day"] = df["start_session"].dt.hour

        common_ports = [
            20,
            21,
            22,
            23,
            25,
            53,
            67,
            68,
            69,
            80,
            110,
            123,
            143,
            161,
            443,
            445,
            993,
            995,
        ]
        df["is_common_port"] = (
            df["source_port"].astype(int).isin(common_ports).astype(int)
        )

        df["has_SYN"] = (df["flag"] == "SYN").astype(int)
        df["has_ACK"] = (df["flag"] == "ACK").astype(int)
        df["has_RST"] = (df["flag"] == "RST").astype(int)
        df["has_FIN"] = (df["flag"] == "FIN").astype(int)

        selected_cols = [
            "duration",
            "total_packets_used",
            "bytes_flow",
            "bytes_per_packet",
            "packets_per_seconds",
            "hour_of_day",
            "is_common_port",
            "has_SYN",
            "has_ACK",
            "has_RST",
            "has_FIN",
        ]

        return df[selected_cols]

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
        except Exception:
            return None

    def generate_histogram(self, feature):
        sns.set(style="whitegrid")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            self.df_train[feature],
            bins=50,
            alpha=0.5,
            label="Train Data",
            color="blue",
            edgecolor="black",
        )
        ax.hist(
            self.df_current[feature],
            bins=50,
            alpha=0.5,
            label="Current Data",
            color="orange",
            edgecolor="black",
        )

        train_mean = self.df_train[feature].mean()
        current_mean = self.df_current[feature].mean()
        ax.axvline(
            train_mean,
            color="blue",
            linestyle="dashed",
            linewidth=2,
            label=f"Train Mean: {train_mean:.2f}",
        )
        ax.axvline(
            current_mean,
            color="orange",
            linestyle="dashed",
            linewidth=2,
            label=f"Current Mean: {current_mean:.2f}",
        )

        ax.set_xlabel(f"{feature} Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Distribution of {feature}", fontsize=14)

        ax.legend(loc="upper right")

        fig.tight_layout()

        return fig
