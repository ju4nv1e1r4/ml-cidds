import os
import io
import datetime
import hashlib
import json
import logging
import pandas as pd

from utils.gcp import CloudStorageOps

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureStore:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.ops = CloudStorageOps(bucket_name)
        self.version_history = {}
        self.features_metadata = {}

    def metadata(
        self,
        feature_group_id,
        feature_group_name,
        feature_group_description,
        source,
        usage,
    ):
        """
        Generates metadata for a feature group

        Args:
        feature_group_id (str): Unique identifier for the feature group
        feature_group_name (str): Name of the feature group
        feature_group_description (str): Description of the feature group
        source (str): Data source
        usage (str): Description of the intended use for ML

        Returns:
        dict: Dictionary with the metadata
        """
        created_at = datetime.datetime.now().isoformat()

        metadata = {
            "feature_group_id": feature_group_id,
            "feature_group_name": feature_group_name,
            "feature_group_description": feature_group_description,
            "created_at": created_at,
            "source": source,
            "usage": usage,
            "features": [],
            "version": "1.0",
        }

        self.features_metadata[feature_group_id] = metadata

        return metadata

    def add_feature_metadata(
        self, feature_group_id, feature_name, feature_type, description=None, stats=None
    ):
        """
        Adds metadata for a specific feature

        Args:
        feature_group_id (str): ID of the feature group this feature belongs to
        feature_name (str): Name of the feature
        feature_type (str): Data type of the feature (int, float, string, etc.)
        description (str, optional): Description of the feature
        stats (dict, optional): Statistics about the feature (avg, min, max, etc.)
        """
        if feature_group_id not in self.features_metadata:
            raise ValueError(f"Feature groups {feature_group_id} not found.")

        feature_info = {
            "name": feature_name,
            "type": feature_type,
            "description": description or "",
            "stats": stats or {},
        }

        self.features_metadata[feature_group_id]["features"].append(feature_info)

    def calculate_dataframe_stats(self, df):
        """
        Calculates basic statistics for each column of the DataFrame

        Args:
        df (pandas.DataFrame): DataFrame to calculate statistics on

        Returns:
        dict: Dictionary with statistics for each column
        """
        stats = {}

        for column in df.columns:
            column_stats = {}

            if pd.api.types.is_numeric_dtype(df[column]):
                column_stats = {
                    "min": (
                        float(df[column].min()) if not df[column].isna().all() else None
                    ),
                    "max": (
                        float(df[column].max()) if not df[column].isna().all() else None
                    ),
                    "mean": (
                        float(df[column].mean())
                        if not df[column].isna().all()
                        else None
                    ),
                    "null_count": int(df[column].isna().sum()),
                    "type": str(df[column].dtype),
                }
            else:
                column_stats = {
                    "unique_values": int(df[column].nunique()),
                    "null_count": int(df[column].isna().sum()),
                    "type": str(df[column].dtype),
                }
                if df[column].nunique() < 10:
                    column_stats["value_counts"] = df[column].value_counts().to_dict()

            stats[column] = column_stats

        return stats

    def version_control(self, feature_group_id, df, version=None):
        """
        Implements simple versioning for data

        Args:
        feature_group_id (str): ID of the feature group
        df (pandas.DataFrame): DataFrame to be versioned
        version (str, optional): Version to assign. If None, will be incremented.

        Returns:
        str: Current version of the data
        """
        df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

        if feature_group_id not in self.version_history:
            self.version_history[feature_group_id] = []

        for v in self.version_history[feature_group_id]:
            if v["hash"] == df_hash:
                return v["version"]

        if version is None:
            if not self.version_history[feature_group_id]:
                new_version = "1.0"
            else:
                last_version = self.version_history[feature_group_id][-1]["version"]
                major, minor = map(int, last_version.split("."))
                new_version = f"{major}.{minor + 1}"
        else:
            new_version = version

        version_info = {
            "version": new_version,
            "hash": df_hash,
            "timestamp": datetime.datetime.now().isoformat(),
            "rows": len(df),
            "columns": list(df.columns),
        }

        self.version_history[feature_group_id].append(version_info)

        if feature_group_id in self.features_metadata:
            self.features_metadata[feature_group_id]["version"] = new_version

        return new_version

    def save_metadata(self, feature_group_id, local_path=None):
        """
        Saves the metadata to a JSON file

        Args:
        feature_group_id (str): Feature group ID
        local_path (str, optional): Local path to save the file

        Returns:
        str: Path to the saved file
        """
        if feature_group_id not in self.features_metadata:
            raise ValueError(f"Grupo de features {feature_group_id} não encontrado")

        metadata = self.features_metadata[feature_group_id]

        if feature_group_id in self.version_history:
            metadata["version_history"] = self.version_history[feature_group_id]

        if local_path is None:
            local_path = f"load/feature_store/metadata_{feature_group_id}.json"

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "w") as f:
            json.dump(metadata, f, indent=2)

        cloud_path = f"feature_store/metadata/{feature_group_id}.json"
        self.ops.upload_file_to_bucket(local_path, cloud_path)

        return local_path

    def grouping_features(
        self, blob, feature_group_id, name, description, source, usage
    ):
        """
        Groups features and saves them into separate files with metadata

        Args:
        blob (str): Path to csv file in bucket
        feature_group_id (str): Feature group ID
        name (str): Feature group name
        description (str): Feature group description
        source (str): Data source
        usage (str): Intended use for ML

        Returns:
        dict: Feature group metadata
        """
        print(f"Processing feature group: {name}")

        file_bytes = self.ops.load_from_bucket(blob)
        df = pd.read_csv(io.BytesIO(file_bytes))

        metadata = self.metadata(
            feature_group_id=feature_group_id,
            feature_group_name=name,
            feature_group_description=description,
            source=source,
            usage=usage,
        )

        feature_groups = {
            "sessions": ["duration", "hour_of_day"],
            "workload": ["bytes_flow", "total_packets_used", "packets_per_seconds"],
            "network": [
                "is_common_port",
                "source_ip_freq",
                "dest_ip_freq",
                "network_protocol_ICMP",
                "network_protocol_TCP",
                "network_protocol_UDP",
            ],
            "flags": ["has_SYN", "has_ACK", "has_RST", "has_FIN", "is_hex_flag"],
            "labels": ["class", "is_attack"],
        }

        for group_name, columns in feature_groups.items():
            valid_columns = [col for col in columns if col in df.columns]

            if valid_columns:
                group_df = df[valid_columns]

                stats = self.calculate_dataframe_stats(group_df)

                for column in valid_columns:
                    feature_type = str(group_df[column].dtype)
                    self.add_feature_metadata(
                        feature_group_id=feature_group_id,
                        feature_name=column,
                        feature_type=feature_type,
                        description=f"Groups feature {group_name}",
                        stats=stats[column],
                    )

                version = self.version_control(
                    f"{feature_group_id}_{group_name}", group_df
                )
                print(
                    f"  - Group {group_name}: version {version}, {len(group_df)} rows, {len(valid_columns)} cols"
                )

                local_path = f"load/feature_store/{group_name}.csv"
                group_df.to_csv(local_path)

                cloud_path = (
                    f"feature_store/{feature_group_id}/{group_name}_v{version}.csv"
                )
                self.ops.upload_to_bucket(local_path, cloud_path)

        self.save_metadata(feature_group_id)

        return metadata
