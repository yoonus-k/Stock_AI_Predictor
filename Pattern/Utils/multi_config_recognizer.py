"""
Multi-Configuration Pattern Recognizer Module

This module supports pattern recognition for multiple configs,
handling different feature dimensions correctly.
"""

import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import logging

# Add parent directory to path for imports
# Add project root to path to ensure imports work
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from Data.Database.db import Database
from Pattern.pip_pattern_miner import Pattern_Miner

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PatternRecognition")


class PatternRecognizer:
    """Base class for pattern recognition algorithms."""

    def __init__(self, name: str = "Base Recognizer", feature_length: int = 5):
        self.name = name
        self.is_trained = False
        self.feature_length = feature_length

    def train(self, patterns: List[np.array], labels: List[int]) -> None:
        """
        Train the pattern recognizer.

        Args:
            patterns: List of normalized pattern arrays
            labels: List of cluster labels corresponding to patterns
        """
        self.is_trained = True

    def predict(self, pattern: np.array) -> Tuple[int, float]:
        """
        Predict the cluster of a pattern.

        Args:
            pattern: Normalized pattern array

        Returns:
            Tuple of (cluster_id, confidence)
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def evaluate_match(self, pattern: np.array, cluster_features: np.array) -> float:
        """
        Evaluate the quality of a pattern match.

        Args:
            pattern: Normalized pattern array
            cluster_features: Features of the matched cluster

        Returns:
            Match quality score (higher is better)
        """
        return -mean_squared_error(pattern, cluster_features)

    def ensure_feature_length(self, pattern: np.array) -> np.array:
        """
        Ensure the pattern has the correct feature length.

        Args:
            pattern: Input pattern array

        Returns:
            Pattern array with correct feature length
        """
        if len(pattern) == self.feature_length:
            return pattern

        if len(pattern) < self.feature_length:
            # Pad with zeros
            padded_pattern = np.zeros(self.feature_length)
            padded_pattern[: len(pattern)] = pattern
            return padded_pattern
        else:
            # Truncate
            return pattern[: self.feature_length]


class SVMPatternRecognizer(PatternRecognizer):
    """Pattern recognizer using Support Vector Machine."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        feature_length: int = 5,
        **kwargs,
    ):
        super().__init__(name="SVM Pattern Recognizer", feature_length=feature_length)
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, **kwargs)
        # logger.info(f"Created SVM recognizer with feature_length={feature_length}")

    def train(self, patterns: np.ndarray, labels: np.ndarray) -> None:
        """
        Train the SVM model.

        Args:
            patterns: Array of normalized pattern arrays
            labels: Array of cluster labels corresponding to patterns
        """
        # Store the feature length from training data
        if patterns is not None and len(patterns) > 0:
            self.feature_length = patterns.shape[1]
            # logger.info(f"Setting feature_length={self.feature_length} from training data")

        self.model.fit(patterns, labels)
        self.is_trained = True
        # logger.info(f"Trained SVM model with {len(patterns)} patterns, feature_length={self.feature_length}")

    def predict(self, pattern: np.array) -> Tuple[int, float]:
        """
        Predict the cluster of a pattern using SVM.

        Args:
            pattern: Normalized pattern array

        Returns:
            Tuple of (cluster_id, probability)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Ensure pattern has correct feature length
        if len(pattern) != self.feature_length:
            logger.warning(
                f"Pattern length mismatch: got {len(pattern)}, expected {self.feature_length}"
            )
            pattern = self.ensure_feature_length(pattern)

        pattern = pattern.reshape(1, -1)
        cluster_id = self.model.predict(pattern)[0]

        probabilities = self.model.predict_proba(pattern)[0]
        confidence = probabilities.max()

        return cluster_id, confidence


class RandomForestPatternRecognizer(PatternRecognizer):
    """Pattern recognizer using Random Forest."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        feature_length: int = 5,
        **kwargs,
    ):
        super().__init__(
            name="Random Forest Pattern Recognizer", feature_length=feature_length
        )
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, **kwargs
        )

    def train(self, patterns: np.ndarray, labels: np.ndarray) -> None:
        """
        Train the Random Forest model.

        Args:
            patterns: Array of normalized pattern arrays
            labels: Array of cluster labels corresponding to patterns
        """
        # Store the feature length from training data        logger.info(f"Training Random Forest with patterns type: {type(patterns)}, shape: {patterns.shape if hasattr(patterns, 'shape') else 'no shape'}")
        # logger.info(f"Labels type: {type(labels)}, shape: {labels.shape if hasattr(labels, 'shape') else 'no shape'}")

        if patterns is not None and len(patterns) > 0:
            self.feature_length = patterns.shape[1]
            # logger.info(f"Setting feature_length={self.feature_length} from training data")

        self.model.fit(patterns, labels)
        self.is_trained = True
        # logger.info(f"Trained Random Forest model with {len(patterns)} patterns, feature_length={self.feature_length}")

    def predict(self, pattern: np.array) -> Tuple[int, float]:
        """
        Predict the cluster of a pattern using Random Forest.

        Args:
            pattern: Normalized pattern array

        Returns:
            Tuple of (cluster_id, probability)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Ensure pattern has correct feature length
        if len(pattern) != self.feature_length:
            logger.warning(
                f"Pattern length mismatch in RF: got {len(pattern)}, expected {self.feature_length}"
            )
            pattern = self.ensure_feature_length(pattern)

        # Reshape for prediction
        pattern = pattern.reshape(1, -1)

        # Get prediction and confidence
        try:
            cluster_id = self.model.predict(pattern)[0]
            probabilities = self.model.predict_proba(pattern)[0]
            confidence = probabilities.max()

            return cluster_id, confidence
        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {e}")
            # Fallback to a default value in case of error
            return -1, 0.0


class ConfigBasedRecognizer:
    """
    A recognizer that handles multiple configurations.
    """

    def __init__(self, db, default_technique="svm"):
        self.db = db
        self.recognizers = {}  # Store recognizers by config_id
        self.feature_lengths = {}  # Store feature lengths by config_id
        self.default_technique = default_technique

    def extract_pips(
        self,
        window: np.array,
        n_pips: int = 5,
        dist_type: int = 2,
    ):
        """
        Extract and normalize PIPs from a price window.

        Args:
            window: Price window to extract PIPs from
            n_pips: Number of PIPs to extract
            dist_type: Distance metric type for PIP extraction algorithm
            expected_length: Expected length of the output feature vector (for padding/truncating)

        Returns:
            tuple: (x indices, normalized y values) or (None, None) if extraction fails
        """
        try:
            
            #window = window.values

            pattern_miner = Pattern_Miner()
            x, y = pattern_miner.find_pips(window, n_pips, dist_type)

            scaler = MinMaxScaler()
            norm_y = scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()

            return x, norm_y
        except Exception as e:
            logger.error(f"Error extracting PIPs: {e}")
            return None, None

    def create_recognizer(
        self, technique: str, feature_length: int = 5, **kwargs
    ) -> PatternRecognizer:
        """
        Create a pattern recognizer based on the technique name.

        Args:
            technique: Name of the recognition technique
            feature_length: Expected length of feature vectors
            **kwargs: Additional parameters for the recognizer

        Returns:
            PatternRecognizer instance
        """
        technique = technique.lower()

        if technique == "svm":
            return SVMPatternRecognizer(feature_length=feature_length, **kwargs)
        elif technique == "random_forest":
            return RandomForestPatternRecognizer(
                feature_length=feature_length, **kwargs
            )
        else:
            raise ValueError(f"Unknown recognition technique: {technique}")

    def parse_feature_array(self, feature_str):
        """
        Parse feature array from string representation.

        Args:
            feature_str: String representation of feature array

        Returns:
            Numpy array of features
        """
        try:
            # Handle JSON-like format with brackets
            if feature_str.startswith("[") and feature_str.endswith("]"):
                # Remove brackets and split by comma
                feature_values = feature_str[1:-1].split(",")
                return np.array([float(x.strip()) for x in feature_values])
            # Handle simple comma-separated format
            else:
                return np.array([float(x) for x in feature_str.split(",")])
        except Exception as e:
            logger.error(f"Error parsing feature array: {e}")
            raise ValueError(f"Invalid feature format: {feature_str}")

    def process_cluster_features(self, clusters_df):
        """
        Process cluster features to ensure consistent dimensions.

        Args:
            clusters_df: DataFrame with cluster information

        Returns:
            Tuple of (processed_features, labels, feature_length)
        """
        if clusters_df.empty:
            raise ValueError("Empty clusters DataFrame provided")

        # Get feature strings from the DataFrame
        feature_strings = clusters_df["avg_price_points_json"].values
        feature_length = len(
            self.parse_feature_array(feature_strings[0])
        )  # Assuming all features have the same length


        # Process features with consistent length
        processed_features = []
        valid_indices = []

        for i, feature_str in enumerate(feature_strings):
            try:
                # Parse the feature array
                features = self.parse_feature_array(feature_str)

                processed_features.append(features)
               
                actual_index = clusters_df.iloc[i]["cluster_id"]
                   

                valid_indices.append(actual_index)
            except Exception as e:
                logger.warning(f"Skipping invalid feature at index {i}: {e}")

        # Create labels using the valid indices
        labels = np.array(valid_indices)

        # Convert processed features to numpy array
        processed_features = np.array(processed_features)

        if len(processed_features) == 0:
            raise ValueError("No valid features could be processed")

        return processed_features, labels, feature_length

    def pattern_matcher(
        self,
        recognizer: PatternRecognizer,
        clusters_df: pd.DataFrame,
        window: np.array,
        n_pips: int = 5,
        dist_type: int = 2,
        mse_threshold: float = 0.03,
    ):

        x, y = self.extract_pips(window, n_pips, dist_type)

        # Predict cluster

        if x is None or y is None:
            logger.warning("Failed to extract PIPs from the window: ", window)
            return None
        cluster_id, confidence = recognizer.predict(y)

        # Handle no match case
        if cluster_id == -1:
            return None

        # Try to map cluster_id to actual index
        cluster_id_to_index = {idx: i for idx, i in enumerate(clusters_df.index)}
        actual_index = cluster_id_to_index.get(cluster_id)

        if actual_index is None:
            return None

        cluster = clusters_df.loc[actual_index].copy()

        # Get cluster features and compare with the pattern

        feature_str = cluster["avg_price_points_json"]
        cluster_features = self.parse_feature_array(
            feature_str
        )  # Ensure same length for comparison

        # calculate MSE between the extracted pattern and cluster features
        mse = mean_squared_error(cluster_features, y)

        # Compare with threshold using direct comparison (avoid Series boolean context)
        if mse > mse_threshold:
            return None

        # plot the window and the matched cluster features side by side
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.plot(window, label='Price Window', color='blue')
        # plt.title('Price Window')
        # plt.xlabel('Time')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.subplot(1, 2, 2)
        # plt.plot(cluster_features, label='Matched Cluster Features', color='orange')
        # plt.title('Matched Cluster Features')
        # plt.xlabel('Time')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        
        cluster["mse"] = mse

        return cluster

    # function to get the recognizer for a specific stock and timeframe
    def get_recognizer(self, stock_id, timeframe_id, config_id=1,clusters = None):
        """
        Get the recognizer for a specific stock and timeframe.

        Args:
            stock_id: Stock ID
            timeframe_id: Timeframe ID
            config_id: Configuration ID or None

        Returns:
            Recognizer instance or None if not found
        """
        key = f"{stock_id}_{timeframe_id}_{config_id}"

        if key in self.recognizers:
            # Return cached recognizer
            recognizer = self.recognizers[key]
            feature_length = self.feature_lengths[key]
            # filter the clusters by stock_id, timeframe_id and config_id
            if clusters is None:
                clusters_df = self.db.get_clusters_by_config(stock_id, timeframe_id, config_id)
            else:
                # filter the dataframe by stock_id, timeframe_id and config_id
                clusters_df = clusters[
                    (clusters["stock_id"] == stock_id)
                    & (clusters["timeframe_id"] == timeframe_id)
                    & (clusters["config_id"] == config_id)
                ]
            
            return recognizer, feature_length, clusters_df
        else:
            logger.warning(f"No recognizer found for {key}. Please create it first.")
            return None, None, None

    def create_new_recognizer(self, stock_id, timeframe_id, config_id=None):
        """
        Get or create a recognizer for the given configuration.

        Args:
            stock_id: Stock ID
            timeframe_id: Timeframe ID
            config_id: Configuration ID or None
            n_pips: Number of PIPs to extract

        Returns:
            Tuple of (recognizer, feature_length, clusters_df)
        """
        key = f"{stock_id}_{timeframe_id}_{config_id}"

        if key in self.recognizers:
            logger.info(
                f"Using cached recognizer for {key} , get it instead of creating a new one"
            )

        # Get clusters for this configuration
        clusters_df = self.db.get_clusters_by_config(stock_id, timeframe_id, config_id)

        if clusters_df.empty:
            raise ValueError(
                f"No clusters found for Stock ID {stock_id}, "
                f"Timeframe ID {timeframe_id}, Config ID {config_id}"
            )

        # Process cluster features
        processed_features, labels, feature_length = self.process_cluster_features(
            clusters_df
        )

        # Create recognizer with the determined feature length
        recognizer = self.create_recognizer(
            technique=self.default_technique, feature_length=feature_length
        )

        # Train the recognizer
        recognizer.train(processed_features, labels)

        # Cache the recognizer
        self.recognizers[key] = recognizer
        self.feature_lengths[key] = feature_length

        # logger.info(f"Created and trained recognizer for {key} with feature_length={feature_length}")

    # function to train the recognizer for a specific timeframe and stock
    def train_recognizer(self, stock_id: int, timeframe_id: int):
        """
        Train a recognizer for the specified stock and timeframe.

        Args:
            stock_id: Stock ID
            timeframe_id: Timeframe ID
            config_id: Configuration ID or None
            n_pips: Number of PIPs to extract

        Returns:
            Tuple of (recognizer, feature_length, clusters_df)
        """
        # get all the configs for the stock and timeframe
        configs = self.db.get_configs_by_stock_and_timeframe(stock_id, timeframe_id)
        if configs.empty:
            raise ValueError(
                f"No configurations found for Stock ID {stock_id}, Timeframe ID {timeframe_id}"
            )

        # loop through each config and train the recognizer , it's a dataframe with config_id and n_pips
        for _, config in configs.iterrows():
            # Extract config_id and n_pips from the row
            # Assuming config is a dictionary-like object with keys 'config_id' and 'n_pips'
            config_id = config["config_id"]
            n_pips = config["n_pips"]
            self.create_new_recognizer(stock_id, timeframe_id, config_id=config_id)
            # logger.info(f"Trained recognizers for Stock ID {stock_id}, Timeframe ID {timeframe_id}")

        # function to predict the best cluster for a given price window accross all configs

    def predict_best_cluster(self, stock_id: int, timeframe_id: int, window: np.array ,configs: pd.DataFrame = None, clusters: pd.DataFrame = None):
        """
        Predict the best matching cluster for a given price window across all configs.

        Args:
            stock_id: Stock ID
            timeframe_id: Timeframe ID
            window: Price window to match
            configs: DataFrame of configurations or None to use all configs
        Returns:
            Dictionary with best matching pattern information or None if no match found
        """
        best_match = None
        best_expected_value = 0.0

        # get the configs for the stock and timeframe
        if configs is None:
            configs = self.db.get_configs_by_stock_and_timeframe(stock_id, timeframe_id)
            if configs.empty:
                raise ValueError(
                    f"No configurations found for Stock ID {stock_id}, Timeframe ID {timeframe_id}"
                )
        # Loop through each config and get the recognizer
        for _, config in configs.iterrows():
            config_id = config["config_id"]
            n_pips = config["n_pips"]
            lookback = config["lookback"]
            match_window = window[-lookback:]

            recognizer, feature_length, clusters_df = self.get_recognizer(
                stock_id, timeframe_id, config_id , clusters
            )

            if recognizer:
                # Match the pattern using the recognizer
                matched_cluster = self.pattern_matcher(
                    recognizer, clusters_df, match_window, n_pips
                )
                if matched_cluster is None:
                    # logger.warning(
                    #     f"No match found for Stock ID {stock_id}, Timeframe ID {timeframe_id}, Config ID {config_id}"
                    # )
                    continue

                if matched_cluster["expected_value"] > best_expected_value:
                    best_expected_value = matched_cluster["expected_value"]
                    best_match = matched_cluster

        return best_match , best_expected_value


if __name__ == "__main__":

    # Connect to database
    db = Database("Data/Storage/data.db")
    print(f"Connected to database: {db.db_path}")

    recognizer = ConfigBasedRecognizer(db, default_technique="svm")

    # Train recognizers for a specific stock and timeframe
    stock_id = 1  # Example stock ID
    timeframe_id = 5  # Example timeframe ID
    recognizer.train_recognizer(stock_id, timeframe_id)

    # Example price window to match
    price_data = db.get_stock_data_range(
        stock_id, timeframe_id, "2025-04-01", "2025-04-30"
    )

    # get only close prices
    price_window = price_data["close_price"].values[
        300:348
    ]  # Last 24 hours as an example

    best_cluster,best_expected_value = recognizer.predict_best_cluster(stock_id, timeframe_id, price_window)
    print(best_cluster)
    
    # normalize the price window
    scaler = MinMaxScaler()
    price_window = scaler.fit_transform(price_window.reshape(-1, 1)).flatten()
    
    # plot the best cluster features and the price window in seperate subplots
    if best_cluster is not None:
        cluster_features = recognizer.parse_feature_array(
            best_cluster["avg_price_points_json"]
        )

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(price_window, label="Price Window", color="blue")
        plt.title("Price Window")
        plt.xlabel("Time")
        plt.ylabel("Normalized Price")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(cluster_features, label="Matched Cluster Features with expected value of : " + str(best_expected_value), color="orange")
        plt.title("Matched Cluster Features")
        plt.xlabel("Time")
        plt.ylabel("Normalized Price")
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("No matching cluster found.")
    
    