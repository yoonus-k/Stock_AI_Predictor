"""
Pattern Recognition Module

This module implements various pattern recognition techniques
for identifying trading patterns in price data.

Usage:
    Import this module to use different pattern recognition algorithms
    in backtesting strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def extract_pips(window: np.array, n_pips: int = 5, dist_type: int = 3) -> Tuple[List[int], np.array]:
    """
    Extract and normalize PIPs from a price window.
    
    Args:
        window: Price window to extract PIPs from
        n_pips: Number of PIPs to extract 
        dist_type: Distance metric type for PIP extraction algorithm
        
    Returns:
        tuple: (x indices, normalized y values) or (None, None) if extraction fails
    """
    try:
        from Pattern.perceptually_important import find_pips
        x, y = find_pips(window, n_pips, dist_type)
        
        if len(x) < n_pips or len(y) < n_pips:
            return None, None
            
        scaler = MinMaxScaler()
        norm_y = scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()
        return x, norm_y
    except Exception as e:
        print(f"Error extracting PIPs: {e}")
        return None, None


class PatternRecognizer:
    """Base class for pattern recognition algorithms."""
    
    def __init__(self, name: str = "Base Recognizer"):
        self.name = name
        self.is_trained = False
        
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


class SVMPatternRecognizer(PatternRecognizer):
    """Pattern recognizer using Support Vector Machine."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', **kwargs):
        super().__init__(name="SVM Pattern Recognizer")
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            **kwargs
        )
        
    def train(self, patterns: List[np.array], labels: List[int]) -> None:
        """
        Train the SVM model.
        
        Args:
            patterns: List of normalized pattern arrays
            labels: List of cluster labels corresponding to patterns
        """
        self.model.fit(patterns, labels)
        self.is_trained = True
        
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
            
        pattern = pattern.reshape(1, -1)
        cluster_id = self.model.predict(pattern)[0]
        probabilities = self.model.predict_proba(pattern)[0]
        confidence = probabilities.max()
        
        return cluster_id, confidence


class RandomForestPatternRecognizer(PatternRecognizer):
    """Pattern recognizer using Random Forest."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, **kwargs):
        super().__init__(name="Random Forest Pattern Recognizer")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            **kwargs
        )
        
    def train(self, patterns: List[np.array], labels: List[int]) -> None:
        """
        Train the Random Forest model.
        
        Args:
            patterns: List of normalized pattern arrays
            labels: List of cluster labels corresponding to patterns
        """
        self.model.fit(patterns, labels)
        self.is_trained = True
        
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
            
        pattern = pattern.reshape(1, -1)
        cluster_id = self.model.predict(pattern)[0]
        probabilities = self.model.predict_proba(pattern)[0]
        confidence = probabilities.max()
        
        return cluster_id, confidence


class EnsemblePatternRecognizer(PatternRecognizer):
    """Pattern recognizer using an ensemble of multiple recognizers."""
    
    def __init__(self, recognizers: List[PatternRecognizer], weights: Optional[List[float]] = None):
        super().__init__(name="Ensemble Pattern Recognizer")
        self.recognizers = recognizers
        
        # Normalize weights if provided, otherwise use equal weights
        if weights is None:
            self.weights = [1/len(recognizers)] * len(recognizers)
        else:
            assert len(weights) == len(recognizers), "Weights must match number of recognizers"
            weight_sum = sum(weights)
            self.weights = [w/weight_sum for w in weights]
        
    def train(self, patterns: List[np.array], labels: List[int]) -> None:
        """
        Train all recognizers in the ensemble.
        
        Args:
            patterns: List of normalized pattern arrays
            labels: List of cluster labels corresponding to patterns
        """
        for recognizer in self.recognizers:
            recognizer.train(patterns, labels)
        self.is_trained = True
        
    def predict(self, pattern: np.array) -> Tuple[int, float]:
        """
        Predict the cluster of a pattern using weighted voting.
        
        Args:
            pattern: Normalized pattern array
            
        Returns:
            Tuple of (cluster_id, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Get predictions and confidences from all recognizers
        predictions = []
        confidences = []
        
        for recognizer in self.recognizers:
            cluster_id, confidence = recognizer.predict(pattern)
            predictions.append(cluster_id)
            confidences.append(confidence)
            
        # Count weighted votes for each cluster
        cluster_votes = {}
        for i, pred in enumerate(predictions):
            weighted_vote = confidences[i] * self.weights[i]
            if pred in cluster_votes:
                cluster_votes[pred] += weighted_vote
            else:
                cluster_votes[pred] = weighted_vote
                
        # Select cluster with highest vote
        best_cluster = max(cluster_votes.items(), key=lambda x: x[1])
        return best_cluster[0], best_cluster[1]


class DistanceBasedPatternRecognizer(PatternRecognizer):
    """Pattern recognizer using distance metrics."""
    
    def __init__(self, distance_metric: str = 'euclidean', max_distance: float = 0.2):
        super().__init__(name="Distance-Based Pattern Recognizer")
        self.distance_metric = distance_metric
        self.max_distance = max_distance
        self.patterns = None
        self.labels = None
        
    def train(self, patterns: List[np.array], labels: List[int]) -> None:
        """
        Store patterns and labels for distance-based matching.
        
        Args:
            patterns: List of normalized pattern arrays
            labels: List of cluster labels corresponding to patterns
        """
        self.patterns = np.array(patterns)
        self.labels = np.array(labels)
        self.is_trained = True
        
    def predict(self, pattern: np.array) -> Tuple[int, float]:
        """
        Predict the cluster of a pattern using distance metrics.
        
        Args:
            pattern: Normalized pattern array
            
        Returns:
            Tuple of (cluster_id, similarity)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Calculate distances to all patterns
        distances = []
        for p in self.patterns:
            if self.distance_metric == 'euclidean':
                dist = np.sqrt(np.sum((pattern - p)**2))
            elif self.distance_metric == 'manhattan':
                dist = np.sum(np.abs(pattern - p))
            elif self.distance_metric == 'cosine':
                dot_product = np.dot(pattern, p)
                norm_pattern = np.linalg.norm(pattern)
                norm_p = np.linalg.norm(p)
                dist = 1 - (dot_product / (norm_pattern * norm_p))
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
                
            distances.append(dist)
            
        distances = np.array(distances)
        
        # Check if any distance is within the threshold
        if np.min(distances) > self.max_distance:
            return -1, 0.0  # No match found
            
        # Get the closest pattern
        closest_idx = np.argmin(distances)
        cluster_id = self.labels[closest_idx]
        
        # Convert distance to similarity (0-1 scale, higher is better)
        similarity = 1 - (distances[closest_idx] / self.max_distance)
        similarity = max(0, min(similarity, 1))  # Clamp to [0, 1]
        
        return cluster_id, similarity


def create_recognizer(technique: str, **kwargs) -> PatternRecognizer:
    """
    Create a pattern recognizer based on the technique name.
    
    Args:
        technique: Name of the recognition technique
        **kwargs: Additional parameters for the recognizer
        
    Returns:
        PatternRecognizer instance
    """
    technique = technique.lower()
    
    if technique == 'svm':
        return SVMPatternRecognizer(**kwargs)
    elif technique == 'random_forest':
        return RandomForestPatternRecognizer(**kwargs)
    elif technique == 'combined':
        svm = SVMPatternRecognizer(**kwargs.get('svm', {}))
        rf = RandomForestPatternRecognizer(**kwargs.get('rf', {}))
        weights = kwargs.get('voting_weights', [0.5, 0.5])
        return EnsemblePatternRecognizer([svm, rf], weights)
    elif technique == 'distance_based':
        return DistanceBasedPatternRecognizer(**kwargs)
    else:
        raise ValueError(f"Unknown recognition technique: {technique}")


def pattern_matcher(
    recognizer: PatternRecognizer,
    clusters_df: pd.DataFrame,
    window: np.array,
    n_pips: int = 5,
    dist_type: int = 3,
    mse_threshold: float = 0.03
) -> Optional[Dict[str, Any]]:
    """
    Match a price window to a pattern cluster.
    
    Args:
        recognizer: Trained pattern recognizer
        clusters_df: DataFrame with cluster information
        window: Price window to match
        n_pips: Number of PIPs to extract
        dist_type: Distance type for PIP extraction
        mse_threshold: Maximum MSE for a valid match
        
    Returns:
        Dictionary with pattern information or None if no match
    """
    # Extract PIPs from the window
    x, y = extract_pips(window, n_pips, dist_type)
    
    if y is None:
        return None
        
    # Predict cluster
    try:
        cluster_id, confidence = recognizer.predict(y)
    except:
        return None
        
    # Handle no match case
    if cluster_id == -1 or confidence == 0:
        return None
        
    # Get cluster information
    try:
        cluster = clusters_df.iloc[cluster_id]
    except:
        cluster_id_to_index = {id: idx for id, idx in enumerate(clusters_df.index)}
        actual_index = cluster_id_to_index.get(cluster_id)
        if actual_index is None:
            return None
        cluster = clusters_df.loc[actual_index]
    
    # Get cluster features
    try:
        cluster_features = np.array(cluster['AVGPricePoints'].split(','), dtype=float)
        
        # Check if the match is good enough
        mse = mean_squared_error(cluster_features, y)
        if mse > mse_threshold:
            return None
    except:
        # If we can't get features, rely on confidence
        if confidence < 0.7:  # Fallback threshold
            return None
    
    # Return pattern information
    pattern_info = {
        'cluster_id': cluster_id,
        'label': cluster['Label'],
        'outcome': cluster['Outcome'],
        'max_gain': cluster['MaxGain'],
        'max_drawdown': cluster['MaxDrawdown'],
        'confidence': confidence,
        'mse': mse if 'mse' in locals() else None
    }
    
    return pattern_info
