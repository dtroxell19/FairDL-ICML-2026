################################################################################################################################################
#This script generates all 32 synthetic datasets described in the paper
################################################################################################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import product
import pickle
from pathlib import Path


class SyntheticDataGenerator:
    def __init__(self, n_samples=40000, n_features=150, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def generate_dataset(
        self,
        group_split,
        group_predictive_power,
        noise_level,
        constraint_tightness,
        data_type,
    ):
        """
        Generate synthetic dataset with specified parameters

        @param group_split: 'low' (20% group 1) or 'balanced' (50% group 1)
        @param group_predictive_power: 'low-medium', 'high'
        @param noise_level: 'medium', 'high'
        @param constraint_tightness: 'loose', 'tight'
        @param data_type: 'linear', 'nonlinear'

        @returns df
        @returns constraint_info: dataset metadata
        """
        np.random.seed(self.random_state)

        #Set group split probability
        if group_split == "low":
            p_group1 = 0.2
        else:
            p_group1 = 0.5

        #group-feature correlation based on predictive power
        if group_predictive_power == "low-medium":
            correlation_strength = 0.3
        else:
            correlation_strength = 0.7

        #generate protected attribute and base features
        x_protected = np.random.binomial(1, p_group1, size=self.n_samples)
        features = self._generate_base_features()
        #add correlation between features and protected attribute
        features = self._add_group_correlation(
            features, x_protected, correlation_strength
        )
        #standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        #generate target variable based on data type
        if data_type == "linear":
            true_score = self._generate_linear_target(features_scaled)
        else:
            true_score = self._generate_nonlinear_target(features_scaled)

        #Add group bias
        if group_predictive_power == "low-medium":
            group_bias = np.where(x_protected == 0, -3, 3)
        else:
            group_bias = np.where(x_protected == 0, -6, 6)

        if noise_level == "medium":
            noise_std = 0.125
        else:
            noise_std = 0.6

        noise = np.random.normal(0, noise_std, self.n_samples)

        #generate final target and standardize
        y_cont = true_score + group_bias + noise
        y_cont = (y_cont - np.mean(y_cont)) / np.std(y_cont)

        #apply different scaling to each group
        group0_mask = x_protected == 0
        y_cont[group0_mask] *= 0.85  #slightly lower variance for group 0
        #constraint bounds
        if constraint_tightness == "loose":
            constraint_lower, constraint_upper = -3.5, 3.5
        else:  #tight
            constraint_lower, constraint_upper = 0, 3.5

        #make final df
        df = pd.DataFrame(
            features_scaled, columns=[f"x{i}" for i in range(self.n_features)]
        )
        df["x_protected"] = x_protected
        df["y_cont"] = y_cont

        #add constraint information
        constraint_info = {
            "lower_bound": constraint_lower,
            "upper_bound": constraint_upper,
            "violation_rate": np.mean(
                (y_cont < constraint_lower) | (y_cont > constraint_upper)
            ),
        }

        return df, constraint_info

    def _generate_base_features(self):
        """
        Generate features with some correlation structure
        """
        np.random.seed(self.random_state)

        #add correlation structure
        n_blocks = max(1, self.n_features // 20)
        block_size = self.n_features // n_blocks
        features = np.zeros((self.n_samples, self.n_features))

        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, self.n_features)
            block_features = end_idx - start_idx

            #generate correlated block
            base_corr = 0.3
            corr_matrix = np.full((block_features, block_features), base_corr)
            np.fill_diagonal(corr_matrix, 1.0)

            try:
                L = np.linalg.cholesky(corr_matrix)
                block_data = np.random.normal(0, 1, (self.n_samples, block_features))
                features[:, start_idx:end_idx] = block_data @ L.T
            except np.linalg.LinAlgError:
                #fallback to independent features
                features[:, start_idx:end_idx] = np.random.normal(
                    0, 1, (self.n_samples, block_features)
                )

            #fill remaining features if any
            if self.n_features % n_blocks != 0:
                remaining_start = n_blocks * block_size
                features[:, remaining_start:] = np.random.normal(
                    0, 1, (self.n_samples, self.n_features - remaining_start)
                )

        return features

    def _add_group_correlation(self, features, x_protected, correlation_strength):
        """
        Add correlation between some features and the protected attribute
        """
        #select subset of features to correlate with protected attribute
        n_correlated_features = min(50, self.n_features // 6)
        correlated_indices = np.random.choice(
            self.n_features, n_correlated_features, replace=False
        )
        #add group-dependent signal to these features
        for idx in correlated_indices:
            group_signal = (
                correlation_strength
                * (x_protected * 2 - 1)
                * np.random.normal(0, 0.5, self.n_samples)
            )
            features[:, idx] += group_signal

        return features

    def _generate_linear_target(self, features):
        """
        Generate linear target with sparse coefficients
        """
        np.random.seed(self.random_state + 1)
        #Select relevant features
        n_relevant_features = min(10, self.n_features // 10)

        coefficients = np.zeros(self.n_features)
        relevant_indices = np.random.choice(
            self.n_features, n_relevant_features, replace=False
        )

        #generate coefficients with varying strengths
        base_coeffs = np.random.normal(0, 1.0, n_relevant_features)
        coefficients[relevant_indices] = base_coeffs

        return np.dot(features, coefficients)

    def _generate_nonlinear_target(self, features):
        """
        Generate nonlinear target with polynomial and interaction terms
        """
        np.random.seed(self.random_state + 2)

        #linear component and then add polynomial terms & interaction terms
        linear_part = self._generate_linear_target(features)
        #polynomial terms
        n_poly_features = min(15, self.n_features // 10)
        poly_indices = np.random.choice(self.n_features, n_poly_features, replace=False)
        poly_coeffs = np.random.normal(0, 0.4, n_poly_features)
        poly_part = np.sum(features[:, poly_indices] ** 2 * poly_coeffs, axis=1)
        #interaction terms
        n_interactions = min(10, self.n_features // 20)
        interaction_part = 0
        for _ in range(n_interactions):
            idx1, idx2 = np.random.choice(self.n_features, 2, replace=False)
            coeff = np.random.normal(0, 0.4)
            interaction_part += coeff * features[:, idx1] * features[:, idx2]

        return 0.7 * linear_part + 0.2 * poly_part + 0.1 * interaction_part


########################################################################################################################
############################################DRIVER FUNCTIONS ##########################################################
########################################################################################################################


def get_dataset_parameters():
    """Get all parameter combinations for the 32 datasets"""
    group_splits = ["low", "balanced"]
    predictive_powers = ["low-medium", "high"]
    noise_levels = ["medium", "high"]
    constraint_tightnesses = ["loose", "tight"]
    data_types = ["linear", "nonlinear"]

    param_combinations = list(
        product(
            group_splits,
            predictive_powers,
            noise_levels,
            constraint_tightnesses,
            data_types,
        )
    )

    return param_combinations


def generate_dataset_by_index(i, n_samples=40000, n_features=150, random_state=123):
    """
    Generate the ith dataset (0-indexed)

    Parameters:
    - i: dataset index (0-31)
    - n_samples: number of samples to generate
    - n_features: number of features
    - random_state: random seed (will be modified by i to ensure different datasets)

    Returns:
    - df: DataFrame with features, protected attribute, and target
    - dataset_info: dictionary with dataset metadata
    """
    param_combinations = get_dataset_parameters()

    if i >= len(param_combinations):
        raise ValueError(
            f"Index {i} out of range. Maximum index is {len(param_combinations)-1}"
        )

    #get parameters for dataset
    group_split, pred_power, noise, constraint, data_type = param_combinations[i]
    #generator with modified random state to ensure different datasets
    generator = SyntheticDataGenerator(n_samples, n_features, random_state + i)

    #make dataset
    df, constraint_info = generator.generate_dataset(
        group_split, pred_power, noise, constraint, data_type
    )

    #Create dataset name + info
    dataset_name = (
        f"dataset_{i+1:02d}_{group_split}_{pred_power}_{noise}_{constraint}_{data_type}"
    )
    dataset_info = {
        "index": i,
        "name": dataset_name,
        "group_split": group_split,
        "predictive_power": pred_power,
        "noise_level": noise,
        "constraint_tightness": constraint,
        "data_type": data_type,
        "constraint_info": constraint_info,
        "group_distribution": np.bincount(df["x_protected"].values),
        "group_means": df.groupby("x_protected")["y_cont"].mean().to_dict(),
        "n_features": n_features,
    }

    return df, dataset_info


def main():

    for dataset_index in range(32):
        df, dataset_info = generate_dataset_by_index(
            dataset_index, n_samples=40000, n_features=150, random_state=dataset_index
        )
        current_dir = Path.cwd()
        name = f"datasets/dataset_{dataset_index}.pkl"
        pickle_path = current_dir / name

        df.to_pickle(pickle_path)

        metadata_name = f"dataset_metadata/dataset_{dataset_index}.pkl"
        with open(metadata_name, "wb") as f:
            pickle.dump(dataset_info, f)


if __name__ == "__main__":
    main()
