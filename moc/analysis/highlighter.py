import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm


class Highlighter:
    def __init__(self, alpha=0.05, significance_level=0.05):
        """
        Initializes the Highlighter.

        Args:
            alpha (float): The significance level for statistical comparisons
                           and the target offset for specific metrics.
                           Defaults to 0.05.
        """
        if not 0 < alpha < 1:
            raise ValueError('alpha must be between 0 and 1')
        self.alpha = alpha
        self.critical_z_one_sided = norm.ppf(1 - significance_level)
        self.critical_z_two_sided = norm.ppf(1 - significance_level / 2)
        self.target_metrics = ['coverage', 'wsc']
        self.highlight_style = 'font-weight: bold;'
        self.dataset_level = 'dataset'
        self.metric_level = 'metric'

    def _extract_stats(self, cell):
        """Safely extracts mean and standard error from a cell."""
        if pd.isna(cell):
            return np.nan, np.nan
        if isinstance(cell, (list, tuple)) and len(cell) == 2:
            mean, se = cell
            return mean, se
        return np.nan, np.nan

    def _get_ordering_scores(self, means_series, metric):
        """Calculates the score for each mean based on the metric type."""
        if means_series.empty:
            return means_series
        if metric in self.target_metrics:
            target = 1.0 - self.alpha
            scores = (means_series - target).abs()
        else:
            scores = means_series
        return scores

    def _find_best_stats(self, means, ses, metric):
        """Finds the best value and its stats within a Series."""
        valid_indices = means.notna() & ses.notna()
        if not valid_indices.any():
            return pd.Series(dtype=float), pd.Series(dtype=float), None, np.nan, np.nan

        means_valid = means[valid_indices]
        ses_valid = ses[valid_indices]

        if means_valid.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float), None, np.nan, np.nan

        ordering_scores = self._get_ordering_scores(means_valid, metric)
        if ordering_scores.empty or ordering_scores.isna().all():
            return means_valid, ses_valid, None, np.nan, np.nan

        try:
            # Use idxmin which ignores NaNs in scores
            idx_best = ordering_scores.idxmin()
            mean_best = means_valid[idx_best]
            se_best = ses_valid[idx_best]
        except ValueError:  # Handle case where ordering_scores might be all NaN after calculation
            return means_valid, ses_valid, None, np.nan, np.nan

        return means_valid, ses_valid, idx_best, mean_best, se_best

    def _compare_to_best(self, means_valid, ses_valid, idx_best, mean_best, se_best, metric):
        """Performs statistical comparison and returns a boolean Series."""
        if means_valid.empty or idx_best is None or pd.isna(idx_best):  # Check if idx_best is valid
            return pd.Series(False, index=means_valid.index)

        highlight_flags = pd.Series(False, index=means_valid.index)

        # Ensure mean_best and se_best are valid numbers before proceeding
        if pd.isna(mean_best) or pd.isna(se_best):
            warnings.warn(
                f"Best value's mean or SE is NaN for metric '{metric}'. Cannot perform comparisons.",
                UserWarning,
            )
            return highlight_flags  # Return all False

        for idx, mean_i in means_valid.items():
            # Skip comparison if the current item's stats are invalid
            if pd.isna(mean_i):
                continue
            se_i = ses_valid[idx]
            if pd.isna(se_i):
                continue

            if idx == idx_best:
                highlight_flags[idx] = True
                continue

            se_diff_sq = se_i**2 + se_best**2
            if se_diff_sq <= 1e-15:
                if np.isclose(mean_i, mean_best):
                    highlight_flags[idx] = True
            else:
                se_diff = np.sqrt(se_diff_sq)
                # Avoid division by zero if se_diff is somehow still zero/very small
                if se_diff < 1e-15:
                    if np.isclose(mean_i, mean_best):
                        highlight_flags[idx] = True
                    continue  # Cannot calculate Z-score

                z_score = (mean_i - mean_best) / se_diff

                if metric in self.target_metrics:
                    if abs(z_score) <= self.critical_z_two_sided:
                        highlight_flags[idx] = True
                elif z_score <= self.critical_z_one_sided:
                    highlight_flags[idx] = True
        return highlight_flags

    # --- Main Highlighting Methods ---

    def highlight_best_per_dataset(self, data):
        """
        Highlights the best value(s) per (dataset, metric) group. NON-STATISTICAL.
        Assumes index has a level named 'dataset'. Handles simple/MultiIndex columns.
        """
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        dataset_level_name = self.dataset_level  # Use the defined level name

        # --- Check Index Structure ---
        if not isinstance(data.index, pd.MultiIndex) or dataset_level_name not in data.index.names:
            warnings.warn(
                f"Index is not a MultiIndex or missing level '{dataset_level_name}'. Cannot group by dataset.",
                UserWarning,
            )
            return styles

        # --- Identify Metrics ---
        if isinstance(data.columns, pd.MultiIndex):
            try:
                metric_level_idx = data.columns.names.index(self.metric_level)
            except (ValueError, AttributeError):
                metric_level_idx = 0
                warnings.warn(
                    f"Column MultiIndex missing level '{self.metric_level}'. Using level {metric_level_idx}.",
                    UserWarning,
                )
            metrics = data.columns.get_level_values(metric_level_idx).unique()
        else:
            metrics = data.columns.unique()

        # --- Iterate and Highlight using GroupBy ---
        try:
            # REMOVED axis=0 from groupby call
            grouped = data.groupby(level=dataset_level_name, observed=True)
        except KeyError:
            warnings.warn(f"Failed to group by index level '{dataset_level_name}'.", UserWarning)
            return styles

        # --- DEBUG: Print identified metrics ---
        # print(f"DEBUG: Identified metrics for per_dataset: {metrics.tolist()}")
        # is_last_metric = False # Flag for debugging

        for dataset_key, dataset_group in grouped:
            # --- DEBUG: Print dataset being processed ---
            # print(f"\nDEBUG: Processing Dataset: {dataset_key}")
            for i, metric in enumerate(metrics):  # Use enumerate to check if it's the last
                # is_last_metric = (i == len(metrics) - 1)
                # --- DEBUG: Print metric being processed (especially if last) ---
                # if is_last_metric: print(f"DEBUG: Processing LAST metric: {metric}")

                if metric not in dataset_group.columns:
                    # if is_last_metric: print(f"DEBUG: Metric '{metric}' not in dataset_group columns.")
                    continue

                metric_series = dataset_group[metric]
                # if is_last_metric: print(f"DEBUG: Last metric series:\n{metric_series}")

                means = metric_series.apply(lambda x: self._extract_stats(x)[0]).astype(float)
                means_valid = means.dropna()
                # if is_last_metric: print(f"DEBUG: Last metric valid means:\n{means_valid}")

                if not means_valid.empty:
                    scores = self._get_ordering_scores(means_valid, metric)
                    # if is_last_metric: print(f"DEBUG: Last metric scores:\n{scores}")

                    if not scores.empty and not scores.isna().all():
                        try:
                            min_score = scores.min()
                            is_best = np.isclose(scores, min_score)
                            best_sub_indices = means_valid[is_best].index
                            # if is_last_metric: print(f"DEBUG: Last metric best indices: {best_sub_indices.tolist()}")

                            for sub_idx in best_sub_indices:
                                try:
                                    # Get the full original index associated with the sub_idx within this group
                                    original_index = dataset_group.loc[[sub_idx]].index[
                                        0
                                    ]  # Use list selection to ensure index is returned
                                    styles.loc[original_index, metric] = self.highlight_style
                                    # if is_last_metric: print(f"DEBUG: Applied style to Index: {original_index}, Metric: {metric}")

                                except Exception as e:
                                    warnings.warn(
                                        f'Style application failed for Index: {original_index}, Metric: {metric}. Error: {e}',
                                        UserWarning,
                                    )
                        except (
                            ValueError
                        ):  # Handle errors during min() or isclose() if scores are problematic
                            warnings.warn(
                                f'Score calculation/comparison failed for Dataset: {dataset_key}, Metric: {metric}.',
                                UserWarning,
                            )
                    # else: # Debugging
                    # if is_last_metric: print(f"DEBUG: Scores empty or all NaN for metric {metric}")
                # else: # Debugging
                # if is_last_metric: print(f"DEBUG: No valid means found for metric {metric}")

        return styles

    def highlight_statistically_similar_to_best_per_dataset(self, data):
        """
        Highlights values statistically similar to the best per (dataset, metric) group.
        Assumes index has a level named 'dataset'. Handles simple/MultiIndex columns.
        """
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        dataset_level_name = self.dataset_level  # Use the defined level name

        # --- Check Index Structure ---
        if not isinstance(data.index, pd.MultiIndex) or dataset_level_name not in data.index.names:
            warnings.warn(
                f"Index is not a MultiIndex or missing level '{dataset_level_name}'. Cannot group by dataset.",
                UserWarning,
            )
            return styles

        # --- Identify Metrics ---
        if isinstance(data.columns, pd.MultiIndex):
            try:
                metric_level_idx = data.columns.names.index(self.metric_level)
            except (ValueError, AttributeError):
                metric_level_idx = 0
                warnings.warn(
                    f"Column MultiIndex missing level '{self.metric_level}'. Using level {metric_level_idx}.",
                    UserWarning,
                )
            metrics = data.columns.get_level_values(metric_level_idx).unique()
        else:
            metrics = data.columns.unique()

        # --- Iterate and Highlight using GroupBy ---
        try:
            # REMOVED axis=0 from groupby call
            grouped = data.groupby(level=dataset_level_name, observed=True)
        except KeyError:
            warnings.warn(f"Failed to group by index level '{dataset_level_name}'.", UserWarning)
            return styles

        # --- DEBUG: Print identified metrics ---
        # print(f"DEBUG STAT: Identified metrics for per_dataset: {metrics.tolist()}")
        # is_last_metric = False # Flag for debugging

        for dataset_key, dataset_group in grouped:
            # --- DEBUG: Print dataset being processed ---
            # print(f"\nDEBUG STAT: Processing Dataset: {dataset_key}")
            for i, metric in enumerate(metrics):
                # is_last_metric = (i == len(metrics) - 1)
                # if is_last_metric: print(f"DEBUG STAT: Processing LAST metric: {metric}")

                if metric not in dataset_group.columns:
                    # if is_last_metric: print(f"DEBUG STAT: Metric '{metric}' not in dataset_group columns.")
                    continue

                metric_series = dataset_group[metric]
                # if is_last_metric: print(f"DEBUG STAT: Last metric series:\n{metric_series}")

                stats = metric_series.apply(self._extract_stats)
                means = stats.apply(lambda x: x[0]).astype(float)
                ses = stats.apply(lambda x: x[1]).astype(float)
                # if is_last_metric: print(f"DEBUG STAT: Last metric means:\n{means}\nDEBUG STAT: Last metric ses:\n{ses}")

                means_valid, ses_valid, idx_best, mean_best, se_best = self._find_best_stats(
                    means, ses, metric
                )
                # if is_last_metric: print(f"DEBUG STAT: Find Best Results: idx={idx_best}, mean={mean_best}, se={se_best}")

                if means_valid.empty or idx_best is None or pd.isna(idx_best):
                    # if is_last_metric: print(f"DEBUG STAT: No valid data or best index found for {metric}.")
                    continue

                highlight_flags = self._compare_to_best(
                    means_valid, ses_valid, idx_best, mean_best, se_best, metric
                )
                # if is_last_metric: print(f"DEBUG STAT: Highlight flags for {metric}:\n{highlight_flags}")

                indices_to_highlight = means_valid[highlight_flags].index
                # if is_last_metric: print(f"DEBUG STAT: Indices to highlight for {metric}: {indices_to_highlight.tolist()}")

                for sub_idx in indices_to_highlight:
                    try:
                        original_index = dataset_group.loc[[sub_idx]].index[0]  # Use list selection
                        styles.loc[original_index, metric] = self.highlight_style
                        # if is_last_metric: print(f"DEBUG STAT: Applied style to Index: {original_index}, Metric: {metric}")

                    except Exception as e:
                        warnings.warn(
                            f'Style application failed for Index: {original_index}, Metric: {metric}. Error: {e}',
                            UserWarning,
                        )

        return styles

    # --- Methods for 'per_metric' highlighting (ROW-WISE comparison) ---
    # (These remain unchanged regarding the 'axis' keyword as they don't use groupby that way)
    # ... [highlight_best_per_metric and highlight_statistically_similar_to_best_per_metric methods] ...
    # Make sure to include the full definitions of these methods from the previous response if needed.
    def highlight_best_per_metric(self, data):
        """
        Highlights the best value(s) per (metric, dataset) group (ROW-WISE across methods).
        Assumes column MultiIndex with 'metric' level. Handles ties.
        """
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        if not isinstance(data.columns, pd.MultiIndex):
            warnings.warn(
                "'highlight_best_per_metric' expects a column MultiIndex. Returning empty styles.",
                UserWarning,
            )
            return styles

        # --- Identify Metric Level ---
        try:
            metric_level_idx = data.columns.names.index(self.metric_level)
        except (ValueError, AttributeError):
            metric_level_idx = 0  # Default to first level
            warnings.warn(
                f"Column MultiIndex missing level '{self.metric_level}'. Using level {metric_level_idx}.",
                UserWarning,
            )

        # --- Iterate Metrics ---
        for metric in data.columns.get_level_values(metric_level_idx).unique():
            try:
                metric_df = data.xs(metric, level=metric_level_idx, axis=1, drop_level=False)
            except KeyError:
                warnings.warn(
                    f"Could not select columns for metric '{metric}' using level index {metric_level_idx}.",
                    UserWarning,
                )
                continue

            if metric_df.empty:
                continue

            for dataset_idx, row_series in metric_df.iterrows():
                means = row_series.apply(lambda x: self._extract_stats(x)[0]).astype(float)
                means_valid = means.dropna()

                if not means_valid.empty:
                    scores = self._get_ordering_scores(means_valid, metric)
                    if not scores.empty and not scores.isna().all():
                        try:
                            min_score = scores.min()
                            is_best = np.isclose(scores, min_score)
                            best_col_sub_indices = means_valid[is_best].index

                            for col_sub_idx in best_col_sub_indices:
                                try:
                                    full_col_idx = col_sub_idx
                                    styles.loc[dataset_idx, full_col_idx] = self.highlight_style
                                except Exception as e:
                                    warnings.warn(
                                        f'Style application failed for Row: {dataset_idx}, Col: {full_col_idx}. Error: {e}',
                                        UserWarning,
                                    )
                        except ValueError:
                            warnings.warn(
                                f'Score calculation/comparison failed for Row: {dataset_idx}, Metric: {metric}.',
                                UserWarning,
                            )

        return styles

    def highlight_statistically_similar_to_best_per_metric(self, data):
        """
        Highlights values statistically similar to the best per (metric, dataset) group (ROW-WISE).
        Assumes column MultiIndex with 'metric' level.
        """
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        if not isinstance(data.columns, pd.MultiIndex):
            warnings.warn(
                "'highlight_statistically_similar_to_best_per_metric' expects a column MultiIndex. Returning empty styles.",
                UserWarning,
            )
            return styles

        # --- Identify Metric Level ---
        try:
            metric_level_idx = data.columns.names.index(self.metric_level)
        except (ValueError, AttributeError):
            metric_level_idx = 0
            warnings.warn(
                f"Column MultiIndex missing level '{self.metric_level}'. Using level {metric_level_idx}.",
                UserWarning,
            )

        # --- Iterate Metrics ---
        for metric in data.columns.get_level_values(metric_level_idx).unique():
            try:
                metric_df = data.xs(metric, level=metric_level_idx, axis=1, drop_level=False)
            except KeyError:
                warnings.warn(
                    f"Could not select columns for metric '{metric}' using level index {metric_level_idx}.",
                    UserWarning,
                )
                continue

            if metric_df.empty:
                continue

            for dataset_idx, row_series in metric_df.iterrows():
                stats = row_series.apply(self._extract_stats)
                means = stats.apply(lambda x: x[0]).astype(float)
                ses = stats.apply(lambda x: x[1]).astype(float)

                means_valid, ses_valid, idx_best, mean_best, se_best = self._find_best_stats(
                    means, ses, metric
                )

                if means_valid.empty or idx_best is None or pd.isna(idx_best):
                    continue

                highlight_flags = self._compare_to_best(
                    means_valid, ses_valid, idx_best, mean_best, se_best, metric
                )
                indices_to_highlight = means_valid[highlight_flags].index

                for col_sub_idx in indices_to_highlight:
                    try:
                        full_col_idx = col_sub_idx
                        styles.loc[dataset_idx, full_col_idx] = self.highlight_style
                    except Exception as e:
                        warnings.warn(
                            f'Style application failed for Row: {dataset_idx}, Col: {full_col_idx}. Error: {e}',
                            UserWarning,
                        )

        return styles
