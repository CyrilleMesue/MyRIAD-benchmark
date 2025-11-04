# Import Packages
import numpy as np
import pandas as pd
from scipy.stats import beta, norm
from math import factorial
pd.set_option('future.no_silent_downcasting', True)

class RobustRankAggregator:    
    """
    A Python implementation of the Robust Rank Aggregation (RRA) algorithm and other
    statistical methods for combining ranked feature lists.

    Supports:
    - RRA (Robust Rank Aggregation)
    - Mean, Min, Median, Geometric Mean
    - Stuart p-value correction (optional)

    This code is an adaptation of the R package 'RobustRankAggreg'
    (Version 1.2, © 2022 Raivo Kolde and Sven Laur), originally released under the GPL-2 license.

    Original R package:
    https://cran.r-project.org/web/packages/RobustRankAggreg/index.html

    Adapted for Python by Cyrille M. Njume, 2025.

    License: GPL-2 (same as original package — this implementation must also remain open-source)
    """

    def __init__(self, method="RRA", full=False, exact=False, top_cutoff=None, normalize_ranks=True):
        """
        Initialize the aggregator.

        Parameters:
        - method (str): Aggregation method ('RRA', 'mean', 'min', 'median', 'geom.mean', 'stuart')
        - full (bool): Whether full rankings are provided (default: False)  
        Used if the the sets of ranked elements do not match perfectlyovided (default: False)
        - exact (bool): Whether to use exact p-value correction (default: False)
        - top_cutoff (list or None): Cutoff thresholds for normalized ranks (optional)
        """
        self.method = method.lower()
        self.full = False if top_cutoff is not None else full
        self.exact = exact
        self.top_cutoff = top_cutoff
        self.normalize_ranks = normalize_ranks

    def normalize_rmat(self, rmat):
        """
        Normalize a rank matrix so that all ranks fall between 0 and 1.
        Each entry r_ij becomes r_ij / n, where n is the number of items ranked.
        """
        n = rmat.shape[0]  # number of items/features
        return rmat / n

    def rank_matrix(self, glist, N=None):
        """
        Convert a list of ranked lists into a normalized rank matrix.

        Parameters:
        - glist (list of lists): Ranked elements
        - N (int or None): Total number of possible rankable items

        Returns:
        - pd.DataFrame: Rank matrix with elements
        """
        # Get all unique features across all lists
        unique_elements = list(set(e for sublist in glist for e in sublist))
    
        # Infer N if not given
        if N is None:
            N = len(unique_elements)
    
        columns = [f"List_{i}" for i in range(len(glist))]
        rank_matrix = pd.DataFrame(index=unique_elements, columns=columns)
    
        if not self.full:
            # Fill missing values with max rank (N)
            rank_matrix.fillna(N, inplace=True)
    
        for i, ranked_list in enumerate(glist):
            for rank, item in enumerate(ranked_list):
                if self.full:
                    value = rank + 1  # Absolute rank
                else:
                    value = (rank + 1) / N  # Normalized rank
                rank_matrix.at[item, f"List_{i}"] = value
        
        return rank_matrix.astype(float)

    def format_output(self, scores, names, ordering="ascending"):
        """
        Format the results into a sorted DataFrame.
    
        Parameters:
        - scores (list or np.array): Score values (e.g., p-values, means, etc.)
        - names (list): Names corresponding to the scores (e.g., gene names)
        - ordering (str): 'ascending' or 'descending'
    
        Returns:
        - pd.DataFrame: Two-column table with sorted results
        """
        df = pd.DataFrame({"Name": names, "Score": scores})
        df = df.sort_values(by="Score", ascending=(ordering == "ascending")).reset_index(drop=True)
        return df

    def sum_stuart(self, v, r):
        """
        Stuart-Aerts helper to compute recursive term in qStuart.
        
        Parameters:
        - v (np.array): Recursive vector v[1:k]
        - r (float): Single rank value r[N - k + 1]
    
        Returns:
        - float: Summed result for this k
        """
        k = len(v)
        l_k = np.arange(1, k + 1)
        ones = (-1) ** (l_k + 1)
        f = np.array([factorial(i) for i in l_k])
        p = r ** l_k
        return np.sum(ones * np.flip(v) * p / f)

    def q_stuart(self,r):
        """
        Calculate the Stuart score for a single sorted rank vector.
        Equivalent to qStuart in R.
        """
        r = np.array(r, dtype=np.float64)
        N = np.sum(~np.isnan(r))
        v = np.zeros(N + 1)
        v[0] = 1
    
        for k in range(1, N + 1):
            v[k] = self.sum_stuart(v[:k], r[N - k])
        return factorial(N) * v[N]
    
    def stuart(self,rmat):
        """
        Apply Stuart method across all rows of a rank matrix.
        Equivalent to stuart(rmat) in R.
        
        Parameters:
        - rmat: 2D np.array where rows are features and columns are scores
    
        Returns:
        - np.array: Stuart scores for each row
        """
        rmat = np.sort(rmat, axis=1)  # sort each row
        return np.apply_along_axis(self.q_stuart, axis=1, arr=rmat)

    def beta_scores(self, r):
        """
        Compute beta scores for a vector of normalized ranks.
    
        This function uses order statistics to compute p-values under the null hypothesis
        that ranks are uniformly distributed in [0, 1].
    
        Parameters:
        - r (np.array or list): Normalized ranks in [0, 1]
    
        Returns:
        - np.array: Vector of p-values (one for each order statistic)
        """
        r = np.array(r, dtype=np.float64)
        r = r[~np.isnan(r)]  # remove NA values
    
        n = len(r)
        if n == 0:
            return np.array([])
    
        r_sorted = np.sort(r)
        k = np.arange(1, n + 1)
        pvals = beta.cdf(r_sorted, k, n - k + 1)
    
        return pvals

    def threshold_beta_score(self, r, k=None, n=None, sigma=None):
        """
        Python implementation of thresholdBetaScore from the RobustRankAggreg R package.
    
        Parameters:
        - r: np.array of observed p-values (normalized ranks)
        - k: list of cutoffs for which to compute beta scores
        - n: total number of items
        - sigma: np.array of top cutoff thresholds (e.g., topCutoff)
    
        Returns:
        - np.array of threshold-adjusted beta scores
        """
        r = np.array(r, dtype=np.float64)
        r = r[~np.isnan(r)]
    
        if n is None:
            n = len(r)
        if k is None:
            k = np.arange(1, n + 1)
        if sigma is None:
            sigma = np.ones(n)
        sigma = np.asarray(sigma)
    
        if len(sigma) != n:
            raise ValueError("The length of sigma does not match n")
        if len(r) != n:
            raise ValueError("The length of r does not match n")
        if np.min(sigma) < 0 or np.max(sigma) > 1:
            raise ValueError("Elements of sigma must be in the range [0,1]")
        if np.any((~np.isnan(r)) & (r > sigma)):
            raise ValueError("Elements of r must be smaller than or equal to corresponding sigma")
    
        x = np.sort(r)
        sigma_sorted = np.sort(sigma)[::-1]  # decreasing order
        beta_ = np.full(len(k), np.nan)
    
        for idx, ki in enumerate(k):
            if ki > n:
                beta_[idx] = 0
                continue
            if ki > len(x):
                beta_[idx] = 1
                continue
            if sigma_sorted[n - 1] >= x[ki - 1]:
                beta_[idx] = beta.cdf(x[ki - 1], ki, n + 1 - ki)
                continue
    
            n0_candidates = np.where(sigma_sorted < x[ki - 1])[0]
            n0 = n0_candidates[0] if len(n0_candidates) > 0 else n
    
            if n0 == 0:
                B = np.zeros(ki + 1)
                B[0] = 1
            elif ki > n0:
                pvals = beta.cdf(x[ki - 1], np.arange(1, n0 + 1), np.arange(n0, 0, -1))
                B = np.concatenate(([1], pvals, np.zeros(ki - n0)))
            else:
                pvals = beta.cdf(x[ki - 1], np.arange(1, ki + 1), n0 + 1 - np.arange(1, ki + 1))
                B = np.concatenate(([1], pvals))
    
            z = sigma_sorted[n0:n]
            for zj in z:
                B[1:ki + 1] = (1 - zj) * B[1:ki + 1] + zj * B[0:ki]
    
            beta_[idx] = B[ki]
    
        return beta_

    def correct_beta_pvalues(self, p, k):
        """
        Apply Bonferroni-like correction: min(p * k, 1)
    
        Parameters:
        - p (float): minimum beta score
        - k (int): number of valid ranks
    
        Returns:
        - float: corrected p-value
        """
        return min(p * k, 1.0)
        
    def correct_beta_pvalues_exact(self, p, k):
        """
        Apply exact correction using Stuart method on reversed beta quantiles.
    
        Parameters:
        - p (float): minimum beta score
        - k (int): number of non-NaN entries
    
        Returns:
        - float: corrected p-value
        """
        if k < 1:
            return 1.0
    
        q_matrix = np.array([1 - beta.ppf(p, i, k - i + 1) for i in range(1, k + 1)])
        q_matrix = q_matrix.reshape(1, -1)  # mimic t() and apply() from R
    
        return 1.0 - self.stuart(q_matrix)
    
    def rho_scores(self, r):
        """
        Compute the corrected p-value (rho score) for a vector of normalized ranks.
    
        Parameters:
        - r (np.array or list): Vector of normalized ranks [0, 1]
        Returns:
        - float: Rho score (corrected minimum beta p-value)
        """
        r = np.array(r, dtype=np.float64)
        r = r[~np.isnan(r)]
        r[r == 1.0] = np.nan

        # Remove NaNs and r == 1.0 from BOTH r and sigma
        mask = (~np.isnan(r)) & (r != 1.0)
        r = r[mask]
        
        if len(r) == 0:
            return 1.0
    
        if self.top_cutoff is None:
            x = self.beta_scores(r)
        else:
            sigma = np.array(self.top_cutoff, dtype=np.float64)
            sigma = sigma[mask]
            x = self.threshold_beta_score(r, sigma=sigma)
    
        min_x = np.nanmin(x) if len(x) else 1.0
        k_val = np.sum(~np.isnan(x))
    
        if self.exact:
            return self.correct_beta_pvalues_exact(min_x, k_val)
        else:
            return self.correct_beta_pvalues(min_x, k_val)

    def aggregate_ranks(self, glist=None, rmat=None, N=None):
        """
        Aggregate ranks using the selected method.

        Parameters:
        - glist (list of li. The glist is by default converted to this format.sts): Optional raw ranked input
        - rmat (pd.DataFrame): Optional precomputed rank matrix
        - N (int): Total number of rankable items
        """
        if rmat is None:
            if glist is None:
                raise ValueError("Either glist or rmat must be provided.")
            rmat = self.rank_matrix(glist, N)
    
        if N is None:
            N = rmat.shape[0]
            
        if self.normalize_ranks:
            rmat = self.normalize_rmat(rmat)
        
        names = rmat.index.tolist()
    
        if self.method == "min":
            scores = rmat.min(axis=1).values
            return self.format_output(scores, names, ordering="ascending")
    
        elif self.method == "median":
            scores = rmat.median(axis=1).values
            return self.format_output(scores, names, ordering="ascending")
    
        elif self.method == "mean":
            scores = rmat.mean(axis=1).values
            n = rmat.notna().sum(axis=1)
            corrected = norm.cdf(scores, loc=0.5, scale=np.sqrt(1 / (12 * n)))
            return self.format_output(corrected, names, ordering="ascending")
    
        elif self.method == "geom.mean":
            scores = rmat.apply(lambda x: np.exp(np.nanmean(np.log(x))), axis=1).values
            return self.format_output(scores, names, ordering="ascending")
    
        elif self.method == "rra":
            scores = rmat.apply(self.rho_scores, axis=1).values
            return self.format_output(scores, names, ordering="ascending")
    
        elif self.method == "stuart":
            scores = self.stuart(rmat.values)
            return self.format_output(scores, rmat.index.tolist(), ordering="ascending")
        
        else:
            raise ValueError(
                f"Unsupported aggregation method: '{self.method}'. "
                "Method should be one of: 'min', 'geom.mean', 'mean', 'median', 'stuart' or 'RRA'."
            )
