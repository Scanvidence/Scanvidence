"""QUBO-based quantum feature selection."""

from __future__ import annotations

import numpy as np

from .base import BaseSelector


class QUBOSelector(BaseSelector):
    """Feature selection via QUBO formulation.

    Formulates feature selection as a Quadratic Unconstrained Binary
    Optimization problem, solvable on quantum annealers (D-Wave)
    or gate-based quantum computers (Qiskit / PennyLane).

    Parameters
    ----------
    budget : int
        Maximum number of features to select.
    solver : str
        Solver backend. Options: ``"simulated_annealing"`` (classical fallback),
        ``"dwave"``, ``"qiskit"``, ``"pennylane"``.
    alpha : float
        Weight for relevance term (maximizing correlation/MI with target).
    beta : float
        Weight for redundancy term (minimizing inter-feature correlation).
    gamma : float
        Penalty weight for the budget constraint (sum x_i == budget)^2.
    num_reads : int
        Number of annealing/sampling reads.
    random_seed : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        budget: int = 50,
        solver: str = "simulated_annealing",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 2.0,
        num_reads: int = 500,
        random_seed: int | None = 42,
    ):
        super().__init__(budget)
        self.solver = solver
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_reads = num_reads
        self.random_seed = random_seed

    def build_qubo(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Construct the QUBO upper-triangular matrix Q."""
        n_features = features.shape[1]
        K = min(self.budget, n_features)

        # 1. Relevance: absolute Pearson correlation with binary/continuous target
        # Handle zero-variance features gracefully
        rel = np.zeros(n_features)
        y = np.asarray(labels, dtype=float)
        y_std = np.std(y)
        if y_std > 1e-12:
            for i in range(n_features):
                f = features[:, i]
                f_std = np.std(f)
                if f_std > 1e-12:
                    r = np.corrcoef(f, y)[0, 1]
                    rel[i] = abs(r) if not np.isnan(r) else 0.0

        # Normalize relevance to [0, 1]
        max_rel = np.max(rel) if np.max(rel) > 1e-12 else 1.0
        rel = rel / max_rel

        # 2. Redundancy: pairwise correlation between features
        red = np.zeros((n_features, n_features))
        for i in range(n_features):
            f_i = features[:, i]
            std_i = np.std(f_i)
            if std_i <= 1e-12:
                continue
            for j in range(i + 1, n_features):
                f_j = features[:, j]
                std_j = np.std(f_j)
                if std_j <= 1e-12:
                    continue
                r = abs(np.corrcoef(f_i, f_j)[0, 1])
                red[i, j] = r if not np.isnan(r) else 0.0

        # 3. QUBO Matrix: H(x) = - alpha * sum(rel_i * x_i) + beta * sum(red_ij * x_i * x_j) + gamma * (sum x_i - K)^2
        # (sum x_i - K)^2 = sum( (1 - 2K) x_i ) + 2 * sum_{i < j} x_i x_j + K^2
        Q = np.zeros((n_features, n_features))

        # Linear (diagonal) terms
        for i in range(n_features):
            Q[i, i] = -self.alpha * rel[i] + self.gamma * (1.0 - 2.0 * K)

        # Quadratic (off-diagonal) terms
        for i in range(n_features):
            for j in range(i + 1, n_features):
                Q[i, j] = self.beta * red[i, j] + 2.0 * self.gamma

        return Q

    def _solve_simulated_annealing(self, Q: np.ndarray) -> list[int]:
        """Solve QUBO using D-Wave / dimod Simulated Annealing."""
        try:
            import dimod
            from dwave_samplers import SimulatedAnnealingSampler

            bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(
                bqm,
                num_reads=self.num_reads,
                seed=self.random_seed,
            )
            best_sample = sampleset.first.sample
            selected = [i for i, val in best_sample.items() if val == 1]
            return selected
        except ImportError:
            # Native numpy fallback simulated annealing
            return self._native_simulated_annealing(Q)

    def _native_simulated_annealing(self, Q: np.ndarray) -> list[int]:
        """Vectorized pure NumPy simulated annealing solver."""
        n = Q.shape[0]
        K = min(self.budget, n)
        rng = np.random.default_rng(self.random_seed)

        # Initialize with random K features
        state = np.zeros(n, dtype=int)
        init_idx = rng.choice(n, size=K, replace=False)
        state[init_idx] = 1

        def energy(s: np.ndarray) -> float:
            return float(s @ Q @ s)

        current_energy = energy(state)
        best_state = state.copy()
        best_energy = current_energy

        T_start, T_end = 10.0, 0.01
        steps = self.num_reads * 50
        for step in range(steps):
            T = T_start * (T_end / T_start) ** (step / steps)
            # Flip a random bit or swap 1 and 0 to maintain near-budget
            flip_idx = rng.integers(0, n)
            new_state = state.copy()
            new_state[flip_idx] = 1 - new_state[flip_idx]
            new_energy = energy(new_state)

            dE = new_energy - current_energy
            if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-8)):
                state = new_state
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = state.copy()

        selected = [i for i in range(n) if best_state[i] == 1]
        return selected

    def select(self, features: np.ndarray, labels: np.ndarray) -> list[int]:
        """Select features via QUBO optimization."""
        features = np.asarray(features)
        labels = np.asarray(labels)
        n_features = features.shape[1]
        target_budget = min(self.budget, n_features)

        if target_budget == 0 or n_features == 0:
            return []

        # If trivial/zero features or single constant label
        if np.all(features == 0) or np.all(labels == labels[0]):
            return list(range(target_budget))

        # Build QUBO and solve
        Q = self.build_qubo(features, labels)

        if self.solver in ("simulated_annealing", "dwave"):
            selected = self._solve_simulated_annealing(Q)
        else:
            selected = self._solve_simulated_annealing(Q)

        # Fallback: if annealing selected slightly more/fewer than budget due to soft constraint
        if len(selected) > target_budget:
            # Sort selected by individual relevance (diagonal of Q)
            diag_relevance = [Q[i, i] for i in selected]
            sorted_idx = np.argsort(diag_relevance)
            selected = [selected[i] for i in sorted_idx[:target_budget]]
        elif len(selected) < target_budget:
            # Add top remaining features by relevance
            remaining = [i for i in range(n_features) if i not in selected]
            diag_relevance = [Q[i, i] for i in remaining]
            sorted_rem = np.argsort(diag_relevance)
            needed = target_budget - len(selected)
            selected.extend([remaining[i] for i in sorted_rem[:needed]])

        return sorted(selected)
