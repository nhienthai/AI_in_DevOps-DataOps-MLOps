"""
A dependency-free SVD (biased matrix factorisation) recommender.

Why this exists
---------------
``scikit-surprise`` ships C extensions and has no prebuilt wheel for every
interpreter/OS combination (notably CPython 3.12 on Windows, where it needs the
MSVC toolchain). Rather than making the whole test suite unrunnable on a
developer machine, this module provides a NumPy-only model that is
*API-compatible* with ``surprise.SVD``:

    prediction = model.predict(user_id, movie_id)
    prediction.est  ->  float

which is the only surface :class:`app.model.MovieRatingModel` depends on. The
serving layer therefore does not care which backend produced the artifact.

The algorithm is the standard Funk-SVD objective optimised with SGD:

    r_hat(u, i) = mu + b_u + b_i + q_i . p_u
"""

from typing import Dict, List, NamedTuple, Sequence, Tuple

import numpy as np

DEFAULT_MIN_RATING = 1.0
DEFAULT_MAX_RATING = 5.0


class Prediction(NamedTuple):
    """Mirrors ``surprise.prediction_algorithms.predictions.Prediction``."""

    uid: str
    iid: str
    r_ui: float | None
    est: float
    details: Dict[str, object]


class TrainsetView:
    """
    Minimal stand-in for ``surprise.Trainset``.

    Only the two lookups the serving layer uses are implemented, with the same
    contract: return the inner id, or raise ``ValueError`` when the raw id was
    never seen. That lets :meth:`app.model.MovieRatingModel.is_known_user` run
    one code path against either backend.
    """

    def __init__(self, user_index: Dict[str, int], item_index: Dict[str, int]) -> None:
        self._user_index = user_index
        self._item_index = item_index

    def to_inner_uid(self, raw_uid: str) -> int:
        """Return the inner user id, or raise ValueError if unknown."""
        try:
            return self._user_index[str(raw_uid)]
        except KeyError:
            raise ValueError(f"User {raw_uid} is not part of the trainset.") from None

    def to_inner_iid(self, raw_iid: str) -> int:
        """Return the inner item id, or raise ValueError if unknown."""
        try:
            return self._item_index[str(raw_iid)]
        except KeyError:
            raise ValueError(f"Item {raw_iid} is not part of the trainset.") from None

    @property
    def n_users(self) -> int:
        """Number of distinct users seen during training."""
        return len(self._user_index)

    @property
    def n_items(self) -> int:
        """Number of distinct items seen during training."""
        return len(self._item_index)


class LocalSVD:
    """
    Biased matrix factorisation trained with stochastic gradient descent.

    Args:
        n_factors: dimensionality of the latent space.
        n_epochs: number of SGD passes over the training data.
        lr_all: learning rate applied to every parameter.
        reg_all: L2 regularisation applied to every parameter.
        random_state: seed, so training is reproducible run to run.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state

        self.global_mean: float = 0.0
        self.user_index: Dict[str, int] = {}
        self.item_index: Dict[str, int] = {}
        self.bu: np.ndarray = np.zeros(0)
        self.bi: np.ndarray = np.zeros(0)
        self.pu: np.ndarray = np.zeros((0, 0))
        self.qi: np.ndarray = np.zeros((0, 0))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, ratings: Sequence[Tuple[str, str, float]]) -> "LocalSVD":
        """
        Fit the model on ``(user_id, movie_id, rating)`` triplets.

        Returns ``self`` so the call can be chained, like scikit-learn estimators.
        """
        if len(ratings) == 0:
            raise ValueError("Cannot fit on an empty ratings set")

        users = [str(u) for u, _, _ in ratings]
        items = [str(i) for _, i, _ in ratings]
        values = np.asarray([float(r) for _, _, r in ratings], dtype=np.float64)

        self.user_index = {u: n for n, u in enumerate(dict.fromkeys(users))}
        self.item_index = {i: n for n, i in enumerate(dict.fromkeys(items))}
        n_users, n_items = len(self.user_index), len(self.item_index)

        u_idx = np.asarray([self.user_index[u] for u in users], dtype=np.int64)
        i_idx = np.asarray([self.item_index[i] for i in items], dtype=np.int64)

        rng = np.random.default_rng(self.random_state)
        self.global_mean = float(values.mean())
        self.bu = np.zeros(n_users, dtype=np.float64)
        self.bi = np.zeros(n_items, dtype=np.float64)
        self.pu = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.qi = rng.normal(0, 0.1, (n_items, self.n_factors))

        lr, reg = self.lr_all, self.reg_all
        order = np.arange(len(values))

        for _ in range(self.n_epochs):
            rng.shuffle(order)
            for n in order:
                u, i, r = u_idx[n], i_idx[n], values[n]
                pu, qi = self.pu[u], self.qi[i]

                err = r - (self.global_mean + self.bu[u] + self.bi[i] + float(pu @ qi))

                self.bu[u] += lr * (err - reg * self.bu[u])
                self.bi[i] += lr * (err - reg * self.bi[i])
                # Update both factor vectors from the same residual.
                pu_old = pu.copy()
                pu += lr * (err * qi - reg * pu)
                qi += lr * (err * pu_old - reg * qi)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def estimate(self, user_id: str, movie_id: str) -> Tuple[float, bool]:
        """
        Return ``(estimate, was_impossible)`` for a user-movie pair.

        Unknown users or movies fall back to the global mean plus whichever
        bias term is known - the usual cold-start behaviour.
        """
        u = self.user_index.get(str(user_id))
        i = self.item_index.get(str(movie_id))

        estimate = self.global_mean
        if u is not None:
            estimate += float(self.bu[u])
        if i is not None:
            estimate += float(self.bi[i])
        if u is not None and i is not None:
            estimate += float(self.pu[u] @ self.qi[i])

        was_impossible = u is None or i is None
        clipped = min(max(estimate, DEFAULT_MIN_RATING), DEFAULT_MAX_RATING)
        return clipped, was_impossible

    def predict(self, uid: str, iid: str, r_ui: float | None = None) -> Prediction:
        """Predict a rating, returning a surprise-shaped ``Prediction``."""
        est, was_impossible = self.estimate(uid, iid)
        details: Dict[str, object] = {"was_impossible": was_impossible}
        if was_impossible:
            details["reason"] = "User and/or item is unknown."
        return Prediction(uid=str(uid), iid=str(iid), r_ui=r_ui, est=est, details=details)

    def test(self, ratings: Sequence[Tuple[str, str, float]]) -> List[Prediction]:
        """Predict every triplet in ``ratings`` (mirrors ``surprise.SVD.test``)."""
        return [self.predict(u, i, r) for u, i, r in ratings]

    @property
    def trainset(self) -> TrainsetView:
        """Surprise-shaped view of what the model saw during training."""
        return TrainsetView(self.user_index, self.item_index)


def rmse(predictions: Sequence[Prediction]) -> float:
    """Root mean squared error over predictions carrying a true rating."""
    errors = np.asarray([p.est - float(p.r_ui) for p in predictions if p.r_ui is not None])
    return float(np.sqrt((errors**2).mean()))


def mae(predictions: Sequence[Prediction]) -> float:
    """Mean absolute error over predictions carrying a true rating."""
    errors = np.asarray([p.est - float(p.r_ui) for p in predictions if p.r_ui is not None])
    return float(np.abs(errors).mean())
