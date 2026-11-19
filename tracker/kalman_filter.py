"""Filtr Kalmana ze stałą prędkością dla bounding-boxów.

Stan 8D: [cx, cy, w, h, vx, vy, vw, vh].
Pomiar 4D: [cx, cy, w, h].

Szumy procesu i obserwacji są skalowane wysokością bb (h) — zgodnie z
DeepSORT/ByteTrack/BoT-SORT. Daje to adaptacyjne kowariancje: dalsze
(mniejsze) obiekty mają mniejsze szumy w pikselach niż obiekty bliskie.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


class KalmanFilter:
    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = np.asarray(measurement, dtype=float)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])
        h = float(measurement[3])
        std = np.array([
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
        ])
        covariance = np.diag(std ** 2)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        std_pos = np.array([
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
        ])
        std_vel = np.array([
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
        ])
        motion_cov = np.diag(np.concatenate([std_pos, std_vel]) ** 2)
        new_mean = self._motion_mat @ mean
        new_cov = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return new_mean, new_cov

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        std = np.array([
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
            self._std_weight_position * h,
        ])
        innovation_cov = np.diag(std ** 2)
        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return projected_mean, projected_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = np.asarray(measurement, dtype=float) - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov
