"""Algebraic real-root enumeration for the QUASI continuity polynomial.

The QUASI C1 relation has degree at most three after its rational endpoint
slopes are cross-multiplied.  Sampling sign changes is not exhaustive: it
misses even-multiplicity roots and can miss several roots in one sample
interval.  This module instead isolates every real root using the roots of the
derivative.  Repeated roots are stationary roots of the polynomial and are
therefore considered explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class AlgebraicRoot:
    """One distinct real polynomial root and its algebraic multiplicity."""

    value: float
    multiplicity: int


@dataclass(frozen=True)
class AlgebraicRootSet:
    """All distinct real roots in a closed interval."""

    roots: Tuple[AlgebraicRoot, ...]
    degree: int
    identically_zero: bool = False

    @property
    def values(self) -> Tuple[float, ...]:
        return tuple(root.value for root in self.roots)


def _trim_and_normalize(
    coefficients: Sequence[float], tolerance: float
) -> Tuple[np.ndarray, bool]:
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Polynomial coefficients must be a nonempty 1D sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("Polynomial coefficients must be finite")

    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return np.asarray([0.0]), True
    values = values / scale
    while values.size > 1 and abs(values[-1]) <= tolerance:
        values = values[:-1]
    if values.size == 1 and abs(values[0]) <= tolerance:
        return np.asarray([0.0]), True
    return values, False


def _evaluate(coefficients: np.ndarray, value: float) -> float:
    return float(np.polynomial.polynomial.polyval(value, coefficients))


def _residual_scale(coefficients: np.ndarray, value: float) -> float:
    powers = np.power(abs(value), np.arange(coefficients.size, dtype=float))
    return max(1.0, float(np.dot(np.abs(coefficients), powers)))


def _is_zero(coefficients: np.ndarray, value: float, tolerance: float) -> bool:
    return abs(_evaluate(coefficients, value)) <= tolerance * _residual_scale(
        coefficients, value
    )


def _bisect_sign_change(
    coefficients: np.ndarray,
    lower: float,
    upper: float,
    tolerance: float,
) -> float:
    f_lower = _evaluate(coefficients, lower)
    f_upper = _evaluate(coefficients, upper)
    if f_lower == 0.0:
        return lower
    if f_upper == 0.0:
        return upper
    if np.signbit(f_lower) == np.signbit(f_upper):
        raise ValueError("Root interval does not bracket a sign change")

    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = _evaluate(coefficients, midpoint)
        if _is_zero(coefficients, midpoint, tolerance):
            return midpoint
        if abs(upper - lower) <= tolerance * max(1.0, abs(midpoint)):
            return midpoint
        if np.signbit(f_lower) == np.signbit(f_midpoint):
            lower = midpoint
            f_lower = f_midpoint
        else:
            upper = midpoint
            f_upper = f_midpoint
    return 0.5 * (lower + upper)


def _deduplicate(values: Sequence[float], tolerance: float) -> Tuple[float, ...]:
    separation = max(32.0 * tolerance, 64.0 * np.finfo(float).eps)
    result = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > separation:
            result.append(float(value))
        elif abs(value) < abs(result[-1]):
            result[-1] = float(value)
    return tuple(result)


def _distinct_root_values(
    coefficients: np.ndarray,
    lower: float,
    upper: float,
    tolerance: float,
) -> Tuple[float, ...]:
    degree = coefficients.size - 1
    if degree <= 0:
        return ()
    if degree == 1:
        root = -coefficients[0] / coefficients[1]
        if lower - tolerance <= root <= upper + tolerance:
            return (min(upper, max(lower, float(root))),)
        return ()

    derivative = np.arange(1, coefficients.size, dtype=float) * coefficients[1:]
    derivative, derivative_zero = _trim_and_normalize(derivative, tolerance)
    critical_points = ()
    if not derivative_zero:
        critical_points = _distinct_root_values(derivative, lower, upper, tolerance)

    candidates = []
    knots = (lower,) + critical_points + (upper,)
    for knot in knots:
        if _is_zero(coefficients, knot, tolerance):
            candidates.append(knot)

    for left, right in zip(knots[:-1], knots[1:]):
        if right - left <= tolerance:
            continue
        f_left = _evaluate(coefficients, left)
        f_right = _evaluate(coefficients, right)
        if _is_zero(coefficients, left, tolerance) or _is_zero(
            coefficients, right, tolerance
        ):
            continue
        if np.signbit(f_left) != np.signbit(f_right):
            candidates.append(
                _bisect_sign_change(
                    coefficients,
                    left,
                    right,
                    tolerance,
                )
            )
    return _deduplicate(candidates, tolerance)


def _multiplicity(coefficients: np.ndarray, root: float, tolerance: float) -> int:
    derivative = coefficients.copy()
    for order in range(1, coefficients.size):
        derivative = np.arange(1, derivative.size, dtype=float) * derivative[1:]
        if not _is_zero(derivative, root, tolerance):
            return order
    return coefficients.size - 1


def enumerate_real_polynomial_roots(
    coefficients: Sequence[float],
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    tolerance: float = 1.0e-12,
) -> AlgebraicRootSet:
    """Enumerate every real root in ``[lower, upper]`` without sampling.

    Coefficients are in ascending order.  The routine is intended for the
    degree-three QUASI relation but works for small higher-degree polynomials
    as well.  The identically-zero relation is represented explicitly because
    every parameter is then a solution and no unique root exists.
    """

    if tolerance <= 0.0:
        raise ValueError("Root tolerance must be positive")
    if not lower < upper:
        raise ValueError("Root interval must have positive length")

    normalized, identically_zero = _trim_and_normalize(coefficients, tolerance)
    if identically_zero:
        return AlgebraicRootSet((), degree=0, identically_zero=True)

    degree = normalized.size - 1
    values = _distinct_root_values(normalized, lower, upper, tolerance)
    roots = tuple(
        AlgebraicRoot(value, _multiplicity(normalized, value, tolerance))
        for value in values
    )
    return AlgebraicRootSet(roots, degree=degree)


__all__ = [
    "AlgebraicRoot",
    "AlgebraicRootSet",
    "enumerate_real_polynomial_roots",
]
