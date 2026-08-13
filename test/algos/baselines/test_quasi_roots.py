import pytest

from main.algos.baselines.quasi_roots import enumerate_real_polynomial_roots


def test_enumerates_distinct_and_repeated_cubic_roots():
    # (x - 0.25)^2 (x - 0.75)
    result = enumerate_real_polynomial_roots(
        [-0.046875, 0.4375, -1.25, 1.0]
    )

    assert [root.value for root in result.roots] == pytest.approx([0.25, 0.75])
    assert [root.multiplicity for root in result.roots] == [2, 1]


def test_excludes_real_roots_outside_the_admissible_edge():
    # (x + 1) (x - 0.5) (x - 2)
    result = enumerate_real_polynomial_roots([1.0, -1.5, -1.5, 1.0])

    assert [root.value for root in result.roots] == pytest.approx([0.5])


def test_reports_an_identically_zero_relation():
    result = enumerate_real_polynomial_roots([0.0, 0.0, 0.0, 0.0])

    assert result.identically_zero
    assert result.roots == ()

