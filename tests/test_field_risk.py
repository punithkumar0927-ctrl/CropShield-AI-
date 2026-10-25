import pytest

from src.cropshield.field_risk import assess_field_risk


def test_field_risk_is_transparent_and_increases_with_reported_risk_factors():
    result = assess_field_risk(8, 20, True, True, True)

    assert result.level == "Elevated"
    assert result.score == 6
    assert result.affected_fraction == 0.4
    assert len(result.factors) == 4


def test_unknown_observations_do_not_add_risk_points():
    result = assess_field_risk(0, 20, None, None, None)

    assert result.level == "Low"
    assert result.score == 0
    assert result.factors == ("No additional spread-favoring conditions were reported.",)


def test_user_reported_rapid_spread_adds_explicit_field_factor():
    result = assess_field_risk(0, 20, None, None, None, rapid_symptom_spread=True)

    assert result.score == 2
    assert result.level == "Moderate"
    assert any("rapid symptom spread" in factor for factor in result.factors)


@pytest.mark.parametrize(
    ("affected", "inspected"),
    [(-1, 10), (11, 10), (0, 0)],
)
def test_field_risk_rejects_invalid_leaf_counts(affected, inspected):
    with pytest.raises(ValueError):
        assess_field_risk(affected, inspected, None, None, None)
