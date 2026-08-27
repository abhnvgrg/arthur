from __future__ import annotations

import pytest

from arthur.tools.convert import ConversionError, convert, supported_units


@pytest.mark.parametrize(
    "value,source,target,expected",
    [
        (1, "km", "m", 1000.0),
        (100, "cm", "m", 1.0),
        (1, "mi", "km", 1.609344),
        (12, "in", "ft", 1.0),
        (1, "kg", "g", 1000.0),
        (1, "lb", "kg", 0.453592),
        (1, "l", "ml", 1000.0),
        (1, "hour", "min", 60.0),
        (1, "day", "hour", 24.0),
        (1, "gb", "mb", 1024.0),
    ],
)
def test_units_convert_correctly(value, source, target, expected):
    assert convert(value, source, target)["result"] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "value,source,target,expected",
    [
        (0, "c", "f", 32.0),
        (100, "c", "f", 212.0),
        (32, "f", "c", 0.0),
        (0, "c", "k", 273.15),
        (273.15, "k", "c", 0.0),
        (-40, "c", "f", -40.0),
    ],
)
def test_temperatures_convert_correctly(value, source, target, expected):
    assert convert(value, source, target)["result"] == pytest.approx(expected, abs=1e-6)


def test_a_conversion_reports_its_family():
    assert convert(1, "km", "m")["family"] == "length"
    assert convert(1, "c", "f")["family"] == "temperature"


def test_converting_to_the_same_unit_is_the_identity():
    assert convert(42.5, "kg", "kg")["result"] == pytest.approx(42.5)


def test_plural_units_are_accepted():
    assert convert(2, "hours", "minutes")["result"] == pytest.approx(120.0)


def test_unit_names_are_case_insensitive():
    assert convert(1, "KM", "M")["result"] == pytest.approx(1000.0)


def test_a_round_trip_returns_the_original():
    metres = convert(5, "mi", "m")["result"]
    assert convert(metres, "m", "mi")["result"] == pytest.approx(5.0)


def test_mixing_families_is_refused():
    with pytest.raises(ConversionError, match="different kinds of quantity"):
        convert(1, "kg", "km")


def test_mixing_temperature_with_anything_else_is_refused():
    with pytest.raises(ConversionError, match="temperature units"):
        convert(1, "c", "km")


@pytest.mark.parametrize("unit", ["furlong", "parsec", "smoot", ""])
def test_unknown_units_are_named_in_the_error(unit):
    with pytest.raises(ConversionError, match="Unknown unit"):
        convert(1, unit or "?", "m")


def test_negative_values_convert():
    assert convert(-5, "km", "m")["result"] == pytest.approx(-5000.0)


def test_zero_converts():
    assert convert(0, "km", "m")["result"] == 0.0


def test_every_family_is_listed():
    units = supported_units()

    assert set(units) == {"length", "mass", "volume", "duration", "data", "temperature"}
    assert "km" in units["length"]
    assert "celsius" in units["temperature"]


def test_every_listed_unit_actually_converts():
    for family, names in supported_units().items():
        if family == "temperature":
            continue
        first = names[0]
        for name in names:
            assert convert(1, name, first)["result"] > 0
