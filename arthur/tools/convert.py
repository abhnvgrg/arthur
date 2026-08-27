from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

LENGTH = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "in": 0.0254, "inch": 0.0254, "ft": 0.3048, "foot": 0.3048,
    "yd": 0.9144, "mi": 1609.344, "mile": 1609.344,
}

MASS = {
    "mg": 0.000001, "g": 0.001, "kg": 1.0, "t": 1000.0,
    "oz": 0.0283495, "lb": 0.453592, "pound": 0.453592, "st": 6.35029,
}

VOLUME = {
    "ml": 0.001, "l": 1.0, "litre": 1.0, "liter": 1.0,
    "cup": 0.236588, "pint": 0.473176, "gal": 3.78541, "gallon": 3.78541,
}

DURATION = {
    "s": 1.0, "sec": 1.0, "second": 1.0,
    "min": 60.0, "minute": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0,
    "day": 86400.0, "week": 604800.0,
}

DATA = {
    "b": 1.0, "byte": 1.0,
    "kb": 1024.0, "mb": 1024.0 ** 2, "gb": 1024.0 ** 3, "tb": 1024.0 ** 4,
}

FAMILIES = {
    "length": LENGTH,
    "mass": MASS,
    "volume": VOLUME,
    "duration": DURATION,
    "data": DATA,
}

TEMPERATURES = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}


class ConversionError(ValueError):
    pass


def _to_celsius(value: float, unit: str) -> float:
    if unit in {"c", "celsius"}:
        return value
    if unit in {"f", "fahrenheit"}:
        return (value - 32.0) * 5.0 / 9.0
    return value - 273.15


def _from_celsius(value: float, unit: str) -> float:
    if unit in {"c", "celsius"}:
        return value
    if unit in {"f", "fahrenheit"}:
        return value * 9.0 / 5.0 + 32.0
    return value + 273.15


def convert(value: float, source: str, target: str) -> dict[str, Any]:
    source_unit = source.strip().lower().rstrip("s") or source.strip().lower()
    target_unit = target.strip().lower().rstrip("s") or target.strip().lower()

    if source_unit in TEMPERATURES or target_unit in TEMPERATURES:
        if source_unit not in TEMPERATURES or target_unit not in TEMPERATURES:
            raise ConversionError(
                f"Cannot convert between {source!r} and {target!r}: "
                "temperature units only convert to other temperature units."
            )
        result = _from_celsius(_to_celsius(value, source_unit), target_unit)
        return {
            "value": value,
            "from": source_unit,
            "to": target_unit,
            "result": result,
            "family": "temperature",
        }

    for family, table in FAMILIES.items():
        if source_unit in table and target_unit in table:
            result = value * table[source_unit] / table[target_unit]
            return {
                "value": value,
                "from": source_unit,
                "to": target_unit,
                "result": result,
                "family": family,
            }

    known = {source_unit: None, target_unit: None}
    for family, table in FAMILIES.items():
        for unit in list(known):
            if unit in table:
                known[unit] = family

    if known[source_unit] and known[target_unit]:
        raise ConversionError(
            f"Cannot convert {source!r} ({known[source_unit]}) to "
            f"{target!r} ({known[target_unit]}): different kinds of quantity."
        )

    unknown = [unit for unit, family in known.items() if family is None]
    raise ConversionError(f"Unknown unit(s): {', '.join(unknown)}")


def supported_units() -> dict[str, list[str]]:
    units = {family: sorted(table) for family, table in FAMILIES.items()}
    units["temperature"] = sorted(TEMPERATURES)
    return units


class ConvertArgs(BaseModel):
    value: float
    from_unit: str = Field(min_length=1, max_length=20)
    to_unit: str = Field(min_length=1, max_length=20)


class NoArgs(BaseModel):
    pass


def register(registry) -> None:
    from arthur.tools.registry import Risk

    @registry.tool(
        name="convert_units",
        description=(
            "Convert a value between units of length, mass, volume, duration, "
            "data size, or temperature."
        ),
        parameters=ConvertArgs,
        risk=Risk.READ_ONLY,
    )
    def convert_units(args: ConvertArgs) -> dict[str, Any]:
        return convert(args.value, args.from_unit, args.to_unit)

    @registry.tool(
        name="list_units",
        description="List every unit that convert_units understands.",
        parameters=NoArgs,
        risk=Risk.READ_ONLY,
    )
    def list_units(_: NoArgs) -> dict[str, Any]:
        return supported_units()
