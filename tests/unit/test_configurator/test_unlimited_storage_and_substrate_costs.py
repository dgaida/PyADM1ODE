"""Unlimited substrate storage and substrate costs.

A storage without a stated stock always has enough substrate, so a plant can run
for months without tracking inventory or refills. Substrates carry a cost; it
can be evaluated for any feed record (measured, estimated, configured) and is
accumulated by the plant from the feed the digesters really received.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyadm1 import Feedstock
from pyadm1.components.feeding.substrate_storage import SubstrateStorage
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.configurator.plant_configurator import PlantConfigurator

SUBSTRATES = ["maize_silage_milk_ripeness", "swine_manure"]
MAIZE, MANURE = SUBSTRATES
Q = [10.0, 5.0] + [0.0] * 8
DT = 1.0 / 24.0


@pytest.fixture
def feedstock() -> Feedstock:
    fs = Feedstock(SUBSTRATES, feeding_freq=24, total_simtime=5)
    fs.set_substrate_costs({MAIZE: 40.0, MANURE: 2.0})
    return fs


def _plant(feedstock: Feedstock) -> tuple[BiogasPlant, PlantConfigurator]:
    plant = BiogasPlant("p")
    cfg = PlantConfigurator(plant, feedstock)
    cfg.add_digester("dig", V_liq=1000.0, V_gas=150.0, Q_substrates=list(Q))
    return plant, cfg


class TestUnlimitedStorage:
    def test_no_stock_given_means_unlimited(self) -> None:
        assert SubstrateStorage("s").unlimited is True
        assert SubstrateStorage("s").available_stock == float("inf")
        assert SubstrateStorage("s", initial_level=0.0).unlimited is False
        assert SubstrateStorage("s", initial_level=0.0).available_stock == 0.0

    def test_it_never_runs_empty_and_never_degrades(self) -> None:
        storage = SubstrateStorage("s", capacity=100.0)

        for day in range(365):
            out = storage.step(float(day), 1.0, {"withdrawal_rate": 50.0})

        assert out["withdrawn_this_step"] == 50.0
        assert out["is_empty"] is False
        assert out["quality_factor"] == 1.0
        assert out["losses_this_step"] == 0.0
        assert storage.cumulative_withdrawals == pytest.approx(365 * 50.0)

    def test_feeder_is_never_starved_in_a_plant(self, feedstock: Feedstock) -> None:
        plant, cfg = _plant(feedstock)
        cfg.add_substrate_storage("silo", MAIZE, capacity=10.0)  # tiny, but unlimited
        cfg.add_feeder("feed", "dig", MAIZE, storage_id="silo", enable_dosing_noise=False)

        for _ in range(48):
            results = plant.step(DT)

        assert results["feed"]["Q_actual"] == pytest.approx(10.0)
        assert plant.components["silo"].cumulative_withdrawals == pytest.approx(10.0 * 2.0)

    def test_flag_round_trips_and_old_files_stay_tracked(self) -> None:
        unlimited = SubstrateStorage("s")
        config = unlimited.to_dict()

        assert SubstrateStorage.from_dict(config).unlimited is True
        del config["unlimited"]
        assert SubstrateStorage.from_dict(config).unlimited is False

    def test_tracked_storage_round_trips_its_level(self) -> None:
        tracked = SubstrateStorage("s", capacity=1000.0, initial_level=400.0)

        restored = SubstrateStorage.from_dict(tracked.to_dict())

        assert restored.unlimited is False
        assert restored.current_level == 400.0


class TestSubstrateCost:
    def test_cost_defaults_to_zero_without_a_price(self) -> None:
        from dataclasses import fields

        from pyadm1.substrates.feedstock import SubstrateParams

        assert next(f for f in fields(SubstrateParams) if f.name == "cost").default == 0.0

    def test_library_price_can_be_overridden(self) -> None:
        fs = Feedstock(SUBSTRATES, feeding_freq=24, total_simtime=5)
        library = fs.substrate_costs

        fs.set_substrate_costs({MAIZE: 47.5})

        assert fs.substrate_costs == {MAIZE: 47.5, MANURE: library[MANURE]}

    def test_setting_costs_does_not_leak_into_another_feedstock(self, feedstock: Feedstock) -> None:
        feedstock.set_substrate_costs({MAIZE: 99.0})
        other = Feedstock(SUBSTRATES, feeding_freq=24, total_simtime=5)

        assert feedstock.substrate_costs[MAIZE] == 99.0
        assert other.substrate_costs[MAIZE] != 99.0

    def test_unknown_substrate_and_negative_cost_are_rejected(self, feedstock: Feedstock) -> None:
        with pytest.raises(ValueError, match="grass"):
            feedstock.set_substrate_costs({"grass": 10.0})
        with pytest.raises(ValueError, match="negative"):
            feedstock.set_substrate_costs({MAIZE: -1.0})

    def test_cost_is_read_from_a_substrate_file(self, tmp_path) -> None:
        import shutil
        from pathlib import Path

        import pyadm1

        source = next(Path(pyadm1.__file__).resolve().parents[1].glob(f"data/substrates/{MAIZE}.yaml"), None)
        if source is None:
            pytest.skip("substrate library not found next to the package")
        target = tmp_path / "priced_maize.yaml"
        shutil.copy(source, target)
        target.write_text(target.read_text(encoding="utf-8") + "\ncost: 37.5\n", encoding="utf-8")

        fs = Feedstock([str(target)], feeding_freq=24, total_simtime=5)

        assert fs.substrate_costs == {"priced_maize": 37.5}


class TestCostFromAnyFeedRecord:
    def test_constant_rates_over_a_period(self, feedstock: Feedstock) -> None:
        costs = feedstock.calc_substrate_costs(Q, dt=30.0)

        # Default Feedstock convention: Q is tonnes per day.
        assert costs["mass_t"][MAIZE] == pytest.approx(300.0)
        assert costs["by_substrate"] == pytest.approx({MAIZE: 12000.0, MANURE: 300.0})
        assert costs["total"] == pytest.approx(12300.0)

    def test_time_series_as_from_a_flow_sensor(self, feedstock: Feedstock) -> None:
        measured = np.array([[9.5, 5.2], [10.4, 4.9], [0.0, 5.0]])  # three daily readings

        costs = feedstock.calc_substrate_costs(measured, dt=1.0)

        assert costs["mass_t"][MAIZE] == pytest.approx(19.9)
        assert costs["total"] == pytest.approx(19.9 * 40.0 + 15.1 * 2.0)

    def test_volume_convention_converts_through_the_density(self) -> None:
        fs = Feedstock(SUBSTRATES, feeding_freq=24, total_simtime=5, simba_q_convention=False)
        fs.set_substrate_costs({MAIZE: 40.0})

        costs = fs.calc_substrate_costs([10.0, 0.0], dt=1.0)

        assert costs["mass_t"][MAIZE] == pytest.approx(10.0 * fs._densities[0] / 1000.0)


class TestCostFromTheSimulation:
    def test_plant_accumulates_the_configured_feed(self, feedstock: Feedstock) -> None:
        plant, _ = _plant(feedstock)

        for _ in range(24):
            plant.step(DT)
        costs = plant.get_substrate_costs()

        assert costs["mass_t"] == pytest.approx({MAIZE: 10.0, MANURE: 5.0})
        assert costs["total"] == pytest.approx(10.0 * 40.0 + 5.0 * 2.0)
        assert costs["specific_cost"] == pytest.approx(410.0 / 15.0)
        # Same figures as pricing the configured feed directly.
        direct = feedstock.calc_substrate_costs(Q, dt=1.0)
        assert costs["by_substrate"] == pytest.approx(direct["by_substrate"])

    def test_plant_prices_what_was_really_fed(self, feedstock: Feedstock) -> None:
        plant, cfg = _plant(feedstock)
        cfg.add_substrate_storage("silo", MAIZE, capacity=500.0, initial_level=0.1, degradation_rate=1e-12)
        cfg.add_feeder("feed", "dig", MAIZE, storage_id="silo", enable_dosing_noise=False)

        for _ in range(24):
            plant.step(DT)
        costs = plant.get_substrate_costs()

        # Only the 0.1 t that were in the silo reached the digester, not 10 t.
        assert costs["mass_t"][MAIZE] == pytest.approx(0.1, abs=1e-6)
        assert costs["by_substrate"][MAIZE] == pytest.approx(4.0, abs=1e-4)

    def test_costs_round_trip_through_json(self, feedstock: Feedstock, tmp_path) -> None:
        plant, _ = _plant(feedstock)
        for _ in range(24):
            plant.step(DT)
        path = tmp_path / "plant.json"

        plant.to_json(str(path))
        restored = BiogasPlant.from_json(str(path), feedstock=feedstock)

        before, after = plant.get_substrate_costs(), restored.get_substrate_costs()
        assert after["total"] == pytest.approx(before["total"])
        assert after["by_substrate"] == pytest.approx(before["by_substrate"])
        assert after["mass_t"] == pytest.approx(before["mass_t"])

    def test_plant_without_feed_costs_nothing(self) -> None:
        assert BiogasPlant("p").get_substrate_costs() == {"total": 0.0, "by_substrate": {}, "mass_t": {}, "specific_cost": 0.0}


class TestLibraryPrices:
    """Every library substrate carries a price, and every file format reads ``cost``."""

    @staticmethod
    def _library():
        from pathlib import Path

        import pyadm1

        root = Path(pyadm1.__file__).resolve().parents[1] / "data" / "substrates"
        if not root.is_dir():
            pytest.skip("substrate library not found next to the package")
        return root

    def test_every_library_substrate_has_a_price(self) -> None:
        from pyadm1.substrates.feedstock import load_substrate

        prices = {p.stem: load_substrate(p).cost for p in sorted(self._library().glob("*.yaml"))}

        assert prices and all(cost > 0.0 for cost in prices.values()), prices
        # Energy crops cost an order of magnitude more than slurry.
        assert prices["maize_silage_milk_ripeness"] > 10.0 * prices["swine_manure"]

    @pytest.mark.parametrize("suffix", [".yaml", ".toml", ".xml"])
    def test_cost_is_read_from_every_file_format(self, suffix: str) -> None:
        from pyadm1.substrates.feedstock import load_substrate

        assert load_substrate(self._library() / "examples" / f"cattle_manure{suffix}").cost == 2.0
