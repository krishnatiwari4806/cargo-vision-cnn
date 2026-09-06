"""
test_vehicle_database.py - Unit Tests for Vehicle Database & Specification Schema
=================================================================================
Project: Cargo Vision Logistics System (Phase 1)
Purpose: Verifies schema integrity, validation constraints, geometric sanity,
         lookup operations, and filtering logic for the vehicle specification database.
"""

import os
import sys
import unittest
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.vehicle_database import VehicleSpec, VehicleDatabase, DEFAULT_VEHICLE_CATALOG, vehicle_db


class TestVehicleDatabase(unittest.TestCase):
    """
    Unit test cases for VehicleSpec and VehicleDatabase.
    """

    def setUp(self):
        self.db = vehicle_db

    def test_catalog_size_and_categories(self):
        """Verify all 6 required vehicle categories are populated."""
        vehicles = self.db.get_all_vehicles()
        self.assertEqual(len(vehicles), 6, "Expected exactly 6 standard fleet vehicle specifications.")

        expected_categories = [
            "3-wheeler cargo auto",
            "Tata Ace / mini tempo",
            "commercial pickup",
            "14-ft truck",
            "19-ft truck",
            "multi-car carrier",
        ]
        actual_categories = [v.category for v in vehicles]
        for cat in expected_categories:
            self.assertIn(cat, actual_categories, f"Missing required category: {cat}")

    def test_required_fields_present_and_typed(self):
        """Verify every vehicle has all required fields with non-null positive values."""
        for v in self.db.get_all_vehicles():
            self.assertIsInstance(v.vehicle_id, str)
            self.assertTrue(v.vehicle_id.startswith("V_"))
            self.assertIsInstance(v.vehicle_name, str)
            self.assertIsInstance(v.category, str)
            self.assertGreater(v.usable_length_cm, 0.0)
            self.assertGreater(v.usable_width_cm, 0.0)
            self.assertGreater(v.usable_height_cm, 0.0)
            self.assertGreater(v.usable_volume_m3, 0.0)
            self.assertGreater(v.floor_area_m2, 0.0)
            self.assertGreater(v.max_payload_kg, 0.0)
            self.assertIsInstance(v.body_type, str)
            self.assertIsInstance(v.suitable_cargo_types, list)
            self.assertGreater(len(v.suitable_cargo_types), 0)

    def test_geometric_and_physical_sanity(self):
        """Verify floor area and usable volume match bounding dimensions within bounding envelope."""
        for v in self.db.get_all_vehicles():
            calc_floor_m2 = (v.usable_length_cm * v.usable_width_cm) / 10000.0
            calc_vol_m3 = (v.usable_length_cm * v.usable_width_cm * v.usable_height_cm) / 1000000.0

            # Stored floor area and volume should closely match physical bounding box dimensions
            self.assertAlmostEqual(v.floor_area_m2, calc_floor_m2, delta=0.05,
                                   msg=f"Floor area mismatch for {v.vehicle_id}")
            self.assertAlmostEqual(v.usable_volume_m3, calc_vol_m3, delta=0.05,
                                   msg=f"Volume mismatch for {v.vehicle_id}")

    def test_payload_tier_hierarchy(self):
        """Verify payload capacities strictly increase across commercial tiers."""
        vehicles = self.db.get_all_vehicles()
        payloads = [v.max_payload_kg for v in vehicles]
        # Check ascending order: 500kg <= 850kg <= 1500kg <= 3500kg <= 7500kg <= 20000kg
        self.assertEqual(payloads, sorted(payloads), "Vehicle catalog should be strictly ordered by payload capacity.")

    def test_lookup_by_id(self):
        """Verify get_vehicle_by_id returns exact matching vehicle or None."""
        v_ace = self.db.get_vehicle_by_id("V_TATA_ACE")
        self.assertIsNotNone(v_ace)
        self.assertEqual(v_ace.category, "Tata Ace / mini tempo")
        self.assertEqual(v_ace.max_payload_kg, 850.0)

        v_none = self.db.get_vehicle_by_id("NON_EXISTENT_ID")
        self.assertIsNone(v_none)

    def test_lookup_by_category(self):
        """Verify get_vehicles_by_category retrieves matching vehicles."""
        pickup_matches = self.db.get_vehicles_by_category("commercial pickup")
        self.assertEqual(len(pickup_matches), 1)
        self.assertEqual(pickup_matches[0].vehicle_id, "V_BOLERO_PICKUP")

    def test_filter_vehicles_dimensions_and_payload(self):
        """Verify filter_vehicles filters by physical dimensions and weight thresholds."""
        # Query requiring at least 400cm length and 3000kg payload (should match 14-ft, 19-ft, and car carrier)
        matches = self.db.filter_vehicles(min_length_cm=400.0, min_payload_kg=3000.0)
        match_ids = [m.vehicle_id for m in matches]
        self.assertEqual(set(match_ids), {"V_TATA_407_14FT", "V_EICHER_19FT", "V_CAR_CARRIER_MULTI"})

        # Query requiring 15000kg payload (should match only multi-car carrier)
        heavy_matches = self.db.filter_vehicles(min_payload_kg=15000.0)
        self.assertEqual(len(heavy_matches), 1)
        self.assertEqual(heavy_matches[0].vehicle_id, "V_CAR_CARRIER_MULTI")

        # Impossible query (should return empty list)
        no_matches = self.db.filter_vehicles(min_payload_kg=50000.0)
        self.assertEqual(len(no_matches), 0)

    def test_validation_rejection_on_invalid_spec(self):
        """Verify VehicleSpec raises ValueError on negative or zero dimensions."""
        with self.assertRaises(ValueError):
            VehicleSpec(
                vehicle_id="V_INVALID",
                vehicle_name="Invalid Vehicle",
                category="test",
                usable_length_cm=-100.0,  # Negative length
                usable_width_cm=100.0,
                usable_height_cm=100.0,
                usable_volume_m3=1.0,
                floor_area_m2=1.0,
                max_payload_kg=500.0,
                body_type="test",
            )

        with self.assertRaises(ValueError):
            VehicleSpec(
                vehicle_id="V_INVALID_PAYLOAD",
                vehicle_name="Invalid Payload",
                category="test",
                usable_length_cm=100.0,
                usable_width_cm=100.0,
                usable_height_cm=100.0,
                usable_volume_m3=1.0,
                floor_area_m2=1.0,
                max_payload_kg=0.0,  # Zero payload
                body_type="test",
            )

    def test_serialization_to_dict_and_json(self):
        """Verify to_dict and to_json serialization produce valid structured data."""
        v = self.db.get_vehicle_by_id("V_BOLERO_PICKUP")
        v_dict = v.to_dict()
        self.assertEqual(v_dict["vehicle_id"], "V_BOLERO_PICKUP")
        self.assertEqual(v_dict["usable_length_cm"], 250.0)

        json_str = self.db.to_json()
        parsed_json = json.loads(json_str)
        self.assertEqual(len(parsed_json), 6)
        self.assertEqual(parsed_json[0]["vehicle_id"], "V_3W_AUTO")

    def test_vehicle_semantic_compatibility(self):
        """Verify semantic compatibility rules for general freight vs specialized car carrier."""
        car_carrier = self.db.get_vehicle_by_id("V_CAR_CARRIER_MULTI")
        self.assertIsNotNone(car_carrier)
        self.assertEqual(car_carrier.vehicle_type, "specialized_car_carrier")

        # Car carrier accepts car/vehicle cargo
        self.assertTrue(car_carrier.is_compatible_with_cargo("car"))
        self.assertTrue(car_carrier.is_compatible_with_cargo("SUV"))
        self.assertTrue(car_carrier.is_compatible_with_cargo("automobile"))

        # Car carrier rejects non-car cargo
        for non_car in ["refrigerator", "box", "bed", "chair", "table", "couch", "desk", "tv"]:
            self.assertFalse(
                car_carrier.is_compatible_with_cargo(non_car),
                f"Car carrier should not be compatible with {non_car}",
            )

        # General freight vehicles accept general cargo
        for v_id in ["V_3W_AUTO", "V_TATA_ACE", "V_BOLERO_PICKUP", "V_TATA_407_14FT", "V_EICHER_19FT"]:
            v = self.db.get_vehicle_by_id(v_id)
            self.assertEqual(v.vehicle_type, "general_freight")
            self.assertTrue(v.is_compatible_with_cargo("box"))
            self.assertTrue(v.is_compatible_with_cargo("refrigerator"))


if __name__ == "__main__":
    unittest.main()
