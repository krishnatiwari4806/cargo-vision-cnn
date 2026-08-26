"""
vehicle_database.py - Structured Vehicle Specification Database & Catalog
==========================================================================
Project: Cargo Vision Logistics System
Purpose: Provides structured, immutable vehicle specifications covering
         dimensions, usable volume, floor footprint area, payload capacity,
         body types, and suitable cargo categories for fleet recommendation.

Rules & Design:
  1. All physical dimensions are strictly in centimeters (cm).
  2. Usable volume is in cubic meters (m^3).
  3. Floor area is in square meters (m^2).
  4. Payload capacity is in kilograms (kg).
  5. Immutable dataclass pattern with schema validation and query methods.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json


@dataclass(frozen=True)
class VehicleSpec:
    """
    Structured Vehicle Specification Dataclass.
    """
    vehicle_id: str
    vehicle_name: str
    category: str
    usable_length_cm: float
    usable_width_cm: float
    usable_height_cm: float
    usable_volume_m3: float
    floor_area_m2: float
    max_payload_kg: float
    body_type: str
    suitable_cargo_types: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Validate non-negative physical values
        if self.usable_length_cm <= 0:
            raise ValueError(f"usable_length_cm must be positive, got {self.usable_length_cm}")
        if self.usable_width_cm <= 0:
            raise ValueError(f"usable_width_cm must be positive, got {self.usable_width_cm}")
        if self.usable_height_cm <= 0:
            raise ValueError(f"usable_height_cm must be positive, got {self.usable_height_cm}")
        if self.usable_volume_m3 <= 0:
            raise ValueError(f"usable_volume_m3 must be positive, got {self.usable_volume_m3}")
        if self.floor_area_m2 <= 0:
            raise ValueError(f"floor_area_m2 must be positive, got {self.floor_area_m2}")
        if self.max_payload_kg <= 0:
            raise ValueError(f"max_payload_kg must be positive, got {self.max_payload_kg}")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes vehicle specification to dictionary."""
        return asdict(self)


# -----------------------------------------------------------------------------
# Standard Commercial Fleet Vehicle Catalog
# -----------------------------------------------------------------------------

DEFAULT_VEHICLE_CATALOG: List[VehicleSpec] = [
    VehicleSpec(
        vehicle_id="V_3W_AUTO",
        vehicle_name="Cargo 3-Wheeler Auto (Piaggio Ape / Bajaj Maxima)",
        category="3-wheeler cargo auto",
        usable_length_cm=145.0,
        usable_width_cm=130.0,
        usable_height_cm=120.0,
        usable_volume_m3=2.26,
        floor_area_m2=1.88,
        max_payload_kg=500.0,
        body_type="covered_box",
        suitable_cargo_types=["box", "suitcase", "stool", "small_appliances", "cartons", "parcels"],
    ),
    VehicleSpec(
        vehicle_id="V_TATA_ACE",
        vehicle_name="Mini Tempo (Tata Ace / Mahindra Supro)",
        category="Tata Ace / mini tempo",
        usable_length_cm=220.0,
        usable_width_cm=145.0,
        usable_height_cm=150.0,
        usable_volume_m3=4.78,
        floor_area_m2=3.19,
        max_payload_kg=850.0,
        body_type="open_or_closed_bed",
        suitable_cargo_types=["box", "chair", "table", "washing_machine", "refrigerator", "desk", "small_furniture"],
    ),
    VehicleSpec(
        vehicle_id="V_BOLERO_PICKUP",
        vehicle_name="8-ft Commercial Pickup (Mahindra Bolero Maxi Truck)",
        category="commercial pickup",
        usable_length_cm=250.0,
        usable_width_cm=170.0,
        usable_height_cm=175.0,
        usable_volume_m3=7.43,
        floor_area_m2=4.25,
        max_payload_kg=1500.0,
        body_type="flatbed_tarpaulin",
        suitable_cargo_types=["couch", "dining_table", "bed", "furniture_sets", "heavy_crates", "office_equipment"],
    ),
    VehicleSpec(
        vehicle_id="V_TATA_407_14FT",
        vehicle_name="14-ft Light Commercial Truck (Tata 407 / Eicher Pro 2049)",
        category="14-ft truck",
        usable_length_cm=420.0,
        usable_width_cm=200.0,
        usable_height_cm=210.0,
        usable_volume_m3=17.64,
        floor_area_m2=8.40,
        max_payload_kg=3500.0,
        body_type="closed_container",
        suitable_cargo_types=["apartment_relocation", "bulk_boxes", "commercial_freight", "multiple_couches", "industrial_goods"],
    ),
    VehicleSpec(
        vehicle_id="V_EICHER_19FT",
        vehicle_name="19-ft Medium Freight Truck (Eicher Pro 2095 / BharatBenz)",
        category="19-ft truck",
        usable_length_cm=580.0,
        usable_width_cm=220.0,
        usable_height_cm=230.0,
        usable_volume_m3=29.35,
        floor_area_m2=12.76,
        max_payload_kg=7500.0,
        body_type="heavy_container",
        suitable_cargo_types=["industrial_machinery", "car_single", "bulk_distribution", "multi_room_furniture", "heavy_pallets"],
    ),
    VehicleSpec(
        vehicle_id="V_CAR_CARRIER_MULTI",
        vehicle_name="Dedicated Multi-Car Carrier (Double-Deck Trailer)",
        category="multi-car carrier",
        usable_length_cm=1800.0,
        usable_width_cm=250.0,
        usable_height_cm=400.0,
        usable_volume_m3=180.0,
        floor_area_m2=45.0,
        max_payload_kg=20000.0,
        body_type="double_deck_open_ramp",
        suitable_cargo_types=["car", "suv", "automobile_batch", "vehicles"],
    ),
]


class VehicleDatabase:
    """
    Vehicle Specification Database Interface.
    Provides query, filtering, and lookup methods for the fleet recommendation engine.
    """

    def __init__(self, catalog: Optional[List[VehicleSpec]] = None):
        self._catalog: List[VehicleSpec] = catalog if catalog is not None else list(DEFAULT_VEHICLE_CATALOG)
        self._id_index: Dict[str, VehicleSpec] = {v.vehicle_id: v for v in self._catalog}
        self._category_index: Dict[str, List[VehicleSpec]] = {}
        for v in self._catalog:
            self._category_index.setdefault(v.category, []).append(v)

    def get_all_vehicles(self) -> List[VehicleSpec]:
        """Returns all vehicle specifications in the database ordered by payload/capacity."""
        return list(self._catalog)

    def get_vehicle_by_id(self, vehicle_id: str) -> Optional[VehicleSpec]:
        """Finds a vehicle specification by its unique ID."""
        return self._id_index.get(vehicle_id)

    def get_vehicles_by_category(self, category: str) -> List[VehicleSpec]:
        """Returns all vehicles matching a given category name."""
        return self._category_index.get(category, [])

    def filter_vehicles(
        self,
        min_length_cm: float = 0.0,
        min_width_cm: float = 0.0,
        min_height_cm: float = 0.0,
        min_volume_m3: float = 0.0,
        min_floor_area_m2: float = 0.0,
        min_payload_kg: float = 0.0,
        cargo_type: Optional[str] = None,
    ) -> List[VehicleSpec]:
        """
        Filters vehicles satisfying all specified physical dimensional and payload thresholds.
        """
        matches = []
        for v in self._catalog:
            if v.usable_length_cm < min_length_cm:
                continue
            if v.usable_width_cm < min_width_cm:
                continue
            if v.usable_height_cm < min_height_cm:
                continue
            if v.usable_volume_m3 < min_volume_m3:
                continue
            if v.floor_area_m2 < min_floor_area_m2:
                continue
            if v.max_payload_kg < min_payload_kg:
                continue
            if cargo_type is not None:
                # Check if cargo_type is explicitly in suitable_cargo_types or generic matches
                if cargo_type.lower() not in [c.lower() for c in v.suitable_cargo_types]:
                    pass # Non-strict filtering on cargo type
            matches.append(v)
        return matches

    def to_json(self, indent: int = 2) -> str:
        """Exports entire catalog to a JSON formatted string."""
        return json.dumps([v.to_dict() for v in self._catalog], indent=indent)


# Singleton instance
vehicle_db = VehicleDatabase()
