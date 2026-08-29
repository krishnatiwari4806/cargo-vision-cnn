def recommend_vehicle(
    self,
    category: str,
    dimensions: Dict[str, Optional[float]],
    cargo_summary: Dict[str, Any],
    quantity: int,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Matches cargo requirements against the structured Vehicle Database.

    Vehicle selection uses:
      1. Physical dimension fit
      2. Volume fit
      3. Floor-area fit
      4. Cargo-type suitability as a ranking signal

    Exact cargo-type matches are preferred over generic vehicles.
    """
    warnings = []
    warnings.append(
        "Payload suitability cannot be verified because physical object weight is unavailable."
    )

    length = dimensions.get("length_cm")
    width = dimensions.get("width_cm")
    height = dimensions.get("height_cm")
    total_vol = cargo_summary.get("total_volume_m3")
    req_floor_area = cargo_summary.get("required_floor_area_m2")

    category_normalized = category.lower().strip()

    all_vehicles = self.vehicle_db.get_all_vehicles()
    suitable_vehicles: List[VehicleSpec] = []

    for v in all_vehicles:
        # ---------------------------------------------------------
        # Constraint 1: Physical dimension fit
        # ---------------------------------------------------------
        if length is not None and length > v.usable_length_cm:
            continue

        if width is not None and width > v.usable_width_cm:
            continue

        if height is not None and height > v.usable_height_cm:
            continue

        # ---------------------------------------------------------
        # Constraint 2: Volume fit
        # ---------------------------------------------------------
        if total_vol is not None and total_vol > v.usable_volume_m3:
            continue

        # ---------------------------------------------------------
        # Constraint 3: Floor area fit
        # ---------------------------------------------------------
        if req_floor_area is not None and req_floor_area > v.floor_area_m2:
            continue

        suitable_vehicles.append(v)

    if not suitable_vehicles:
        largest_v = all_vehicles[-1]

        return {
            "vehicle_id": largest_v.vehicle_id,
            "vehicle_name": largest_v.vehicle_name,
            "reason": (
                f"Cargo requirement ({quantity} {category}s) exceeds "
                f"standard single-vehicle capacity. Multi-trip or fleet "
                f"batching using {largest_v.vehicle_name} is required."
            ),
            "alternatives": [],
        }, warnings

    # -------------------------------------------------------------
    # Cargo suitability ranking
    #
    # Exact category match gets highest priority.
    # Special aliases such as car_single are treated as compatible
    # with a single car.
    # -------------------------------------------------------------
    def suitability_score(vehicle: VehicleSpec) -> int:
        cargo_types = {
            cargo.lower().strip()
            for cargo in vehicle.suitable_cargo_types
        }

        # Exact match: strongest signal
        if category_normalized in cargo_types:
            return 3

        # Single-car compatibility
        if category_normalized == "car" and "car_single" in cargo_types:
            return 2

        # Generic vehicle compatibility
        if category_normalized in {"suv", "vehicle", "vehicles"}:
            if "vehicles" in cargo_types:
                return 2

        return 0

    # -------------------------------------------------------------
    # Rank:
    #   1. Cargo suitability
    #   2. Smaller vehicle tier / capacity
    #
    # Stable ordering is preserved for equal scores.
    # -------------------------------------------------------------
    suitable_vehicles.sort(
        key=lambda v: (
            -suitability_score(v),
            v.usable_volume_m3,
        )
    )

    primary_v = suitable_vehicles[0]

    # Alternatives: next suitable vehicles
    alternatives = [
        v.vehicle_name
        for v in suitable_vehicles[1:3]
    ]

    # -------------------------------------------------------------
    # Construct explanation
    # -------------------------------------------------------------
    reason_parts = [
        f"Accommodates {quantity} {category}(s)"
    ]

    if total_vol is not None:
        reason_parts.append(
            f"requiring {total_vol:.2f} m³ usable space "
            f"(vehicle capacity: {primary_v.usable_volume_m3:.2f} m³)"
        )

    if req_floor_area is not None:
        reason_parts.append(
            f"and {req_floor_area:.2f} m² floor bed area "
            f"(vehicle bed: {primary_v.floor_area_m2:.2f} m²)"
        )

    reason_str = ", ".join(reason_parts) + "."

    recommendation = {
        "vehicle_id": primary_v.vehicle_id,
        "vehicle_name": primary_v.vehicle_name,
        "reason": reason_str,
        "alternatives": alternatives,
    }

    return recommendation, warnings