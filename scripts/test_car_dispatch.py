from ultralytics import YOLO
from scripts.dimension_estimator import DimensionEstimator
from scripts.vehicle_database import vehicle_db

model = YOLO("yolov8n.pt")
result = model.predict("data/car.jpg", conf=0.25, verbose=False)[0]

box = max(result.boxes, key=lambda x: float(x.conf[0]))
cls = model.names[int(box.cls[0])]
conf = float(box.conf[0])

d = DimensionEstimator().estimate(category=cls)
dims = d.dimensions_cm

vol = dims["length"] * dims["width"] * dims["height"] / 1_000_000
area = dims["length"] * dims["width"] / 10_000

matches = [
    v for v in vehicle_db.get_all_vehicles()
    if dims["length"] <= v.usable_length_cm
    and dims["width"] <= v.usable_width_cm
    and dims["height"] <= v.usable_height_cm
    and vol <= v.usable_volume_m3
    and area <= v.floor_area_m2
]

# For ONE car, prefer a vehicle explicitly marked for a single car.
single_car = [
    v for v in matches
    if "car_single" in [x.lower() for x in v.suitable_cargo_types]
]

if cls.lower() == "car" and single_car:
    primary = min(single_car, key=lambda v: v.usable_volume_m3)
else:
    exact = [
        v for v in matches
        if cls.lower() in [x.lower() for x in v.suitable_cargo_types]
    ]
    primary = min(exact or matches, key=lambda v: v.usable_volume_m3)

print()
print("=" * 78)
print("CARGO VISION - CAR TRANSPORT RECOMMENDATION")
print("=" * 78)
print(f"Detected Object : {cls.upper()}")
print(f"Confidence      : {conf * 100:.2f}%")
print(f"Bounding Box    : {[round(float(x), 1) for x in box.xyxy[0]]}")

print()
print("PHYSICAL DIMENSIONS (CATEGORY PRIOR)")
print(f"Length          : {dims['length']:.1f} cm")
print(f"Width           : {dims['width']:.1f} cm")
print(f"Height          : {dims['height']:.1f} cm")
print(f"Volume          : {vol:.3f} m3")
print(f"Floor Area      : {area:.2f} m2")

print()
print("TRANSPORT VEHICLE RECOMMENDATION")
print(f"Vehicle ID      : {primary.vehicle_id}")
print(f"Vehicle         : {primary.vehicle_name}")
print(f"Capacity        : {primary.usable_volume_m3:.2f} m3")
print(f"Bed Area        : {primary.floor_area_m2:.2f} m2")
print(f"Max Payload     : {primary.max_payload_kg:.0f} kg")

print()
print(f"Reason          : Accommodates 1 {cls}, requiring {vol:.3f} m3")
print(f"                  and {area:.2f} m2 floor area.")
print("=" * 78)
