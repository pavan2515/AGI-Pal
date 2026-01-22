import json
import os
from utils.distance import haversine_distance

def load_cold_storage():
    """Load cold storage data from JSON file"""
    try:
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, '..', 'data', 'cold_storage.json')

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # ✅ FIX: correct key name
            return data.get('cold_storage', [])

    except FileNotFoundError:
        print(f"Error: cold_storage.json not found at {data_path}")
        return []

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

def get_nearby_cold_storage(latitude, longitude, radius=20):
    """
    Get cold storage facilities within specified radius
    """
    facilities = load_cold_storage()
    nearby_facilities = []

    for facility in facilities:
        try:
            facility_lat = float(facility['latitude'])
            facility_lon = float(facility['longitude'])

            distance = haversine_distance(latitude, longitude, facility_lat, facility_lon)

            if distance <= radius:
                facility_copy = facility.copy()
                facility_copy['distance_km'] = round(distance, 2)
                nearby_facilities.append(facility_copy)

        except (ValueError, TypeError, KeyError) as e:
            print(f"Error processing facility: {e}")
            continue

    nearby_facilities.sort(key=lambda x: x['distance_km'])

    return {
        "count": len(nearby_facilities),
        "radius_km": radius,
        "user_location": {
            "latitude": latitude,
            "longitude": longitude
        },
        # ✅ frontend expects this key
        "cold_storage": nearby_facilities
    }
