import json
import os
from utils.distance import haversine_distance


def load_markets():
    try:
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, '..', 'data', 'markets.json')

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # 🔴 FIX: Handle BOTH formats safely
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get('markets', [])
            else:
                return []

    except Exception as e:
        print(f"[MARKETS LOAD ERROR]: {e}")
        return []


def get_nearby_markets(latitude, longitude, radius=20):
    markets = load_markets()
    nearby = []

    for market in markets:
        try:
            m_lat = float(market.get('latitude'))
            m_lon = float(market.get('longitude'))

            distance = haversine_distance(latitude, longitude, m_lat, m_lon)

            if distance <= radius:
                m_copy = market.copy()
                m_copy['distance_km'] = round(distance, 2)
                nearby.append(m_copy)

        except Exception as e:
            print(f"[MARKET PROCESS ERROR]: {e}")

    nearby.sort(key=lambda x: x['distance_km'])

    return {
        "count": len(nearby),
        "radius_km": radius,
        "markets": nearby
    }
