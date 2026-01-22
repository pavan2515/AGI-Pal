from utils.distance import haversine_distance

def calculate_distance(lat1, lon1, lat2, lon2):
    return haversine_distance(lat1, lon1, lat2, lon2)
