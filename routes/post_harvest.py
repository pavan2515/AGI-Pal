from flask import Blueprint, request, jsonify
from services.agro_shops_service import get_nearby_agro_shops
from services.markets_service import get_nearby_markets
from services.storage_service import get_nearby_cold_storage

post_harvest_bp = Blueprint("post_harvest", __name__, url_prefix='/post-harvest')

@post_harvest_bp.route("/agro-shops", methods=["POST"])
def agro_shops():
    """Get nearby agro shops (government and organic)"""
    data = request.json
    
    if not data or "latitude" not in data or "longitude" not in data:
        return jsonify({"error": "latitude and longitude are required"}), 400
    
    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        radius = float(data.get("radius", 20))
        
        result = get_nearby_agro_shops(latitude, longitude, radius)
        return jsonify(result), 200
        
    except ValueError:
        return jsonify({"error": "Invalid coordinate values"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@post_harvest_bp.route("/markets", methods=["POST"])
def markets():
    """Get nearby markets"""
    data = request.json
    
    if not data or "latitude" not in data or "longitude" not in data:
        return jsonify({"error": "latitude and longitude are required"}), 400
    
    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        radius = float(data.get("radius", 20))
        
        result = get_nearby_markets(latitude, longitude, radius)
        return jsonify(result), 200
        
    except ValueError:
        return jsonify({"error": "Invalid coordinate values"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@post_harvest_bp.route("/storage", methods=["POST"])
def storage():
    """Get nearby cold storage facilities"""
    data = request.json
    
    if not data or "latitude" not in data or "longitude" not in data:
        return jsonify({"error": "latitude and longitude are required"}), 400
    
    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        radius = float(data.get("radius", 20))
        
        result = get_nearby_cold_storage(latitude, longitude, radius)
        return jsonify(result), 200
        
    except ValueError:
        return jsonify({"error": "Invalid coordinate values"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500