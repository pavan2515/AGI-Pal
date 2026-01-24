
from flask import Flask, request, render_template, jsonify, url_for, redirect, flash
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import os
import io
import uuid
import json
import logging
import cv2
import shutil  # ✅ ADD THIS - Required for file operations in predict route
import traceback  # ✅ ADD THIS - Required for detailed error logging
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote_plus
from datetime import datetime
import random
from segment2 import segment_analyze_plant
# ✅ POST-HARVEST BLUEPRINT IMPORTS (ADD THESE)
from routes.post_harvest import post_harvest_bp
from routes.schemes import schemes_bp
import signal
import sys
import socket

# Add this function near the top of your file, after imports
def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
# ✅ REGISTER POST-HARVEST BLUEPRINTS (ADD THESE 2 LINES)
app.register_blueprint(post_harvest_bp)
app.register_blueprint(schemes_bp)
# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'agripal-secret-key'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Upload folder configured at: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")

GEMINI_API_KEY = "AIzaSyAL_7MfAGGI8HBpyUhAvyzUl9hPIWJk4bk"
genai.configure(api_key=GEMINI_API_KEY)

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocess image for model input
def preprocess_image(image):
    try:
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

# Load the TensorFlow model
try:
    model_path = 'plant_diseases_model.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully!")
    else:
        logger.error(f"Model file not found at: {os.path.abspath(model_path)}")
        model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Load disease treatments from JSON file
def load_disease_treatments():
    try:
        treatment_path = 'disease_treatments.json'
        if os.path.exists(treatment_path):
            with open(treatment_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded disease treatments from {treatment_path}")
                return data
        else:
            logger.error(f"Disease treatments file not found at: {os.path.abspath(treatment_path)}")
            return {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading disease treatments: {e}")
        return {}

# Class names for plant diseases
class_names = [
    "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
    "Blueberry_healthy", "Cherry_(including_sour)Powdery_mildew", "Cherry(including_sour)_healthy",
    "Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot", "Corn(maize)_Common_rust",
    "Corn_(maize)Northern_Leaf_Blight", "Corn(maize)_healthy", "Grape_Black_rot",
    "Grape_Esca_(Black_Measles)", "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape_healthy",
    "Orange_Haunglongbing_(Citrus_greening)", "Peach_Bacterial_spot", "Peach_healthy",
    "Pepper_bell_Bacterial_spot", "Pepper_bell_healthy", "Potato_Early_blight",
    "Potato_Late_blight", "Potato_healthy", "Raspberry_healthy", "Soybean_healthy",
    "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry_healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two-spotted_spider_mite", "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato_healthy"
]

# Add this constant after your class_names list
CONFIDENCE_THRESHOLD = 50.0  # Minimum confidence for valid prediction
SUPPORTED_PLANTS = {
    'Apple': ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy'],
    'Blueberry': ['Blueberry_healthy'],
    'Cherry': ['Cherry_(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy'],
    'Corn (Maize)': ['Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)_Common_rust', 
                     'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy'],
    'Grape': ['Grape_Black_rot', 'Grape_Esca_(Black_Measles)', 'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape_healthy'],
    'Orange': ['Orange_Haunglongbing_(Citrus_greening)'],
    'Peach': ['Peach_Bacterial_spot', 'Peach_healthy'],
    'Pepper (Bell)': ['Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy'],
    'Potato': ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy'],
    'Raspberry': ['Raspberry_healthy'],
    'Soybean': ['Soybean_healthy'],
    'Squash': ['Squash_Powdery_mildew'],
    'Strawberry': ['Strawberry_Leaf_scorch', 'Strawberry_healthy'],
    'Tomato': ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato_Target_Spot',
               'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']
}

# Common Agricultural Questions Database
COMMON_QUESTIONS = {
    "plant_diseases": [
        "What are the most common tomato diseases?",
        "How do I identify powdery mildew?",
        "What causes yellow leaves on plants?",
        "How to prevent fungal diseases in plants?",
        "What are the signs of bacterial infection in crops?",
        "How to identify viral diseases in plants?",
        "What causes leaf spots on vegetables?",
        "How to detect early blight in tomatoes?"
    ],
    "treatment_methods": [
        "What are organic pest control methods?",
        "How to make homemade fungicide?",
        "What is integrated pest management?",
        "How to use neem oil for plant diseases?",
        "What are the best copper-based fungicides?",
        "How to apply systemic pesticides safely?",
        "What is the difference between preventive and curative treatments?",
        "How to rotate pesticides to prevent resistance?"
    ],
    "crop_management": [
        "When is the best time to plant tomatoes?",
        "How much water do vegetables need daily?",
        "What is crop rotation and why is it important?",
        "How to improve soil fertility naturally?",
        "What are companion plants for tomatoes?",
        "How to prepare soil for planting?",
        "What are the signs of nutrient deficiency?",
        "How to manage weeds organically?"
    ],
    "seasonal_advice": [
        "What crops to plant in monsoon season?",
        "How to protect plants from extreme heat?",
        "What are winter crop management tips?",
        "How to prepare garden for rainy season?",
        "What vegetables grow best in summer?",
        "How to manage greenhouse in different seasons?",
        "What are post-harvest handling best practices?",
        "How to store seeds for next season?"
    ],
    "technology_agriculture": [
        "How can AI help in agriculture?",
        "What are smart farming techniques?",
        "How to use drones in agriculture?",
        "What are precision agriculture tools?",
        "How does satellite imagery help farmers?",
        "What are IoT applications in farming?",
        "How to use weather data for crop planning?",
        "What are digital farming platforms?"
    ]
}

# Load disease treatments
disease_treatments = load_disease_treatments()
logger.info(f"Loaded {len(disease_treatments)} disease treatments")

def normalize_disease_info(disease_info):
    """
    Map old JSON field names to standardized template field names
    This allows backward compatibility with your existing JSON structure
    """
    if not disease_info or 'pesticide' not in disease_info:
        return disease_info
    
    # Create deep copy to avoid modifying original
    import copy
    normalized = copy.deepcopy(disease_info)
    
    logger.info("=" * 80)
    logger.info("📄 NORMALIZING DISEASE INFO FIELDS")
    logger.info("=" * 80)
    
    # Process both chemical and organic treatments
    for treatment_type in ['chemical', 'organic']:
        if treatment_type not in normalized['pesticide']:
            logger.warning(f"⚠️ No {treatment_type} treatment found")
            continue
            
        treatment = normalized['pesticide'][treatment_type]
        logger.info(f"📦 Processing {treatment_type.upper()} treatment...")
        
        # ===== FIELD MAPPING =====
        
        # 1. Map: application_frequency -> frequency
        if 'application_frequency' in treatment and 'frequency' not in treatment:
            treatment['frequency'] = treatment['application_frequency']
            logger.info(f"  ✅ Mapped application_frequency -> frequency")
            logger.info(f"     Value: {treatment['frequency'][:50]}...")
        elif 'frequency' not in treatment or not treatment.get('frequency'):
            treatment['frequency'] = "Apply according to product label recommendations and disease pressure."
            logger.warning(f"  ⚠️ No frequency field found, added fallback")
        
        # 2. Map: precautions -> safety
        if 'precautions' in treatment and 'safety' not in treatment:
            treatment['safety'] = treatment['precautions']
            logger.info(f"  ✅ Mapped precautions -> safety")
            logger.info(f"     Value: {treatment['safety'][:50]}...")
        elif 'safety' not in treatment or not treatment.get('safety'):
            if treatment_type == 'chemical':
                treatment['safety'] = "Wear protective equipment. Follow all label precautions. Keep away from water sources."
            else:
                treatment['safety'] = "Safe for beneficial insects when used as directed. Apply during cooler parts of day."
            logger.warning(f"  ⚠️ No safety field found, added fallback")
        
        # 3. Ensure usage exists and has content
        if 'usage' not in treatment or not treatment.get('usage') or len(treatment.get('usage', '').strip()) < 10:
            treatment['usage'] = f"Apply as directed on product label. Ensure thorough coverage of all affected plant surfaces. Repeat applications as needed based on disease pressure."
            logger.warning(f"  ⚠️ Missing or short usage, added fallback")
        
        # 4. Ensure all required fields exist
        required_fields = {
            'name': f"{treatment_type.title()} Treatment",
            'dosage_per_hectare': 0.0,
            'unit': 'L',
            'usage': 'Apply as directed',
            'frequency': 'As needed',
            'safety': 'Follow product label instructions'
        }
        
        for field, default_value in required_fields.items():
            if field not in treatment or not treatment.get(field):
                treatment[field] = default_value
                logger.warning(f"  ⚠️ Missing {field}, added default: {default_value}")
        
        # Validate field lengths
        logger.info(f"  📊 Field lengths:")
        logger.info(f"     - Name: {len(treatment.get('name', ''))} chars")
        logger.info(f"     - Usage: {len(treatment.get('usage', ''))} chars")
        logger.info(f"     - Frequency: {len(treatment.get('frequency', ''))} chars")
        logger.info(f"     - Safety: {len(treatment.get('safety', ''))} chars")
    
    logger.info("=" * 80)
    logger.info("✅ NORMALIZATION COMPLETE")
    logger.info("=" * 80)
    
    return normalized
# Enhanced function to get disease information with better video source handling
def get_disease_info(disease_name):
    """
    Enhanced function with field normalization and detailed logging
    """
    try:
        logger.info("=" * 80)
        logger.info(f"🔍 DISEASE LOOKUP: {disease_name}")
        logger.info("=" * 80)
        logger.info(f"📚 Database has {len(disease_treatments)} diseases")
        
        # Try exact match first
        disease_info = disease_treatments.get(disease_name, None)
        
        # If no exact match, try variations
        if not disease_info:
            logger.info(f"⚠️ No exact match, trying variations...")
            cleaned_name = disease_name.replace('_', ' ').replace('(', '').replace(')', '').strip()
            
            for key, value in disease_treatments.items():
                if cleaned_name.lower() in key.lower() or key.lower() in cleaned_name.lower():
                    disease_info = value
                    logger.info(f"✅ Found match with key: {key}")
                    break
        
        if not disease_info:
            logger.error(f"❌ NO DISEASE INFO FOUND for: {disease_name}")
            available = list(disease_treatments.keys())[:5]
            logger.info(f"📝 Available diseases (first 5): {available}")
            return None
        
        logger.info(f"✅ Raw disease info found")
        logger.info(f"📋 Raw keys: {list(disease_info.keys())}")
        
        # ===== NORMALIZE FIELD NAMES (backward compatibility) =====
        disease_info = normalize_disease_info(disease_info)
        
        # ===== FINAL VALIDATION =====
        logger.info("=" * 80)
        logger.info("📊 FINAL VALIDATION")
        logger.info("=" * 80)
        logger.info(f"✅ Disease Name: {disease_info.get('name')}")
        logger.info(f"✅ Description: {len(disease_info.get('description', ''))} chars")
        logger.info(f"✅ Treatment Steps: {len(disease_info.get('treatment', []))}")
        logger.info(f"✅ Severity: {disease_info.get('severity')}")
        
        if 'pesticide' in disease_info:
            for treatment_type in ['chemical', 'organic']:
                if treatment_type in disease_info['pesticide']:
                    t = disease_info['pesticide'][treatment_type]
                    logger.info(f"")
                    logger.info(f"📦 {treatment_type.upper()}:")
                    logger.info(f"  Name: {t.get('name')}")
                    logger.info(f"  Usage: {len(t.get('usage', ''))} chars - {bool(t.get('usage'))}")
                    logger.info(f"  Frequency: {len(t.get('frequency', ''))} chars - {bool(t.get('frequency'))}")
                    logger.info(f"  Safety: {len(t.get('safety', ''))} chars - {bool(t.get('safety'))}")
                    logger.info(f"  Dosage: {t.get('dosage_per_hectare')} {t.get('unit')}/hectare")
        
        # ===== PROCESS VIDEO SOURCES =====
        if 'pesticide' in disease_info:
            for treatment_type in ['chemical', 'organic']:
                if treatment_type not in disease_info['pesticide']:
                    continue
                
                treatment = disease_info['pesticide'][treatment_type]
                
                if 'video_sources' in treatment:
                    video_sources = treatment['video_sources']
                    
                    # Add YouTube search URLs
                    if 'search_terms' in video_sources:
                        search_urls = []
                        for term in video_sources['search_terms']:
                            search_urls.append({
                                'term': term,
                                'url': f"https://www.youtube.com/results?search_query={quote_plus(term)}"
                            })
                        video_sources['search_urls'] = search_urls
                        logger.info(f"✅ Added {len(search_urls)} YouTube URLs for {treatment_type}")
                    
                    # Process reliable channels
                    if 'reliable_channels' in video_sources:
                        channel_urls = []
                        for channel in video_sources['reliable_channels']:
                            channel_urls.append({
                                'name': channel,
                                'url': f"https://www.youtube.com/results?search_query={quote_plus(channel + ' ' + disease_name.replace('_', ' '))}"
                            })
                        video_sources['channel_urls'] = channel_urls
                        logger.info(f"✅ Added {len(channel_urls)} channel URLs for {treatment_type}")
        
        logger.info("=" * 80)
        logger.info("✅ DISEASE INFO PROCESSING COMPLETE")
        logger.info("=" * 80)
        
        return disease_info
        
    except Exception as e:
        logger.error(f"❌ ERROR in get_disease_info: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    
def combine_disease_treatments(unique_diseases):
    """
    Combine treatments from multiple diseases of the same plant
    Returns a merged treatment plan with intelligent deduplication
    """
    logger.info("=" * 80)
    logger.info("🔀 COMBINING TREATMENTS FROM MULTIPLE DISEASES")
    logger.info("=" * 80)
    
    combined = {
        'diseases': [],
        'description': '',
        'treatment': [],
        'severity': 'Unknown',
        'pesticide': {
            'chemical': {
                'name': 'Combined Chemical Treatment',
                'usage': [],
                'frequency': [],
                'safety': [],
                'dosage_per_hectare': 0,
                'unit': 'L',
                'video_sources': {
                    'search_terms': [],
                    'reliable_channels': []
                }
            },
            'organic': {
                'name': 'Combined Organic Treatment',
                'usage': [],
                'frequency': [],
                'safety': [],
                'dosage_per_hectare': 0,
                'unit': 'L',
                'video_sources': {
                    'search_terms': [],
                    'reliable_channels': []
                }
            }
        },
        'additional_resources': {
            'step_by_step_guide': [],
            'extension_guides': []
        }
    }
    
    severity_levels = {'Low': 1, 'Moderate': 2, 'Medium': 2, 'High': 3, 'Severe': 4}
    max_severity_score = 0
    
    # Track unique items to avoid duplicates
    unique_chemical_names = set()
    unique_organic_names = set()
    unique_treatments = set()
    unique_guides = set()
    
    logger.info(f"📊 Processing {len(unique_diseases)} diseases...")
    
    for disease, data in unique_diseases.items():
        disease_info = data['disease_info']
        if not disease_info:
            logger.warning(f"⚠️ No disease info for {disease}")
            continue
        
        logger.info(f"   Processing: {disease}")
        
        # Track diseases
        combined['diseases'].append({
            'name': disease,
            'display_name': disease.replace('_', ' '),
            'count': data['count'],
            'avg_confidence': data['total_confidence'] / data['count']
        })
        
        # Combine descriptions
        if disease_info.get('description'):
            combined['description'] += f"**{disease.replace('_', ' ')}**: {disease_info['description']}\n\n"
        
        # Combine treatment steps (with section headers)
        if disease_info.get('treatment'):
            header = f"=== Treatment for {disease.replace('_', ' ')} ==="
            if header not in unique_treatments:
                combined['treatment'].append(header)
                unique_treatments.add(header)
                
                for step in disease_info['treatment']:
                    if step and step not in unique_treatments:
                        combined['treatment'].append(step)
                        unique_treatments.add(step)
                
                combined['treatment'].append("")  # Spacer
        
        # Track highest severity
        disease_severity = disease_info.get('severity', 'Unknown')
        severity_score = severity_levels.get(disease_severity, 0)
        if severity_score > max_severity_score:
            max_severity_score = severity_score
            combined['severity'] = disease_severity
            logger.info(f"   Updated max severity: {disease_severity}")
        
        # Combine pesticide info
        if 'pesticide' in disease_info:
            for treatment_type in ['chemical', 'organic']:
                if treatment_type not in disease_info['pesticide']:
                    continue
                
                treatment = disease_info['pesticide'][treatment_type]
                unique_set = unique_chemical_names if treatment_type == 'chemical' else unique_organic_names
                
                # Collect unique treatment names and usage
                treatment_name = treatment.get('name', '')
                if treatment_name and treatment_name not in unique_set:
                    unique_set.add(treatment_name)
                    usage_text = f"**{treatment_name}** ({disease.replace('_', ' ')}): {treatment.get('usage', 'Apply as directed')}"
                    combined['pesticide'][treatment_type]['usage'].append(usage_text)
                    
                    logger.info(f"      Added {treatment_type}: {treatment_name}")
                
                # Collect frequencies
                if treatment.get('frequency'):
                    freq = treatment['frequency'].strip()
                    if freq not in combined['pesticide'][treatment_type]['frequency']:
                        combined['pesticide'][treatment_type]['frequency'].append(freq)
                
                # Collect safety info
                if treatment.get('safety'):
                    safety = treatment['safety'].strip()
                    if safety not in combined['pesticide'][treatment_type]['safety']:
                        combined['pesticide'][treatment_type]['safety'].append(safety)
                
                # Sum dosages (will be averaged later)
                dosage = treatment.get('dosage_per_hectare', 0)
                combined['pesticide'][treatment_type]['dosage_per_hectare'] += dosage
                
                # Combine video sources
                if treatment.get('video_sources'):
                    video_sources = treatment['video_sources']
                    
                    if 'search_terms' in video_sources:
                        for term in video_sources['search_terms']:
                            if term not in combined['pesticide'][treatment_type]['video_sources']['search_terms']:
                                combined['pesticide'][treatment_type]['video_sources']['search_terms'].append(term)
                    
                    if 'reliable_channels' in video_sources:
                        for channel in video_sources['reliable_channels']:
                            if channel not in combined['pesticide'][treatment_type]['video_sources']['reliable_channels']:
                                combined['pesticide'][treatment_type]['video_sources']['reliable_channels'].append(channel)
        
        # Combine additional resources
        if 'additional_resources' in disease_info:
            resources = disease_info['additional_resources']
            
            if 'step_by_step_guide' in resources:
                for step in resources['step_by_step_guide']:
                    if step not in combined['additional_resources']['step_by_step_guide']:
                        combined['additional_resources']['step_by_step_guide'].append(step)
            
            if 'extension_guides' in resources:
                for guide in resources['extension_guides']:
                    if guide not in unique_guides:
                        combined['additional_resources']['extension_guides'].append(guide)
                        unique_guides.add(guide)
    
    # Format combined fields
    logger.info("📝 Formatting combined treatment data...")
    
    for treatment_type in ['chemical', 'organic']:
        # Format usage
        if combined['pesticide'][treatment_type]['usage']:
            combined['pesticide'][treatment_type]['usage'] = "\n\n".join(
                combined['pesticide'][treatment_type]['usage']
            )
        else:
            combined['pesticide'][treatment_type]['usage'] = "Apply treatments according to product labels for each specific disease."
        
        # Format frequency
        if combined['pesticide'][treatment_type]['frequency']:
            unique_freq = list(set(combined['pesticide'][treatment_type]['frequency']))
            if len(unique_freq) == 1:
                combined['pesticide'][treatment_type]['frequency'] = unique_freq[0]
            else:
                combined['pesticide'][treatment_type]['frequency'] = " OR ".join(unique_freq)
        else:
            combined['pesticide'][treatment_type]['frequency'] = "Follow individual disease treatment schedules"
        
        # Format safety
        if combined['pesticide'][treatment_type]['safety']:
            combined['pesticide'][treatment_type]['safety'] = " • ".join(
                list(set(combined['pesticide'][treatment_type]['safety']))
            )
        else:
            combined['pesticide'][treatment_type]['safety'] = "Follow all safety guidelines on product labels. Wear protective equipment."
        
        # Average dosages
        num_diseases = len(unique_diseases)
        if num_diseases > 0 and combined['pesticide'][treatment_type]['dosage_per_hectare'] > 0:
            combined['pesticide'][treatment_type]['dosage_per_hectare'] /= num_diseases
            logger.info(f"   {treatment_type.title()} avg dosage: {combined['pesticide'][treatment_type]['dosage_per_hectare']:.2f}")
        
        # Process video sources for URLs
        video_sources = combined['pesticide'][treatment_type]['video_sources']
        if video_sources['search_terms']:
            search_urls = []
            for term in video_sources['search_terms']:
                search_urls.append({
                    'term': term,
                    'url': f"https://www.youtube.com/results?search_query={quote_plus(term)}"
                })
            video_sources['search_urls'] = search_urls
        
        if video_sources['reliable_channels']:
            channel_urls = []
            for channel in video_sources['reliable_channels']:
                channel_urls.append({
                    'name': channel,
                    'url': f"https://www.youtube.com/results?search_query={quote_plus(channel + ' multiple plant diseases')}"
                })
            video_sources['channel_urls'] = channel_urls
    
    logger.info("=" * 80)
    logger.info("✅ COMBINED TREATMENT PLAN READY")
    logger.info(f"   Diseases: {len(combined['diseases'])}")
    logger.info(f"   Treatment steps: {len(combined['treatment'])}")
    logger.info(f"   Overall severity: {combined['severity']}")
    logger.info("=" * 80)
    
    return combined

# Enhanced dosage calculation function with better error handling
def calculate_dosage(area, area_unit, pesticide_info):
    """Calculate pesticide dosage based on area and unit with enhanced error handling"""
    logger.info(f"Calculating dosage for area: {area} {area_unit}")
    
    try:
        chemical_dosage = None
        organic_dosage = None
        hectare_conversion = 0
        
        # Safely get pesticide information
        chemical_info = pesticide_info.get("chemical", {}) if pesticide_info else {}
        organic_info = pesticide_info.get("organic", {}) if pesticide_info else {}
        
        # Get dosage per hectare with safe defaults
        chemical_dosage_per_hectare = float(chemical_info.get("dosage_per_hectare", 0))
        organic_dosage_per_hectare = float(organic_info.get("dosage_per_hectare", 0))
        
        # Convert area to hectares with validation
        try:
            area_float = float(area) if area else 0
            if area_float <= 0:
                logger.warning(f"Invalid area value: {area}")
                return None, None, 0
                
            # Conversion factors to hectares
            conversion_factors = {
                'hectare': 1.0,
                'acre': 0.404686,
                'square_meter': 0.0001,
                'square_feet': 0.0000092903
            }
            
            hectare_conversion = area_float * conversion_factors.get(area_unit, 1.0)
            logger.info(f"Converted {area_float} {area_unit} to {hectare_conversion} hectares")
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting area to float: {e}")
            return None, None, 0
        
        # Calculate required dosage
        if chemical_dosage_per_hectare > 0 and hectare_conversion > 0:
            chemical_dosage = chemical_dosage_per_hectare * hectare_conversion
            logger.info(f"Calculated chemical dosage: {chemical_dosage}")
        
        if organic_dosage_per_hectare > 0 and hectare_conversion > 0:
            organic_dosage = organic_dosage_per_hectare * hectare_conversion
            logger.info(f"Calculated organic dosage: {organic_dosage}")
        
        return chemical_dosage, organic_dosage, hectare_conversion
        
    except Exception as e:
        logger.error(f"Error in dosage calculation: {e}")
        return None, None, 0



# Enhanced image validation function
def is_plant_image(image_path):
    """
    Enhanced function to check if the uploaded image is likely a plant image
    using multiple validation techniques
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Could not read image file")
            return False
            
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. GREEN COLOR ANALYSIS (More Strict)
        # Define multiple green ranges to catch different types of plant greens
        green_ranges = [
            # Bright green (healthy leaves)
            ([35, 50, 50], [85, 255, 255]),
            # Dark green (mature leaves)
            ([25, 30, 30], [75, 255, 200]),
            # Yellow-green (some diseased leaves)
            ([15, 40, 40], [35, 255, 255])
        ]
        
        total_green_pixels = 0
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_green_pixels += cv2.countNonZero(mask)
        
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = total_green_pixels / total_pixels
        
        # 2. TEXTURE ANALYSIS - Plants have organic textures
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate Local Binary Pattern variance (texture measure)
        texture_variance = np.var(gray)
        
        # 3. EDGE ANALYSIS - Natural vs artificial edges
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        edge_ratio = edge_pixels / total_pixels
        
        # 4. COLOR DISTRIBUTION ANALYSIS
        # Plants typically have more natural color distribution
        color_std = np.std(rgb, axis=(0, 1))
        color_mean = np.mean(color_std)
        
        # 5. BRIGHTNESS AND CONTRAST CHECKS
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # 6. SHAPE ANALYSIS - Look for leaf-like shapes
        # Use contour detection to find organic shapes
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        organic_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Organic shapes are neither too circular nor too geometric
                    if 0.1 < circularity < 0.8:
                        organic_shapes += 1
        
        # 7. GEOMETRIC PATTERN DETECTION (to reject posters/documents)
        # Look for straight lines (common in posters/documents)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        straight_lines = len(lines) if lines is not None else 0
        
        # 8. TEXT DETECTION (basic) - Posters often have text
        # Simple text detection based on horizontal edge patterns
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        text_like_pixels = cv2.countNonZero(horizontal_lines)
        text_ratio = text_like_pixels / total_pixels
        
        # DECISION LOGIC (STRICT CRITERIA)
        height, width = img.shape[:2]
        
        # Basic size check
        is_reasonable_size = height > 100 and width > 100
        
        # Green content check (STRICTER)
        has_significant_green = green_ratio > 0.12  # At least 12% green
        
        # Texture check
        has_organic_texture = texture_variance > 500  # Organic textures have variation
        
        # Edge analysis
        has_natural_edges = 0.02 < edge_ratio < 0.25  # Not too sharp, not too smooth
        
        # Color variety check
        has_natural_colors = color_mean > 15  # Natural variation in colors
        
        # Brightness check
        reasonable_brightness = 30 < brightness < 220  # Not too dark or overexposed
        
        # Contrast check
        good_contrast = contrast > 20  # Some contrast indicating detail
        
        # Organic shapes check
        has_organic_shapes = organic_shapes > 0
        
        # Reject if too many straight lines (indicates documents/posters)
        not_too_geometric = straight_lines < 10
        
        # Reject if too much text-like content
        not_text_heavy = text_ratio < 0.05  # Less than 5% text-like content
        
        # FINAL SCORING SYSTEM
        score = 0
        criteria_met = []
        
        if has_significant_green:
            score += 3
            criteria_met.append("green_content")
        
        if has_organic_texture:
            score += 2
            criteria_met.append("organic_texture")
            
        if has_natural_edges:
            score += 2
            criteria_met.append("natural_edges")
            
        if has_natural_colors:
            score += 1
            criteria_met.append("natural_colors")
            
        if reasonable_brightness:
            score += 1
            criteria_met.append("good_brightness")
            
        if good_contrast:
            score += 1
            criteria_met.append("good_contrast")
            
        if has_organic_shapes:
            score += 2
            criteria_met.append("organic_shapes")
            
        if not_too_geometric:
            score += 1
            criteria_met.append("not_geometric")
            
        if not_text_heavy:
            score += 1
            criteria_met.append("not_text_heavy")
        
        # Log detailed analysis
        logger.info(f"Plant image analysis for {image_path}:")
        logger.info(f"  - Green ratio: {green_ratio:.3f} (threshold: 0.12)")
        logger.info(f"  - Texture variance: {texture_variance:.1f} (threshold: 500)")
        logger.info(f"  - Edge ratio: {edge_ratio:.3f} (range: 0.02-0.25)")
        logger.info(f"  - Color variation: {color_mean:.1f} (threshold: 15)")
        logger.info(f"  - Brightness: {brightness:.1f} (range: 30-220)")
        logger.info(f"  - Contrast: {contrast:.1f} (threshold: 20)")
        logger.info(f"  - Organic shapes: {organic_shapes}")
        logger.info(f"  - Straight lines: {straight_lines} (threshold: <10)")
        logger.info(f"  - Text ratio: {text_ratio:.3f} (threshold: <0.05)")
        logger.info(f"  - Total score: {score}/14")
        logger.info(f"  - Criteria met: {criteria_met}")
        
        # Require minimum score of 7/14 and must have green content
        is_plant = (score >= 7 and has_significant_green and is_reasonable_size)
        
        logger.info(f"  - Final decision: {'PLANT' if is_plant else 'NOT PLANT'}")
        
        return is_plant
        
    except Exception as e:
        logger.error(f"Error in enhanced plant image validation: {e}")
        return False  # Fail safe - reject if analysis fails

# 2. ADD this new function after the is_plant_image function:

def validate_plant_type(predicted_class, confidence):
    """
    Additional validation to ensure predicted class makes sense
    """
    try:
        # Check if predicted class is in our supported classes
        if predicted_class not in class_names:
            logger.warning(f"Predicted class {predicted_class} not in supported classes")
            return False, "Predicted class not recognized"
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"Confidence {confidence:.2f}% below threshold {CONFIDENCE_THRESHOLD}%")
            return False, f"Low confidence prediction ({confidence:.1f}%)"
        
        # Additional checks can be added here
        # For example, checking if the prediction pattern makes sense
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error in plant type validation: {e}")
        return False, str(e)
# Enhanced preprocess function with validation
def preprocess_image_with_validation(image, image_path):
    """
    Enhanced preprocessing with strict validation
    """
    try:
        logger.info(f"Starting image validation for: {image_path}")
        
        # First check if it's likely a plant image with enhanced validation
        if not is_plant_image(image_path):
            logger.warning("Image failed plant validation - not a plant image")
            return None, False
            
        logger.info("Image passed plant validation")
        
        # Additional file format validation
        try:
            # Verify image can be opened and is valid
            image_test = Image.open(image_path)
            image_test.verify()  # This will raise an exception if image is corrupted
        except Exception as e:
            logger.error(f"Image file validation failed: {e}")
            return None, False
            
        # Original preprocessing
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info("Image preprocessing completed successfully")
        return img_array, True
        
    except Exception as e:
        logger.error(f"Error in enhanced image preprocessing: {e}")
        return None, False

# Enhanced prediction function
def make_enhanced_prediction(processed_image):
    """
    Enhanced prediction with multiple validation layers
    """
    try:
        logger.info("Starting enhanced prediction")
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index]) * 100
        
        logger.info(f"Raw prediction - Index: {predicted_class_index}, Confidence: {confidence:.2f}%")
        
        # Get predicted class name
        if predicted_class_index >= len(class_names):
            logger.error(f"Prediction index {predicted_class_index} out of range (max: {len(class_names)-1})")
            return None, confidence, "Prediction index out of range"
        
        predicted_class = class_names[predicted_class_index]
        logger.info(f"Predicted class: {predicted_class}")
        
        # Validate the prediction
        is_valid, validation_error = validate_plant_type(predicted_class, confidence)
        if not is_valid:
            logger.warning(f"Prediction validation failed: {validation_error}")
            return None, confidence, validation_error
        
        # Additional confidence analysis - check top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_confidences = [predictions[0][i] * 100 for i in top_3_indices]
        
        logger.info("Top 3 predictions:")
        for i, (idx, conf) in enumerate(zip(top_3_indices, top_3_confidences)):
            logger.info(f"  {i+1}. {class_names[idx]}: {conf:.2f}%")
        
        # Check if there's a clear winner (gap between 1st and 2nd should be reasonable)
        if len(top_3_confidences) > 1:
            confidence_gap = top_3_confidences[0] - top_3_confidences[1]
            if confidence_gap < 10 and confidence < 70:  # If predictions are too close and confidence is low
                logger.warning(f"Ambiguous prediction - confidence gap: {confidence_gap:.2f}%")
                return None, confidence, f"Ambiguous prediction (confidence gap: {confidence_gap:.1f}%)"
        
        logger.info(f"Prediction validated successfully: {predicted_class} ({confidence:.2f}%)")
        return predicted_class, confidence, None
            
    except Exception as e:
        logger.error(f"Error in enhanced prediction: {e}")
        return None, 0.0, str(e)
def generate_gradcam(img_array, model, predicted_class_index, layer_name=None):
    """
    Generate GradCAM heatmap for disease visualization - FIXED VERSION
    
    Args:
        img_array: Preprocessed image (1, 128, 128, 3)
        model: Trained Keras model
        predicted_class_index: Index of predicted class
        layer_name: Name of last conv layer (auto-detected if None)
    
    Returns:
        heatmap: GradCAM heatmap (128, 128)
        superimposed_img: Original image with heatmap overlay
    """
    try:
        logger.info("🎯 Starting GradCAM generation...")
        
        # AUTO-DETECT last convolutional layer
        if layer_name is None:
            # Strategy 1: Look for 'conv' in name with 4D output
            conv_layers = [layer.name for layer in model.layers 
                          if 'conv' in layer.name.lower() and 
                          len(layer.output_shape) == 4]
            
            # Strategy 2: If no 'conv', find any 4D output layer
            if not conv_layers:
                conv_layers = [layer.name for layer in model.layers 
                              if len(layer.output_shape) == 4]
            
            if not conv_layers:
                logger.error("❌ No convolutional layers found in model")
                return None, None
            
            layer_name = conv_layers[-1]
            logger.info(f"✅ Auto-detected conv layer: {layer_name}")
        
        # Verify layer exists
        try:
            target_layer = model.get_layer(layer_name)
            logger.info(f"✅ Using layer: {layer_name} (output shape: {target_layer.output_shape})")
        except:
            logger.error(f"❌ Layer {layer_name} not found in model")
            return None, None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.input],
            outputs=[target_layer.output, model.output]
        )
        
        # Cast to float32 for gradient computation
        img_array = tf.cast(img_array, tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, predicted_class_index]
        
        # Get gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            logger.error("❌ Gradients are None - GradCAM failed")
            return None, None
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight channels by gradient importance
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        
        # Multiply each channel by importance and sum
        heatmap = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
        for i in range(pooled_grads.shape[0]):
            heatmap += conv_outputs[:, :, i].numpy() * pooled_grads[i]
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        
        # Resize to original image size (128x128)
        heatmap_resized = cv2.resize(heatmap, (128, 128))
        
        # Apply colormap - JET (Red = High attention, Blue = Low)
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        
        # Get original image (denormalize from 0-1 to 0-255)
        original_img = (img_array[0].numpy() * 255).astype(np.uint8)
        original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(
            original_img_bgr, 0.6,  # Original image weight
            heatmap_colored, 0.4,   # Heatmap weight
            0
        )
        
        logger.info("✅ GradCAM generated successfully!")
        logger.info(f"   - Heatmap shape: {heatmap_resized.shape}")
        logger.info(f"   - Heatmap range: [{heatmap_resized.min():.3f}, {heatmap_resized.max():.3f}]")
        
        return heatmap_resized, superimposed_img
        
    except Exception as e:
        logger.error(f"❌ Error generating GradCAM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def save_gradcam_image(superimposed_img, original_filename):
    """
    Save GradCAM visualization to uploads folder with ENHANCED ERROR HANDLING
    
    Returns:
        gradcam_filename: Filename of saved GradCAM image
    """
    try:
        # Generate filename
        base_name = os.path.splitext(original_filename)[0]
        gradcam_filename = f"{base_name}_gradcam.jpg"
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        
        # Verify folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save image with high quality
        cv2.imwrite(gradcam_path, superimposed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Verify file was saved
        if os.path.exists(gradcam_path):
            file_size = os.path.getsize(gradcam_path)
            logger.info(f"✅ GradCAM saved: {gradcam_path} ({file_size} bytes)")
            return gradcam_filename
        else:
            logger.error(f"❌ GradCAM file not found after save: {gradcam_path}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Error saving GradCAM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_detailed_error_message(error_type, image_analysis=None):
    """
    Generate detailed error messages for different validation failures
    """
    if error_type == "not_plant":
        return {
            "title": "Not a Plant Image",
            "message": "The uploaded image doesn't appear to be a plant photograph.",
            "suggestions": [
                "Upload a clear photo of plant leaves",
                "Ensure the image shows actual plant matter (not drawings or posters)",
                "Make sure leaves are visible with any disease symptoms",
                "Use good lighting and focus on the affected plant parts"
            ],
            "technical_details": image_analysis
        }
    elif error_type == "low_confidence":
        return {
            "title": "Unable to Identify Plant Disease",
            "message": "The image quality or plant type may not be suitable for accurate analysis.",
            "suggestions": [
                "Try uploading a clearer, higher quality image",
                "Ensure the plant is one of our supported types",
                "Focus on leaves showing clear disease symptoms",
                "Check if lighting is adequate"
            ]
        }
    elif error_type == "unsupported_plant":
        return {
            "title": "Unsupported Plant Type",
            "message": "This plant type may not be in our current database.",
            "suggestions": [
                "Check our supported plants list",
                "Try with Apple, Tomato, Potato, Corn, Grape, Peach, Pepper, or Strawberry plants",
                "Ensure the image clearly shows the plant type"
            ]
        }
    else:
        return {
            "title": "Analysis Error",
            "message": "An error occurred during image analysis.",
            "suggestions": [
                "Try uploading the image again",
                "Ensure the image file is not corrupted",
                "Use a different image format (JPG, PNG)"
            ]
        }

def initialize_enhanced_gemini():
    """Enhanced Gemini AI initialization with better error handling"""
    try:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your-api-key-here":
            logger.error("Gemini API key not configured properly")
            return False, "API key not configured"
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        test_model = genai.GenerativeModel('models/gemini-1.5-flash-001')
        test_prompt = "What is the most important factor in plant health?"
        
        test_response = test_model.generate_content(test_prompt)
        
        if test_response and test_response.text:
            logger.info("✅ Gemini AI connected successfully!")
            logger.info(f"Test response: {test_response.text[:100]}...")
            return True, "Connected successfully"
        else:
            logger.error("❌ Gemini AI test failed - no response received")
            return False, "No response from API"
            
    except Exception as e:
        logger.error(f"❌ Gemini AI initialization failed: {str(e)}")
        return False, str(e)

def get_enhanced_chatbot_response(message, detected_disease=None, conversation_history=None):
    """Enhanced chatbot with improved AI integration and common questions"""
    
    original_message = message
    message = message.lower().strip()
    
    logger.info(f"Enhanced chatbot processing: {original_message}")
    
    # Handle system commands first
    if message in ["help", "/help", "commands", "/commands"]:
        return generate_help_response()
    
    elif message in ["questions", "/questions", "common questions", "examples"]:
        return generate_common_questions_response()
    
    elif message.startswith("/category "):
        category = message.replace("/category ", "").strip()
        return generate_category_questions(category)
    
    # Handle date and time requests
    elif any(keyword in message for keyword in ["date", "time", "today", "current date", "current time"]):
        current_datetime = datetime.now()
        if "time" in message:
            return f"🕐 Current time: {current_datetime.strftime('%H:%M:%S')} IST"
        elif "date" in message:
            return f"📅 Today's date: {current_datetime.strftime('%B %d, %Y (%A)')}"
        else:
            return f"📅🕐 Current date and time: {current_datetime.strftime('%B %d, %Y %H:%M:%S (%A)')}"
    
    # Handle greeting responses
    elif any(greeting in message for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "namaste", "start"]):
        greeting_response = """🌱 **Namaste! Welcome to AgriPal AI!** 

I'm your intelligent agricultural assistant powered by advanced AI. I can help you with:

🔍 **Disease Detection** - Upload images for instant plant disease identification
💊 **Treatment Plans** - Get specific, science-based treatment recommendations  
🧮 **Dosage Calculator** - Calculate exact pesticide amounts for your farm size
🌿 **Organic Solutions** - Eco-friendly pest and disease management
📊 **Crop Management** - Seasonal advice and farming best practices
🤖 **AI-Powered Q&A** - Ask any agricultural question, get expert answers

**Quick Start Commands:**
- Type `questions` to see common agricultural questions
- Type `help` to see available commands
- Ask specific questions like "How to treat tomato blight?"

What would you like to explore today? 🚀"""
        return greeting_response
    
    # Handle goodbye messages
    elif any(farewell in message for farewell in ["bye", "goodbye", "see you", "thanks", "thank you", "dhanyawad"]):
        return """🙏 **Thank you for using AgriPal AI!** 

**Remember these key farming tips:**
- Monitor your crops regularly for early disease detection
- Maintain good field hygiene and crop rotation
- Keep learning about sustainable farming practices

Happy farming! 🌾🚜✨"""
    
    # For all other questions, use Gemini AI
    else:
        try:
            ai_prompt = create_agricultural_prompt(original_message, detected_disease, conversation_history)
            
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 500,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-001",
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            response = model.generate_content(ai_prompt)
            
            if response and response.text:
                ai_response = response.text.strip()
                formatted_response = f"🤖 **AgriPal AI Expert Response:**\n\n{ai_response}\n\n---\n💡 *Need plant disease identification? Upload an image using our detection tool!*"
                return formatted_response
            else:
                return get_fallback_response(original_message, detected_disease)
                
        except Exception as e:
            logger.error(f"❌ Gemini AI error: {str(e)}")
            return get_fallback_response(original_message, detected_disease, str(e))


def get_common_questions_by_category(category=None, limit=5):
    """Get common questions by category or random selection"""
    if category and category in COMMON_QUESTIONS:
        questions = COMMON_QUESTIONS[category]
        return random.sample(questions, min(limit, len(questions)))
    else:
        all_questions = []
        for cat_questions in COMMON_QUESTIONS.values():
            all_questions.extend(cat_questions)
        return random.sample(all_questions, min(limit, len(all_questions)))

def create_agricultural_prompt(user_message, detected_disease=None, conversation_history=None):
    """Create a comprehensive prompt for agricultural AI assistant"""
    
    base_context = """You are AgriPal AI, an expert agricultural assistant specializing in:
- Plant disease identification and treatment
- Crop management and farming techniques
- Pest control and integrated pest management
- Soil health and nutrition management
- Seasonal farming advice
- Sustainable agriculture practices

Your responses should be:
- Scientifically accurate and evidence-based
- Practical and actionable for farmers
- Safe and environmentally conscious
- Appropriate for different farming scales
- Under 400 words but comprehensive"""

    disease_context = ""
    if detected_disease:
        disease_context = f"\n\nCurrent Context: The user recently detected '{detected_disease}'. This should inform your responses."

    history_context = ""
    if conversation_history:
        recent_messages = conversation_history[-3:]
        history_summary = " ".join([msg.get('text', '')[:100] for msg in recent_messages])
        history_context = f"\n\nRecent conversation: {history_summary}"

    user_context = f"\n\nUser's question: {user_message}"
    
    return base_context + disease_context + history_context + user_context

def generate_help_response():
    """Generate help response with available commands"""
    return """🆘 **AgriPal AI Help Center**

**Available Commands:**
- `help` - Show this help menu
- `questions` - View common agricultural questions
- `/category [name]` - Get questions by category

**Categories:**
- `plant_diseases` - Disease identification
- `treatment_methods` - Treatment options
- `crop_management` - Farming practices
- `seasonal_advice` - Season-specific guidance
- `technology_agriculture` - Modern farming tech

**Example Questions:**
- "What causes yellow leaves in tomatoes?"
- "How to make organic pesticide?"
- "Best time to plant vegetables?"

Just type your question naturally! 🌱"""

def generate_common_questions_response():
    """Generate response with common questions"""
    questions = get_common_questions_by_category(limit=8)
    
    response = "❓ **Popular Agricultural Questions**\n\n"
    
    for i, question in enumerate(questions, 1):
        response += f"**{i}.** {question}\n"
    
    response += "\n**More Help:** Type `/category plant_diseases` for specific topics!"
    return response

def generate_category_questions(category):
    """Generate questions for a specific category"""
    if category not in COMMON_QUESTIONS:
        available_categories = ", ".join(COMMON_QUESTIONS.keys())
        return f"❓ Category '{category}' not found.\n\n**Available:** {available_categories}"
    
    questions = COMMON_QUESTIONS[category]
    category_title = category.replace('_', ' ').title()
    
    response = f"📚 **{category_title} - Questions**\n\n"
    
    for i, question in enumerate(questions, 1):
        response += f"**{i}.** {question}\n"
    
    return response

def get_fallback_response(original_message, detected_disease=None, error_msg=None):
    """Enhanced fallback response when AI is unavailable"""
    
    fallback = f"""🤖 **AgriPal AI Assistant** *(Offline Mode)*

**Your question:** "{original_message}"

"""
    
    if detected_disease:
        fallback += f"**Detected disease:** {detected_disease}\n\n"
    
    message_lower = original_message.lower()
    
    if any(word in message_lower for word in ["disease", "fungus", "infection"]):
        fallback += """**For plant diseases:**
🔍 Take clear photos of affected areas
✂️ Remove diseased plant parts
🌿 Apply appropriate treatment
📞 Consult agricultural extension officer"""
    
    elif any(word in message_lower for word in ["treatment", "pesticide", "spray"]):
        fallback += """**Treatment guidelines:**
🧪 Use registered pesticides as per label
🌱 Try organic options (neem oil, copper sulfate)
⏰ Apply during cool hours
⚠️ Always wear protective equipment"""
    
    fallback += "\n\n**Try:** `questions` for common topics or `help` for commands"
    
    return fallback

def startup_gemini_check():
    """Check Gemini AI status on startup"""
    logger.info("🚀 Initializing Enhanced AgriPal Chatbot...")
    
    success, message = initialize_enhanced_gemini()
    
    if success:
        logger.info(f"✅ Chatbot Ready: {message}")
    else:
        logger.warning(f"⚠️ AI Limited Mode: {message}")
    
    return success

# Enhanced function to get disease information with better debugging
def get_disease_info(disease_name):
    """Enhanced function to get disease information with proper JSON structure handling"""
    try:
        logger.info(f"Looking for disease info: {disease_name}")
        logger.info(f"Available diseases in JSON: {list(disease_treatments.keys())[:5]}...")  # Log first 5 keys
        
        # First, try exact match
        disease_info = disease_treatments.get(disease_name, None)
        
        # If no exact match, try with some variations
        if not disease_info:
            logger.info(f"No exact match for {disease_name}, trying variations...")
            # Try removing underscores and parentheses for matching
            cleaned_name = disease_name.replace('_', ' ').replace('(', '').replace(')', '').strip()
            for key, value in disease_treatments.items():
                if cleaned_name.lower() in key.lower() or key.lower() in cleaned_name.lower():
                    disease_info = value
                    logger.info(f"Found match with key: {key}")
                    break
        
        if disease_info:
            # Validate the structure
            required_keys = ['name', 'description', 'treatment', 'severity', 'pesticide']
            missing_keys = [key for key in required_keys if key not in disease_info]
            if missing_keys:
                logger.warning(f"Missing keys in disease info: {missing_keys}")
            
            # Process video sources for better YouTube integration
            if 'pesticide' in disease_info:
                for treatment_type in ['chemical', 'organic']:
                    if treatment_type in disease_info['pesticide']:
                        treatment = disease_info['pesticide'][treatment_type]
                        
                        # Process video sources if they exist
                        if 'video_sources' in treatment:
                            video_sources = treatment['video_sources']
                            
                            # Add YouTube search URLs for each search term
                            if 'search_terms' in video_sources:
                                search_urls = []
                                for term in video_sources['search_terms']:
                                    search_urls.append({
                                        'term': term,
                                        'url': f"https://www.youtube.com/results?search_query={quote_plus(term)}"
                                    })
                                video_sources['search_urls'] = search_urls
                                logger.info(f"Added {len(search_urls)} search URLs for {treatment_type}")
                            
                            # Process reliable channels for direct links
                            if 'reliable_channels' in video_sources:
                                channel_urls = []
                                for channel in video_sources['reliable_channels']:
                                    channel_urls.append({
                                        'name': channel,
                                        'url': f"https://www.youtube.com/results?search_query={quote_plus(channel + ' ' + disease_name.replace('_', ' '))}"
                                    })
                                video_sources['channel_urls'] = channel_urls
                                logger.info(f"Added {len(channel_urls)} channel URLs for {treatment_type}")
            
            logger.info(f"Successfully processed disease info for: {disease_name}")
            return disease_info
        
        logger.warning(f"No disease info found for: {disease_name}")
        return None
        
    except Exception as e:
        logger.error(f"Error processing disease info for {disease_name}: {e}")
        return None



# All your existing routes
@app.route('/')
def index():
    logger.info("Rendering index page")
    return render_template('index.html')
# Enhanced chatbot routes

@app.route('/chatbot')
def chatbot_page():
    """Render the chatbot HTML page"""
    logger.info("Rendering chatbot interface page")
    return render_template('chatbot.html')

@app.route('/api/chat/enhanced', methods=['POST'])
def enhanced_chat_api():
    """Enhanced API endpoint with better features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        conversation_history = data.get('history', [])
        detected_disease = data.get('detected_disease')
        
        # Try to get detected disease from file if not provided
        if not detected_disease:
            try:
                with open('detected_disease.json', 'r') as f:
                    disease_data = json.load(f)
                    detected_disease = disease_data.get('disease')
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        response_text = get_enhanced_chatbot_response(
            user_message, 
            detected_disease, 
            conversation_history
        )
        
        return jsonify({
            'success': True,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'detected_disease': detected_disease,
            'ai_status': 'online' if gemini_status else 'offline'
        })
        
    except Exception as e:
        logger.error(f"Enhanced Chat API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your message',
            'details': str(e)
        }), 500

@app.route('/api/chat/common-questions')
def get_common_questions_api():
    """API to get common questions by category"""
    try:
        category = request.args.get('category')
        limit = int(request.args.get('limit', 10))
        
        if category:
            if category not in COMMON_QUESTIONS:
                return jsonify({
                    'success': False,
                    'error': f'Category not found: {category}',
                    'available_categories': list(COMMON_QUESTIONS.keys())
                }), 404
            
            questions = get_common_questions_by_category(category, limit)
            return jsonify({
                'success': True,
                'category': category,
                'questions': questions,
                'total': len(questions)
            })
        else:
            all_categories = {}
            for cat, questions in COMMON_QUESTIONS.items():
                all_categories[cat] = {
                    'title': cat.replace('_', ' ').title(),
                    'sample_questions': questions[:3],
                    'total_questions': len(questions)
                }
            
            return jsonify({
                'success': True,
                'categories': all_categories,
                'total_questions': sum(len(q) for q in COMMON_QUESTIONS.values())
            })
            
    except Exception as e:
        logger.error(f"Common questions API error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
@app.route('/api/chat/system-status')
def chat_status():
    """Get current chat status and context"""
    try:
        detected_disease = None
        detection_time = None
        
        try:
            with open('detected_disease.json', 'r') as f:
                disease_data = json.load(f)
                detected_disease = disease_data.get('disease')
                detection_time = disease_data.get('timestamp')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return jsonify({
            'success': True,
            'gemini_available': GEMINI_API_KEY is not None,
            'detected_disease': detected_disease,
            'detection_time': detection_time,
            'model_loaded': model is not None,
            'supported_plants': len(SUPPORTED_PLANTS)
        })
    except Exception as e:
        logger.error(f"Chat status error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/detection-tool')
def detection_tool():
    logger.info("Rendering detection tool page")
    return render_template('detection-tool.html')

@app.route('/detection')
def detection():
    logger.info("Rendering detection page")
    return render_template('detection-tool.html')

@app.route('/about-us')
def about_us():
    logger.info("Rendering about-us page")
    return render_template('about-us.html')

@app.route('/contact')
def contact():
    logger.info("Rendering contact page")
    return render_template('contact.html')

@app.route('/library')
def library():
    logger.info("Rendering library page")
    return render_template('library.html')

# ✅ POST-HARVEST MANAGEMENT ROUTES (ADD THESE)

@app.route('/post-harvest')
def post_harvest_page():
    """Render post-harvest management page"""
    logger.info("Rendering post-harvest management page")
    return render_template('post-harvest.html')

@app.route('/schemes')
def schemes_page():
    """Render government schemes page"""
    logger.info("Rendering schemes page")
    return render_template('schemes.html')

@app.route('/api/info')
def api_info():
    """Enhanced API information endpoint"""
    return jsonify({
        'message': 'AGRI_PAL Unified API',
        'version': '2.0',
        'status': 'running',
        'endpoints': {
            'disease_detection': {
                'predict': 'POST /predict',
                'supported_plants': 'GET /api/supported-plants',
                'treatment': 'GET /api/treatment/<disease_name>'
            },
            'post_harvest': {
                'agro_shops': 'POST /post-harvest/agro-shops',
                'markets': 'POST /post-harvest/markets',
                'storage': 'POST /post-harvest/storage'
            },
            'schemes': {
                'all_schemes': 'GET /api/schemes',
                'categories': 'GET /api/schemes/categories',
                'by_category': 'GET /api/schemes/category/<category>',
                'by_id': 'GET /api/schemes/<scheme_id>',
                'search': 'GET /api/schemes/search?q=<query>'
            },
            'chatbot': {
                'chat': 'POST /api/chat/enhanced',
                'common_questions': 'GET /api/chat/common-questions',
                'status': 'GET /api/chat/system-status'
            }
        }
    })

@app.route('/plant-library')
def plant_library():
    logger.info("Rendering plant library page")
    return render_template('library.html')

@app.route('/process_audio', methods=['POST'])


# Line 1260
@app.route('/api/supported-plants')
def get_supported_plants():
    """API endpoint to get list of supported plants"""
    return jsonify({
        'supported_plants': SUPPORTED_PLANTS,
        'total_plants': len(SUPPORTED_PLANTS),
        'total_conditions': len(class_names)
    })
# Line 1267 ends here
# Line 1268 (NEW)
@app.route('/upload')
def upload_file():
    """
    Route for upload page - alias for detection tool
    This fixes the BuildError in error.html
    """
    logger.info("Upload file route accessed - redirecting to detection tool")
    return detection_tool()
# Line 1275 (NEW) ends here

@app.route('/predict', methods=['POST'])
def analyze():
    """Enhanced predict endpoint with FastSAM segmentation and multi-disease detection"""
    
    # Clear any previous detection at the start
    try:
        if os.path.exists('detected_disease.json'):
            os.remove('detected_disease.json')
    except Exception as e:
        logger.error(f"Error removing old detection file: {e}")
    
    logger.info("=" * 80)
    logger.info("🚀 ENHANCED PREDICT ENDPOINT - MULTI-DISEASE DETECTION")
    logger.info("=" * 80)
    
    if model is None:
        logger.error("Model not loaded, cannot process prediction")
        flash("Error: Model not loaded. Please check server logs.", "error")
        error_context = {
            'error': 'Model not loaded',
            'error_message': 'The AI model could not be loaded. Please check server configuration.',
            'back_link': '/detection-tool'
        }
        return render_template('error.html', **error_context)
    
    if 'image' not in request.files:
        logger.error("No image file in request")
        flash("Error: No image file uploaded", "error")
        error_context = {
            'error': 'No image file',
            'error_message': 'No image file was uploaded. Please select an image.',
            'back_link': '/detection-tool'
        }
        return render_template('error.html', **error_context)
    
    image_file = request.files['image']
    logger.info(f"📷 Image file received: {image_file.filename}")
    
    if image_file.filename == '':
        logger.error("Empty filename submitted")
        flash("Error: No image selected", "error")
        error_context = {
            'error': 'Empty filename',
            'error_message': 'No image was selected. Please choose a file.',
            'back_link': '/detection-tool'
        }
        return render_template('error.html', **error_context)
    
    if not allowed_file(image_file.filename):
        logger.error(f"File type not allowed: {image_file.filename}")
        flash("Error: File type not allowed. Please upload a PNG, JPG, or JPEG image.", "error")
        error_context = {
            'error': 'Invalid file type',
            'error_message': 'File type not allowed. Please upload a PNG, JPG, or JPEG image.',
            'back_link': '/detection-tool'
        }
        return render_template('error.html', **error_context)
    
    try:
        # STEP 1: SAVE UPLOADED IMAGE
        image_filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)
        logger.info(f"✅ Image saved to: {image_path}")
        
        # STEP 2: FASTSAM SEGMENTATION - EXTRACT LEAVES
        logger.info("🔬 Starting GrabCut + Watershed segmentation with severity analysis...")
        
        try:
            leaf_results, plant_severity, plant_level = segment_analyze_plant(image_path)
            
            logger.info(f"✅ Segmentation completed - {len(leaf_results)} leaves found")
            logger.info(f"🌱 Plant Severity: {plant_severity}% ({plant_level})")
            
            leaves_dir = os.path.join("static", "individual_leaves")
            segmented_dir = os.path.join("static", "segmented_output")
            heatmap_path = os.path.join(segmented_dir, "segmented_leaf_heatmap.png")
            has_heatmap = os.path.exists(heatmap_path)
            
            logger.info(f"📊 Heatmap available: {has_heatmap}")
                    
        except Exception as seg_error:
            logger.error(f"❌ Segmentation failed: {seg_error}")
            logger.error(traceback.format_exc())
            
            # Fallback: use original image
            logger.warning("⚠️ Falling back to original image")
            leaves_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'leaves')
            os.makedirs(leaves_dir, exist_ok=True)
            
            fallback_leaf = os.path.join(leaves_dir, 'leaf_1.jpg')
            shutil.copy(image_path, fallback_leaf)
            
            leaf_results = []
            plant_severity = 0.0
            plant_level = "Unknown"
            has_heatmap = False
        
        # STEP 3: GET LEAF PATHS WITH SEVERITY DATA
        if leaf_results:
            leaf_paths = [result["leaf"] for result in leaf_results]
            logger.info(f"🍃 Processing {len(leaf_paths)} leaves with severity data")
        else:
            leaf_paths = [
                os.path.join(leaves_dir, f)
                for f in os.listdir(leaves_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            logger.info(f"🍃 Processing {len(leaf_paths)} leaves (no severity data)")
        
        if not leaf_paths:
            logger.error("❌ No leaves detected after segmentation")
            error_context = {
                'error': 'No leaves detected',
                'error_message': 'No usable leaves detected. Try uploading a clearer plant image.',
                'error_details': get_detailed_error_message("not_plant"),
                'image_url': url_for('static', filename=f'uploads/{image_filename}'),
                'supported_plants': SUPPORTED_PLANTS,
                'back_link': '/detection-tool'
            }
            return render_template('error.html', **error_context)
        
        # STEP 4: PROCESS EACH LEAF - DISEASE DETECTION WITH PLANT TYPE TRACKING
        predictions = []
        leaf_severities = []
        plant_types_detected = {}  # Track different plant types
        
        logger.info("🔍 Starting disease detection on each leaf...")
        
        for idx, leaf_path in enumerate(leaf_paths, 1):
            logger.info(f"   Processing leaf {idx}/{len(leaf_paths)}: {os.path.basename(leaf_path)}")
            
            try:
                leaf_image = Image.open(leaf_path).convert("RGB")
                processed_image, is_valid_plant = preprocess_image_with_validation(leaf_image, leaf_path)

                if not is_valid_plant:
                    logger.warning(f"   ⚠️ Leaf {idx} failed plant validation - skipping")
                    continue

                predicted_class, confidence, error_message = make_enhanced_prediction(processed_image)

                if predicted_class and error_message is None:
                    # Extract plant type from predicted class (e.g., "Tomato" from "Tomato_Early_blight")
                    plant_type = predicted_class.split('_')[0]
                    
                    disease_info = get_disease_info(predicted_class)
                    severity = disease_info.get('severity', 'Unknown') if disease_info else 'Unknown'
                    
                    color_severity = None
                    color_severity_level = "Unknown"

                    if leaf_results:
                        for result in leaf_results:
                            if os.path.basename(leaf_path) == os.path.basename(result["leaf"]):
                                color_severity = result["severity_percent"]
                                color_severity_level = result["severity_level"]
                                break

                    prediction_data = {
                        "leaf": os.path.basename(leaf_path),
                        "leaf_number": idx,
                        "predicted_class": predicted_class,
                        "plant_type": plant_type,  # Track plant type
                        "confidence": confidence,
                        "model_severity": severity,
                        "color_severity": color_severity,
                        "color_severity_level": color_severity_level,
                        "leaf_area": result.get("leaf_area", 0) if leaf_results else 0,
                        "error": error_message,
                        "disease_info": disease_info  # Store full disease info
                    }
                    
                    predictions.append(prediction_data)
                    
                    # Track plant types
                    if plant_type not in plant_types_detected:
                        plant_types_detected[plant_type] = []
                    plant_types_detected[plant_type].append(prediction_data)

                    logger.info(f"   ✅ Leaf {idx}: {predicted_class} ({confidence:.1f}%) - "
                            f"Plant: {plant_type}, Model: {severity}, Color: {color_severity_level}")
                    
                    if severity != 'Unknown':
                        leaf_severities.append(severity)
                else:
                    logger.warning(f"   ⚠️ Leaf {idx}: Prediction failed - {error_message}")
                    
            except Exception as leaf_error:
                logger.error(f"   ❌ Error processing leaf {idx}: {leaf_error}")
                continue
        
        logger.info(f"📊 Disease detection completed: {len(predictions)}/{len(leaf_paths)} leaves successfully analyzed")
        logger.info(f"🌿 Plant types detected: {list(plant_types_detected.keys())}")
        
        if not predictions:
            logger.error("❌ No valid predictions from any leaf")
            error_context = {
                'error': 'Low confidence',
                'error_message': 'Leaves detected, but disease could not be predicted clearly.',
                'error_details': get_detailed_error_message("low_confidence"),
                'image_url': url_for('static', filename=f'uploads/{image_filename}'),
                'supported_plants': SUPPORTED_PLANTS,
                'back_link': '/detection-tool'
            }
            return render_template('error.html', **error_context)

        # STEP 5: FILTER - KEEP ONLY DOMINANT PLANT TYPE
        if len(plant_types_detected) > 1:
            logger.info("🔍 Multiple plant types detected - filtering to dominant type...")
            
            # Find the plant type with highest total confidence
            dominant_plant_type = max(
                plant_types_detected.items(), 
                key=lambda x: sum(p['confidence'] for p in x[1])
            )[0]
            
            logger.info(f"🎯 Dominant plant type: {dominant_plant_type}")
            
            # Filter predictions to only include dominant plant type
            filtered_predictions = [p for p in predictions if p['plant_type'] == dominant_plant_type]
            rejected_count = len(predictions) - len(filtered_predictions)
            
            if rejected_count > 0:
                logger.warning(f"⚠️ Filtered out {rejected_count} leaves from different plant types")
                logger.info(f"   Keeping {len(filtered_predictions)} {dominant_plant_type} leaves")
                
                # Log rejected plant types
                for plant_type, leaves in plant_types_detected.items():
                    if plant_type != dominant_plant_type:
                        logger.info(f"   Rejected: {len(leaves)} {plant_type} leaves")
            
            # Update predictions to filtered list
            predictions = filtered_predictions
        else:
            dominant_plant_type = list(plant_types_detected.keys())[0]
            rejected_count = 0
            logger.info(f"✅ Single plant type detected: {dominant_plant_type}")

        # STEP 6: COLLECT UNIQUE DISEASES & THEIR DATA
        unique_diseases = {}
        for pred in predictions:
            disease = pred['predicted_class']
            if disease not in unique_diseases:
                unique_diseases[disease] = {
                    'count': 0,
                    'total_confidence': 0,
                    'disease_info': pred['disease_info'],
                    'severities': [],
                    'leaves': []
                }
            unique_diseases[disease]['count'] += 1
            unique_diseases[disease]['total_confidence'] += pred['confidence']
            unique_diseases[disease]['leaves'].append(pred['leaf_number'])
            if pred['model_severity'] != 'Unknown':
                unique_diseases[disease]['severities'].append(pred['model_severity'])

        logger.info(f"🦠 Unique diseases detected: {len(unique_diseases)}")
        for disease, data in unique_diseases.items():
            avg_conf = data['total_confidence'] / data['count']
            logger.info(f"   - {disease}: {data['count']} leaves (avg confidence {avg_conf:.1f}%), leaves: {data['leaves']}")

        # STEP 7: SELECT PRIMARY DISEASE (highest count, then confidence)
        primary_disease = max(
            unique_diseases.items(), 
            key=lambda x: (x[1]['count'], x[1]['total_confidence'])
        )[0]

        best = max(predictions, key=lambda x: x["confidence"])
        predicted_class = primary_disease  # Use most common disease
        confidence = best["confidence"]

        logger.info(f"🎯 Primary disease: {predicted_class} ({confidence:.1f}%)")

        # STEP 8: COMBINE TREATMENTS IF MULTIPLE DISEASES
        combined_treatments = None
        if len(unique_diseases) > 1:
            logger.info("🔀 Multiple diseases detected - combining treatments...")
            combined_treatments = combine_disease_treatments(unique_diseases)
            logger.info(f"✅ Combined treatment plan created for {len(unique_diseases)} diseases")
        else:
            logger.info("✅ Single disease detected - using standard treatment")
        
        # Calculate overall severity
        severity_levels = {'Low': 1, 'Moderate': 2, 'Medium': 2, 'High': 3, 'Severe': 4}
        if leaf_severities:
            severity_scores = [severity_levels.get(s, 0) for s in leaf_severities]
            avg_severity_score = sum(severity_scores) / len(severity_scores)
            
            if avg_severity_score <= 1.5:
                overall_severity = 'Low'
            elif avg_severity_score <= 2.5:
                overall_severity = 'Moderate'
            elif avg_severity_score <= 3.5:
                overall_severity = 'High'
            else:
                overall_severity = 'Severe'
        else:
            overall_severity = best.get('model_severity', 'Unknown')
        
        logger.info(f"📈 Overall Severity: {overall_severity}")
        
        # STEP 9: GENERATE GRADCAM
        gradcam_filename = None
        try:
            logger.info("🎨 Generating GradCAM heatmap...")
            
            full_image = Image.open(image_path).convert("RGB")
            full_processed, is_valid = preprocess_image_with_validation(full_image, image_path)
            
            if full_processed is not None:
                predicted_class_index = class_names.index(predicted_class)
                heatmap, superimposed_img = generate_gradcam(full_processed, model, predicted_class_index)
                
                if superimposed_img is not None:
                    original_size = full_image.size
                    superimposed_resized = cv2.resize(superimposed_img, original_size)
                    gradcam_filename = save_gradcam_image(superimposed_resized, image_filename)
                    
                    if gradcam_filename:
                        logger.info(f"✅ GradCAM saved: {gradcam_filename}")
                    else:
                        logger.warning("⚠️ GradCAM save failed")
                        
        except Exception as gradcam_error:
            logger.error(f"❌ GradCAM error: {gradcam_error}")
            logger.error(traceback.format_exc())
            gradcam_filename = None

        # STEP 10: GET LAND AREA & LOCATION INFO
        area = request.form.get('area', '0')
        area_unit = request.form.get('area_unit', 'hectare')
        location = request.form.get('location', '')
        logger.info(f"🏡 Land area: {area} {area_unit}, Location: {location}")
        
        try:
            area_float = float(area) if area else 0.0
        except ValueError:
            area_float = 0.0
            logger.warning(f"Invalid area value: {area}, setting to 0")
        
        # STEP 11: GET DISEASE INFO & CALCULATE DOSAGE
        disease_info = get_disease_info(predicted_class)
        if disease_info:
            import copy
            disease_info = copy.deepcopy(disease_info)
            logger.info(f"✅ Found disease info for: {predicted_class}")
        else:
            logger.warning(f"⚠️ No disease info found for: {predicted_class}")
        
        chemical_dosage = None
        organic_dosage = None
        hectare_conversion = None
        
        if disease_info and 'pesticide' in disease_info and area_float > 0:
            try:
                chemical_dosage, organic_dosage, hectare_conversion = calculate_dosage(
                    area_float, area_unit, disease_info['pesticide']
                )
                logger.info(f"💊 Dosages - Chemical: {chemical_dosage}, Organic: {organic_dosage}")
            except Exception as dosage_error:
                logger.error(f"❌ Dosage calculation error: {dosage_error}")
        
        # STEP 12: SAVE DETECTION FOR CHATBOT
        try:
            detection_data = {
                "disease": predicted_class,
                "timestamp": str(datetime.now()),
                "confidence": confidence,
                "severity": overall_severity,
                "area": area,
                "area_unit": area_unit,
                "total_leaves_analyzed": len(predictions),
                "plant_type": dominant_plant_type,
                "multiple_diseases": len(unique_diseases) > 1,
                "unique_disease_count": len(unique_diseases)
            }
            
            if len(unique_diseases) > 1:
                detection_data["all_diseases"] = [
                    {
                        "name": disease,
                        "count": data['count'],
                        "avg_confidence": data['total_confidence'] / data['count']
                    }
                    for disease, data in unique_diseases.items()
                ]
            
            with open('detected_disease.json', 'w') as f:
                json.dump(detection_data, f, indent=2)
            logger.info(f"💾 Saved detection context for chatbot")
        except Exception as save_error:
            logger.error(f"❌ Error saving detection: {save_error}")
                
        # STEP 13: PREPARE & RENDER RESULTS
        template_vars = {
            # ===== IMAGE URLS =====
            'image_url': url_for('static', filename=f'uploads/{image_filename}'),
            'gradcam_url': url_for('static', filename=f'uploads/{gradcam_filename}') if gradcam_filename else None,
            'heatmap_url': url_for('static', filename='segmented_output/segmented_leaf_heatmap.png') if has_heatmap else None,
            'segmented_image_url': url_for('static', filename='segmented_output/segmented_leaf.png') if os.path.exists(os.path.join('static', 'segmented_output', 'segmented_leaf.png')) else None,
            
            # ===== PREDICTION DATA =====
            'predicted_class': predicted_class,
            'confidence': confidence,
            'severity': overall_severity,
            'total_leaves': len(predictions),
            'all_predictions': predictions,  # Filtered to same plant type
            
            # ===== PLANT & DISEASE TRACKING =====
            'dominant_plant_type': dominant_plant_type,
            'filtered_leaf_count': rejected_count if rejected_count > 0 else None,
            'unique_diseases': unique_diseases if len(unique_diseases) > 1 else None,
            'combined_treatments': combined_treatments,
            'is_multi_disease': len(unique_diseases) > 1,
            
            # ===== PLANT SEVERITY (Color-based) =====
            'plant_severity': plant_severity,
            'plant_severity_level': plant_level,
            
            # ===== DISEASE INFORMATION =====
            'result': disease_info,  # Primary disease info
            
            # ===== FARM DATA =====
            'area': area,
            'area_unit': area_unit,
            'location': location,
            
            # ===== DOSAGE CALCULATIONS =====
            'chemical_dosage': chemical_dosage,
            'organic_dosage': organic_dosage,
            'hectare_conversion': hectare_conversion,
            
            # ===== LEAF RESULTS =====
            'leaf_results': leaf_results
        }

        # DEBUG LOGGING
        logger.info("=" * 80)
        logger.info("📦 TEMPLATE VARIABLES FOR RENDERING")
        logger.info("=" * 80)
        logger.info(f"✅ Predicted: {predicted_class}")
        logger.info(f"✅ Confidence: {confidence:.1f}%")
        logger.info(f"✅ Severity: {overall_severity}")
        logger.info(f"✅ Total Leaves: {len(predictions)}")
        logger.info(f"✅ Plant Type: {dominant_plant_type}")
        logger.info(f"✅ Filtered Leaves: {rejected_count}")
        logger.info(f"✅ Unique Diseases: {len(unique_diseases)}")
        logger.info(f"✅ Multi-Disease: {len(unique_diseases) > 1}")
        logger.info("=" * 80)
        return render_template('result1.html', **template_vars)

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"❌ CRITICAL ERROR in predict endpoint:")
        logger.error(error_details)
        
        error_context = {
            'error': str(e),
            'error_message': f"An error occurred during analysis: {str(e)}",
            'error_type': type(e).__name__,
            'error_details': {
                'reason': 'Processing error',
                'issues': [str(e)]
            },
            'back_link': '/detection-tool'
        }
        
        flash(f"An error occurred: {str(e)}", "error")
        
        try:
            return render_template('error.html', **error_context), 500
        except Exception as template_error:
            logger.error(f"Error template rendering failed: {template_error}")
            return f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Error - AgriPal</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        padding: 20px;
                        margin: 0;
                    }}
                    .error-container {{
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
                        max-width: 600px;
                        width: 100%;
                        padding: 40px;
                        text-align: center;
                    }}
                    .error-icon {{ font-size: 60px; margin-bottom: 20px; }}
                    h1 {{ color: #333; margin-bottom: 15px; }}
                    p {{ color: #666; line-height: 1.6; margin-bottom: 25px; }}
                    .error-details {{
                        background: #f8f9fa;
                        border-left: 4px solid #ff6b6b;
                        padding: 15px;
                        margin: 20px 0;
                        text-align: left;
                        border-radius: 5px;
                    }}
                    .btn {{
                        display: inline-block;
                        padding: 12px 24px;
                        margin: 10px 5px;
                        border-radius: 25px;
                        text-decoration: none;
                        font-weight: 600;
                        transition: all 0.3s ease;
                    }}
                    .btn-primary {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }}
                    .btn-primary:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
                    }}
                </style>
            </head>
            <body>
                <div class="error-container">
                    <div class="error-icon">⚠️</div>
                    <h1>Error Occurred</h1>
                    <p>An error occurred while processing your request:</p>
                    <div class="error-details">
                        <strong>Error Type:</strong> {type(e).__name__}<br>
                        <strong>Message:</strong> {str(e)}
                    </div>
                    <p>Please try again or contact support if the problem persists.</p>
                    <div>
                        <a href="/detection-tool" class="btn btn-primary">🔄 Try Again</a>
                        <a href="/" class="btn btn-primary">🏠 Go Home</a>
                    </div>
                </div>
            </body>
            </html>
            """, 500

@app.route('/health')
def health_check():
    health_status = {
        "status": "ok",
        "model_loaded": model is not None,
        "treatments_loaded": len(disease_treatments) > 0,
        "upload_dir_exists": os.path.exists(app.config['UPLOAD_FOLDER']),
        "total_diseases": len(disease_treatments)
    }
    return jsonify(health_status)


@app.route('/api/treatment/<disease_name>')
def treatment_api(disease_name):
    try:
        logger.info(f"Treatment API called for disease: {disease_name}")
        disease_info = get_disease_info(disease_name)
        if disease_info:
            return jsonify(disease_info)
        else:
            logger.warning(f"No disease info found for: {disease_name}")
            return jsonify({'error': 'Disease information not found'}), 404
    except Exception as e:
        logger.error(f"Treatment API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources/<disease_name>')
def resources_api(disease_name):
    try:
        logger.info(f"Resources API called for disease: {disease_name}")
        disease_info = get_disease_info(disease_name)
        if disease_info and 'additional_resources' in disease_info:
            return jsonify(disease_info['additional_resources'])
        else:
            return jsonify({'error': 'Additional resources not found'}), 404
    except Exception as e:
        logger.error(f"Resources API error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/chat/direct-ai', methods=['POST'])
def direct_ai_chat():
    """Direct Gemini AI chat without AgriPal formatting"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        history = data.get('history', [])
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
            
        # Simple direct prompt to Gemini
        model = genai.GenerativeModel('gemini-1.5-flash-001')
        
        # Build conversation context
        conversation_context = ""
        if history:
            recent_messages = history[-3:]  # Last 3 exchanges
            for msg in recent_messages:
                role = "Human" if msg['role'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['text']}\n"
        
        # Create direct prompt
        full_prompt = f"""You are Gemini AI, a helpful and knowledgeable AI assistant. Respond naturally and comprehensively to the user's question.
        {conversation_context}
        Human: {message}
        Assistant:"""
        
        response = model.generate_content(full_prompt)
        
        if response and response.text:
            return jsonify({
                'success': True,
                'response': response.text.strip(),
                'timestamp': datetime.now().isoformat(),
                'ai_status': 'online',
                'mode': 'direct_ai'
            })
        else:
            return jsonify({'success': False, 'error': 'No response from AI'}), 500
            
    except Exception as e:
        logger.error(f"Direct AI chat error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# New endpoint for dosage calculation
@app.route('/api/calculate-dosage', methods=['POST'])
def calculate_dosage_api():
    try:
        data = request.json
        disease_name = data.get('disease_name')
        area = data.get('area')
        area_unit = data.get('area_unit', 'hectare')
        
        disease_info = get_disease_info(disease_name)
        if disease_info and 'pesticide' in disease_info:
            chemical_dosage, organic_dosage, hectare_conversion = calculate_dosage(
                area, area_unit, disease_info['pesticide']
            )
            return jsonify({
                'chemical_dosage': chemical_dosage,
                'organic_dosage': organic_dosage,
                'hectare_conversion': hectare_conversion,
                'area': area,
                'area_unit': area_unit
            })
        else:
            return jsonify({'error': 'Disease or pesticide information not found'}), 404
    except Exception as e:
        logger.error(f"Dosage calculation API error: {str(e)}")
        return jsonify({'error': str(e)}), 500



# Add missing import for datetime
from datetime import datetime
# Initialize enhanced system
gemini_status = startup_gemini_check()

# Add this function for graceful shutdown
def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\n" + "="*80)
    logger.info("🛑 SHUTDOWN SIGNAL RECEIVED (Ctrl+C)")
    logger.info("="*80)
    logger.info("Cleaning up resources...")
    
    # Clean up temporary files if needed
    try:
        temp_files = ['detected_disease.json']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"✅ Cleaned up: {temp_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    logger.info("👋 AgriPal Server Stopped Successfully!")
    logger.info("="*80)
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Replace the existing if __name__ == '__main__' block at the end of your file
if __name__ == '__main__':
    local_ip = get_local_ip()
    
    logger.info("="*80)
    logger.info("🌱 STARTING AGRIPAL APPLICATION 🌱")
    logger.info("="*80)
    logger.info(f"Model status: {'✅ Loaded' if model else '❌ Not loaded'}")
    logger.info(f"Disease treatments loaded: {len(disease_treatments)}")
    logger.info(f"Gemini AI status: {'✅ Online' if gemini_status else '⚠️ Offline'}")
    logger.info("="*80)
    logger.info("📱 ACCESS URLs:")
    logger.info(f"   Local:    http://127.0.0.1:5000")
    logger.info(f"   Network:  http://{local_ip}:5000")
    logger.info("="*80)
    logger.info("📱 MOBILE ACCESS:")
    logger.info(f"   1. Connect your phone to the SAME WiFi network")
    logger.info(f"   2. Open browser on phone")
    logger.info(f"   3. Go to: http://{local_ip}:5000")
    logger.info("="*80)
    logger.info("🛑 SHUTDOWN: Press Ctrl+C to stop the server")
    logger.info("="*80)
    
    try:
        # Run server on all interfaces (0.0.0.0) to allow mobile access
        app.run(
            host='0.0.0.0',  # This allows external connections
            port=5000,
            debug=True,
            use_reloader=True
        )
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"❌ Server error: {e}")

        sys.exit(1)
