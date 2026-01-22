""" 
AgriPal RAG System - MySQL Database Setup 
Creates agricultural database with structured farm data 
"""
import mysql.connector
from mysql.connector import Error
import config
import logging
from datetime import datetime, timedelta
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgriMySQLSetup:
    """Setup MySQL database for agricultural data"""
    
    def __init__(self):
        self.config = config.MYSQL_CONFIG
        self.conn = None
        self.cursor = None
    
    def create_database(self):
        """Create the agripal_rag database"""
        try:
            conn = mysql.connector.connect(
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password'],
                port=self.config['port']
            )
            cursor = conn.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS agripal_rag")
            logger.info("✅ Database 'agripal_rag' created/verified")
            cursor.close()
            conn.close()
        except Error as e:
            logger.error(f"❌ Error creating database: {e}")
            raise
    
    def connect(self):
        """Connect to the agripal_rag database"""
        try:
            self.conn = mysql.connector.connect(**self.config)
            self.cursor = self.conn.cursor()
            logger.info("✅ Connected to database")
        except Error as e:
            logger.error(f"❌ Error connecting: {e}")
            raise
    
    def create_tables(self):
        """Create all agricultural tables"""
        tables = {
            'farmers': """
                CREATE TABLE IF NOT EXISTS farmers (
                    farmer_id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    phone VARCHAR(15),
                    village VARCHAR(100),
                    district VARCHAR(100),
                    state VARCHAR(100),
                    land_size DECIMAL(10,2),
                    land_unit VARCHAR(20) DEFAULT 'acre',
                    soil_type VARCHAR(100),
                    irrigation_type VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_state (state),
                    INDEX idx_district (district),
                    UNIQUE KEY unique_phone (phone)
                )
            """,
            
            'crops': """
                CREATE TABLE IF NOT EXISTS crops (
                    crop_id INT AUTO_INCREMENT PRIMARY KEY,
                    crop_name VARCHAR(100) NOT NULL UNIQUE,
                    crop_type VARCHAR(50),
                    season VARCHAR(50),
                    duration_days INT,
                    water_requirement VARCHAR(50),
                    soil_type VARCHAR(100),
                    climate VARCHAR(100),
                    avg_yield_per_acre DECIMAL(10,2),
                    yield_unit VARCHAR(20),
                    INDEX idx_season (season)
                )
            """,
            
            'farmer_crops': """
                CREATE TABLE IF NOT EXISTS farmer_crops (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    farmer_id INT NOT NULL,
                    crop_id INT NOT NULL,
                    area DECIMAL(10,2),
                    area_unit VARCHAR(20) DEFAULT 'acre',
                    planting_date DATE,
                    expected_harvest DATE,
                    status VARCHAR(50) DEFAULT 'growing',
                    FOREIGN KEY (farmer_id) REFERENCES farmers(farmer_id) ON DELETE CASCADE,
                    FOREIGN KEY (crop_id) REFERENCES crops(crop_id) ON DELETE CASCADE,
                    INDEX idx_status (status),
                    INDEX idx_farmer (farmer_id)
                )
            """,
            
            'disease_detections': """
                CREATE TABLE IF NOT EXISTS disease_detections (
                    detection_id INT AUTO_INCREMENT PRIMARY KEY,
                    farmer_id INT,
                    crop_name VARCHAR(100),
                    disease_name VARCHAR(200) NOT NULL,
                    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    severity VARCHAR(50),
                    confidence DECIMAL(5,2),
                    image_path VARCHAR(500),
                    location VARCHAR(200),
                    treatment_applied TEXT,
                    notes TEXT,
                    FOREIGN KEY (farmer_id) REFERENCES farmers(farmer_id) ON DELETE SET NULL,
                    INDEX idx_disease (disease_name),
                    INDEX idx_date (detection_date)
                )
            """,
            
            'government_schemes': """
                CREATE TABLE IF NOT EXISTS government_schemes (
                    scheme_id INT AUTO_INCREMENT PRIMARY KEY,
                    scheme_name VARCHAR(200) NOT NULL UNIQUE,
                    scheme_type VARCHAR(100),
                    state VARCHAR(100),
                    eligibility TEXT,
                    benefits TEXT,
                    subsidy_percentage DECIMAL(5,2),
                    application_process TEXT,
                    website VARCHAR(500),
                    active BOOLEAN DEFAULT TRUE,
                    start_date DATE,
                    end_date DATE,
                    INDEX idx_state (state),
                    INDEX idx_active (active)
                )
            """,
            
            'fertilizers': """
                CREATE TABLE IF NOT EXISTS fertilizers (
                    fertilizer_id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(200) NOT NULL UNIQUE,
                    type VARCHAR(50),
                    npk_ratio VARCHAR(50),
                    application_rate VARCHAR(100),
                    suitable_crops TEXT,
                    price_per_kg DECIMAL(10,2),
                    manufacturer VARCHAR(200),
                    organic BOOLEAN DEFAULT FALSE
                )
            """,
            
            'market_prices': """
                CREATE TABLE IF NOT EXISTS market_prices (
                    price_id INT AUTO_INCREMENT PRIMARY KEY,
                    crop_name VARCHAR(100) NOT NULL,
                    mandi_name VARCHAR(200),
                    district VARCHAR(100),
                    state VARCHAR(100),
                    price_per_quintal DECIMAL(10,2),
                    price_date DATE NOT NULL,
                    min_price DECIMAL(10,2),
                    max_price DECIMAL(10,2),
                    modal_price DECIMAL(10,2),
                    INDEX idx_crop (crop_name),
                    INDEX idx_date (price_date),
                    INDEX idx_mandi (mandi_name),
                    UNIQUE KEY unique_price_entry (crop_name, mandi_name, price_date)
                )
            """,
            
            'weather_data': """
                CREATE TABLE IF NOT EXISTS weather_data (
                    weather_id INT AUTO_INCREMENT PRIMARY KEY,
                    location VARCHAR(200) NOT NULL,
                    district VARCHAR(100),
                    state VARCHAR(100),
                    date DATE NOT NULL,
                    temperature_min DECIMAL(5,2),
                    temperature_max DECIMAL(5,2),
                    rainfall DECIMAL(6,2),
                    humidity DECIMAL(5,2),
                    wind_speed DECIMAL(5,2),
                    INDEX idx_location (location),
                    INDEX idx_date (date),
                    UNIQUE KEY unique_weather_entry (location, date)
                )
            """,
            
            'chatbot_history': """
                CREATE TABLE IF NOT EXISTS chatbot_history (
                    chat_id INT AUTO_INCREMENT PRIMARY KEY,
                    farmer_id INT,
                    query TEXT NOT NULL,
                    response TEXT,
                    detected_intent VARCHAR(100),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rating INT,
                    FOREIGN KEY (farmer_id) REFERENCES farmers(farmer_id) ON DELETE SET NULL,
                    INDEX idx_timestamp (timestamp)
                )
            """
        }
        
        logger.info("\n📋 Creating tables...")
        for table_name, create_statement in tables.items():
            try:
                self.cursor.execute(create_statement)
                logger.info(f"   ✅ {table_name}")
            except Error as e:
                logger.error(f"   ❌ {table_name}: {e}")
                raise
        
        self.conn.commit()
    
    def clear_sample_data(self):
        """Clear existing sample data - USE ONLY IN DEVELOPMENT"""
        logger.info("\n🗑️  Clearing existing data...")
        
        # Disable foreign key checks temporarily
        self.cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        tables_to_clear = [
            'chatbot_history',
            'disease_detections', 
            'farmer_crops',
            'market_prices',
            'weather_data',
            'farmers',
            'crops',
            'fertilizers',
            'government_schemes'
        ]
        
        for table in tables_to_clear:
            try:
                self.cursor.execute(f"TRUNCATE TABLE {table}")
                logger.info(f"   ✅ Cleared {table}")
            except Error as e:
                logger.warning(f"   ⚠️  Could not clear {table}: {e}")
        
        # Re-enable foreign key checks
        self.cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        self.conn.commit()
        logger.info("   ✅ All data cleared")
    
    def insert_sample_data(self):
        """Insert comprehensive sample data with duplicate handling"""
        logger.info("\n📥 Inserting sample data...")
        
        # Sample Farmers - using INSERT IGNORE to handle duplicates
        farmers_data = [
            ('Ramesh Kumar', '9876543210', 'Dharwad', 'Dharwad', 'Karnataka', 5.0, 'acre', 'Black Soil', 'Drip'),
            ('Lakshmi Devi', '9876543211', 'Hassan', 'Hassan', 'Karnataka', 3.5, 'acre', 'Red Soil', 'Borewell'),
            ('Suresh Patil', '9876543212', 'Belgaum', 'Belgaum', 'Karnataka', 10.0, 'acre', 'Mixed Soil', 'Canal'),
            ('Manjula Gowda', '9876543213', 'Mysore', 'Mysore', 'Karnataka', 2.0, 'acre', 'Red Soil', 'Rainwater'),
            ('Prakash Reddy', '9876543214', 'Bellary', 'Bellary', 'Karnataka', 15.0, 'acre', 'Black Soil', 'Canal')
        ]
        
        insert_farmer = """
            INSERT INTO farmers 
            (name, phone, village, district, state, land_size, land_unit, soil_type, irrigation_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                village = VALUES(village),
                district = VALUES(district),
                state = VALUES(state),
                land_size = VALUES(land_size),
                soil_type = VALUES(soil_type),
                irrigation_type = VALUES(irrigation_type)
        """
        
        self.cursor.executemany(insert_farmer, farmers_data)
        logger.info("   ✅ Farmers data inserted/updated")
        
        # Sample Crops - using ON DUPLICATE KEY UPDATE
        crops_data = [
            ('Tomato', 'Vegetable', 'Rabi', 90, 'Medium', 'Red/Black Soil', 'Warm', 15.0, 'ton/acre'),
            ('Potato', 'Vegetable', 'Rabi', 100, 'Medium', 'Sandy Loam', 'Cool', 8.0, 'ton/acre'),
            ('Rice', 'Cereal', 'Kharif', 120, 'High', 'Clay/Loam', 'Warm & Humid', 2.5, 'ton/acre'),
            ('Cotton', 'Cash Crop', 'Kharif', 180, 'Low', 'Black Soil', 'Hot', 10.0, 'quintal/acre'),
            ('Onion', 'Vegetable', 'Rabi', 110, 'Medium', 'Well-drained', 'Mild', 12.0, 'ton/acre'),
            ('Maize', 'Cereal', 'Kharif', 90, 'Medium', 'Well-drained', 'Warm', 3.0, 'ton/acre'),
            ('Sugarcane', 'Cash Crop', 'Annual', 365, 'Very High', 'Loamy', 'Tropical', 35.0, 'ton/acre'),
            ('Chilli', 'Spice', 'Rabi', 120, 'Medium', 'Sandy Loam', 'Warm', 2.0, 'ton/acre')
        ]
        
        insert_crop = """
            INSERT INTO crops 
            (crop_name, crop_type, season, duration_days, water_requirement, 
             soil_type, climate, avg_yield_per_acre, yield_unit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                crop_type = VALUES(crop_type),
                season = VALUES(season),
                duration_days = VALUES(duration_days),
                water_requirement = VALUES(water_requirement),
                soil_type = VALUES(soil_type),
                climate = VALUES(climate),
                avg_yield_per_acre = VALUES(avg_yield_per_acre),
                yield_unit = VALUES(yield_unit)
        """
        
        self.cursor.executemany(insert_crop, crops_data)
        logger.info("   ✅ Crops data inserted/updated")
        
        # Sample Government Schemes
        schemes_data = [
            ('PM-KISAN', 'Direct Benefit', 'All India', 'All farmers', '₹6000/year in 3 installments', 100.0, 
             'Online at pmkisan.gov.in', 'https://pmkisan.gov.in', True, '2019-01-01', None),
            ('Pradhan Mantri Fasal Bima Yojana', 'Insurance', 'All India', 'All farmers', 
             'Crop insurance at 2% premium', 98.0, 'Through banks/CSC', 'https://pmfby.gov.in', True, '2016-01-01', None),
            ('Karnataka Raitha Suraksha Scheme', 'Insurance', 'Karnataka', 'Karnataka farmers', 
             'Zero premium crop insurance', 100.0, 'Automatic enrollment', 'https://raitamitra.karnataka.gov.in', 
             True, '2023-04-01', None),
            ('Soil Health Card Scheme', 'Advisory', 'All India', 'All farmers', 
             'Free soil testing and recommendations', 100.0, 'Through agriculture department', 
             'https://soilhealth.dac.gov.in', True, '2015-02-01', None)
        ]
        
        insert_scheme = """
            INSERT INTO government_schemes 
            (scheme_name, scheme_type, state, eligibility, benefits, subsidy_percentage, 
             application_process, website, active, start_date, end_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                scheme_type = VALUES(scheme_type),
                state = VALUES(state),
                eligibility = VALUES(eligibility),
                benefits = VALUES(benefits),
                subsidy_percentage = VALUES(subsidy_percentage),
                application_process = VALUES(application_process),
                website = VALUES(website),
                active = VALUES(active)
        """
        
        self.cursor.executemany(insert_scheme, schemes_data)
        logger.info("   ✅ Government schemes inserted/updated")
        
        # Sample Fertilizers
        fertilizers_data = [
            ('Urea', 'Nitrogen', '46-0-0', '100-150 kg/acre', 'Rice, Wheat, Maize, Sugarcane', 6.50, 'IFFCO', False),
            ('DAP', 'Phosphate', '18-46-0', '50-75 kg/acre', 'All crops', 27.00, 'IFFCO', False),
            ('Potash (MOP)', 'Potassium', '0-0-60', '25-50 kg/acre', 'Potato, Tomato, Banana', 17.50, 'IPL', False),
            ('Vermicompost', 'Organic', 'Variable', '500-1000 kg/acre', 'All crops', 8.00, 'Local', True),
            ('Neem Cake', 'Organic', '5-1-1', '200-400 kg/acre', 'All crops', 25.00, 'Various', True)
        ]
        
        insert_fertilizer = """
            INSERT INTO fertilizers 
            (name, type, npk_ratio, application_rate, suitable_crops, price_per_kg, manufacturer, organic)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                type = VALUES(type),
                npk_ratio = VALUES(npk_ratio),
                application_rate = VALUES(application_rate),
                suitable_crops = VALUES(suitable_crops),
                price_per_kg = VALUES(price_per_kg),
                manufacturer = VALUES(manufacturer),
                organic = VALUES(organic)
        """
        
        self.cursor.executemany(insert_fertilizer, fertilizers_data)
        logger.info("   ✅ Fertilizers data inserted/updated")
        
        # Sample Market Prices (last 7 days)
        logger.info("   📊 Inserting market prices...")
        prices_data = []
        crops_for_price = ['Tomato', 'Potato', 'Onion', 'Rice']
        mandis = [
            ('Yeshwanthpur Mandi', 'Bangalore', 'Karnataka'),
            ('Dharwad APMC', 'Dharwad', 'Karnataka'),
            ('Mysore Mandi', 'Mysore', 'Karnataka')
        ]
        
        for i in range(7):
            date = datetime.now().date() - timedelta(days=i)
            for crop in crops_for_price:
                for mandi_name, district, state in mandis:
                    base_price = random.randint(1000, 3000)
                    prices_data.append((
                        crop, mandi_name, district, state, base_price, date,
                        base_price - 100, base_price + 150, base_price + 50
                    ))
        
        insert_price = """
            INSERT INTO market_prices 
            (crop_name, mandi_name, district, state, price_per_quintal, 
             price_date, min_price, max_price, modal_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                price_per_quintal = VALUES(price_per_quintal),
                min_price = VALUES(min_price),
                max_price = VALUES(max_price),
                modal_price = VALUES(modal_price)
        """
        
        self.cursor.executemany(insert_price, prices_data)
        logger.info(f"   ✅ Market prices inserted/updated ({len(prices_data)} records)")
        
        # Sample Weather Data
        logger.info("   🌤️  Inserting weather data...")
        weather_data = []
        locations = [
            ('Dharwad', 'Dharwad', 'Karnataka'),
            ('Bangalore', 'Bangalore', 'Karnataka'),
            ('Mysore', 'Mysore', 'Karnataka')
        ]
        
        for i in range(7):
            date = datetime.now().date() - timedelta(days=i)
            for location, district, state in locations:
                weather_data.append((
                    location, district, state, date,
                    random.uniform(15, 22),  # min temp
                    random.uniform(28, 35),  # max temp
                    random.uniform(0, 25),   # rainfall
                    random.uniform(60, 85),  # humidity
                    random.uniform(5, 15)    # wind speed
                ))
        
        insert_weather = """
            INSERT INTO weather_data 
            (location, district, state, date, temperature_min, temperature_max, 
             rainfall, humidity, wind_speed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                temperature_min = VALUES(temperature_min),
                temperature_max = VALUES(temperature_max),
                rainfall = VALUES(rainfall),
                humidity = VALUES(humidity),
                wind_speed = VALUES(wind_speed)
        """
        
        self.cursor.executemany(insert_weather, weather_data)
        logger.info(f"   ✅ Weather data inserted/updated ({len(weather_data)} records)")
        
        # Sample Disease Detections - Get farmer IDs first
        self.cursor.execute("SELECT farmer_id FROM farmers LIMIT 5")
        farmer_ids = [row[0] for row in self.cursor.fetchall()]
        
        if farmer_ids:
            detections_data = [
                (farmer_ids[0], 'Tomato', 'Tomato_Early_blight', 'High', 92.5, 'Copper fungicide applied', 'Leaves removed'),
                (farmer_ids[1], 'Potato', 'Potato_Late_blight', 'Severe', 88.3, 'Mancozeb spray', 'Immediate action needed'),
                (farmer_ids[0], 'Tomato', 'Tomato_Leaf_Mold', 'Moderate', 85.7, 'Organic neem treatment', 'Under monitoring'),
                (farmer_ids[2], 'Cotton', 'Cotton_Leaf_Curl', 'Low', 78.9, 'Vector control', 'Preventive spray'),
            ]
            
            insert_detection = """
                INSERT INTO disease_detections 
                (farmer_id, crop_name, disease_name, severity, confidence, treatment_applied, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.executemany(insert_detection, detections_data)
            logger.info("   ✅ Disease detections inserted")
        
        self.conn.commit()
        logger.info("   ✅ All sample data committed")
    
    def create_views(self):
        """Create useful views for analysis"""
        logger.info("\n👁️  Creating views...")
        
        views = {
            'farmer_disease_summary': """
                CREATE OR REPLACE VIEW farmer_disease_summary AS
                SELECT 
                    f.farmer_id,
                    f.name,
                    f.district,
                    f.state,
                    COUNT(d.detection_id) as total_detections,
                    COUNT(DISTINCT d.disease_name) as unique_diseases,
                    MAX(d.detection_date) as last_detection,
                    AVG(d.confidence) as avg_confidence
                FROM farmers f
                LEFT JOIN disease_detections d ON f.farmer_id = d.farmer_id
                GROUP BY f.farmer_id, f.name, f.district, f.state
            """,
            
            'crop_price_trends': """
                CREATE OR REPLACE VIEW crop_price_trends AS
                SELECT 
                    crop_name,
                    mandi_name,
                    state,
                    DATE(price_date) as date,
                    AVG(price_per_quintal) as avg_price,
                    MIN(min_price) as lowest_price,
                    MAX(max_price) as highest_price
                FROM market_prices
                GROUP BY crop_name, mandi_name, state, DATE(price_date)
                ORDER BY date DESC
            """,
            
            'active_schemes_summary': """
                CREATE OR REPLACE VIEW active_schemes_summary AS
                SELECT 
                    state,
                    COUNT(*) as total_schemes,
                    SUM(CASE WHEN scheme_type = 'Insurance' THEN 1 ELSE 0 END) as insurance_schemes,
                    SUM(CASE WHEN scheme_type = 'Direct Benefit' THEN 1 ELSE 0 END) as benefit_schemes,
                    AVG(subsidy_percentage) as avg_subsidy
                FROM government_schemes
                WHERE active = TRUE
                GROUP BY state
            """
        }
        
        for view_name, view_sql in views.items():
            try:
                self.cursor.execute(view_sql)
                logger.info(f"   ✅ {view_name}")
            except Error as e:
                logger.error(f"   ❌ {view_name}: {e}")
        
        self.conn.commit()
    
    def test_queries(self):
        """Run test queries"""
        logger.info("\n🧪 Running test queries...\n")
        
        # Test 1: Farmer summary
        self.cursor.execute("SELECT * FROM farmer_disease_summary")
        results = self.cursor.fetchall()
        print("📊 Farmer Disease Summary:")
        print("   " + "-"*70)
        for row in results:
            print(f"   {row[1]} ({row[2]}, {row[3]}): {row[4]} detections, {row[5]} unique diseases")
        
        # Test 2: Recent prices
        self.cursor.execute("""
            SELECT crop_name, mandi_name, price_per_quintal, price_date 
            FROM market_prices 
            WHERE price_date = (SELECT MAX(price_date) FROM market_prices)
            LIMIT 5
        """)
        results = self.cursor.fetchall()
        print(f"\n💰 Recent Market Prices:")
        print("   " + "-"*70)
        for row in results:
            print(f"   {row[0]} @ {row[1]}: ₹{row[2]:.2f}/quintal ({row[3]})")
        
        # Test 3: Active schemes
        self.cursor.execute("""
            SELECT scheme_name, state, benefits 
            FROM government_schemes 
            WHERE active = TRUE
        """)
        results = self.cursor.fetchall()
        print(f"\n📜 Active Government Schemes:")
        print("   " + "-"*70)
        for row in results:
            benefits = row[2][:60] + "..." if len(row[2]) > 60 else row[2]
            print(f"   {row[0]} ({row[1]})")
            print(f"   {benefits}")
        
        # Test 4: Database stats
        self.cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM farmers) as total_farmers,
                (SELECT COUNT(*) FROM crops) as total_crops,
                (SELECT COUNT(*) FROM disease_detections) as total_detections,
                (SELECT COUNT(*) FROM government_schemes WHERE active=TRUE) as active_schemes,
                (SELECT COUNT(*) FROM market_prices) as price_records
        """)
        stats = self.cursor.fetchone()
        print(f"\n📈 Database Statistics:")
        print("   " + "-"*70)
        print(f"   Total Farmers: {stats[0]}")
        print(f"   Total Crops: {stats[1]}")
        print(f"   Disease Detections: {stats[2]}")
        print(f"   Active Schemes: {stats[3]}")
        print(f"   Price Records: {stats[4]}")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("\n✅ Database connection closed")


def main():
    """Main setup pipeline"""
    print("\n" + "="*70)
    print("🌾 AgriPal RAG - MySQL Database Setup")
    print("="*70)
    
    # Validate configuration
    if not config.validate_config():
        return
    
    db = AgriMySQLSetup()
    
    try:
        # Create database
        logger.info("\n🗄️  Creating database...")
        db.create_database()
        
        # Connect
        logger.info("\n🔌 Connecting to database...")
        db.connect()
        
        # Ask user if they want to clear existing data
        print("\n" + "="*70)
        print("⚠️  DATA HANDLING OPTIONS")
        print("="*70)
        print("1. Keep existing data and update/insert new records (RECOMMENDED)")
        print("2. Clear all existing data and start fresh (DEVELOPMENT ONLY)")
        print("="*70)
        
        choice = input("Enter your choice (1 or 2) [default: 1]: ").strip() or "1"
        
        if choice == "2":
            confirm = input("⚠️  Are you sure? This will DELETE all data! (yes/no): ").strip().lower()
            if confirm == "yes":
                db.clear_sample_data()
            else:
                logger.info("   ℹ️  Skipping data clear")
        
        # Create tables
        db.create_tables()
        
        # Insert sample data
        db.insert_sample_data()
        
        # Create views
        db.create_views()
        
        # Test queries
        db.test_queries()
        
        print("\n" + "="*70)
        print("✅ MYSQL SETUP COMPLETE!")
        print("="*70)
        print(f"📍 Database: {config.MYSQL_CONFIG['database']}")
        print(f"🏠 Host: {config.MYSQL_CONFIG['host']}:{config.MYSQL_CONFIG['port']}")
        print("="*70)
        print("\n💡 Next Steps:")
        print("   1. Run your RAG application")
        print("   2. Connect to MySQL to view data: mysql -u root -p agripal_rag")
        print("   3. Check views: SELECT * FROM farmer_disease_summary;")
        print("="*70)
        
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()