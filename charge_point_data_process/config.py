#config.py
import os
from dotenv import load_dotenv
from geopy.geocoders import GoogleV3
from azure.storage.blob.aio import BlobServiceClient
import adlfs
import asyncio
import logging

# Load environment variables from .env file
load_dotenv()
###Configurations
# Azure Storage connection string
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Get API key from environment variable
GOOGLE_PLACES_API_KEY = os.getenv('GOOGLE_PLACES_API_KEY')

# Database parameters
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Connection pool for PostgreSQL
connection_params = {
    'host': DB_HOST,
    'port': DB_PORT,
    'dbname': DB_NAME,
    'user': DB_USER,
    'password': DB_PASSWORD
}

# Time interval for operations
INTERVAL =  30
freq = f'{INTERVAL}m' 

# Initialize Google Geocoding API
google_geocoder = GoogleV3(api_key=GOOGLE_PLACES_API_KEY)
# Initialize BlobServiceClient using the connection string from config.py
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
#fs = adlfs.AzureBlobFileSystem(connection_string=AZURE_STORAGE_CONNECTION_STRING, asynchronous = True) # Adjust authentication as needed
# Limit to 10 concurrent requests
semaphore = asyncio.Semaphore(8)

# In-memory cache (dictionary)
cache = {}

# connect to PostGres Reporting Database, query location data
query_location = """
                    SELECT DISTINCT
                        points.evse_id,
                        ports.evse_port_id,
                        operators.name AS cpo_name,
                        locations.address_street_1 AS address, 
                        locations.address_suburb_town AS suburb, 
                        locations.address_postcode AS postcode,
                        locations.lga_name AS lga_name,
                        locations.address_state AS state,
                        locations.latitude AS latitude,
                        locations.longitude AS longitude,
                        ports.power_kilowatts AS power
                    FROM     
                        charge_point_operators AS operators
                    INNER JOIN 
                        charge_locations AS locations  
                        ON operators.id=locations.cpo_id 
                    INNER JOIN charge_points AS points
                        ON locations.id=points.location_id
                    INNER JOIN charge_point_ports AS ports 
                        ON points.id = ports.charge_point_id;
                """
                #    WHERE locations.address_state IN %s
                
query_status = """  WITH RankedHistory AS (
                        SELECT
                            points.evse_id,
                            ports.evse_port_id,
                            operators.name AS cpo_name,
                            types.name AS status,
                            history.updated_timestamp AS update_time,
                            LAG(history.updated_timestamp) OVER (PARTITION BY points.evse_id, ports.evse_port_id ORDER BY history.updated_timestamp) AS prev_update_time
                        FROM     
                            charge_point_operators AS operators
                        INNER JOIN charge_locations AS locations  
                            ON operators.id=locations.cpo_id 
                        INNER JOIN charge_points AS points
                            ON locations.id=points.location_id
                        INNER JOIN charge_point_ports AS ports 
                            ON points.id = ports.charge_point_id
                        INNER JOIN port_status_history AS history
                            ON ports.id = history.port_id
                        INNER JOIN port_status_types AS types
                            ON history.status_type_id=types.id
                        ),
                        BoundaryIntervals AS (
                        SELECT 
                            evse_id,
                            evse_port_id,
                            MIN(prev_update_time) AS min_prev_update_time
                        FROM RankedHistory
                        WHERE 
                            update_time > (%s - INTERVAL '11 hour') AND  
                            update_time < (%s + INTERVAL '1 day' - INTERVAL '11 hour') AND
                            (evse_id, evse_port_id) IN %s
                        GROUP BY evse_id, evse_port_id
                        ),
                        FilteredHistory AS (
                        SELECT 
                            rh.*
                        FROM 
                            RankedHistory AS rh
                        INNER JOIN 
                            BoundaryIntervals AS bi
                            ON rh.evse_id = bi.evse_id AND rh.evse_port_id = bi.evse_port_id
                        WHERE 
                            rh.update_time BETWEEN bi.min_prev_update_time AND (%s + INTERVAL '1 day' - INTERVAL '11 hour')                                        
                         )
                        SELECT
                            evse_id,
                            evse_port_id,
                            cpo_name,
                            status,
                            update_time
                        FROM FilteredHistory
                        ORDER BY
                            cpo_name, update_time;
                """
query_offline = """
                    SELECT 
                        operators.name AS cpo_name,
                        offline.offline_timestamp,
                        offline.online_timestamp 
                    FROM     
                        charge_point_operators AS operators
                    INNER JOIN charge_point_operators_offline AS offline
                        ON operators.id=offline.cpo_id  
                    WHERE 
                        offline.offline_timestamp > (%s - INTERVAL '11 hour') AND  
                        offline.offline_timestamp < (%s + INTERVAL '1 day' - INTERVAL '11 hour') 
                    ORDER BY
                        offline_timestamp                                       
            """
state_postcodes={
    'NSW':[x for x in range(2000,2600)]+[x for x in range(2619,2900)]+[x for x in range(2921,3000)],
    'ACT':[x for x in range(2600,2619)]+[x for x in range(2900,2921)],
    'VIC':[x for x in range(3000,3997)],
    'QLD':[x for x in range(4000,5000)],
    'SA':[x for x in range(5000,5800)],
    'WA':[x for x in range(6000,6798)],
    'TAS':[x for x in range(7000,7800)],
    'NT':[x for x in range(800,900)]
    }

# storage local or cloud
local = False

logger = logging.getLogger('sql_process')
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(level=logging.DEBUG)
formatter =  logging.Formatter('%(levelname)s : %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)
