# config.py
from pathlib import Path
from datetime import datetime,timedelta
import pytz
import os
import faicons as fa
import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

#app_dir = Path(__file__).resolve().parent
# Load environment variables from .env file
load_dotenv( Path(__file__).resolve().parent / ".env")
# Azure Storage connection string
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# Azure Acount key
AccountKey = os.getenv("AccountKey")
# Initialize BlobServiceClient using the connection string from config.py
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
#Current month
month = (datetime.now().replace(day=1) - timedelta(days=1)).strftime('%B')

# Period select from date range
# Convert to string format required by input_date_range
date_range_start = datetime(year = 2025, month = 3, day = 1).strftime("%Y-%m-%d")
date_range_end = datetime.now().strftime("%Y-%m-%d")

month_to_number = {
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'may': 5,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12
}

USER1 = os.getenv("USER1")
USER2 = os.getenv("USER2")

user_dict = {
    USER1:'WA',
    USER2:'NSW'}

#colour mix ratio
colour_mix = 0.3
# variable labels
var_labels = {
    "postcode": "Postcode",
    "suburb_name": "Suburb Name",
    "Out of order_label": "Out of Order ",
    "Utilisation_label":"Utilisation ",
    "Uptime_label":"Uptime ",
    "Unavailability_label":"Unavailability ",
    "Charging_label":"Charging ",
    "Finishing_label":"Finishing ",
    "Reserved_label":"Reserved ",
    "Available_label":"Available ",
    "Unavailable_label":"Unavailable ",
    "Missing_label":"Missing ",
    "Unknown_label":"Unknown ",
    "evse_port_count": "Charge point count",
}

# inverse status labels
inv_var_labels = {v: k for k, v in var_labels.items()}
# Create the choropleth map
state_capitals = {'NSW': {'lat': -33.8688, 'lon': 151.2093},
                  'ACT': {'lat': -35.2802, 'lon': 149.1310},
                  'TAS': {'lat':-42.8826, 'lon': 147.3257},
                  'VIC': {'lat':-37.8136, 'lon':144.9631},
                  'SA': {'lat':-34.9285, 'lon':138.6007},
                  'NT': {'lat':-12.4637, 'lon':130.8444},
                  'WA': {'lat':-31.9514, 'lon':115.8617},
                  'QLD': {'lat':-27.4705, 'lon':153.0260}} 

# Define time zones
timezone_mappings = {
    "NSW": pytz.timezone("Australia/Sydney"),  # AEST (Australia Eastern Standard Time, adjusts for daylight savings)
    "VIC":pytz.timezone("Australia/Melbourne"),
    "WA": pytz.timezone("Australia/Perth"),    # AWST (Australia Western Standard Time, no daylight savings)
    "SA": pytz.timezone("Australia/Adelaide"),
    "ACT": pytz.timezone("Australia/Canberra"),
    "QLD": pytz.timezone("Australia/Brisbane"),
    "TAS": pytz.timezone("Australia/Hobart"),
    "NT": pytz.timezone("Australia/Darwin")
}

# Define timezone mappings
timezone_mappings2 = {
    "NSW": "Australia/Sydney",
    "VIC": "Australia/Melbourne",
    "QLD": "Australia/Brisbane",
    "SA": "Australia/Adelaide",
    "WA": "Australia/Perth",
    "TAS": "Australia/Hobart",
    "NT": "Australia/Darwin",
    "ACT": "Australia/Sydney",
}

#Define a list of style configurations to apply in a loop
cpo_styles = {
  "Exploren":{'fill': "#6229C3",
              'text': '#FFC400',
              'icon': 'exploren.png'},   # Apply to specific columns later if needed
  "ChargeN'Go": {'fill':"#BC160B",
                 'text':'#FFFFFF',
                 'icon':'chargengo.png'},
  "Evie Networks":{'fill':"#028EA9",
                  'text':'#FFFFFF',
                   'icon':'evienetworks.png'},
  "Wevolt":{'fill':"#091F40",
            'text':'#89DC65',
            'icon':"wevolt.png"},
  "Charge Post":{'fill':"#FFFFFF",
            'text':'#19405C',
            'icon':"chargepost.png"},
  "Sonic Charge":{'fill':"#FFFFFF",
            'text':'#03c1c1',
            'icon':"soniccharge.png"},
  "CasaCharge":{'fill':"#32a524",
            'text':'#FFFFFF',
            'icon':"casacharge.png"},
  "Charge Hub":{'fill':'#FFFFFF',
            'text':'#4cd538',
            'icon':"chargehub.png"},
  "All CPOs":{'fill':'#4E95D9',
             'text':'#FFFFFF',
             'icon':"all_cpos.png"},
  "Overall":{'fill':'#FFFFFF',
             'text':'#000000',
             'icon':"overall.png"}
            }

cpo_anonymity = {
  "Exploren": False,   
  "ChargeN'Go": False,
  "Wevolt": False,
  "Charge Post": False,
  "Sonic Charge":False,
  "CasaCharge":False,
  "Charge Hub":False,
            }

# ICONS
icons = {
    "charger": fa.icon_svg("bolt", "solid"), 
    "ellipsis": fa.icon_svg("ellipsis"),
    "tooltip": fa.icon_svg("circle-info") 
}

# Convert the selectize input into a timedelta object
interval_mapping = {
    "60min": pd.Timedelta(minutes=60),
    "1440min": pd.Timedelta(days=1),
    "10080min": pd.Timedelta(weeks=1),
}

interval_options = {"60min": "Hourly",
                    "1440min":"Daily",
                    "10080min":"Weekly",
                    "ME":"Monthly",
                    "Q":"Quarterly",
                    "Y":"Annually"                            
                    }
# inverse interval options dictionary
interval_options_inverse = { v:k for k,v in interval_options.items()}

interval_options2 = {"60min": "hour",
                    "1440min":"day",
                    "10080min":"week",
                    "ME":"month",
                    "Q":"quarter",
                    "Y":"annum"                            
                    }

# Utilisation Status variables
utilisation_status = {
    'Metrics':{
     "Utilisation":"Utilisation",
     "Uptime":"Uptime",
     "Unavailability":"Unavailability"
     },
    'Statuses':{
        "Charging":"Charging",
        "Finishing":"Finishing",
        "Reserved":"Reserved",
        "Available":"Available",
        "Unavailable":"Unavailable",
        "Out of order":"Out of Order",
        "Unknown":"Unknown",
        "Missing":"Missing"
        }
    }
status_metric_vars =['Utilisation','Uptime','Unavailability','Charging','Finishing','Reserved','Available','Unavailable','Out of order','Unknown','Missing'] 
location_vars = ['cpo_name','address','suburb','postcode','latitude','longitude','evse_port_count']
funded_chargers = {"1": "Grant funded chargers",
                   "2": "Non-grant funded chargers",
                   "3": "All chargers"}
processing_speed=60000
LOCAL = os.getenv('LOCAL')

INTERVAL = 30
