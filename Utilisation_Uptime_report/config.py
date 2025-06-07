# config.py
from datetime import datetime,timedelta
import pytz
import faicons as fa
import pandas as pd

#Current month
month = (datetime.now().replace(day=1) - timedelta(days=1)).strftime('%B')
#colour mix ratio
colour_mix = 0.3
# variable labels
var_labels = {"cpo_name":"Charge point operators",
              "variable":'Status Types',
              "address1":'address',
              "address2":'suburb',
              "value":"Number of Chargepoints",
              "interval": "Observed Period",
              'Utilisation_prop':'Utilisation',
              'Charging_prop':'Charging',
              'Finishing_prop':'Finishing',
              'Reserved_prop':'Reserved',
              'out_of_order_prop':'Out of Order',
              'Out of order_prop':'Out of Order',
              'Available_prop':'Available',
              'Unavailable_prop':'Unavailable',
              'Unavailability_prop':'Unavailability',
              'Unknown_prop':'Unknown',
              'Missing_prop':'Missing',
              'Uptime_prop':'Uptime',
              'evse_port_site_count':"Number of Chargepoints"
              }

# Create the choropleth map
centroids = {'NSW': (-32.18411400724141, 147.03133362435045),
            'ACT': (-35.358507545252465, 149.0542630937404),
            'TAS': (-42.01736484286484, 146.59203874206906),
            'VIC': (-36.86343052563767, 144.32403629108708),
            'SA': (-30.166993250838594, 135.8519581102913),
            'NT': (-19.361408727457572, 133.36304560465538),
            'WA': (-32.72515110600491, 117.53822006487553),
            'SEQ': (-26.73908093151759, 151.13527845901),
            'QLD': (-22.035640275321153, 143.77089064612295)} 

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


#Define a list of style configurations to apply in a loop
cpo_styles = {
  "Exploren":{'fill': "#6229C3",
              'text': '#FFC400',
              'icon': 'exploren.png'},   # Apply to specific columns later if needed
  "ChargeN'Go": {'fill':"#BC160B",
                 'text':'#FFFFFF',
                 'icon':'chargengo.png'},
  "EVie Networks":{'fill':"#028EA9",
                  'text':'#FFFFFF',
                   'icon':'evienetworks.png'},
  "Wevolt":{'fill':"#091F40",
            'text':'#89DC65',
            'icon':"wevolt.png"},
  "Overall":{'fill':'#FFFFFF',
             'text':'#000000',
             'icon':"overall.png"}
            }

# ICONS
icons = {
    "charger": fa.icon_svg("bolt", "solid"), 
    "ellipsis": fa.icon_svg("ellipsis"),
    "tooltip": fa.icon_svg("circle-info") 
}

# Convert the selectize input into a timedelta object
interval_mapping = {
    "30min": pd.Timedelta(minutes=30),
    "60min": pd.Timedelta(minutes=60),
    "120min": pd.Timedelta(minutes=120),
    "240min": pd.Timedelta(minutes=240),
    "360min": pd.Timedelta(minutes=360),
    "720min": pd.Timedelta(minutes=720),
    "1440min": pd.Timedelta(days=1),
    "2880min": pd.Timedelta(days=2),
    "10080min": pd.Timedelta(weeks=1),
    "20160min": pd.Timedelta(weeks=2)
}

interval_options = {"60min": "Hourly",
                    "1440min":"Daily",
                    "10080min":"Weekly",
                    "ME":"Monthly",
                    "Q":"Quarterly",
                    "Y":"Annually"                            
                    }

interval_options2 = {"60min": "hour",
                    "1440min":"day",
                    "10080min":"week",
                    "ME":"month",
                    "Q":"quarter",
                    "Y":"annum"                            
                    }

