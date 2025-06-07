#utilities.py
#import module
import pandas as pd
import polars as pl
import geopandas as gpd
import numpy as np
import plotly.express as px
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
#import functions
from plotly import graph_objects as go
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from io import BytesIO
from config import * 

#selected state
def select_state(user_code,user_dict):
    state = user_dict.get(user_code)
    return state

# Helper function to generate SAS URL
def generate_image_url(image_file, container_name="data"):
    """
    Generates a SAS URL for a blob in Azure Blob Storage.

    Parameters:
    - container_name (str): The name of the Azure Blob Storage container.
    - blob_name (str): The name of the blob (image).
    - expiry_minutes (int): How long the SAS URL should be valid, in minutes.

    Returns:
    - str: A full SAS URL for the blob.
    """
    try:        # Get the blob client
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=f"img/{image_file}")
        # Set the expiry to 1 year from now
        expiry_time = datetime.now() + timedelta(days=1)
        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=f"img/{image_file}",
            account_key=AccountKey,
            permission=BlobSasPermissions(read=True),  # Allow read access
            expiry=expiry_time
        )
        # Combine the blob URL with the SAS token
        img_path = f"{blob_client.url}?{sas_token}"
        return img_path
    except Exception as e:
                print(f"Error downloading img {image_file}: {e}")
                return None  # Return an empty DataFrame or handle as needed

# list of months for ui input values    
def generate_month_dates(df: pl.DataFrame, datetime_col: str) -> list[datetime]:
    """
    Generates a list of datetime objects for the start of each month
    within the range of years found in the specified datetime column.

    Args:
        df (pl.DataFrame): Polars DataFrame with a datetime column.
        datetime_col (str): The name of the datetime column.

    Returns:
        list[datetime]: List of datetime objects for the start of each month.
    """
    # Ensure datetime_col is in datetime format
    df = df.with_columns(pl.col(datetime_col).cast(pl.Datetime))

    # Extract year and month bounds
    min_year = df.select(pl.col(datetime_col).dt.year().min()).item()
    max_year = df.select(pl.col(datetime_col).dt.year().max()).item()

    # Start from Jan of the min year to Jan of the year after the max
    start_date = datetime(min_year, 1, 1)
    end_date = datetime(max_year + 1, 1, 1)

    # Use Pandas to generate month starts (Polars lacks `date_range(freq='MS')`)
    month_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    return month_dates.to_list()


def convert_dataframe_timezone(df: pl.DataFrame, datetime_col: str, state: str) -> pl.DataFrame:
    """
     Converts a Polars DataFrame datetime column from UTC to the target timezone based on the provided state.
    
    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        datetime_col (str): The name of the datetime column in the DataFrame.
        state_col (str): The name of the state column in the DataFrame.

    Returns:
        pl.DataFrame: Polars DataFrame with the datetime column converted to the target timezone.
    """
    # Ensure datetime_column is in datetime format with UTC timezone
    df = df.with_columns(pl.col(datetime_col).dt.replace_time_zone("UTC"))
    # Get the corresponding timezone
    tz = timezone_mappings2.get(state, "UTC")
    # Apply timezone conversion in bulk
    df = df.with_columns(pl.col(datetime_col).dt.convert_time_zone(tz))
    df = df.with_columns(pl.col(datetime_col).dt.replace_time_zone(None))  # Removes timezone info

    return df

def clean_string(value):
    """
    Removes non-permitted characters from a string.
    Only allows letters, numbers, and underscores in the value.
    
    Parameters:
    value (str): The string to clean.
    
    Returns:
    str: The cleaned string.
    """
    if not isinstance(value, str):
        return "unknown"  # Fallback for non-string or None values
    # Define regex pattern to keep only letters, numbers, and underscores
    permitted_pattern = re.compile(r'[^a-zA-Z0-9_]')
    # Apply the regex to remove non-permitted characters
    cleaned_value = re.sub(permitted_pattern, '', value)
    return cleaned_value

# Ensure we use an executor for synchronous operations
executor = ThreadPoolExecutor()

async def load_and_prepare_data(selected_state, start_date, end_date,p,container_name="data"):
    """
    Loads and prepares data files stored in Azure Blob Storage or locally.

    Parameters:
        selected_state (str): The state identifier (e.g., 'wa', 'ny').
        start_date (str): The start date for filtering files, in 'YYYY-MM-DD' format.
        end_date (str): The end date for filtering files, in 'YYYY-MM-DD' format.
        local_env (bool, optional): Flag indicating if the function should run in a local environment. Default is False.
        container_name (str, optional): Name of the Azure Blob Storage container. Default is 'data'.

    Returns:
        tuple: A tuple containing:
            - months_data (pd.DataFrame): Combined data from selected files.
            - lga_geogcoord_dict (dict): Dictionary of LGA geographic coordinates.
            - poa_suburb (dict): Dictionary mapping postcodes to suburbs.
            - geodf_filter_lga (gpd.GeoDataFrame): Filtered GeoJSON data for LGAs.
            - geodf_filter_poa (gpd.GeoDataFrame): Filtered GeoJSON data for POAs.
    """
    state_file_name = selected_state.lower() + "_"
    
    ##### Cloud Environment
    def get_blob_size(blob_name):
        try:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            # Get blob properties
            blob_properties = blob_client.get_blob_properties()
            # Retrieve blob size
            blob_size = blob_properties.size  # Size in bytes
            return blob_size
        except Exception as e:
            print(f"Error reading blob {blob_name} properties: {e}")
            return None  # Return an empty DataFrame or handle as needed
            
    # Function to download and read a blob file as a DataFrame
    def download_blob_to_dataframe(blob_name, file_type):
        try:
            
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            file = BytesIO(blob_client.download_blob().readall())
            # Conditional file type    
            if file_type == 'json':
                return gpd.read_file(file)
            elif file_type == 'csv':
                return pd.read_csv(file)
            elif file_type == 'parquet':    
                return pl.read_parquet(file)
            else:
                print(f"Unsupported file type: {file_type}")
                return pl.DataFrame()
        except Exception as e:
            print(f"Error downloading blob {blob_name}: {e}")
            return pl.DataFrame()  # Return an empty DataFrame or handle as needed
        
    # Download CSV and GeoJSON files
    p.set(2,message=" Please wait...", detail="loading LGA data from blob storage")
    geodf_filter_lga = download_blob_to_dataframe("geodf_filter_lga.json", 'json').set_index('LGA_name')
    p.set(3,message=" Please wait...", detail="loading postcode area data from blob storage")
    geodf_filter_poa = download_blob_to_dataframe("geodf_filter_poa.json", 'json').set_index('postcode')
    p.set(4,message=" Please wait...", detail="loading LGA coordinates data from blob storage")
    lga_geogcoord_df = download_blob_to_dataframe("geogcoord_lga_df.csv", 'csv')
    
    # List all blobs in the container and filter for month data files
    blobs = blob_service_client.get_container_client(container_name).list_blobs()
    p.set(5,message=" Please wait...", detail="finding monthly charge point status data in blob storage")
    # Collect list of month data files from blob storage
    month_data_blobs = []
    for blob in blobs:
        blob_name = blob.name
        if state_file_name in blob_name:
            try:
                # Extract year and month from the blob name          
                parts = blob_name.split("_")
                year = int(parts[1])
                month_str = parts[2].lower()
                month = month_to_number.get(month_str)
                blob_date = datetime(year, month, 1) #tzinfo = start_date.tzinfo)  # Parse the selected start date
                # Check if blob_date is within the range
                if start_date.replace(day = 1) <= blob_date <= end_date.replace(day = 1):
                    month_data_blobs.append(blob_name)
            except (ValueError, IndexError):
                continue  # Skip blobs that do not match the naming pattern
    # Load and concatenate all month data
    # Initialize progress bar
    num_blobs = len(month_data_blobs)
    progress_per_blob = (83 - len(cpo_anonymity)) // num_blobs
    
    p.set(6,message=" Please wait...", detail="loading monthly charge point status data from storage")
    data_frames = []
    
    for i, blob_name in enumerate(month_data_blobs):
        blob_size = get_blob_size(blob_name)
        # Calculate duration per step based on blob size
        duration = (blob_size / processing_speed ) / 100  # Divide total time by 100 steps
        # Update progress bar incrementally
        async def update_progress():
            for step in range(100):  # Simulate smaller steps for each blob
                # Extract the file name without extension
                base_name, _ = os.path.splitext(blob_name)  # Remove the extension
                p.set(
                    6 + i * progress_per_blob + step * (progress_per_blob / 100),
                    message="Please wait...",
                    detail=f"downloading and processing file: {base_name} ({step}%)"
                )
                await asyncio.sleep(duration)  # Simulate smaller progress steps
        # Run download and progress update concurrently
        download_task = asyncio.get_event_loop().run_in_executor(
            executor,
            download_blob_to_dataframe,
            blob_name,
            'parquet')
        await update_progress()
        # Wait for both to finish
        data_frame = await download_task
                   
        data_frames.append(data_frame)
    # Concatenate all the data synchronously
    p.set(90 - len(cpo_anonymity), message="Please wait...", detail="Combining data frames")
    if len(data_frames) > 1:
        months_data = pl.concat(data_frames, how="vertical")
    else:
        months_data = data_frames[0]   
    #rename columns
    months_data = months_data.rename({'update_time':'interval',
                                                'status':'variable',
                                                'status_duration':'value'})
    # Convert to Pandas
    #months_data = months_data.to_pandas(use_pyarrow_extension_array=True)
    # Apply timezone conversion
    #print(months_data.head(10))
    p.set(91 - len(cpo_anonymity),message=" Please wait...", detail="Converting date time to local timezone")
    months_data = convert_dataframe_timezone(months_data, 'interval', selected_state)
    #months_data.to_csv('month_data_example.csv')
    # LGA and lon lat coords
    lga_geogcoord_dict = {
    row['lga_name']: {  # Outer dictionary key from col1
        'lat': row['lat'],  # Inner dictionary key from col2
        'lon': row['lon']   # Inner dictionary key from col3
    }
    for _, row in lga_geogcoord_df.iterrows()
    }
    # Remove duplicates and create dictionary
    unique_df = months_data.select(["postcode", "suburb"]).unique()
    poa_suburb = dict(zip(unique_df["postcode"].to_list(), unique_df["suburb"].to_list()))
    p.set(92 - len(cpo_anonymity),message=" Please wait...", detail="All data loaded")
    return months_data, lga_geogcoord_dict, poa_suburb, geodf_filter_lga, geodf_filter_poa

### 08/05/25 - Addition to meet compliance with DSA
# Filter out LGAs that only have a single CPO providing public Charging to maintain anonymity
# CPO Anonymity function
def anonymise_cpos1(filtered_data_set: pl.DataFrame) -> pl.DataFrame:
    # Step 1: Filter for non-funded sites
    cpo_non_funded_sites = filtered_data_set.filter(
        pl.col("government_funded") == "not_funded"
    )
    # Step 2: Count unique CPOs per LGA
    cpo_lga_counts = (
        cpo_non_funded_sites
        .group_by("lga_name")
        .agg(pl.col("cpo_name").n_unique().alias("cpo_count"))
    )
    # Step 3: Filter LGA names where cpo_count > 1
    anonymous_cpos_lga_list = (
        cpo_lga_counts
        .filter(pl.col("cpo_count") > 1)
        .select("lga_name")
        .to_series()
        .to_list()
    )
    # Step 4: Keep rows that are:
    # - either in one of those LGA names with > 1 CPO
    # - OR not government-funded == 'not_funded'
    filtered_data_set2 = filtered_data_set.filter(
        (pl.col("lga_name").is_in(anonymous_cpos_lga_list)) |
        (pl.col("government_funded") != "not_funded")
    )
    return filtered_data_set2

def anonymise_cpos_in_on_demand_df(filtered_data_set2,p):
    cpo_counter = 1  # To assign CPO 1, CPO 2, ...
    for num,(cpo,should_anonymise)  in enumerate(cpo_anonymity.items()):
        mask_cpo = filtered_data_set2['cpo_name'] == cpo
        unique_locations = filtered_data_set2.loc[mask_cpo,'location'].unique()
        location_map = {loc: loc for loc in unique_locations}
        # Conditionally anonymise CPO name and location
        if should_anonymise :
            location_map = {
                loc: f'Location {i+1}' for i, loc in enumerate(unique_locations)
            }
            # Replace location names
            filtered_data_set2.loc[mask_cpo, 'location'] = (
                filtered_data_set2.loc[mask_cpo, 'location'].map(location_map)
            )
            # Replace CPO name
            filtered_data_set2.loc[mask_cpo, 'cpo_name'] = f'CPO {cpo_counter}'
            cpo_counter += 1  
         # Always anonymise evse_id and evse_port_id per location
        for loc in unique_locations:
            mask_loc = (
                (filtered_data_set2.cpo_name == (f'CPO {cpo_counter - 1}' if should_anonymise else cpo)) & 
                (filtered_data_set2.location == (location_map[loc] if should_anonymise else loc))
            )
            # Anonymise Charge Station (evse_id)
            unique_stations = filtered_data_set2.loc[mask_loc,'evse_id'].unique()
            station_map = {station: f'station {j+1}' for j, station in enumerate(unique_stations)}
            filtered_data_set2.loc[mask_loc, 'charge station'] = (
                filtered_data_set2.loc[mask_loc, 'evse_id'].map(station_map)
            )
            
            # Anonymise Connector per (cpo_name, location, evse_id)
            for station in unique_stations:
                mask_evse = mask_loc & (filtered_data_set2['evse_id'] == station)
                unique_connectors = filtered_data_set2.loc[mask_evse, 'evse_port_id'].unique()
                connector_map = {conn: f'Port {k+1}' for k, conn in enumerate(unique_connectors)}
                filtered_data_set2.loc[mask_evse, 'connector'] = (
                    filtered_data_set2.loc[mask_evse, 'evse_port_id'].map(connector_map)
                )
        p.set(97 - (len(cpo_anonymity) - (num + 1)),message=" Please wait...", detail=f"Re-labelled {cpo} stations and plugs")            
    return filtered_data_set2           

def prepare_on_demand_data(df,p):
    # Filter data to match desired output
    mask = (
        (df.government_funded != 'not_funded') &
        (df.variable == 'Charging') &
        (df.value > 0))
    df_filtered = df.loc[mask,:].copy()
    p.set(96 - len(cpo_anonymity),message=" Please wait...", detail="filtered for government funded evse, charging sessions")
    #Feature engineer
    # define the location column
    df_filtered['location'] = df_filtered['address']+", "+df_filtered['suburb']+", "+df_filtered['postcode']
    # Sort by location and datetime to prepare for grouping
    #print(df_filtered.columns)
    df_filtered = df_filtered.sort_values(by=['cpo_name', 'location', 'evse_id', 'evse_port_id', 'interval'])
    # Identify gaps larger than 1 hour as new sessions
    df_filtered['time_diff'] = df_filtered.groupby(['cpo_name', 'location', 'evse_id', 'evse_port_id'])['interval'].diff()
    df_filtered['new_session'] = (df_filtered['time_diff'] > pd.Timedelta(minutes = INTERVAL)).cumsum()
    # Aggregate sessions
    session_data = (
        df_filtered.groupby(['cpo_name', 'location', 'evse_id', 'evse_port_id','new_session','power','government_funded'])[['interval','value']]
        .agg(session_start_interval=('interval', 'min'),
             charge_duration=('value', 'sum'))
        .reset_index()
    )
    p.set(97 - len(cpo_anonymity),message=" Please wait...", detail="Identify and aggregate along charging sessions")
    #Ensure 'interval' column is of datetime datatype and convert it to string with timezone
    session_data['Date'] = pd.to_datetime(session_data['session_start_interval'], errors='coerce').dt.strftime('%Y-%m-%d')
    #Ensure 'interval' column is of datetime datatype and convert it to string with timezone
    session_data['Start interval'] = pd.to_datetime(session_data['session_start_interval'], errors='coerce').dt.strftime('%H:%M:%S%z')              
    #df_filtered.to_csv('test_data.csv')
    # anonymise locations and cpos if required (cpo didn't give permission to show charge points) 
    session_data = anonymise_cpos_in_on_demand_df(session_data,p)
    p.set(97,message=" Please wait...", detail="Identify and aggregate along charging sessions")
    session_data = session_data[['Start interval','Date','cpo_name','location','charge station','connector','charge_duration','power','government_funded']]
    session_data['max usage (kWh)'] = np.round(session_data['power']  *  (session_data['charge_duration']/(60*60))) 
    session_data['charge_duration (min)'] = np.round(session_data['charge_duration']/60,1)
    session_data = session_data.drop(columns=['power','charge_duration'])     
    return session_data
 
 ### evse - level aggregattion ###
def evse_level_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    # Define grouping columns
    grouping_var = [
        'cpo_name', 'address', 'suburb', 'postcode', 'lga_name',
        'latitude', 'longitude', 'power', 'government_funded', 'evse_id', 'interval'
    ]
    # Step 1: Compute unique evse_port_id count per group
    evse_port_count = (
        df.groupby(grouping_var)
        .agg({'evse_port_id': pd.Series.nunique})
        .rename(columns={'evse_port_id': 'value'})
        .reset_index()
    )
    evse_port_count['variable'] = 'evse_port_count'
    #logger.debug("Step 3 processing: Aggregate at site level") 
    # Step 2: Aggregate 'value' per group and variable
    evse_interval_aggregation_df = (
        df.groupby(grouping_var + ['variable'])['value']
        .sum()
        .reset_index()
    )
    # Step 3: Concatenate both DataFrames
    evse_aggregate_df = pd.concat([evse_interval_aggregation_df, evse_port_count], ignore_index=True)
    # Step 4: Sort by grouping columns and variable
    evse_aggregate_df = evse_aggregate_df.sort_values(by=grouping_var + ['variable']).reset_index(drop=True)
    return evse_aggregate_df    
           
# uptime and utilisation metrics calculation function
def uptime_utilisation_unavailability(df1: pd.DataFrame):
    # Engineer variables - combined columns
    
    df1['Utilisation'] = (
        df1.get('Charging', 0) +
        df1.get('Finishing', 0) +
        df1.get('Reserved', 0)
    )
    df1['Unavailability'] = (
        df1.get('Unavailable', 0) +
        df1.get('Out of order', 0) +
        df1.get('Inoperative', 0) +
        df1.get('Blocked', 0)
    )
    df1['Uptime'] = (
        df1.get('Available', 0) + df1.get('Utilisation',0)
    )
    metric_cols = ['Utilisation', 'Unavailability', 'Uptime']
    print("Computed metrics: Utilisation, Unavailability, and Uptime.")
    return df1, metric_cols

# weighted mean
def weighted_mean(df1):
    
    return None
    
    


 
# geograph statistics function
def geography_statistics_data(filtered_data_set: pd.DataFrame,
                         filtered_data_set2: pd.DataFrame,
                         plugshare_df: pd.DataFrame) -> dict:
    """
    Gives summary statistics on the geographics coverage of charge points represented and reported on via Charge@Large. Forms comparison to population estimate
    from Plugshare data.
    Args:
        filtered_data_set (pd.DataFrame): filtered dataset based on time interval only, showing all available LGAs
        filtered_data_set2 (pd.DataFrame): filtered dataset based on time interval and LGAs with more than 1 CPO with non-funded EVSE or any funded EVSE,
        reporting LGAs that maintains CPO anonymity when aggreggated.
        plugshare_df (pd.DataFrame): Full EVSE population estimates across LGAs, Postcodes and Suburbs based on Plugshare dataset. 
    Returns:
        geography_statistics_df (pd.DataFrame): dataframe of summary statistics of geographic coverage of EVSE that are reported, and compared to filtered lGAs
        and whole LGA population estimate.
    """
    geo_stats_dict = {}
    # Sets of LGAs
    reporting_lga_set = set(filtered_data_set2.lga_name.unique())
    non_reporting_lga_set = set(filtered_data_set.lga_name.unique()).difference(reporting_lga_set)     
    lga_pop_set = set(plugshare_df.lga_name.unique())
    # proportions
    # proportion of LGAs being reported on based on anonymity provisions LGAs with more than 1 CPO provider.
    reporting_prop = len(reporting_lga_set)/len(non_reporting_lga_set)
    # proportion of LGAs captured by Charge@Large App out of EVSE population estimate
    app_coverage_prop = len(set(filtered_data_set.lga_name.unique()))/len(lga_pop_set)
    # non-reporting LGAs dataframe
    filtered_data_set.loc[~(filtered_data_set.lga_name.isin(reporting_lga_set)),:]
    return None
    
    


# Process Dataframe based on filters
def process_data(df: pd.DataFrame,
                 agg_cols: list,
                 interval_option: object,
                 time_col = 'interval',
                 var_col = 'variable') -> pd.DataFrame:
    """
    Process the input DataFrame by aggregating data based on the specified columns and interval option.
    Defines the columns to aggregate and the interval option to resample the data.
    
    Args:
        df (pd.DataFrame): _description_
        agg_cols (list): _description_
        interval_option (object): _description_
        time_col (str, optional): _description_. Defaults to 'interval'.
        status_col (str, optional): _description_. Defaults to 'variable'.

    Returns:
        pd.DataFrame: _description_
    """
    try:
        # Validate inputs
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        agg_statuses = [col for col in df[var_col].unique() if col != 'evse_port_count']  
        # aggregate data on agg list cols
        df_agg  = (
            df.
            groupby(agg_cols+[time_col,var_col])['value']
            .sum()
            .reset_index()
        )
        # list indexing cols
        index_cols = [col for col in df_agg.columns if col not in ['value',var_col]]
        # Convert summary data to wide form       
        df_wide = df_agg.pivot(index = index_cols, columns = var_col, values = 'value').reset_index()
        
        df_wide,metric_cols = uptime_utilisation_unavailability(df_wide)
        #### Aggregate over different intervals
        # Resample data based on the specified interval
        df_wide[time_col] = pd.to_datetime(df_wide[time_col], errors='coerce')
        # if government funded - no cpo_name field to aggregate on,
        if not agg_cols:
            df_resampled = (
            df_wide
            .set_index(time_col)
            .groupby(['evse_port_count'])[agg_statuses+metric_cols]
            .resample(interval_option)
            .sum()
            .reset_index()
            )
        else:
            df_resampled = (
                df_wide
                .set_index(time_col)
                .groupby(agg_cols + ['evse_port_count'])[agg_statuses+metric_cols]
                .resample(interval_option)
                .sum()
                .reset_index()
            )
        print("Resampled data based on interval.") 
        # Exclude 'Total' from the list of columns to process
        columns_to_process = [col for col in metric_cols+agg_statuses if col != 'Total'] 
        # Form proportion of Total column
        for col in columns_to_process:
            temp_col = f"{col}_temp"  # Temporary column name with '_temp' suffix
            if col in metric_cols:
                # Calculate metric column proportions excluding Missing and Unknown
                df_resampled[temp_col] = df_resampled[col]*100 / (df_resampled['Total'] - df_resampled['Missing'] - df_resampled['Unknown'])
            else:
                df_resampled[temp_col] = df_resampled[col]*100 / df_resampled['Total']
        print("Metric calculated.")
        # Rename temporary columns to the original column names
        for col in columns_to_process:
            temp_col = f"{col}_temp"
            if temp_col in df_resampled.columns:
                # Remove the original column
                df_resampled.drop(columns=[col], inplace=True, errors='ignore')
                # Rename the temp column to the original column name
                df_resampled.rename(columns={temp_col: col}, inplace=True)
        
        # Drop unnecessary columns
        processed_data = df_resampled.drop(columns=['Total'], errors='ignore')
        print('Calculated proportions') 
        return processed_data  
    except Exception as e:
        print(f"Error in process_data: {e}")
        raise ValueError(f"Error in process_data: {e}")
    
# Custom function to calculate standard error
def standard_error(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))

def calculate_cpo_statistics(data, cpo, selected_dates):
        """
        Calculate key statistics for a specific Charge Point Operator (CPO).
        
        Args:
            data (DataFrame): The dataset containing CPO data.
            cpo (str): The name of the CPO to calculate statistics for.
            selected_dates (tuple): A tuple containing the start and end dates for the analysis period.

        Returns:
            dict: A dictionary containing the calculated statistics.
        """
        try:
            # Filter data for the selected CPO
            cpo_data = data.loc[data.cpo_name == cpo]
            
            # Calculate statistics
            average_uptime = cpo_data['Uptime'].mean()
            minimum_uptime = cpo_data['Uptime'].min()
            average_utilisation = cpo_data['Utilisation'].mean()
            max_utilisation = cpo_data['Utilisation'].max()
            maximum_unavailability = cpo_data['Unavailability'].max()
            average_unavailability = cpo_data['Unavailability'].mean()
            evse_count = cpo_data['evse_port_count'].max()  # Assuming max as total count
            # Calculate period duration
            period_duration = (selected_dates[1] - selected_dates[0]).days          
            # Check if the data contains only a single record
            if len(cpo_data) == 1:
                print(f"Single record detected for CPO: {cpo}")
                # Use a default interval assumption, e.g., 1 month (1440 minutes)
                 # Get the month and year from the single record
                single_date = cpo_data['interval'].iloc[0]
                period = pd.Period(single_date, freq='M')  # Convert to a Period object with monthly frequency
                # Calculate the number of days in the month using pandas.Period
                days_in_month = period.days_in_month
                # Calculate interval minutes dynamically based on the month
                interval_minutes = days_in_month * 1440  # 1440 minutes in a day
                total_observed_time = interval_minutes  # Observation covers one month
                missing_duration = cpo_data['Missing'].iloc[0]  # Use the single record's missing value directly
            else:
                # Calculate interval minutes and total observed time
                cpo_data = cpo_data.copy()  # Avoid modifying the original dataframe
                #print(f"Length of CPO data for {cpo}: {len(cpo_data)} records")
                cpo_data['interval_minutes'] = cpo_data['interval'].diff().dt.total_seconds() / 60
                cpo_data['interval_minutes'] = cpo_data['interval_minutes'].bfill()  # Fill NaN with backfill
                #print(f'interval minutes is calculated to be: {cpo_data['interval_minutes']}')
                total_observed_time = (
                    cpo_data['interval'].max() - cpo_data['interval'].min()
                ).total_seconds() / 60  # Convert to minutes

                # Calculate missing duration as a percentage of total observed time
                missing_duration = (
                    cpo_data[['Missing', 'interval_minutes']]
                    .eval('(Missing / 100) * interval_minutes')
                    .sum() / total_observed_time
                ) * 100

            # Debugging logs (optional)
            #print(f"Total observed time for {cpo}: {total_observed_time:.2f} minutes")
            #print(f"Missing duration for {cpo}: {missing_duration:.2f}%")

            # Return the calculated statistics as a dictionary
            return {
                "average_uptime": average_uptime,
                "minimum_uptime": minimum_uptime,
                "average_utilisation": average_utilisation,
                "max_utilisation": max_utilisation,
                "maximum_unavailability": maximum_unavailability,
                "average_unavailability": average_unavailability,
                "charge_station_count": evse_count,
                "period_duration": period_duration,
                "total_observed_time": total_observed_time,
                "missing_duration": missing_duration,
            }
        except Exception as e:
            print(f"Error calculating statistics for {cpo}: {e}")
            return {
                "average_uptime": None,
                "minimum_uptime": None,
                "average_utilisation": None,
                "max_utilisation": None,
                "maximum_unavailability": None,
                "average_unavailability": None,
                "charge_station_count": None,
                "period_duration": None,
                "total_observed_time": None,
                "missing_duration": None,
            }
        
#Prepare average plot data
def prepare_plot_average_data(data,group_vars):
    try:
        plot_data = (data
                    .groupby(group_vars)[status_metric_vars]
                    .agg(['mean',standard_error]))
        
        plot_data.columns = ['_'.join(col).strip() for col in plot_data.columns.values]
        for col in status_metric_vars:
            plot_data[f'{col}_label'] = (
                plot_data[f'{col}_mean'].round(2).astype(str)
            + ' ± '
            + plot_data[f'{col}_standard_error'].round(2).astype(str)
        )  
        # Rename {col}_mean to just {col}
        renaming_dict = {f'{col}_mean': col for col in status_metric_vars}
        plot_data = plot_data.rename(columns=renaming_dict)   
        # Drop {col}_standard_error columns
        drop_columns = [f'{col}_standard_error' for col in status_metric_vars]
        plot_data = plot_data.drop(columns=drop_columns) 
        plot_data = plot_data.reset_index()
        print('plot data prepared')
        return plot_data
    except Exception as e:
        print(f"Error in prepare_plot_average_data: {e}")
        raise

#helper function to plot chloropleth map
def plot_choropleth_map(df1: pd.DataFrame,
                         df2: pd.DataFrame,
                         geo_df:pd.DataFrame,
                         #lga_geogcoord_dict: dict,
                         status_prop: str,
                         gov_funded: int,
                         lga_name: str,
                         poa_suburb: dict,
                         state: str,
                         hover_data1: dict,
                         hover_data2: dict
                         ) -> go.Figure:
    """
    Plots a choropleth map with a scatter plot overlay for the specified status property.
    The choropleth map displays the status property for each postcode, while the scatter plot
    shows the location of each charging station. The color scale is based on the status property.
    Args:
        df (pd.DataFrame): 
        geo_df (pd.Dataframe): _description_
        lga_geogcoord_dict (dict): _description_
        status_prop (str): _description_
    Returns:
        object: _description_
    """
    try:
        # Add 'suburb_name' column by mapping 'postcode' to 'poa_suburb'
        df1.loc[:,'suburb_name'] = df1['postcode'].map(poa_suburb).fillna("Unknown Suburb")      
        color_scale = {"Utilisation": "Greens",
                        "Uptime": "Blues", 
                        "Unavailability": "Reds"                       
                        }.get(status_prop, "Viridis")
        range_color = (0, df1[status_prop].max())
        
        # Find the correct status_prop within utilisation_status
        if status_prop in utilisation_status['Metrics']:
            status_label = utilisation_status['Metrics'][status_prop]
        elif status_prop in utilisation_status['Statuses']:
            status_label = utilisation_status['Statuses'][status_prop]
        else:
            status_label = status_prop  # Default to the raw value if not found
        
        center_coord = {  
        'lat': df2["latitude"].iloc[0],  
        'lon': df2["longitude"].iloc[0]   
            }
        map_fig = px.choropleth_mapbox(
                df1,
                geojson = geo_df.geometry,
                locations = 'postcode',
                color = status_prop,
                color_continuous_scale = color_scale,
                range_color = range_color,
                labels = var_labels,
                hover_name = 'suburb_name',
                hover_data=hover_data1,
                opacity=0.5,
                center=center_coord,
                #center=lga_geogcoord_dict.get(lga_name, state_capitals[state]),  # Default to Australia's center
                mapbox_style="open-street-map",
                #title = f"{status_label} by test",
                #mapbox_style="carto-positron",
                zoom=10,
            )
        # Hide color bar in choropleth
        map_fig.update_layout(coloraxis_showscale=False) 
        if gov_funded == "1":
        # Add scatter plot (coordinates) on top of the choropleth map
            scatter_fig = px.scatter_mapbox(
                df2, 
                lat="latitude", 
                lon="longitude", 
                text="cpo_name",
                labels = var_labels,
                hover_data= hover_data2
            )
            # Set showlegend=False on the scatter plot
            scatter_fig.update_traces(showlegend=False)    
            # Combine the two plots (choropleth and scatter points)
            map_fig.add_trace(scatter_fig.data[0])
        map_fig.update_layout(
            title=dict(
            y=1.0,  # Position title higher up
            x=0.5,   # Center the title horizontally
            xanchor="center",
            yanchor="top",
            font=dict(size=16)  # Customize title font size
            ),
            showlegend=False,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            autosize=True,  # Automatically adjust size
        )
        print('print graphs')
        #return map_fig1
        #print(f"Center on {lga_geogcoord_dict.get(lga_name, state_capitals[state])}")
        return map_fig
    except Exception as e:
        print(f"Error in plot_chloropleth_map: {e}")
        raise


# Adding a custom function for week of the month calculation
def week_of_month(date):
    #first_day = date.replace(day=1)
    return (date.day - 1) // 7 + 1

#helper function to plot chloropleth map
def plot_column_graph(df1: pd.DataFrame,
                      status_prop: str,
                      interval_option: str,
                      lga_name: str,
                      ) -> go.Figure:
    """
    Plots a bar graph showing the mean and standard deviation of a specified status property 
    for each period based on the given interval option, with a threshold line.

    Args:
        df1 (pd.DataFrame): Input data containing 'interval' and 'status_prop' columns.
        status_prop (str): Column name for the status property to be plotted.
        threshold (int): Threshold value for the horizontal line.
        interval_option (str): Interval option key for grouping the data.

    Returns:
        go.Figure: A Plotly Figure object with the bar chart and threshold line.
    """
    try:
        # Validate input DataFrame
        if df1.empty:
            raise ValueError("Input DataFrame is empty.")
        # Ensure your DataFrame is sorted by 'cpo_name' and 'interval' (or any other ordering you prefer)
        df1 = df1.sort_values(by=['cpo_name', 'interval'])
        # Define a mapping for interval options to datetime attributes
        interval_extraction = {
            "60min": df1['interval'].dt.hour + 1,  # Extract hour of day
            "1440min": df1['interval'].dt.dayofweek + 1,  # Extract day of the week
            "10080min": df1['interval'].apply(week_of_month),  # Extract week of the month
            "ME": df1['interval'].dt.month,  # Extract month
            "Q": df1['interval'].dt.quarter,  # Extract quarter
            "Y": df1['interval'].dt.year  # Extract year
        }  
        # Use groupby and cumcount to generate a sequential number for each unique 'interval' within each 'cpo_name' group

        df1['period_number'] = interval_extraction.get(interval_option, df1['interval'].dt.hour)
        # plot average across period number
        plot_data = (df1
                    .groupby(['cpo_name','period_number'])[status_prop]
                    .agg(mean_status = 'mean',  std_status = 'std', count_status = 'count')
                    .reset_index()
                    )
        plot_data['mean_status'] = np.round(plot_data['mean_status'],2)
        plot_data['std_err'] = np.round(plot_data['std_status']/np.sqrt(plot_data['count_status']),2)
        label = {'period_number':interval_options2[interval_option],status_prop: f'Mean {status_prop}', 'mean_status':f'Average {status_prop}'}   
         # Find the correct status_prop within utilisation_status
        if status_prop in utilisation_status['Metrics']:
            status_label = utilisation_status['Metrics'][status_prop]
        elif status_prop in utilisation_status['Statuses']:
            status_label = utilisation_status['Statuses'][status_prop]
        else:
            status_label = status_prop  # Default to the raw value if not found
        
        col_fig = px.bar(
            plot_data,
            x = 'period_number',
            y = 'mean_status',
            labels = var_labels | label,
            color = 'cpo_name',
            error_y = 'std_err',
            barmode="group",
            title=f"Average {status_label} by {interval_options2[interval_option].capitalize()} for {lga_name} "  
            )
        
        # Update layout to position the legend at the top
        col_fig.update_layout(
        title=dict(
            y=1.0,  # Position title higher up
            x=0.5,   # Center the title horizontally
            xanchor="center",
            yanchor="top",
            font=dict(size=16)  # Customize title font size
        ),
        legend=dict(
            title = None,
            orientation="h",    # Horizontal orientation for the legend
            yanchor="top",      # Align the bottom of the legend box
            y=1.05,             # Position the legend below the title
            xanchor="center",   # Center the legend horizontally
            x=0.5               # Center position horizontally
        ),
        #margin={"r": 0, "t": 0, "l": 0, "b": 0},
        #margin=dict(t=60, b=40)  # Adjust margins to fit the title and legend
        )
        
        # Add a horizontal dashed red line for the threshold
        #col_fig.add_shape(
        #    type="line",
        #    x0=0, x1=plot_data['period_number'].max()+1,
        #    y0=threshold, y1=threshold,
        #    line=dict(color="red", width=2, dash="dash"),
        #    name="Threshold"
        #)
        
        # Add an annotation near the threshold line
        #col_fig.add_annotation(
        #    x=0.99,  # Position in the middle of the x-axis
        #    y=threshold*0.95,
        #    xref="paper",  # x is relative to the plot width
        #    yref="y",  # y is on the y-axis scale
        #    text=f"Threshold: {threshold: 0.1f}%",
        #    showarrow=False,
        #    font=dict(color="red"),
        #    align="center",
        #    bgcolor="rgba(255,255,255,0.6)",  # Semi-transparent background
        #    bordercolor="red",
        #    borderwidth=1
        #)
        # Ensure all x-axis labels are shown
        col_fig.update_xaxes(
            tickmode="linear",
            
        )
        return col_fig
    except Exception as e:
        print(f"Error in plot_column_graph: {e}")
        raise
