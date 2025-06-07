#utils.py
#import libraries
import pandas as pd
import polars as pl
import numpy as np
import aiohttp
import asyncio
import psycopg2
import os
#import functions
from pathlib import Path
from datetime import datetime, timezone
from io import StringIO, BytesIO
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import KDTree
from azure.storage.blob import ContentSettings
# Importing variables from config.py
from config import *


# Helper function to get start_date and end_date for the current month
def get_current_month_dates():
    """
    This function returns the start_date and end_date based on the previous month.
    
    - start_date: The first day of the previous month.
    - end_date: The first day of the current month.
    """
    now = datetime.now(timezone.utc)
    # Get the first day of the current month
    end_date = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    # Get the first day of the next month
    # If the current month is December, increment the year
    if now.month == 1:
        start_date = datetime(now.year - 1, 12, 1, tzinfo=timezone.utc)
    else:
        start_date = datetime(now.year, now.month - 1, 1, tzinfo=timezone.utc)
    return start_date, end_date

# Helper function to generate dynamic blob name based on month
def generate_blob_name(start_date, state):
    """
    Helper function to generate dynamic blob file name as a parquet based on month and state.
    
    Parameters:
    - start_date (datetime): The start date to extract year and month.
    - state (str): The state code for the blob name.
    
    Returns:
    - str: The generated blob name as a parquet file.
    """
    state = state[0]
    year = start_date.strftime("%Y") #get year from start_date
    month_name = start_date.strftime("%B")  # Get month name from start_date
    blob_name = f"{state.lower()}_{year}_{month_name.lower()}_processed_data.parquet"
    return blob_name

# reading a csv file from Azure Blob Storage
async def read_blob_to_dataframe(container_name, blob_name,local):
    """
    Reads a CSV file from Azure Blob Storage and converts it into a Pandas DataFrame.#

    Parameters:
    - container_name (str): The name of the Azure Blob Storage container.
    - blob_name (str): The name of the CSV file to be read from Blob Storage.
    - local (boolean): A boolean flag to indicate local storage of data. Set True for local and False for cloud
    Returns:
    - pd.DataFrame: The DataFrame created from the CSV file.
    """
    try:
        if local:
            cwd = os.getcwd() 
            path = Path(cwd) / container_name 
            file = path / blob_name
        else:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            # Asynchronously download blob content
            blob_data = await blob_client.download_blob()
            blob_content = await blob_data.readall()
            # Decode the content and convert to Pandas DataFrame
            file = StringIO(blob_content.decode('utf-8'))
                
        data = pd.read_csv(file)
        return data
    except Exception as e:
        logger.error(f"Error reading from Blob: {e}")
        return pd.DataFrame()  # Return empty DataFrame if blob not found
    
    # Asynchronous SQL query      
async def sql_query_data(query_dict,key):
    '''
    Asynchronously connects to the PostgreSQL database and executes the query, using connection pooling for PostgreSQL queries.
    Parameters:
    - pool (object): sql connect pool
    - query (str): PostGreSQL Query
    - query_vars (list): variables embedded in query
    Returns:
    - pd.DataFrame: The DataFrame created from the PostGreSQL query
    '''
    conn = None
    try:
        conn = psycopg2.connect(**connection_params)  # Get a connection from the pool
        logger.info("PostgreSQL connection is opened")
        cur = conn.cursor()
        cur.execute(query_dict[key]['query'],query_dict[key]['vars'])
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        if key == 'location':
            # Create a DataFrame from the rows and column names
            return pd.DataFrame(rows, columns=colnames)
        else:
            return pl.DataFrame(rows, schema=colnames, orient="row")
        # Print the DataFrame    
    except psycopg2.Error as e:
        logger.error(f"Error connecting to the database: {e}")
        raise
    finally:
        # Closing the connection
        if conn:
            cur.close()
            conn.close()
            logger.debug("PostgreSQL connection is closed")
           
# Matching EVSEs with Government Funded EVSEs
def match_evse(
    df1, df2, lat_col='latitude', lon_col='longitude', merge_columns=['state', 'postcode', 'cpo_name', 'suburb']):
    """
    Finds the closest point in df2 for each point in df1 based on geographical proximity and merges based on specified columns.
    Adds a 'government_funded' column to df2 with True for matching rows within the two closest distances in each group and with partial matches in 'lga_name' or 'address'.
    
    Parameters:
    - df1: DataFrame containing the first set of coordinates
    - df2: DataFrame containing the second set of coordinates
    - lat_col, lon_col: Column names for latitude and longitude, assumed to be the same in both DataFrames
    - merge_columns: List of columns to use for initial exact merge

    Returns:
    - df2 with a new column 'government_funded', where rows matched with df1 and within the two closest distances are marked True, others False.
    """
    # Ensure all columns used for merging are strings in both dataframes
    df1[merge_columns] = df1[merge_columns].astype(str)
    df2[merge_columns] = df2[merge_columns].astype(str)
    
    # Filter df1 to keep only unique rows across merge_columns
    df1 = df1.drop_duplicates(subset=merge_columns)    
    # Step 1: Perform exact merge to filter rows by merge_columns, retaining original df2 indices for tracking
    exact_merged_df = pd.merge(
        df1, 
        df2.reset_index(),  # Retain original df2 index for tracking
        on=merge_columns, 
        suffixes=('_df1', '_df2')
    )
    funded_sites = []  # To store the original indices of df2 rows that are government funded
    
    # Step 2: Perform exact match and group by merge_columns
    for _, group in exact_merged_df.groupby(merge_columns):
        # Extract coordinates for KDTree for matched rows
        coords1 = group[[f"{lat_col}_df1", f"{lon_col}_df1"]].to_numpy()
        coords2 = group[[f"{lat_col}_df2", f"{lon_col}_df2"]].to_numpy()

        # Convert coordinates to radians for KDTree
        coords1_radians = np.radians(coords1)
        coords2_radians = np.radians(coords2)

        # Step 3: Build KDTree and find closest distances
        tree = KDTree(coords1_radians)
        distances, indices = tree.query(coords2_radians)
        
        # Store distances and the original indices of corresponding df2 rows
        group_distances = pd.DataFrame({
            'original_df2_index': group['index'],  # Use the retained original index of df2
            'distance_km': distances * 6371  # Convert to kilometers
        })

        # Step 4: Filter to retain rows with the two smallest distances
        closest_two = group_distances.nsmallest(2, 'distance_km')
                
        # Filter group to include only rows with original indices in closest_df2_indices
        filtered_group = group[group['index'].isin(closest_two['original_df2_index'])]

        ## Interim Step: Perform regex matching for 'lga_name' and 'address' within each group
        def partial_match(row, min_common_tokens=1):
            """
            Checks if there is a partial match based on common words between 'lga_name' and 'address' columns.
            """
            def get_tokens(text):
                # Convert the string to lowercase, split by spaces, and remove any empty tokens
                return set(str(text).lower().split()) if pd.notna(text) else set()

            # Get tokens (words) for lga_name and address fields
            lga_tokens_df1 = get_tokens(row['lga_name_df1'])
            lga_tokens_df2 = get_tokens(row['lga_name_df2'])
            address_tokens_df1 = get_tokens(row['address_df1'])
            address_tokens_df2 = get_tokens(row['address_df2'])
            
            # Check if there is sufficient token overlap
            lga_match = len(lga_tokens_df1.intersection(lga_tokens_df2)) >= min_common_tokens
            address_match = len(address_tokens_df1.intersection(address_tokens_df2)) >= min_common_tokens
            
            return lga_match or address_match

        # Apply partial matching within the filtered group
        filtered_group = filtered_group[filtered_group.apply(partial_match, axis=1)]
        

        # Collect original indices for rows that pass both the distance and regex match
        funded_sites.extend(filtered_group['index'].tolist())
    print("collected funded sites that passes both regex and distance match")
    new_name_df = exact_merged_df.loc[exact_merged_df['index'].isin(funded_sites),['index','program','round']]
    new_name_df['government_funded'] = new_name_df.apply(lambda row: ' '.join([str(row['program']),str(row['round'])]), axis = 1)
    print("Assigned program and round to funded sites")
    new_name_df = new_name_df.set_index('index')
    df2.loc[funded_sites, 'government_funded'] = new_name_df['government_funded'].reindex(df2.index)
    df2['government_funded'] = df2['government_funded'].fillna('not_funded')
    return df2
#Postcode check to return state
def postcode_state(df):
    postcode_to_state = {code: state for state, codes in state_postcodes.items() for code in codes}
    df['postcode'] = df['postcode'].astype(int)
    df['state'] = df['postcode'].map(postcode_to_state)
    return df

# Helper function to get LGA name from Google Places API
# Asynchronous HTTP request to Google Places API
async def get_lga_address_state(lat, long, api_key, column, session,cache):
    """
    Performs reverse geocoding to retrieve address details (LGA name or other locality information) based on latitude and longitude.
    This function sends a request to the Google Geocoding API using latitude and longitude
    coordinates, and extracts either the LGA name (if specified) or the locality from the result.
    Parameters:
    - lat (float): Latitude of the location.
    - long (float): Longitude of the location.
    - api_key (str): The Google Places API key.
    - column (str): The name of the column to be filled ('lga_name' or another address component).
    Returns:
    - str: The LGA name or locality name if found, otherwise `None`.
    """
    cache_key = f"{lat}-{long}"
    # Suburbs of ACT
    act_locality_list = ['Dickson', 'Mitchell', 'Fyshwick', 'Phillip', 'Canberra', 'Bruce',
       'Braddon', 'Turner', 'Calwell', 'Kingston', 'Beard', 'Weston',
       'Strathnairn', 'Kambah', 'Symonston']
    # Check if the result is already cached
    if cache_key in cache and column in cache[cache_key]:
        logger.debug(f"Cache hit for {column} at coordinates: {lat}, {long}")
        return cache[cache_key][column]
    
    async with semaphore, session.get('https://maps.googleapis.com/maps/api/geocode/json', params={'latlng': f'{lat},{long}', 'key': api_key}) as response:
        try:
            response.raise_for_status()
            results = (await response.json()).get('results', [])
            # initialise cache
            if cache_key not in cache:
                    cache[cache_key] = {}
            # lga look up values
            if column == 'lga_name':
                for result in results:
                    for component in result.get('address_components', []):
                        if 'administrative_area_level_2' in component.get('types', []):
                            lga_name = component.get('long_name') 
                            # Cache the result before returning
                            cache[cache_key]['lga_name'] = lga_name  # Store in cache
                            #logger.debug(f'returning {lga_name}.')
                            return lga_name  
                # If 'administrative_area_level_2' not found, search for 'locality'
                for result in results:
                    for component in result.get('address_components', []):
                        if ('locality' in component.get('types', [])) and (component.get('short_name', []) in act_locality_list):
                            # If the locality is in the ACT, return 'Unincorporated ACT'
                            lga_name = 'Unincorporated ACT'
                            # Cache the result before returning
                            cache[cache_key]['lga_name'] = lga_name
                            logger.debug(f'returning {lga_name}.')
                            return lga_name     
                # If nothing found, cache and return None
                cache[cache_key]['lga_name'] = None
                return None
            # lookup state values
            elif column == 'state':
                    for result in results:
                        for component in result.get('address_components', []):
                            if 'administrative_area_level_1' in component.get('types', []):
                                state = component.get('short_name')
                                cache[cache_key]['state'] = state
                                
                                return state          
            elif column == 'address':
                #reverse geocoding to get address details.
                location = google_geocoder.reverse(f'{lat},{long}')
                if location and len(location) > 0:
                    # Cache the result before returning
                    cache[cache_key]['address'] = location[0]
                    logger.debug(f'returning {location[0]}.')
                    return location[0]
                
                cache[cache_key]['address'] = None
                return None
        except (aiohttp.ClientError, ValueError) as e:
            logger.error(f"Error fetching geocode data: {e}")
            return None

# Helper function for Filling missing values using Google Places API
async def fill_missing_values(row, lookup_df, column, api_key, session,cache):
    """
    Fills missing values in a specific column (e.g., LGA name) using either:
    - A lookup from a DataFrame (`lookup_df`).
    - Reverse geocoding if the value is not found in the lookup table.

    Parameters:
    - row (pd.Series): A row from a Pandas DataFrame containing the data, including latitude and longitude.
    - lookup_df (pd.DataFrame): A DataFrame containing known LGA names and corresponding data for lookups.
    - column (str): The name of the column to be filled (e.g., 'lga_name').

    Returns:
    - str: The filled value for the column (LGA name or locality) if found, otherwise `None`.
    """
    lga_num = 0
    state_num = 0
    address_num = 0
    if column == 'state' and pd.notnull(row[column]):
        google_state_check = await get_lga_address_state(row['latitude'], row['longitude'], api_key, column, session,cache)
        if row[column] == google_state_check:
            return row[column]
        else:
            logger.debug(f'The state {row[column]} is incorrect, instead replaced with {google_state_check}')
            state_num += 1
            return google_state_check  
    elif column != 'state' and pd.notnull(row[column]):
        return row[column]
    else:
        match column:
            case 'lga_name':
                google_lga = await get_lga_address_state(row['latitude'], row['longitude'], api_key, column, session,cache)
                if google_lga is None:
                    result = None
                lga_name = lookup_df.loc[lookup_df['google_lga_name'] == google_lga, 'lga_name'].values
                result = lga_name[0] if len(lga_name) > 0 else lga_name
                lga_num += 1
                logger.debug(f'The lga {result} is imputed.')  
            case 'address':
                result = await get_lga_address_state(row['latitude'], row['longitude'], api_key, column, session,cache)
                logger.debug(f'The address {result} is imputed.') 
                address_num += 1 
            case 'state':
                result = await get_lga_address_state(row['latitude'], row['longitude'], api_key, column, session,cache)
                logger.debug(f'The state {result} is imputed.')
                state_num += 1
            case _:
                logger.debug("Column other than lga_name or address is missing data.") 
                result = None
    return result
    
##### Upsample Status Dataframe to 1min interval, forward fill records, downsample to 60min intervals ####
# Updated data management protocol as of DEC 2024
# amend start and end of intervals at boundary datetimes
# Using Polars Library
def add_offline_online_events(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:   
    # Step 1: Filter out records of updated status during offline window
    # Do a join to get matched windows by cpo_name
    joined = df1.join(df2, on="cpo_name", how="inner")
    # Keep only rows where update_time is **not** in any window
    condition = (
        (pl.col("update_time") > pl.col("offline_timestamp")) &
        (pl.col("update_time") < pl.col("online_timestamp"))
        )
    to_remove = joined.filter(condition).select(df1.columns)
    df_clean = df1.filter(~pl.struct(df1.columns).is_in(to_remove))  
    # Step 2 (optimized): Only include cpo_names that exist in df2
    evse_pairs = (
        df_clean
        .filter(pl.col("cpo_name").is_in(df2.select("cpo_name").unique()))
        .select(["cpo_name", "evse_id", "evse_port_id"])
        .unique()
    )
    # Step 3a: Synthetic rows for ONLINE timestamps (no status)
    # Join to match each cpo_name group in df2 with all their evse entries
    joined2 = evse_pairs.join(df2, on="cpo_name", how="inner")
    # Use online_timestamp as the synthetic update_time, and assign synthetic status
    synthetic_online_df = joined2.select([
        pl.col("evse_id"),
        pl.col("evse_port_id"),
        pl.col("cpo_name"),
        pl.lit(None, dtype=pl.Utf8).alias("status"),
        pl.col("online_timestamp").alias("update_time")
    ])
    combined_df = pl.concat([df_clean, synthetic_online_df], how="vertical")   
     # Step 3b: Sort and forward fill status per group
    combined_df = (
        combined_df
        .sort(["cpo_name", "evse_id", "evse_port_id", "update_time"])
        .with_columns([
            pl.col("status")
            .forward_fill()
            .over(["cpo_name", "evse_id", "evse_port_id"])
        ])
    )
    #Step 4: Synthetic rows for OFFLINE timestamps — set status = "Missing"
    synthetic_offline_df = joined2.select([
        pl.col("evse_id"),
        pl.col("evse_port_id"),
        pl.col("cpo_name"),
        pl.lit("Missing").alias("status"),
        pl.col("offline_timestamp").alias("update_time")
    ])
    # Step 5: Stack with original df1
    combined_df = pl.concat([combined_df, synthetic_offline_df], how="vertical")  
    combined_df = combined_df.sort(["cpo_name", "evse_id", "evse_port_id", "update_time"])  
    return combined_df


def amend_start_append_end_update_time(df: pl.DataFrame, start_time, end_time) -> pl.DataFrame:
    '''
    Determines the indices of starting and ending update times for each evse and evse_port location, amends the starting update time
    with the start time boundary and appends a new record for the end time boundary.
    Parameters:
    - df (pd.DataFrame): Status timeseries Dataframe per location.
    - start_time (datetime): start time boundary for analysis
    - end_time (datetime): end time boundary for analysis
    
    Returns:
    - pd.DataFrame: The Status timeseries per location Dataframe with amended start and end times.
    '''
    location_cols = ['evse_id', 'evse_port_id']
    # Step 1: Find the indices of the minimum update_time per group
    df_min = df.group_by(location_cols).agg(pl.col("update_time").min().alias("min_update_time"))
    logger.debug('amend_start_append_end_update_time Step 1 complete')
    ##### New EVSE ######
    # Step 2: Identify new EVSE locations
    new_evse_df = (   
        df
        .join(df_min, on=location_cols, how="inner")  # Join to get min_update_time
        .filter((pl.col("update_time") == pl.col("min_update_time")) &
                (pl.col("update_time") > start_time))  # Keep only rows where update_time == min
        .select(location_cols)  # Keep relevant columns
        )
    logger.debug('amend_start_append_end_update_time Step 2 complete')
    # Step 3: Define the `new_evse` column
    df = df.with_columns(
        pl.when(pl.struct(location_cols).is_in(new_evse_df))
        .then(1)
        .otherwise(0)
        .alias("new_evse")
    )
    logger.debug('amend_start_append_end_update_time Step 3 complete')
    # Step 4: Update `update_time` for rows where it equals the minimum
    df_updated = (
        df
        .join(df_min, on=location_cols, how="inner")
        .with_columns(
            pl.when((pl.col("new_evse")  == 0) &
                    (pl.col("update_time") == pl.col("min_update_time")))
        .then(start_time)
        .otherwise(pl.col("update_time"))
        .alias("update_time")
        )
        .drop("min_update_time")
    )
    logger.debug('amend_start_append_end_update_time Step 4 complete')
    # Step 5: Find the indices of the maximum update_time per group
    df_max = df.group_by(location_cols).agg(pl.col("update_time").max().alias("max_update_time"))
    logger.debug('amend_start_append_end_update_time Step 5 complete')
    # Step 6: Create new rows for max update_time values with new end_time
    df_new_rows = (
        df
        .join(df_max, on=location_cols, how="inner")  # Join to get min_update_time
        .filter((pl.col("update_time") == pl.col("max_update_time")))
        .with_columns(
            end_time.alias("update_time")
        )
        .drop('max_update_time')
    )
    logger.debug('amend_start_append_end_update_time Step 6 complete')
    # Step 7: Join the updated start times and new rows of end times.
    # Concatenate and floor update_time to nearest minute
    df_final = (
        pl.concat([df_updated, df_new_rows])
        .sort(location_cols + ["update_time"])
        .with_columns(pl.col("update_time").dt.truncate("1m"))
    )
    logger.debug('amend_start_append_end_update_time Step 7 complete')
    return df_final

# Upsamples the dataframe to a 1min frequency
def expand_time_index(df_wide: pl.DataFrame) -> pl.DataFrame:
    """
    Generates a per-group 1-minute time index for each unique location pair in df_wide.
    Parameters:
    - df (pd.DataFrame): Status timeseries per location in wide form.
    Returns:
    - pd.DataFrame: The upsampled Status timeseries per location Dataframe in wide form.
    """
    # Define the location identifier columns
    location_cols = ['evse_id', 'evse_port_id']
    #### Existing EVSE ####
    existing_evse_df = df_wide.filter(pl.col('new_evse') == 0)
    # Get unique location pairs
    existing_evse_locations = existing_evse_df.select(location_cols).unique()   
    # Get min and max timestamps
    logger.debug('expand_time_index Step 1 complete')
    min_time, max_time = (
        existing_evse_df
        .select([
            pl.col('update_time').min().alias("min_time"), 
            pl.col('update_time').max().alias("max_time")
        ])
        .row(0)  # Extract values as tuple
    )
    logger.debug('expand_time_index Step 2 complete')
    # Generate the time range
    existing_evse_time_range = pl.DataFrame(pl.datetime_range(start = min_time, end = max_time, interval="30s",eager=True).alias('update_time'))
    # Perform a cross join to get all combinations
    expanded_existing_evse_df = existing_evse_locations.join(existing_evse_time_range, how="cross")
    logger.debug('expand_time_index Step 3 complete')
    #### New EVSE ####
    new_evse_df = df_wide.filter(pl.col('new_evse') == 1)
    # Get unique location pairs
    new_evse_locations = new_evse_df.select(location_cols).unique()   
    logger.debug('expand_time_index Step 4 complete')
    # Create an empty list to store new EVSE dataframes
    #filtered_df  = new_evse_df.join(new_evse_locations, on = location_cols, how = "inner")
    
    # Compute min and max update_time per location
    time_bounds = (
        new_evse_df
        .group_by(location_cols)
        .agg([
            pl.col("update_time").min().alias("start_time"),
            pl.col("update_time").max().alias("end_time")
        ])
    )
    logger.debug('expand_time_index Step 5 complete')
    # Generate per-location time ranges
    expanded_new_evse_df = (
        time_bounds
        .group_by(location_cols)
        .agg(
            pl.datetime_range(
                start=pl.col("start_time"), 
                end=pl.col("end_time"), 
                interval="30s"                
            )
            .alias("update_time")
        )
        .explode("update_time")# Expands each time range into multiple rows
    )
    logger.debug('expand_time_index Step 6 complete')
    # Concatenate both DataFrames
    df_expanded = pl.concat([expanded_existing_evse_df, expanded_new_evse_df], how="vertical")
    logger.debug('expand_time_index Step 7 complete')
    return df_expanded

# Converts to df wide form, upsamples to 1min frequency, forward fills statuses, downsamples and aggregates to 60min frequency. 
def aggregate_duration_across_interval(df: pl.DataFrame, end_time, interval=freq) -> pl.DataFrame:
    """
    Converts a time-series DataFrame into wide form, up-samples it to a 1-minute frequency, forward-fills missing statuses,
    then down-samples and aggregates to a specified interval (default: 60 minutes).

    This function performs the following operations:
    1. Converts the input DataFrame from long to wide format using pivoting, ensuring status counts are preserved.
    2. Generates a full per-location time range at a 1-minute frequency to fill in missing timestamps.
    3. Merges the expanded time index with the original data and forward-fills missing values.
    4. Down-samples the data to the desired interval (default: 60 minutes) and aggregates status counts.
    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing status time-series data per location. 
        It must contain the following columns:
        - 'evse_id' (Location identifier)
        - 'evse_port_id' (Port identifier)
        - 'update_time' (Timestamp of status update)
        - 'status' (Categorical status variable)
    
    interval : str, optional
        The resampling interval for down-sampling the data (default is '60min').
        Accepts any Pandas-compatible time frequency string (e.g., '30min', '1H', etc.).

    Returns:
    -------
    pd.DataFrame
        A resampled and aggregated DataFrame where:
        - 'update_time' represents the start of each interval.
        - 'evse_id' and 'evse_port_id' define unique locations.
        - Each status column contains the aggregated duration of that status in the specified interval.

    Notes:
    ------
    - The function ensures that all locations remain in the dataset by merging with an expanded time index.
    - Forward filling (`ffill()`) ensures continuity in status values across missing timestamps.
    - The final output retains the structure of the original dataset but in aggregated time intervals.

    Example:
    --------
    >>> df_resampled = aggregate_duration_across_interval(df, interval='30min')
    >>> print(df_resampled.head())

    """
    location_cols = ['evse_id', 'evse_port_id']
    # Convert status to wide format
        
    # Expand time index
    expanded_df = expand_time_index(df)
    
    # Merge and forward-fill missing values
    df_filled = (
        expanded_df
        .join(df, on=["evse_id", "evse_port_id", "update_time"], how="left")
        .fill_null(strategy="forward")
        .drop('new_evse')
    )
    logger.debug('aggregate_duration_across_interval Step 1 complete')
    # Compute time differences and shift up (optimized)
    df_filled = (
        df_filled
        .with_columns(
            pl.col('update_time')
            .diff()
            .dt.total_seconds()
            .cast(pl.Int32)
            .over(location_cols)
            .shift(-1)
            .alias("count")
        )
     ) # Add count column
    logger.debug('aggregate_duration_across_interval Step 2 complete')
    df_filled = df_filled.filter(pl.col("update_time") < end_time)
        # convert to wide form
    #df_wide = df_filled.pivot(
    #    'status',
    #    index=["evse_id", "evse_port_id", "update_time"],
    #    values="count",
    #    aggregate_function="sum"
    #).fill_null(0)  # Fill missing values with 0
    logger.debug('aggregate_duration_across_interval Step 3 complete')
    # Keep long format and aggregate directly
    df_resampled = (
        df_filled
        .group_by_dynamic("update_time", every=interval, group_by=["evse_id", "evse_port_id", "status"])
        .agg(pl.col("count").sum().alias("status_duration"))  # Sum durations
        .sort(["evse_id", "evse_port_id", "update_time", "status"])
    )
    logger.debug('aggregate_duration_across_interval Step 4 complete')
    return df_resampled

# Adjust Status Durations
def adjust_status_duration(df: pl.DataFrame, location_time_cols, target_status: str, total_duration: int = INTERVAL*60) -> pl.DataFrame:
    # Step 1: Compute sum of all other statuses per location/time
    other_status_sum = (
        df.
        filter(~pl.col("status").is_in(["Total", target_status]))
        .group_by(location_time_cols)
        .agg(pl.col("status_duration").sum().alias("other_status_sum"))
    )
    # Step 2: Compute new value for the target status
    updated_target = (
        df
        .join(other_status_sum, on=location_time_cols, how="left")
        .with_columns(
            (
                pl.col("status_duration") + 
                (pl.lit(total_duration) - pl.col("other_status_sum"))
            ).alias("new_duration")
        )
        .select(location_time_cols + ["status", "new_duration"])
    )
    # Step 3: Merge updated values back into original DataFrame
    df_updated = (
        df
        .join(updated_target, on=location_time_cols + ["status"], how="left")
        .with_columns(
            pl.when(pl.col("status") == target_status)
            .then(pl.col("new_duration"))
            .otherwise(pl.col("status_duration"))
            .alias("status_duration")
        )
        .drop("new_duration")
    )
    return df_updated

# Expand status dataframe to cover all locations across whole period, filling with Missing column. 
def expand_and_fill(df: pl.DataFrame, start_time) -> pl.DataFrame:
    location_cols = ["evse_id", "evse_port_id"]
    # Step 1: Get all unique location pairs
    unique_locations = df.select(location_cols).unique()
    # Step 2: Get unique timestamps instead of generating a new range
    unique_time = df.select("update_time").unique()    
    # Step 3: Get unique statuses and ensure "Missing" & "Unknown" are included
    existing_statuses = df.select("status").unique()
    # Define required statuses
    required_statuses = pl.DataFrame({"status": ["Unknown"]})
    all_statuses = pl.concat([existing_statuses, required_statuses], how="vertical")
    # Drop duplicate rows
    all_statuses = all_statuses.unique()
    # Generate all possible combinations
    expanded_df = (
        unique_locations
        .join(unique_time, how="cross")
        .join(all_statuses, how="cross")
    )
    # Step 4: Merge expanded DataFrame with the original DataFrame
    df_filled = expanded_df.join(df, on=["evse_id", "evse_port_id", "update_time", "status"], how="left")
    # Step 5: Compute minimum update_time per location group from the original df
    df_min = (
        df.group_by(location_cols)
        .agg(pl.col("update_time").min().alias("min_update_time"))
    )
    df_filled = df_filled.join(df_min, on=location_cols, how="left")
    # Step 6a: Conditionally fill missing values in `status_duration` for periods less than min_update_time
    df_filled = df_filled.with_columns(
        pl.when((pl.col("status") == "Missing") & (pl.col("update_time") < pl.col("min_update_time")))
        .then(INTERVAL*60)  # If update_time < min_update_time, "Missing" gets 60
        .otherwise(pl.col("status_duration").fill_null(0))  # Otherwise, fill everything with 0
        .alias("status_duration")
    )
    # Step 6b: Conditionally adjust 'status_duration' for Missing status.
    condition = (pl.col("update_time") == pl.col("min_update_time")) & (pl.col("update_time") > start_time)
    adjusted = adjust_status_duration(df_filled.filter(condition), location_cols+['update_time'], 'Missing')
    unchanged = df_filled.filter(~condition)
    result = pl.concat([adjusted, unchanged])

    # Drop min_update_time as it's no longer needed
    result = result.drop("min_update_time")
    result = result.sort(location_cols+['update_time','status']) 
    return result

### Cleaning data
# Cleaning data functions
def clean_wevolt(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("evse_id").str.split(" - ").list.get(0).alias("evse_id")
    )

def clean_chargehub(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("postcode").is_not_null())

# Mapping
cpo_cleaners = {
    "Wevolt": clean_wevolt,
 #   "Charge Hub": clean_chargehub,
}

# Apply cleaner
def apply_cpo_cleaning(df: pl.DataFrame, cpo_cleaners: dict[str, callable]) -> pl.DataFrame:
    """
    Applies cleaning functions to CPO-specific subsets of a Polars DataFrame.

    Args:
        df (pl.DataFrame): The full DataFrame with a 'cpo_name' column.
        cpo_cleaners (dict): A dictionary mapping 'cpo_name' to a cleaning function (func(df_chunk) -> pl.DataFrame).

    Returns:
        pl.DataFrame: The cleaned Polars DataFrame.
    """
    cleaned_chunks = []
    for cpo in df["cpo_name"].unique().to_list():
        chunk = df.filter(pl.col("cpo_name") == cpo)
        # Apply cleaner if available, otherwise keep chunk unchanged
        if cpo in cpo_cleaners:
            cleaned_chunk = cpo_cleaners[cpo](chunk)
            cleaned_chunks.append(cleaned_chunk)
        else:
            cleaned_chunks.append(chunk)
    return pl.concat(cleaned_chunks, how="vertical")
         
# Helper function to clean and process the data
def clean_process(location_df: pl.DataFrame, status_df: pl.DataFrame, start_time, end_time) -> pl.DataFrame:
    """
    Processes location and status dataframes, merging them and generating the final dataset.
    
    Parameters:
    - location_df (pd.DataFrame): The location DataFrame.
    - status_df (pd.DataFrame): The status DataFrame.
    
    Returns:
    - pd.DataFrame: The final merged and processed DataFrame.
    """
    location_cols = ['evse_id', 'evse_port_id']
    new_start_datetime = start_time - pl.duration(hours=11)  # Equivalent to start_time - 11h
    new_end_datetime = end_time + pl.duration(days=1) - pl.duration(hours=11)
    status_var = status_df.select(pl.col("status").unique())["status"].to_list()
    try:
        logger.debug('Processing')
        # adjust start and end datetime to give precise timezone data and inclusive of whole 24 period of end date.
        status_df_edit  = amend_start_append_end_update_time(status_df, new_start_datetime, new_end_datetime)
        # upsample to once per minute, forward fill status, determine duration, and aggregate to 60 min intervals.
        status_df_edit  = aggregate_duration_across_interval(status_df_edit, new_end_datetime)
        # expand out status dataframe to cover all datetime stamps across whole period for every location, fill Missing
        status_df_edit = expand_and_fill(status_df_edit,new_start_datetime)
        logger.debug("Step 1 processing complete. Dataframe expanded and downsampled to 60min buckets")
        # Form Total type in status variable with corresponding duration value of 60.
        if "Total" not in status_var:     
            total_rows  = (
                status_df_edit.select(location_cols+['update_time']).unique()
                .with_columns(
                    pl.lit('Total').alias('status'),
                    pl.lit(INTERVAL*60).alias("status_duration")
                    )
            )
            status_df_edit = pl.concat([status_df_edit,total_rows ], how = "vertical").sort(location_cols+['update_time','status'])  
        # Amend Unknown column such that new value = current value + (Total - sum of all other status columns)
        status_df_edit =  adjust_status_duration(status_df_edit, location_cols+['update_time'], 'Unknown')
        #status_df_edit = adjust_unknown_status(status_df_edit,location_cols+['update_time'])
        logger.debug("Step 2 processing. Total and Unknown status amended")
        # merge with locations dataset.
        location_status_df = status_df_edit.join(location_df, on=['evse_id','evse_port_id'], how="left")
        logger.debug("Step 3 processing: Combined dataframes")
        #modular cleaning via CPO
        location_status_df = apply_cpo_cleaning(location_status_df,cpo_cleaners)                 
        return location_status_df
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        return pd.DataFrame()

# Main asynchronous function to handle parallel processing
async def process_all_states_async(processed_df):
    """
    filters the main dataframe to state specific dataframes and returns each dataframe in dict.
    Parameters:
    - processed_df (pd.DataFrame): The processed dataframe for all states
    Returns:
    - dict(pd.Dataframe): A dictionary of pandas dataframes, with each state as the keys.
    """
    state_dfs_dict = {}  # Dictionary to store results
    tasks = []

    # Function to process data for individual states
    def process_state_group(state, group, state_dfs_dict):
        """
        Processes data for an individual state and updates the result dictionary.

        Parameters:
        - state (str): The state being processed.
        - group (pd.DataFrame): The dataframe group for the state.
        - state_dfs_dict (dict): The dictionary to store the results.
        """
        processed_group = group.drop('state')
        state_dfs_dict[state] = processed_group
    # Create a thread pool executor
    with ThreadPoolExecutor() as executor:
        # Gather all processing tasks
        for state, group in processed_df.partition_by(by = 'state', maintain_order=True, as_dict = True).items():
            tasks.append(asyncio.to_thread(process_state_group, state, group, state_dfs_dict))
        await asyncio.gather(*tasks)
    return state_dfs_dict

# Helper function to save a DataFrame to a CSV file in Azure Blob Storage
async def save_dataframe_to_blob_async(dataframe: pl.DataFrame, container_name: str, blob_name: str, local: bool):
    """
    Asynchronously saves a Pandas DataFrame as a CSV file to Azure Blob Storage.

    Parameters:
    - dataframe (pd.DataFrame): The current month's DataFrame to save.
    - container_name (str): The Blob container name.
    - blob_name (str): The name of the file (including .parquet extension) for the current month.
    Returns:
    - None
    """
    try:
        if local:
            cwd = os.getcwd() 
            path = Path(cwd) / container_name 
            file = path / blob_name
            dataframe.write_parquet(file)
        else:
            # Convert Polars DataFrame to a Parquet file in memory
            file = BytesIO()
            dataframe.write_parquet(file)
            # Reset buffer position (CRUCIAL STEP)
            file.seek(0)
        # Get the blob client
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            # Asynchronously write to Azure Blob Storage
            #f = await fs.open_async(destination, mode="wb")
            #try:
            #    await f.write(parquet_buffer.getvalue())  # Upload parquet bytes
            #finally:
            #    await f.close()  # Ensure the file is properly closed

        # upload data
            await blob_client.upload_blob(file,
                                        overwrite=True, 
                                        content_settings=ContentSettings(content_type="application/x-parquet"),
                                        max_concurrency=12,
                                        timeout=1000 )
        logger.debug(f"Saved DataFrame as {blob_name} in {'local path' if local else 'cloud path'} {container_name} container. Fully uploaded")
    except Exception as e:
        logger.error(f"Error saving DataFrame to Blob: {e}")


# Helper function to save a DataFrame to a CSV file in Azure Blob Storage
# async uploads
async def upload_all_states(state_dfs_dict: dict[str, pl.DataFrame], start_date: str):
    """
    Asynchronously uploads all state-specific Polars DataFrames to Azure Blob Storage.

    Parameters:
    - state_dfs_dict (dict[str, pl.DataFrame]): Dictionary of state names mapped to Polars DataFrames.
    - start_date (str): The start date used to generate blob names.

    Returns:
    - None
    """
    tasks = []
    for state, df in state_dfs_dict.items():
        blob_name = generate_blob_name(start_date, state)
        task = save_dataframe_to_blob_async(df, 'data', blob_name,local)
        tasks.append(task)
    await asyncio.gather(*tasks)
