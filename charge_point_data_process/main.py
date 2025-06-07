import aiohttp
import asyncio

#from datetime import timezone, datetime
from config import *
from utils import *

# Define the function to read CSV data from Azure Blob Storage, and SQL query, process and return as a DataFrame to Azure Blob Storage
# The function is triggered by a timer trigger

async def sql_process() -> None:
    # Get the current month's start_date and end_date
    #start_date, end_date = get_current_month_dates()
    start_date = datetime(2025, 5, 1)# tzinfo=timezone.utc)
    end_date = datetime(2025, 5, 31)# tzinfo=timezone.utc) 
    logger.debug(f'Start Date: {start_date}, End Date: {end_date}')
    # Read CSV data from Blob Storage using input bindings
    try:
        lga_lookup = await read_blob_to_dataframe('data', 'lga_name_google_places_lga_code_lookup.csv',local)
        logger.debug('LGA Lookup data retrieved from Azure Blob Storage.')
        required_states = await read_blob_to_dataframe('data', 'required_states.csv',local)
        logger.debug('Required States data retrieved from Azure Blob Storage.')
        government_funded = await read_blob_to_dataframe('data', 'gov_funded.csv',local)
        logger.debug('Government Funded data retrieved from Azure Blob Storage.')   
        states = tuple(required_states['state'])
        # Using aiohttp.ClientSession with `with` block to ensure proper closure
        async with aiohttp.ClientSession() as session:
            # Define the query variables with the dynamic start_date and end_date
            query = {'location':{'query':query_location,
                                 'vars':[]},
                     'status':{'query':query_status,
                               'vars':[start_date, end_date, end_date]},
                     'offline':{'query':query_offline,
                                'vars':[start_date, end_date]}
            }
            # Perform PostgreSQL Query of Charge@Large Reporting Database for Location data
            location_df = await sql_query_data(query,'location')
            logger.debug('Location data retrieved from PostgreSQL database.')
            # Verify and update  State values based on Postcode.
            location_df = postcode_state(location_df)
            logger.debug('Imputed missing and erroneous State values based on Postcodes.')
            # filtered locations df via state
            location_df = location_df.loc[location_df.state.isin(states),:]
            logger.debug('Filtered EVSE locations data by required State clients.')
            # Assign Governemnt funded EVSE
            location_df = match_evse(government_funded, location_df)
            logger.debug('Government funded EVSE assigned to Location data.')
            # collect EVSE and PORT IDs
            select_evse = tuple(zip(location_df.evse_id,location_df.evse_port_id))
            query['status']['vars'].insert(2,select_evse)
            # Perform PostgreSQL Query of Charge@Large Reporting Database for Status data
            status_df = await sql_query_data(query,'status') 
            # query ofline and online windows
            offline_online_df = await sql_query_data(query,'offline')
            logger.debug('offline and online window data retrieved from PostgreSQL database.')
            status_df_edit = add_offline_online_events(status_df, offline_online_df)
            logger.debug('Added synthetic Missing events to status dataset.')
            # Missing Value Imputation
            missing_cols = ['lga_name', 'address']
            # Loop through the missing values count along with column names
            for col in missing_cols:
                # check if location_df has missing values
                #if location_df[col].isnull().any():
                location_df[col] = await asyncio.gather(
                    *[fill_missing_values(row, lga_lookup, col, GOOGLE_PLACES_API_KEY, session,cache) for _, row in location_df.iterrows()]
                )
            logger.debug(f'Filled missing values of Location data.')
                #else:
                #    logger.debug(f'No missing values in column: {col} of Location data to fill.')
            location_df = pl.from_pandas(location_df)
            logger.debug('Convert to polars dataframe')
            #Clean and process location and sattus data, merge dataframes and return complete dataset concurrently
            processed_df = clean_process(location_df, status_df_edit,start_date, end_date)
            logger.debug('Data is cleaned and processed.')
            # Process all states asynchronously
            state_dfs_dict = await process_all_states_async(processed_df)
            logger.debug(f'Processed all states asynchronously. Saving data for {state_dfs_dict.keys()} to Azure Blob Storage.')
            # Call the upload function
            await upload_all_states(state_dfs_dict, start_date)
            logger.debug('Final DataFrame saved to Azure Blob Storage. Finished Function')
    except Exception as e:
        logger.error(f"[ERROR] Function encountered an error: {e}")

if __name__ == "__main__":
    logger.debug('Function started.')
    asyncio.run(sql_process())
    logger.debug('Function completed')