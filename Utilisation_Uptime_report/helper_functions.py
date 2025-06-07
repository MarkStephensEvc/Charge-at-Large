# helper_functions.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from config import month, colour_mix, var_labels, centroids, cpo_styles  # Import necessary configs

#Helper Functions
# Define a function to lighten colors by blending with white
def lighten_colour(colour, amount=colour_mix):
  """
  Lightens the given color by blending it with white.
  amount=0.0 returns the original color, amount=1.0 returns white.
  """
  try:
    c = mcolors.cnames[colour]
  except:
    c = colour
  c = mcolors.to_rgb(c)
  white = (1, 1, 1)
  return mcolors.to_hex([(1 - amount) * c[i] + amount * white[i] for i in range(3)])

# Process data
def process_data1(data_df,grouping_vars,interval_column='interval', variable_column='variable'):
    """
    Processes the input DataFrame by grouping, aggregating, and reshaping data for 
    reporting on chargepoint operator (CPO) utilization and availability. The function 
    calculates various summary metrics, including the maximum electric vehicle supply 
    equipment (EVSE) counts and the proportion of available, unavailable, and in-use 
    chargepoints.

    Parameters:
    ----------
    data_df : pandas.DataFrame
        The input DataFrame containing chargepoint data with columns such as 'value', 
        'interval', and 'variable', representing the metrics to be aggregated.

    grouping_vars : list
        A list of column names by which the data will be grouped, such as ['cpo_name'] 
        or ['lga_name', 'cpo_name'].

    interval_column : str, default 'interval'
        The name of the column in `data_df` that represents time intervals. The function 
        uses the maximum value in this column to determine the latest time interval for 
        EVSE counts.

    variable_column : str, default 'variable'
        The name of the column in `data_df` that represents different status metrics 
        (e.g., 'Charging', 'Available', etc.) to be used in calculating proportions.

    Returns:
    -------
    pandas.DataFrame
        A processed DataFrame that includes aggregated metrics and proportions for 
        in-use, available, unavailable, and out-of-order chargepoints, grouped by 
        specified `grouping_vars`. The DataFrame is sorted by the 'in_use' column in 
        descending order.

    Notes:
    ------
    - The function groups data by `grouping_vars` and computes the sum of values in 
      each group, also calculating the maximum EVSE counts for the latest interval.
    - If no specific grouping columns are provided for state-level EVSE counts, the 
      function sums the 'value' column without additional grouping.
    - The resulting DataFrame is reshaped to include calculated proportions of each 
      status type (e.g., 'in_use', 'Available', 'Unavailable') for each chargepoint 
      operator, with values sorted by the 'in_use' metric.

    Example:
    --------
    >>> processed_data = process_data1(data_df, ['cpo_name'], interval_column='interval', variable_column='variable')
    >>> print(processed_data.head())
    """
    grouping_vars1 = [variable_column] + grouping_vars
    grouped_data = (data_df
                  .groupby(grouping_vars1)['value']
                  .sum()
                  .reset_index())

    # Combine for no group overall counts, include as Total for cpo_name
    no_cpo_group = [x for x in grouping_vars1 if x != 'cpo_name']
    no_group_data = (data_df
                .groupby(no_cpo_group)['value']
                .sum()
                .reset_index()
                .assign(cpo_name = 'Overall'))
    grouped_data_combined = pd.concat([grouped_data,no_group_data], ignore_index=True)
    # maximum evse count
    max_group_evse = (data_df
                    .loc[(data_df[interval_column] == data_df[interval_column].max()) &
                        (data_df[variable_column] == 'evse_port_site_count'),
                        grouping_vars+['value']]
                    .groupby(grouping_vars)['value']
                    .sum()
                    .reset_index())
    # max evse based on counting observation
    no_cpo_variable_group = [x for x in grouping_vars1 if x not in ['cpo_name','variable']]
    if no_cpo_variable_group:
      max_state_evse = (data_df
                      .loc[(data_df[interval_column] == data_df[interval_column].max()) &
                          (data_df[variable_column] == 'evse_port_site_count'),
                          grouping_vars+['value']]
                      .groupby(no_cpo_variable_group)['value']    
                      .sum()
                      .reset_index()
                      .assign(cpo_name = 'Overall'))
    else:
      # If empty, just sum the 'value' column without grouping
      total_value = (data_df
                        .loc[(data_df[interval_column] == data_df[interval_column].max()) &
                            (data_df[variable_column] == 'evse_port_site_count'),
                            'value']
                        .sum()
                        )
      max_state_evse = pd.DataFrame({'value': [total_value], 'cpo_name': ['Overall']})
    max_evse_combined = pd.concat([max_group_evse,max_state_evse], ignore_index=True)
    index_cols = [col for col in grouped_data_combined.columns if col not in ['variable', 'value']]
    grouped_data_wide = grouped_data_combined.pivot(index = index_cols,columns = 'variable',values = 'value')
    grouped_data_wide = grouped_data_wide.reset_index()
    # Include Unknown variable
    if 'Unknown' not in grouped_data_wide.columns:
        grouped_data_wide = grouped_data_wide.assign(Unknown = 0)
    # Defining metrics     
    grouped_data_wide['Utilisation'] = grouped_data_wide['Charging'] + grouped_data_wide['Finishing'] + grouped_data_wide['Reserved']
    grouped_data_wide['Uptime'] = (grouped_data_wide['Available'] + grouped_data_wide['Utilisation'])
    grouped_data_wide['Unavailability'] = (grouped_data_wide['Unavailable'] + grouped_data_wide['Out of order'])
    #scatter_plot_data_wide_edit = scatter_plot_data_wide.drop(columns = ['Charging','Finishing', 'Reserved'])
    
    # Form Proportions
    # Form proportions of statuses, excluding metrics
    metrics = ['Utilisation','Uptime','Unavailability']
    grouped_data_wide_edit = grouped_data_wide.copy()
    no_prop_vars = ['Total','evse_port_site_count'] + grouping_vars
    status_prop_cols = [col for col in grouped_data_wide_edit.columns if col not in no_prop_vars ]
    # Form proportions of metrics
    for col in status_prop_cols:
        if col in metrics:
            grouped_data_wide_edit[f"{col}_prop"] = grouped_data_wide_edit[col] / (grouped_data_wide_edit['Total'] - grouped_data_wide_edit['Unknown'] - grouped_data_wide_edit['Missing'])  
        else:
            grouped_data_wide_edit[f"{col}_prop"] = grouped_data_wide_edit[col] / grouped_data_wide_edit['Total']     
        
    #clean dataframe
    grouped_data_wide_edit = grouped_data_wide_edit.drop(columns = ['Total']+status_prop_cols)
    #order_cols = grouping_vars + ['Charging','Finishing','Reserved']
    #grouped_data_wide_edit = grouped_data_wide_edit[order_cols] 
    grouped_data_wide_edit = grouped_data_wide_edit.rename(columns = {'Out of order_prop':'out_of_order_prop'})
    grouped_data_wide_edit = grouped_data_wide_edit.sort_values(by = ['cpo_name'], ascending=True).reset_index(drop=True)
    snapshot_data = grouped_data_wide_edit.merge(max_evse_combined, on = grouping_vars)
    return snapshot_data 

# Process data
def process_data2(data_df,interval_column='interval', variable_column='variable'):

    interval_count = data_df[interval_column].nunique()
    index_cols1 = [col for col in data_df.columns if col not in ['value']+[interval_column]]
    scatter_plot_data = (data_df
    .groupby(index_cols1)['value']
    .sum()
    .reset_index())
    
    index_cols2 = [col for col in scatter_plot_data.columns if col not in ['value']+[variable_column]]
    scatter_plot_data_wide = scatter_plot_data.pivot(index = index_cols2,columns = variable_column,values = 'value')
    scatter_plot_data_wide = scatter_plot_data_wide.reset_index()
    
    # Defining metrics    
     # Include Unknown variable
    if 'Unknown' not in scatter_plot_data_wide.columns:
        scatter_plot_data_wide = scatter_plot_data_wide.assign(Unknown = 0)
    scatter_plot_data_wide['Utilisation'] = scatter_plot_data_wide['Charging'] + scatter_plot_data_wide['Finishing'] + scatter_plot_data_wide['Reserved']
    scatter_plot_data_wide['Uptime'] = (scatter_plot_data_wide['Available'] + scatter_plot_data_wide['Utilisation'])
    scatter_plot_data_wide['Unavailability'] = (scatter_plot_data_wide['Unavailable'] + scatter_plot_data_wide['Out of order'])
    #scatter_plot_data_wide_edit = scatter_plot_data_wide.drop(columns = ['Charging','Finishing', 'Reserved'])
    
    # Form proportions of statuses, excluding metrics
    metrics = ['Utilisation','Uptime','Unavailability']
    scatter_plot_data_wide_edit = scatter_plot_data_wide.copy()
    no_prop_vars = ['Total','evse_port_site_count'] + index_cols1
    status_prop_cols = [var for var in scatter_plot_data_wide_edit.columns if var not in no_prop_vars ]
    # Form proportions of metrics
    for col in status_prop_cols:
        if col in metrics:
            scatter_plot_data_wide_edit[f"{col}_prop"] = scatter_plot_data_wide_edit[col] / (scatter_plot_data_wide_edit['Total'] - scatter_plot_data_wide_edit['Unknown'] - scatter_plot_data_wide_edit['Missing'])  
        else:
            scatter_plot_data_wide_edit[f"{col}_prop"] = scatter_plot_data_wide_edit[col] / scatter_plot_data_wide_edit['Total']     
        
    #clean dataframe
    scatter_plot_data_wide_edit = scatter_plot_data_wide_edit.drop(columns = status_prop_cols+['Total'])
    scatter_plot_data_wide_edit = scatter_plot_data_wide_edit.sort_values(by = ['cpo_name'], ascending=True).reset_index(drop=True)
    scatter_plot_data_wide_edit['evse_port_site_count'] = scatter_plot_data_wide_edit['evse_port_site_count'] / interval_count
    no_prop_vars2 = ['Total','evse_port_site_count'] + index_cols1
    status_prop_cols2 = [var for var in scatter_plot_data_wide_edit.columns if var not in no_prop_vars2 ]
    scatter_plot_data_wide_edit.loc[:,status_prop_cols2] = np.round(scatter_plot_data_wide_edit.loc[:,status_prop_cols2]*100,1)
    return scatter_plot_data_wide_edit


def stacked_bar_chart(plot_data,state,month):
    status_cols = ['Charging_prop','Finishing_prop','Reserved_prop','Available_prop','Unavailable_prop','out_of_order_prop','Unknown_prop','Missing_prop','cpo_name']
    plot_data = plot_data[status_cols]
    plot_data = plot_data.rename(columns = {'Charging_prop':'Charging',
                                            'Finishing_prop':'Finishing',
                                            'Reserved_prop':'Reserved',
                                            'Available_prop':'Available',
                                            'Unavailable_prop':'Unavailable',
                                            'out_of_order_prop':'Out of Order',
                                            'Unknown_prop':'Unknown',
                                            'Missing_prop':'Missing'})
    plot_data = pd.melt(plot_data,id_vars = 'cpo_name')
    # Format float columns as percentage to 1 decimal place (excluding 'interval' and 'Total')
    plot_data.loc[:,'value'] = plot_data.loc[:,'value'].apply(lambda x: round(x*100,1))

    fig_stacked = px.bar(plot_data,
                        x='cpo_name',
                        y='value',
                        color='variable',
                        hover_name = "variable",
                        text = 'value',
                        hover_data = {'cpo_name': True,
                                      'value':True,
                                      'variable':False},
                        title=f'The segmentation of Utilisation, Uptime and Unavailability statuses of chargepoints.', 
                        labels=var_labels)
    # Customize text appearance (optional)
    fig_stacked.update_traces(textposition='inside', texttemplate='%{text:.1f}')
    

    fig_stacked.update_layout(
        autosize=True,
        bargap=0.4,
        bargroupgap=0.1,
        legend_title_text='Status Types',
        legend=dict(
            x=0.98,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='Black',
            borderwidth=2,
            font=dict(
                family="Arial",
                size=13,
                color="black"
                )
            ),
        title_font=dict(
                    family="Arial",
                    size=18,
                    color='black'
                ),
    )
    return fig_stacked

def parallel_box_plot(data,state,status):
    
    box_color = {k:v['fill'] for k,v in cpo_styles.items()}
    #set colour based on status
    if status == 'Utilisation_prop':
        plot_data = data.copy()
        hover_data = {
            'cpo_name': False,
            'lga_name': False,  # We use 'hover_name' for lga_name, so no need here
            status: True,
            'Charging_prop':True,
            'Finishing_prop':True,
            'Reserved_prop':True,
            'value': True
        }
        # Format hover data columns as percentages
        for col in ['Charging_prop', 'Finishing_prop', 'Reserved_prop', status]:
            plot_data[col] = plot_data[col].apply(lambda x: np.round(x * 100,1))
        
    elif status == 'Uptime_prop':
        plot_data = data.copy()   
        hover_data = {
            'cpo_name': False,
            'lga_name': False,  # We use 'hover_name' for lga_name, so no need here
            status: True,
            'Charging_prop':True,
            'Finishing_prop':True,
            'Reserved_prop':True,
            'Available_prop': True,
            'value': True
        }
        # Format hover data columns as percentages
        for col in ['Charging_prop', 'Finishing_prop', 'Reserved_prop', 'Available_prop',status]:
            plot_data[col] = plot_data[col].apply(lambda x: np.round(x * 100,1))
    else:
        plot_data = data.copy()
        hover_data = {
            'cpo_name': False,
            'lga_name': False,  # We use 'hover_name' for lga_name, so no need here
            status: True,
            'Unavailable_prop': True,
            'out_of_order_prop': True,
            'Unknown_prop':True,
            'Missing_prop':True,
            'value': True
        }
        # Format hover data columns as percentages
        for col in ['Unavailable_prop', 'out_of_order_prop', 'Unknown_prop', 'Missing_prop',status]:
            plot_data[col] = plot_data[col].apply(lambda x: np.round(x * 100,1))
    # Count values per category
    category_counts = plot_data['cpo_name'].value_counts()

    # Separate categories based on count
    large_categories = category_counts[category_counts >= 5].index
    small_categories = category_counts[category_counts < 5].index

    # Data for large and small categories
    data_large = plot_data[plot_data['cpo_name'].isin(large_categories)]
    data_small = plot_data[plot_data['cpo_name'].isin(small_categories)]

    fig_box = px.box(data_large,
                     x = "cpo_name", 
                     y = status,
                     points = "all",
                     color = 'cpo_name', 
                     color_discrete_map = box_color,
                     labels = var_labels,
                     hover_data=hover_data,
                     hover_name='lga_name')
    
    # Add scatter points for small categories
    for category in small_categories:
        filtered_data = data_small[data_small['cpo_name'] == category]
        fig_box.add_trace(
           go.Scatter(
                x=[category] * len(filtered_data), 
                y=filtered_data[status],
                mode='markers',
                name=category,
                marker=dict(color='black', size=8),
                showlegend=False  # Avoid legend duplication
            )
        )
       
    fig_box.update_layout(
    title={
        'text': f'{var_labels[status]} across {state} LGAs.',
        'y':0.99,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    title_font=dict(
        family="Arial",
        size=20,
        color='black'
    ),
    autosize=True,
    margin={"r": 40, "t": 30, "l": 40, "b": 30},
    legend=dict(
        x=0.99,  # Position legend to the left
        y=0.95,  # Position legend to the top
        bgcolor='rgba(255, 255, 255, 0.5)',  # Background color with transparency
        bordercolor='Black',
        borderwidth=2
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Arial"
    ),
    boxmode="group"
    )
       
    return fig_box
    
  
def chloropleth_map(plot_data,geo_data,state,status,centroids = centroids,month = month):
    
     #set colour based on status
    if (status == 'Utilisation_prop'):
        status_colour = "Blues"
        max_colour_value = plot_data[status].max()
    elif (status == 'Uptime_prop'):
        status_colour = "Greens"
        max_colour_value = plot_data[status].max()
    else:
        status_colour = "Reds"
        max_colour_value = plot_data[status].max()
    
    # Set up hover data with all necessary columns formatted as percentages
    hover_data = {
        'cpo_name': False,
        'lga_name': False,  # We use 'hover_name' for lga_name, so no need here
        'Utilisation_prop': True,   # Format 'status' as a percentage with 1 decimal place
        'Available_prop': True,
        'Unavailable_prop':True,
        'out_of_order_prop':True,
        'value': True
    }
    map_fig = px.choropleth_mapbox(
        plot_data,
        geojson = geo_data.geometry,
        locations = 'lga_name',
        color = status,
        color_continuous_scale=status_colour,
        range_color=(0,max_colour_value),
        labels=var_labels,
        hover_name="lga_name",
        hover_data=hover_data,
        opacity=0.5,
        center=dict(lat=centroids[state][0], lon=centroids[state][1]), #center of state
        mapbox_style="open-street-map",
        #mapbox_style="carto-positron",
        zoom=5,
    )
    
    map_fig.update_layout(
    title={
        'text': f'{var_labels[status]} metric across LGAs for {month}',
        'y':0.99,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    title_font=dict(
        family="Arial",
        size=20,
        color='black'
    ),
    autosize=True,
    margin={"r": 40, "t": 30, "l": 40, "b": 30},
    legend=dict(
        x=0.01,  # Position legend to the left
        y=0.95,  # Position legend to the top
        bgcolor='rgba(255, 255, 255, 0.5)',  # Background color with transparency
        bordercolor='Black',
        borderwidth=2
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Arial")
    )
    return map_fig

def scatter_map(plot_data,status):
    
    # Set up hover data with all necessary columns formatted as percentages
    hover_data = {
        'address': True,
        'suburb': True,
        'cpo_name': False,
        'lga_name': False,  # We use 'hover_name' for lga_name, so no need here
        'latitude': False,
        'longitude': False,
        'Utilisation_prop': True,   # Format 'status' as a percentage with 1 decimal place
        'Available_prop': True,
        'Unavailable_prop':True,
        'Out of order_prop':True,
        'evse_port_site_count': True
    }
    cpo_styles_fill = {k:v['fill'] for k,v in cpo_styles.items()}
    map_fig = px.scatter_mapbox(
        plot_data,
        lat = "latitude",
        lon = "longitude",
        color = "cpo_name",
        size = status,
        color_discrete_map = cpo_styles_fill,
        size_max = 17,
        hover_data = hover_data,
        hover_name = 'cpo_name',
        labels=var_labels
    )
    return map_fig

