#app.py
import faicons as fa
import pandas as pd
import asyncio
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
from shinywidgets import output_widget, render_plotly
import pyarrow
from config import *
from utilities import *


# ------------------------------------------------------------------------
# Define user interface
# ------------------------------------------------------------------------

app_ui = ui.page_fillable(
    ui.tags.style("""
        @media (max-width: 768px) {
            .sidebar {
                display: block; /* Stack items vertically on small screens */
            }
            .graphs-container {
                flex-direction: column; /* Switch to vertical layout */
            }
        }
        @media (min-width: 769px) {
            .sidebar {
                display: flex; /* Restore horizontal layout for larger screens */
            }
        }
    """),
    ui.page_navbar(
        ui.nav_spacer(),
        ui.nav_panel(
            'Login',
            ui.layout_columns(
                ui.card(
                    ui.input_text('user_code',label = "Please enter user code."),
                    ui.output_text_verbatim("check", placeholder=False),
                    ui.tooltip(
                        ui.div(
                            [
                                ui.input_date_range(
                                    "daterange",
                                    "Please select the date range (inclusive)",
                                    format = "dd-mm-yyyy",
                                    start = date_range_start,
                                    end = date_range_end
                                ),
                                fa.icon_svg("circle-info").add_class("ms-2"),
                            ],
                            style="display: flex; align-items: center; gap: 0.5rem;"
                        ),

                        ui.markdown(
                            '''
                            **Please Note:** the greater the selected date range, the longer it will take to load the data 
                            and increase the memory usage to run the dashboard application.<br> To ensure the  memory usage and computation time 
                            does not exceed platform capacity, please do no exceed a **4 month period** with the selected date range.
                            '''
                        ),
                        placement='right'
                    ),
                    ui.output_ui("load_data_button"),  # Display the button
                    ui.output_ui("load_status_message")                    
                ),
                ui.card(
                    ui.row(
                        ui.column(
                            9,
                            ui.output_ui("loaded_data_banner"),
                            style="display: flex; align-items: center;"  # Center vertically
                        ),
                        ui.column(
                            3,
                            ui.div(
                                ui.output_ui("dashboard_button"),
                                ui.br(),  # Adds a line break for spacing
                                ui.output_ui("download_button"),
                                style="display: flex; flex-direction: column; align-items: center; gap: 10px;"  # Stack vertically with spacing
                                ),
                            style="display: flex; align-items: center; justify-content: center;"  # Center vertically and horizontally
                        ),
                    ),
                    ui.output_ui("conditional_summary_ui")
                    ,
                    fill = True,
                    full_screen = True
                ),
                col_widths=(3,9),
                class_ = 'gap-1',
                style="padding: 0; margin: 0;"  # Reduces padding and removes outer margin between cards
            )
        ),
        ui.nav_panel(
            "Dashboard",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_selectize(
                        "interval_select",
                        label = "Select an interval option below:",
                        choices = interval_options,  
                        selected = "ME",
                        multiple= False
                    ),
                    ui.input_radio_buttons(
                        'gov_funded',
                        label='Select category of charge points to observe',
                        choices=funded_chargers,
                        selected="All chargers"
                    ),
                    ui.input_select(
                        'lga_name',
                        label = 'Select a local government area',
                        choices = {"NSW":{"Sydney":"Sydney"}},#state_lga_dict,
                        selected = 'Sydney'#state_lga_dict
                    ),
                    ui.input_selectize(
                        "cpo_name",
                        label="Select Charge Point Operator",
                        choices=['Exploren'],  
                        selected=['Exploren'],
                        multiple=True
                    ), 
                    ui.input_action_button(
                        "apply_filter",
                        "Apply the filters",
                        class_="btn-success"
                    ),
                    open="open",
                    title="Please select required option for each filter.",
                    bg = "#E5F6FA",
                    style="""
                        display: flex;
                        flex-direction: column;
                        padding: 1px; 
                        margin: 0;
                        height: auto; /* Ensure height adapts to content */
                        box-sizing: border-box;
                    """
                ),
                ui.layout_columns(
                    ui.output_ui("value_boxes"),        
                    # Unpack the list of value boxes
                    fill=False,
                    class_="gap-0",
                    style="""
                        padding: 0;  /* Remove padding inside layout_columns */
                        margin: 0;   /* Remove margin around layout_columns */
                    """
                ),
                ui.card(
                    ui.output_ui("card_header_heatmap_no_data_message"),
                    ui.output_ui("graphs_container"),        
                    # This section applies gap-3 to space the cards horizontally
                    style="""
                        padding: 0; 
                        margin: 0; 
                        height: auto; 
                        width: 100%; 
                        box-sizing: border-box;
                    """
                ),
            ),
        ),
        ui.nav_panel(
            "debug",
            ui.output_text('debug_vars'),
        ),
        id="navset",
        title=ui.popover(
            ui.div(
                [
                    ui.h4(
                        ui.output_text("selected_period"),
                        style="margin: 0;"  # Ensure no unnecessary margin
                    ),
                    fa.icon_svg("circle-info").add_class("ms-2"),
                ],
                style="display: flex; align-items: center; gap: 0.5rem;"
            ),
            ui.markdown(
                '''
                **Uptime**: the proportion of time the charger is Charging, Finishing, Reserved or Available out of the observable period.<br>
                **Utilisation**: the proportion of time the charger is charging, finishing or reserved out of the observable period.<br>
                **Unavailability**: the proportion of time the charger is unavailable or Out of Order out of the observable period.<br>
                '''
            ),
            placement='right'
        ),
        window_title = "Charge@Large: Charge point Utilisation and Uptime Dashboard",     
        fillable=True
    )
)
# ------------------------------------------------------------------------
# Server logic
# ------------------------------------------------------------------------
def server(input: Inputs, output: Outputs, session: Session):
    
   # global  cpo_data_filtered,cpo_data_filtered_combined,cpo_data_filtered_prop,cpo_data_filtered_combined_prop 
    # Reactive value to store computed data
    selected_state = reactive.Value(None)
    cpo_data = reactive.Value(None)
    location_data = reactive.Value(None)
    postcode_data = reactive.Value(None)
    computation_count = reactive.Value(0)
    # Signal to track if compute() has completed
    data_load_unload = reactive.Value(False)
    is_processing = reactive.Value(False)
    slider_updated = reactive.Value(False)
    A_completed = reactive.Value(False)
    B_completed = reactive.Value(False)
    C_completed = reactive.Value(False)
    #react_off =  reactive.Value(False)
    # Cache for value_boxes output
    #cached_value_boxes = reactive.Value(None)
    # Cache for choropleth map output
    #cached_choropleth_map = reactive.Value(None)
    # Cache for column graph output
    #cached_column_graph = reactive.Value(None)
    # Define a reactive value to track the button state
    button_state = reactive.Value("Load Data")
    status_message = reactive.Value("")
    #read data variables
    filtered_data = reactive.Value(None)
    download_data = reactive.Value(None)
    lga_geogcoord_dict = reactive.Value(None)
    poa_suburb = reactive.Value(None)
    geodf_filter_lga = reactive.Value(None)
    geodf_filter_poa = reactive.Value(None)
   
    month_dates = reactive.Value(None)

    ### Login and Load panel ###

    @reactive.effect
    @reactive.event(input.user_code)
    def user_code():
        user_code = input.user_code()
        selected_state.set(select_state(user_code,user_dict))

    @render.text
    @reactive.event(input.user_code)
    def check():
        if input.user_code() and selected_state.get() is not None:
            return f"{selected_state.get()} user access"
        elif input.user_code():
            return "No access. Please enter correct username."
    
    @output
    @render.ui
    def load_data_button():
        # Render a button to trigger data loading
        return ui.input_action_button("toggle_button",button_state(),class_="btn btn-primary")
    
    #@reactive.effect
    #def update_load_data_button():
    #    if selected_state.get() is None:
    #        return ui.update_action_button("toggle_button",)
    #    else:
    #        return ui.update_action_button("toggle_button",disabled=False)
        
    # Function to load data
    @reactive.Effect
    @reactive.event(input.toggle_button)
    async def loads_data_date_values():
        if input.user_code() is not None:
            data_load_unload.set(False)
            if selected_state.get() is not None and button_state() == "Load Data":
                # Toggle the button state between "Load Data" and "Unload Data"
                with ui.Progress(min=1, max=100) as p:
                    try:
                        #ui.update_action_button("toggle_button", disabled=True)
                        # Load data and compute static values
                        p.set(1,message=" Please wait...", detail="loading and processing data")
                        # Local timezoe
                        #local_tz = timezone_mappings.get(selected_state.get(), pytz.UTC)
                        # Convert user-selected dates from local timezone to UTC
                        start_date = pd.to_datetime(input.daterange()[0])#  # Parse the selected start date
                        end_date = pd.to_datetime(input.daterange()[1])#.tz_localize(local_tz)  # Parse the selected end date

                        outputs = await load_and_prepare_data(selected_state.get(),start_date,end_date,p) # load_and_prepare_data(local_env=local_env)
                        loaded_data = outputs[0]
                        lga_geogcoord_dict.set(outputs[1])
                        poa_suburb.set(outputs[2])
                        geodf_filter_lga.set(outputs[3])
                        geodf_filter_poa.set(outputs[4])
                        #get months data
                        p.set(93 - len(cpo_anonymity),message=" Please wait...", detail="assigning variables")
                        # Store generated month dates in reactive value 
                        month_dates.set(generate_month_dates(loaded_data, 'interval'))
                        # Filter across the selected time period using the converted dates
                        # Local timezoe
                        #local_tz = timezone_mappings.get(selected_state.get(), pytz.UTC)
                        filtered_monthly_df = loaded_data.filter(
                            (pl.col("interval") >= start_date) &
                            (pl.col("interval") <= end_date)
                            )
                        p.set(94 - len(cpo_anonymity),message=" Please wait...", detail="filter by time period")
                        ### 08/05/25 - Addition to meet compliance with DSA
                        # Filter out LGAs that only have a single CPO providing public Charging to maintain anonymity
                        #filtered_data_set.to_csv('filtered_monthly_data.csv')
                        filtered_monthly_df = anonymise_cpos1(filtered_monthly_df)                   
                        # Convert to pandas
                        filtered_monthly_df = filtered_monthly_df.to_pandas()
                        p.set(95 - len(cpo_anonymity),message=" Please wait...", detail="Filtered out LGAs to ensure CPO anonymity")  
                        # preparing the on_demand dataset
                        filtered_monthly_df2 = prepare_on_demand_data(filtered_monthly_df,p)
                        download_data.set(filtered_monthly_df2)
                        # Aggregate to station - level
                        filtered_monthly_df = evse_level_aggregation(filtered_monthly_df)
                        #assigning variable for downstream analysis
                        filtered_data.set(filtered_monthly_df)
                        p.set(98,message=" Please wait...", detail="prepared on demand charging session data")
                        # reporting coverage across geographies statistics
                        #geography_statistics(filtered_data_set, filtered_data_set2, plugshare_df)
                        p.set(99,message=" Please wait...", detail="determining statistics on geographic coverage for reporting")  
                        # update inputs
                        #update period slider
                        available_period = filtered_monthly_df['interval'].max().month - filtered_monthly_df['interval'].min().month
                        if available_period < 3:
                            available_intervals = {k: v for k, v in interval_options.items() if k not in ['Q', 'Y']}
                        else:
                            available_intervals = interval_options
                        ui.update_selectize(
                            'interval_select',
                            label = "Select an interval option below:",
                            choices = available_intervals, 
                            selected = "ME"
                            ) 
                        #State and LGA dictionary
                        state_lga_dict = {
                            selected_state.get():{lga:lga for lga in sorted(filtered_monthly_df['lga_name'].unique())}
                                    }
                        # Update lga select
                        ui.update_select(
                            "lga_name",
                            label='Select a local government area',
                            choices=state_lga_dict,
                            selected=next(iter(state_lga_dict[selected_state.get()]))
                        )
                        # update cpo selectize
                        ui.update_selectize(
                            "cpo_name",
                            label="Select Charge Point Operator",
                            choices= list(filtered_monthly_df['cpo_name'].unique()),
                            selected=list(filtered_monthly_df['cpo_name'].unique())                    
                        )
                        # Call compute explicitly after setting up the data
                        p.set(100,message=" Please wait...", detail="Finished loading...")
                        data_load_unload.set(True)
                        is_processing.set(False)
                        button_state.set("Unload Data")
                        ui.update_action_button("toggle_button")
                        status_message.set(f"Data loaded for {selected_state.get()} from {filtered_monthly_df['interval'].min().strftime('%d-%b-%Y')} to {filtered_monthly_df['interval'].max().strftime('%d-%b-%Y')}")
                    except Exception as e:
                        status_message.set(f"Error loading data: {str(e)}")
                        raise
            else:
                button_state.set("Load Data")
                # Unload data logic here
                filtered_data.set(None)
                lga_geogcoord_dict.set(None)
                poa_suburb.set(None)
                geodf_filter_lga.set(None)
                geodf_filter_poa.set(None)
                month_dates.set(None)
                # Add your data unloading logic here  
                cpo_data.set(None)
                location_data.set(None)
                postcode_data.set(None)
                # update interval
                ui.update_selectize(
                        'interval_select',
                        label = "Select an interval option below:",
                        choices = interval_options, 
                        selected = "ME"
                        ) 
                # Update lga select
                ui.update_select(
                "lga_name",
                label='Select a local government area',
                choices = {"NSW":{"Sydney":"Sydney"}},#state_lga_dict,
                selected = 'Sydney'#state_lga_dict
                )
                # update cpo selectize
                ui.update_selectize(
                "cpo_name",
                label="Select Charge Point Operator",
                choices=['Exploren'],  
                selected=['Exploren'],
                server=True,
                )
                data_load_unload.set(False)
                is_processing.set(False)
                # Reset variables as before
                #ui.update_action_button("toggle_button", disabled=False)
                status_message.set("Data unloaded successfully.")
    
    @output                            
    @render.ui
    def load_status_message():
        if is_processing.get():
            return ui.div("Processing... Please wait.", class_="text-warning")
        elif data_load_unload.get():
            return ui.div(
                [
                    status_message.get()
                ],
                class_="text-success"
            )
        return ui.div(status_message.get(), class_="text-muted")
    
    @output
    @render.text
    def selected_period():
        # Get the selected range of months from the slider and convert to month names
        if data_load_unload.get():
            data = filtered_data.get()
            start = data['interval'].min().strftime('%d %B')
            end = data['interval'].max().strftime('%d %B')
            return f"Charge point Utilisation, Uptime and Availability Dashboard - {start} to {end}"
        else:
            return "Charge point Utilisation, Uptime and Availability Dashboard"

    #load data banner
    @output
    @render.ui
    def loaded_data_banner():
        if not selected_state.get():
            return ui.div(
                [
                    ui.p("Enter correct user code..."),
                    ui.tags.img(src=generate_image_url("charge@large.jpg"), width="100%", height = "100%")
                ]
                    )
        if selected_state.get() is not None and (button_state() == "Load Data" or data_load_unload.get()):
            return ui.div(
                [
                    ui.p("Please select the date range..."),
                    ui.tags.img(src=generate_image_url(f"logo_{selected_state.get().lower()}.png"), width="100%", height = "100%")
                ]
                    )
        
    @output
    @render.ui
    def conditional_summary_ui():
        if data_load_unload.get():
            return ui.div(
                ui.h4("On-demand Summary of Charging sessions for funded Charge Points."),
                ui.output_data_frame('sample_dataframe'),
                style="display: flex; flex-direction: column; align-items: center; gap: 5px;"
            )
        else:
            return None
    
    @render.data_frame  
    def sample_dataframe():
        if data_load_unload.get():
            try:
                # Retrieve filtered data
                df = download_data.get()
                if df is None or df.empty:
                    raise ValueError("Filtered data is empty or not available.")                    
                return render.DataTable(df.head(3), height = None)
            except Exception as e:
                print(f"Error in rendering sample_dataframe: {e}")
                return None
        
    @output 
    @render.ui
    def dashboard_button():
        if data_load_unload.get():
            return ui.input_action_button('click_dashboard',"Go to Dashboard",class_="btn-success")
    @output 
    @render.ui
    def download_button():
        if data_load_unload.get():
            return ui.download_button('downloadData',"Download the dataset",class_="btn-success")

    @reactive.Effect    
    @reactive.event(input.click_dashboard)
    def navigate_to_dashboard():
        # Navigate to the Dashboard tab
        ui.update_navs("navset", selected="Dashboard")
    
    @render.download(filename = lambda: f"charging_sessions_{pd.to_datetime(input.daterange()[0]).strftime('%b-%y')}.csv")
    async def downloadData():
        await asyncio.sleep(0.25)
        # Navigate to the Dashboard tab
        df = download_data.get()
        yield df.to_csv(index=False)
        
    @reactive.Effect
    @reactive.event(input.gov_funded)
    def _():
        # Ensure data is loaded
        if not data_load_unload.get():
            print("Data is not loaded. Cannot apply filters.")
            return None
        if is_processing.get():
            return
        try:
            is_processing.set(True)  # Disable UI interactions
            # Retrieve current data and inputs
            filtered_data_val = filtered_data.get().copy()
            print(f'Length of the original dataset {len(filtered_data_val)}')
            cpo_selected = input.cpo_name()
            gov_funded = input.gov_funded()
            #map gov_funded input to values
            gov_funded_map = {
                "1":(filtered_data_val['government_funded'] != 'not_funded'),
                "2":(filtered_data_val['government_funded'] == 'not_funded'),
                "3":(filtered_data_val['government_funded'].notna())  # Match all rows with non-NA funding
                }       
            mask = ((filtered_data_val['cpo_name'].isin(cpo_selected)) &
                    gov_funded_map[gov_funded]
                )
            #Initial filter of data
            filtered_data_val2  = filtered_data_val.loc[mask,:]
            print(f"Filtered data size: {len(filtered_data_val2)}")
            # check for data
            if not len(filtered_data_val2) > 0:
                print("No data available after filtering.")
                return ui.notification_show(
                    f'No observations for {funded_chargers[gov_funded]} EVSE with {len(cpo_selected)} CPOs selected',
                    type = 'warning',
                    duration = 4
                )   
            #State and LGA dictionary
            state_lga_dict = {
                selected_state.get():{lga:lga for lga in sorted(filtered_data_val2['lga_name'].unique())}
                        }
            # Update lga select
            ui.update_select(
                "lga_name",
                label='Select a local government area',
                choices=state_lga_dict,
                selected=next(iter(state_lga_dict[selected_state.get()]))
            )
        except Exception as e:
            print(f"Error in compute: {e}")
        finally:
            # Signal that compute() has completed
            is_processing.set(False)    

    ### Dashboard Panel ####

    #@reactive.Effect
    #def no_cpo_selected():
    #    if not input.cpo_name():
    #        return ui.update_action_button("apply_filter",disabled=True)
    #    return ui.update_action_button("apply_filter",disabled=False)
    
    #@reactive.Effect
    #def update_apply_filter_state():
    #    if is_processing.get():
    #        ui.update_action_button("apply_filter", disabled=True)
    #    else:
    #        ui.update_action_button("apply_filter", disabled=False)

    
    @reactive.Effect
    @reactive.event(input.apply_filter)
    def compute():
        # Reset completion flag and cpo_data_filtered
        # Ensure data is loaded
        if not data_load_unload.get():
            print("Data is not loaded. Cannot apply filters.")
            return None
        if is_processing.get():
            return
        try:
            is_processing.set(True)  # Disable UI interactions
            # Retrieve current data and inputs
            filtered_data_val = filtered_data.get().copy()
            print(f'Length of the original dataset {len(filtered_data_val)}')
            lga_name = input.lga_name()
            cpo_selected = input.cpo_name()
            interval_option = input.interval_select()
            gov_funded = input.gov_funded()
            #map gov_funded input to values
            gov_funded_map = {
                "1":(filtered_data_val['government_funded'] != 'not_funded'),
                "2":(filtered_data_val['government_funded'] == 'not_funded'),
                "3":(filtered_data_val['government_funded'].notna())  # Match all rows with non-NA funding
                }       
            with ui.Progress(min=1, max=7) as p:
                p.set(message="Filtering data", detail="Applying filter.....")
                # CPO, LGA, time period and interval filtered data
                # Filter across the selected time period using the converted dates
                print(f"Applying filters for CPOs: {cpo_selected}")
                mask = ((filtered_data_val['lga_name'] == lga_name) &
                        (filtered_data_val['cpo_name'].isin(cpo_selected)) &
                        gov_funded_map[gov_funded]
                )
                print(f"Mask applied: {mask.sum()} rows match criteria.")

                #Initial filter of data
                filtered_data_val2  = filtered_data_val.loc[mask,:]
                print(f"Filtered data size: {len(filtered_data_val2)}")
                p.set(message="Filtering data", detail=f'size of filtered data is :{len(filtered_data_val2)}')
                # check for data
                if not len(filtered_data_val2) > 0:
                    print("No data available after filtering.")
                    return ui.notification_show(
                        f'No observations for {funded_chargers[gov_funded]} in {lga_name}',
                        type = 'warning',
                        duration = 4
                    )   
                #check for government funded chargers only
                if gov_funded != '1':
                    grouping_list = []
                else:
                    grouping_list = ['cpo_name']
                # Aggregate and process data
                p.set(message="Processing data...", detail = "Defining metrics based on interval size, processing data for visuals" )
                data_list = [
                    process_data(filtered_data_val2, grouping_list, interval_option),
                    process_data(filtered_data_val2, ['cpo_name', 'address', 'suburb', 'postcode', 'latitude', 'longitude'], interval_option),
                    process_data(filtered_data_val2, ['postcode'], interval_option)
                ]
                # Update reactive values
                print(f'length of cpo_data is {len(data_list[0])}')
                cpo_data.set(data_list[0])
                #data_list[0].to_csv('cpo_data_inspect.csv')
                print(f'length of location_data is {len(data_list[1])}')
                location_data.set(data_list[1])
                #data_list[1].to_csv('location_data_inspect.csv')
                print(f'length of postcode_data is {len(data_list[2])}')
                postcode_data.set(data_list[2])
                #data_list[1].to_csv('postcode_data_inspect.csv')
                print("Compute completed successfully.")
                # Enable reactivity for the outputs
                #react_off.set(False)  
                # Increment computation_count to trigger output refresh
                computation_count.set(computation_count.get() + 1)

        except Exception as e:
            print(f"Error in compute: {e}")
        finally:
            # Signal that compute() has completed
            is_processing.set(False)
            
    @output                          
    @render.ui
    def card_header_heatmap_no_data_message():
        if not data_load_unload.get():
            return ui.div("Enter correct user code...")
        if not computation_count.get() > 0:
            return ui.card_header(
                ui.div("Data is yet to be filtered...")
            )
        return ui.card_header(
            'Map and column graphs showing',
            ui.output_ui('card_header_heatmap'),
            ui.output_text('card_header_heatmap_lga'),
            #ui.popover(icons["ellipsis"],ui.input_slider('threshold','Set threshold for utilisation',0,100,50),placement="top"),
            class_="d-flex align-items-center gap-1")
    
    @output
    @render.ui
    def graphs_container():
        if not data_load_unload.get():
            return ui.div("Enter correct user code...", style="margin: 0; padding: 0;")
        if not computation_count.get() > 0:
            return ui.tags.img(
                src=generate_image_url(f"logo_{selected_state.get().lower()}.png"),
                width="80%",
                style="margin: 0; padding: 0;"
                )
        return ui.layout_columns(
            ui.div(
                output_widget("choropleth_map"),
                ui.row(
                    ui.div(
                        ui.input_switch('interval_mode',"View average observation", value = True),
                        ui.output_ui(
                            'interval_slider_ui',
                            style="flex-grow: 1; margin: 0; padding: 0;"),
                        style = """
                        display: flex;
                        align-items: center;
                        gap: 1vw;
                        width: 100%;
                        box-sizing: border-box;
                        margin: 0; /* Ensure no extra spacing below */
                    """
                    ),
                    style="display: flex; align-items: center; gap: 0; margin: 0; padding: 0;"
                ),
                style="""
                display: flex; 
                flex-direction: column; 
                justify-content: center; 
                align-items: stretch; 
                margin: 0; 
                padding: 0; 
                min-height: auto; 
                width: 100%;  /* Ensures the container spans full width */
                
                """
            ),
            output_widget("column_graph"),
            fill=True,
            fillable=True,
            class_="gap-0",  # Use 'gap-0' to eliminate extra spacing
            style="""
                margin: 0; 
                padding: 0; 
                height: auto; /* Adapts height to content */
                width: 100%;  /* Adapts to parent container */
                box-sizing: border-box; /* Considers padding for responsive layouts */
            """
        )
            
    @output
    @render.ui
    def card_header_heatmap():
        # Get the selected range of months from the slider and convert to month names
        return ui.input_select("status_prop", None, utilisation_status, width="auto"),   

    @output
    @render.ui
    def interval_slider_ui():
        """
        Render a conditional interval slider that appears when interval_mode is False.
        """   
        # Python logic to check the length of the dataframe
        data = postcode_data.get().dropna()
        is_single_data_point = len(data) == 1  # Check if there's only one data point
        interval_option = input.interval_select()
        
        step = interval_mapping.get(input.interval_select(),pd.Timedelta(days=30))
        start = pd.to_datetime(data['interval']).min()
        end = pd.to_datetime(data['interval']).max()
        # Determine time format
        if interval_option == "60min":
            time_format = "%m-%d %H"
        elif interval_option in ["1440min", "10080min"]:
            time_format = "%m-%d"
        elif interval_option in ["ME", "Q"]:
            time_format = "%b"
        else:
            time_format = "%Y"
            
        return ui.panel_conditional(
            # Condition to show the slider (when interval_mode is False)
            f"input.interval_mode === false && {str(not is_single_data_point).lower()}",
            ui.input_slider(
                "interval_slider",
                label=None,
                min=0,  # Placeholder values
                max=100,
                value=0,
                step=step,
                ticks=True,
                width="100%",
                time_format=time_format,   
            )
        )
    
    @reactive.Effect    
    @reactive.event(input.interval_mode)
    def update_interval_slider():
        """
        Toggle visibility of the interval slider based on interval_mode.
        """
        slider_updated.set(False)  # Reset the flag before starting the update
        print(f"interval_mode = {input.interval_mode()}")
        if not input.interval_mode():
            try:
                # Get the data
                data = postcode_data.get().dropna() 
                interval_option = input.interval_select()
                # Ensure 'interval' column is timezone-aware
                #if not pd.api.types.is_datetime64_any_dtype(data['interval']):
                    # Convert to datetime if not already in datetime format
                #    data['interval'] = pd.to_datetime(data['interval'], errors='coerce')
                if not hasattr(data['interval'].iloc[0], 'tzinfo') or data['interval'].iloc[0].tzinfo is None:
                    local_tz = timezone_mappings.get(selected_state, 'UTC')
                    data['interval'] = pd.to_datetime(data['interval'], utc=True).dt.tz_convert(pytz.timezone(local_tz))
                    print(f'converted timezones from uts to {local_tz}')
                
                start = pd.to_datetime(data['interval']).min()
                end = pd.to_datetime(data['interval']).max()
                # Log properties for debugging
                print(f"Slider properties: start={start}, end={end}")
                # Update the slider values and make it visible
                ui.update_slider(
                    "interval_slider",
                    min=start,
                    max=end,
                    value=start
                    #time_format=time_format
                )
                print("Updated slider")
                if len(data) == 1:
                    print("Skipping slider update: Single data point detected.")
                    return ui.notification_show(
                        'Only average observation available for single month. Please re-select.',
                        type = 'warning',
                        duration = 4
                    ) 
                slider_updated.set(True)  # Mark slider update as complete
            except Exception as e:
                print(f"Error preparing interval slider: {e}")
                slider_updated.set(True)  # Avoid indefinite waiting in case of errors
        
    @output
    @render.text
    @reactive.event(input.apply_filter)
    def card_header_heatmap_lga():
        lga_name = input.lga_name()
        return f"for {lga_name}."
    
    ##### Value Boxes of CPO Statistics######
        
    # Generate dynamic outputs for each CPO based on selection
    def create_output_func(data,cpo,lga_name,interval_option, selected_dates):
        """
        Dynamically generate UI content for a given CPO.
        """
        try:
            # Extract stats for the CPO
            stats = calculate_cpo_statistics(data, cpo, selected_dates)
            # Generate the rendered UI content
            if (stats['period_duration'] > 31) or (interval_option != 'ME'):
                    rendered_ui = ui.div(
                        ui.HTML(
                            f"""
                            <div style="font-size: 0.4em; line-height: 1.1;">
                                <strong>{cpo}'s {interval_options[interval_option]} statistics for</strong><br>
                                <strong>{lga_name}'s {int(stats['charge_station_count'])} charge points</strong><br>
                                <strong>Average Uptime:</strong> {stats['average_uptime']: .1f}% per {interval_options2[interval_option]}<br>
                                <strong>Minimum Uptime:</strong> {stats['minimum_uptime']: .1f}% <br>
                                <strong>Average Utilisation:</strong> {stats['average_utilisation']: .1f}% <br>
                                <strong>Maximum Utilisation:</strong> {stats['max_utilisation']: .1f}% <br>
                                <strong>Average Unavailability:</strong> {stats['average_unavailability']: .1f}% <br>
                                <strong>Maximum Unavailability:</strong> {stats['maximum_unavailability']: .1f}% <br>
                                <strong>Missing observations:</strong> {stats['missing_duration']: .1f}% <br>
                            </div>
                            """
                        ),
                        style = "display: flex; align-items: flex-start; padding: 0; margin: 0;"
                    )
            elif (interval_option == 'ME') and (stats['period_duration'] <= 31):
                rendered_ui = ui.div(
                    ui.HTML(
                        f"""
                        <div style="font-size: 0.4em; line-height: 1.1;">
                            <strong>{cpo}'s {interval_options[interval_option]} statistics for</strong><br>
                            <strong>{lga_name}'s {int(stats['charge_station_count'])} charge points</strong><br>
                            <strong>Uptime:</strong> {stats['average_uptime']: .1f}%<br>
                            <strong>Utilisation:</strong> {stats['average_utilisation']: .1f}%<br>
                            <strong>Unavailability:</strong> {stats['maximum_unavailability']: .1f}%<br>
                            <strong>Missing observations:</strong> {stats['missing_duration']: .1f}%<br>
                        </div>
                        """
                    ),
                    style = "display: flex; align-items: flex-start; padding: 0; margin: 0;"
                )  
            return rendered_ui             
        except Exception as e:
            print(f"Error creating output for {cpo} in value box: {e}")
        return ui.div(f"Error generating content for {interval_options[interval_option]} statistics")
    
   
    # Function to generate value boxes based on selected CPOs 
    def generate_value_boxes(cpo_selected,df,lga_name, interval_option, selected_dates):
        """
        Dynamically generate value boxes based on categories in the DataFrame.
        """
        max_width = "25vw"  # Adjust this as needed
        value_boxes = []       
        # Generate boxes for each selected CPO
        cpos_in_lga = df.cpo_name.unique()
        for cpo in cpo_selected:
            if cpo not in cpo_styles or cpo not in cpos_in_lga:
                print(f"Skipping CPO: {cpo}. Not found or invalid.")
                continue
            rendered_ui = create_output_func(df, cpo, lga_name, interval_option, selected_dates)
            icon_url = generate_image_url(cpo_styles[cpo]['icon'])
            value_boxes.append(
                ui.div(
                    ui.value_box(
                        title = '',
                        value = rendered_ui,
                        showcase=ui.img(
                            src=icon_url, 
                            style = "width: 5vw; margin-left: 1vw; padding: 0;"
                        ),
                        theme = ui.value_box_theme(
                            fg = cpo_styles[cpo]['text'],
                            bg=cpo_styles[cpo]['fill']
                            ),
                        style=(
                            "padding: 0; margin: 0; display: flex; "
                            "align-items: center; justify-content: flex-start; "
                            "gap: 1vw;"  # Add spacing between value and image
                        )  # Tight padding & alignment in value box
                        ),
                    style=f"max-width: {max_width}; padding: 0; margin: 0;",  # External margin for box container
                    class_="gap-0"  # Applies smaller Bootstrap gap class
                )
            )
        return value_boxes    

    @output
    @render.ui
    @reactive.event(computation_count)
    def value_boxes():
        """
        Generate and return the value boxes, respecting caching and reactivity states.
        """
        # Make this output dependent on computation_count
        #computation_count.get()
        
        # Ensure compute() has completed before proceeding
        if selected_state.get() is None:
            return ui.div("Enter correct user code...")
        if button_state() == "Load Data" and selected_state.get() is not None:
            return ui.div("Click the button to load data")
        if not computation_count.get() > 0:
            return ui.div("Click the Apply Filter button.")
        
         # Use cached outputs if reactivity is off
        #if react_off.get():
        #    cached_outputs = cached_value_boxes.get()
        #    if cached_outputs:
        #        print('returning cached data for value boxes')
        #        return cached_outputs
        #    return ui.div("No data available.")
        #Generate new boxes
        try:        
            cpo_selected = input.cpo_name() or []  # Get the selected CPOs
            gov_funded = input.gov_funded()   
            lga_name = input.lga_name()
            interval_option = input.interval_select()
            selected_dates = input.daterange()                    
            data = cpo_data.get()
            
            if gov_funded != '1':
                cpo = "All CPOs"
                data = data.assign(cpo_name = cpo)
                cpo_selected = [cpo]
            # Generate value boxes based on `cpo_data`
            new_output = ui.layout_columns(
                *generate_value_boxes(cpo_selected, data, lga_name, interval_option, selected_dates),
                class_="gap-1"
            )
            # Cache the new output and return it
            #cached_value_boxes.set(new_output)
            A_completed.set(True)  # Mark dynamic outputs as created
            return new_output
        except Exception as e:
            print(f"Error generating value_boxes output: {e}")
            A_completed.set(False)
            return ui.div("Error generating value boxes.")
        
                                                      
    # Render chloropleth map based on inputs
    @output
    @render_plotly
    @reactive.event(computation_count,input.status_prop,slider_updated ,input.interval_slider)
    def choropleth_map():
                # If react_off is True, return the cached output
        #if react_off.get():
        #    print('returning cached data for choropleth graph')
        #    return cached_choropleth_map.get()
                
        try:
            gov_funded = input.gov_funded()
            status_prop = input.status_prop()
            lga_name = input.lga_name()
            interval_mode = input.interval_mode()
            time_interval = input.interval_slider()
            state = selected_state.get() 
            data2 = location_data.get()
            data2 = data2.dropna()
            data1 = postcode_data.get()
            data1 = data1.dropna()
            geodf_filter_poa_val = geodf_filter_poa.get()
            #lga_geogcoord_dict_val = lga_geogcoord_dict.get()
            poa_suburb_val = poa_suburb.get()
            #Generate and return the map
            if interval_mode:
                group_vars1 = ['postcode','evse_port_count']
                plot_data1 = prepare_plot_average_data(data1,group_vars1)
                plot_data2 = prepare_plot_average_data(data2,location_vars)
                print(f'Average data lengths: Plot1={len(plot_data1)}, Plot2={len(plot_data2)}')
                hover_data1 = {
                    "evse_port_count": True,
                    'postcode': True,
                    f'{status_prop}_label': status_prop not in ['Utilisation', 'Uptime','Unavailability'],
                    'Utilisation_label': status_prop == 'Utilisation',
                    'Uptime_label': status_prop == 'Uptime',
                    'Unavailability_label': status_prop == 'Unavailability',
                    'Available_label': status_prop == 'Uptime',
                    'Unavailable_label': status_prop == 'Unavailability',
                    'Out of order_label': status_prop == 'Unavailability',
                    'Reserved_label': status_prop in ['Utilisation', 'Uptime'],
                    'Charging_label': status_prop in ['Utilisation', 'Uptime'],
                    'Finishing_label': status_prop in ['Utilisation', 'Uptime']                    
                }
                hover_data2 = {
                    "evse_port_count": True,
                    'postcode': False,
                    'address': True,
                    'suburb': True,
                    'latitude': False, 
                    'longitude':False,
                    f'{status_prop}_label': status_prop not in ['Utilisation', 'Uptime','Unavailability'],
                    'Utilisation_label': status_prop == 'Utilisation' and interval_mode,
                    'Uptime_label': status_prop == 'Uptime' and interval_mode,
                    'Unavailability_label': status_prop == 'Unavailability' and interval_mode,
                }
            elif slider_updated.get():
                # Ensure interval_slider is valid before proceeding
                
                selected_interval_value = pd.to_datetime(time_interval)
                # Convert to UTC and ensure timezone awareness
                #if selected_interval_value.tzinfo is None:
                #    tz_info = timezone_mappings.get(selected_state, pytz.UTC)
                #    selected_interval_value = selected_interval_value.tz_localize(pytz.UTC).astimezone(pytz.timezone(tz_info))
                        
                plot_data1 = data1.loc[data1.interval == selected_interval_value, :].copy()
                plot_data2 = data2.loc[data2.interval == selected_interval_value, :].copy()
                # for decimal places
                for var in status_metric_vars:
                    plot_data1.loc[:,var] = plot_data1.loc[:,var].round(2)
                    plot_data2.loc[:,var] = plot_data2.loc[:,var].round(2)
                print(f'slider_updated: {slider_updated.get()}')            
                print(f'Filtered data lengths: Plot1={len(plot_data1)}, Plot2={len(plot_data2)}')
                hover_data1={
                    status_prop: status_prop not in ['Utilisation', 'Uptime','Unavailability'],
                    'Utilisation': status_prop == 'Utilisation',
                    'Uptime': status_prop == 'Uptime',
                    'Unavailability': status_prop == 'Unavailability',
                    'Available': status_prop == 'Uptime',
                    'Unavailable': status_prop == 'Unavailability',
                    'Out of order': status_prop == 'Unavailability',
                    'Reserved': status_prop in ['Utilisation', 'Uptime'],
                    'Charging': status_prop in ['Utilisation', 'Uptime'],
                    'Finishing': status_prop in ['Utilisation', 'Uptime']
                }
                hover_data2={
                    "evse_port_count": True,
                    'postcode': False,
                    'address': True,
                    'suburb': True,
                    'latitude': False, 
                    'longitude':False,
                    status_prop: status_prop not in ['Utilisation', 'Uptime','Unavailability'],
                    'Utilisation': status_prop == 'Utilisation',
                    'Uptime': status_prop == 'Uptime',
                    'Unavailability': status_prop == 'Unavailability'
                }
            else:
                # If interval_slider is missing, log a message and return a placeholder
                print("interval_slider not available, cannot render map.")
                return 
            new_output = plot_choropleth_map(
                plot_data1,
                plot_data2,
                geodf_filter_poa_val,
                #lga_geogcoord_dict_val,
                status_prop,
                gov_funded,
                lga_name,
                poa_suburb_val,
                state,
                hover_data1,
                hover_data2
            )
            # Cache the new output and return it
            #cached_choropleth_map.set(new_output)
            B_completed.set(True)
            return new_output
        
        except Exception as e:
            print(f"Error generating chloropleth map: {e}")
            B_completed.set(False)
            
                        
    @output
    @render_plotly
    @reactive.event(computation_count,input.status_prop)
    def column_graph():
        # Ensure computation_count triggers reactivity
        #computation_count.get()
        # If react_off is True, return the cached output
        #if react_off.get():
        #    print('returning cached data for column graph')
        #    return cached_column_graph.get()
        try:
            # select data
            data3 = cpo_data.get().copy()
            data3 = data3.dropna()
            # Redefine inputs if reactivity is enabled
            status_prop = input.status_prop()
            interval_option = input.interval_select()
            gov_funded = input.gov_funded()    
            lga_name = input.lga_name()
            #If grant funded
            if gov_funded != '1':
                cpo = "All CPOs"
                data3 = data3.assign(cpo_name = cpo)
            
            # Generate the column graph    
            new_output = plot_column_graph(
                df1 = data3,
                status_prop = status_prop,
                interval_option = interval_option,
                lga_name = lga_name
                ) 
            # Cache the new output and return it
            #cached_column_graph.set(new_output)
            C_completed.set(True) 
            return new_output
                 
        except Exception as e:
            print(f"Error generating column graph: {e}")
            C_completed.set(False)
                    
            
    #@reactive.effect
    #def flag_reset():
    #    flag_mask = A_completed.get() and B_completed.get() and C_completed.get()
    #    if flag_mask:
    #        reactive.flush()
   
    @output
    @render.text
    def debug_vars():
        return f"""
            apply filter count: {computation_count.get()}
            value boxes completed: {A_completed.get()}
            choropleth map rendered: {B_completed.get()}
            column graph rendered: {C_completed.get()}
            """
            #reactive switch: {react_off.get()}
                           

# Call App() to combine app_ui and server() into an interactive app
app = App(app_ui, server, debug = False)

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 3838)