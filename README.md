# ⚡ Charge@Large Platform Initiative by the Electric Vehicle Council (EVC)

Welcome to the Charge@Large initiative! This platform, spearheaded by the Electric Vehicle Council (EVC), invites the participation of Charge Point Operators (CPOs) across Australia to enhance the visibility and reliability of public EV charging infrastructure. This README provides an overview of the platform’s goals, features, data collection, and participation benefits.

## 🎯 Project Purpose

The Charge@Large platform serves two main purposes:
1. **Live Status Visualization**: A portal displaying the real-time availability of public EV charging equipment across Australia.
2. **Uptime and Utilization Reporting**: Provides critical uptime and utilization data to government bodies, beginning with the DCCEEW in New South Wales (NSW).

This initiative aims to bridge the existing gap between the live status provided by CPO networks and the aggregated, static information found on existing platforms like Plugshare. By consolidating live availability data from multiple networks, the platform will provide a seamless experience for EV drivers across Australia. Additionally, the uptime and utilization data will support government programs in promoting reliable public EV charging infrastructure.

## 🌟 Key Features

- **Live Availability**: Displays the real-time status of chargers across participating CPO networks to help EV drivers quickly locate operational charging points.
- **Uptime and Utilization Reporting**: Logs status observations to generate reports at the EVSE connector, EV charger, and site levels for government reporting. This data is intended for government use only.
- **Secure Data Collection**: Data is collected via secure methods, including a RESTful API endpoint, OCPI protocol, or WebSocket as per mutual agreement with CPOs.

## 📊 Data Collection Overview

To fulfill uptime and utilization reporting requirements, the platform will collect:
- **EV Charger Site Location Data**: Includes site coordinates and addresses for CPO networks.
- **EVSE Connector Details**: Connector type and power rating.
- **Connector Status Observations**: Type, date, and time of status updates.
  - **Frequency**: Ideally, status updates are collected at a high frequency (once per minute); however, data collection intervals may vary based on CPO agreements.
- **Data Collection Methods**: RESTful API endpoint, OCPI protocol, or WebSocket, ensuring flexibility and security in data transfer.

---


## 📂 Repository Structure

This repository contains two key projects developed as part of the Charge@Large initiative by the Electric Vehicle Council (EVC). These projects are designed to support the visualization and reporting capabilities of the platform.

```
charge@Large
│   README.md
└───Report 1 app
│   │   app.py
│   │   utilities.py
|   |   config.py
|   |   requirements.txt
|   |  data.csv # relevant data files in csv
|   |  data.json  # relevant data files in json
│   │
│   └───img
│       │   image1.png
│       │   image2.png
│       │   ...
│   
└───function app
    │   function_app.py
    │   utils.py
    |   config.py
    |   host.json
    |   requirements.txt   
```
## 🛠️ Functionality Overview

### Azure Function App
- **Trigger**: A monthly timer trigger.
- **Input Data**:
- SQL Query: Retrieves structured data from a SQL database.
- Static CSV: Reads supplemental data from Azure Blob Storage.
- **Process**:
- Cleans and processes the data.
- Combines input sources into a single dataframe.
- **Output**:
- Stores the final processed dataframe back into Azure Blob Storage.

### Shiny for Python App
- **Input Data**:
- Uses pre-processed data stored in CSV and JSON formats.
- **Dashboard Features**:
- Displays live status and historical trends of EV charger availability and utilization.
- Interactive charts and tables for user exploration.
- **Images**:
- Contains visual assets under the `img/` directory for dashboard customization.

---

Clone this repository:
 ```bash
 git clone https://github.com/YourUsername/ChargeAtLarge.git
 cd ChargeAtLarge
```
For questions or support, please contact the EVC team. mark@evc.org.au

