# How to use the provided Excel File

## 1. Available Data

### The Excel file contains the following sheets:

### General_Information
Contains general information about the Community:
- Community ID
- Simulation Periods
- Period Duration (min)
- Number of Owners

For each possible Community ID, the following information is provided as a time-series:
- Maximum allowed import (kW)
- Maximum allowed export (kW)
- Energy buy price (€/kWh)
- Energy sell price (€/kWh)
- Generator Coefficients (A, B, C)

### Network_Info
Contains information about the physical network:
- Voltage limits
- Branch Information
- Cable Characteristics

### Peers_Info[_CommunityID]
Contains information about the peers, with each peer containing:
- Peer ID
- Type of Contract
- Owner ID

Additionally, each peer can specify the desired energy buy price import and export for each period.
The maximum allowed importation and exportation for each peer is also specified.

### Load[_CommunityID]
Contains information about the loads of the Community:
- Load ID
- Internal Bus Location
- Charge Type
- Owner ID
- Manager ID
- Type of Contract
- Contracted Power (kW)
- Angle (deg)

Complemented with the following time-series:
- Load Power Forecast (kW)
- Load Reactive Power Forecast (kVAr)
- Power Reduce (kW)
- Power Cut (kW)
- Power Move (kW)
- Power In Move (kW)
- Cost Reduce (€)
- Cost Cut (€)
- Cost Move (€)
- Cost Energy Not Supplied (€)

### Generator[_CommunityID]
Contains information about the generators of the Community:
- Generator ID
- Internal Bus Location
- Generator Type
- Owner
- Manager
- Type of Contract
- Maximum Generated Power (kW)
- Minimum Generated Power (kW)
- Maximum Generated Reactive Power (kVAr)
- Minimum Generated Reactive Power (kVAr)

Complemented with the following time-series:
- Generator Power Forecast (kW)
- Cost Parameter A
- Cost Parameter B
  - This is the value used for the cost function of the generator
- Cost Parameter C
- Cost Excess Power (€)
- Green House Gas Coefficient A (kgCO2eq/kWh)
- Green House Gas Coefficient B (kgCO2eq/kWh)
- Green House Gas Coefficient C (kgCO2eq/kWh)

### Storage[_CommunityID]
Contains information about the storages of the Community:
- Storage ID
- Internal Bus Location
- Battery Type
- Owner
- Manager
- Type of Contract
- Energy Capacity (kVAh)
- Minimum State of Charge (%)
- Charge Efficiency (%)
- Discharge Efficiency (%)
- Initial State of Charge (%)
- Maximum Charge Power (kW)
- Maximum Discharge Power (kW)

Complemented with the following time-series:
- Charge Power (kW)
- Discharge Power (kW)
- Charge Price (€/kWh)
- Discharge Price (€/kWh)

### Vehicle[_CommunityID]
Contains information about the vehicles of the Community:
- Vehicle ID
- Type of Vehicle
- Owner
- Manager
- Type of Contract
- Battery Capacity (kWh)
- Maximum Charge Power (kW)
- Maximum Discharge Power (kW)
- Charge Efficiency Coefficient
- Discharge Efficiency Coefficient
- Initial State of Charge (%)
- Minimum State of Charge (%)

Complemented with the following event information:
- Arrival Time Period
- Departure Time Period
- Charging Station ID
- Arrival State of Charge (%)
- Departure Required State of Charge(%)
- Maximum Charge Power (kW)
- Maximum Discharge Power (kW)
- Charge Price (€/kWh)
- Discharge Price (€/kWh)

### CStation[_CommunityID]
Contains information about the charging stations of the Community:
- Charging Station ID
- Internal Bus Location
- Owner
- Manager
- Type of Contract
- Maximum Charge Power (kW)
- Maximum Discharge Power (kW)
- Charge Efficiency Coefficient
- Discharge Efficiency Coefficient
- Maximum Allowed Power (kW)
- Place Start
- Place End

Complete with the following time-series:
- Maximum Charge Power (kW)
- Maximum Discharge Power (kW)

### Information
Contains information about:
- Type of Generator
- Type of Contract


## 2. Adding Resources

### 2.1. Example - Adding a new Load
To add a new Load, the following steps must be followed:
1. Copy the 12 rows corresponding to a Load profile
2. Paste the 12 rows at the end of the Load sheet
3. Change the Load ID to a new one
4. Change the required information

### 2.2. Considerations
- The Resource ID must be a unique, consecutive number
- The Resource ID should be consecutive

## 3. Adding Events (Vehicles)
To add a new event to a vehicle, the following steps must be followed:
1. Add a new column to the Vehicle sheet, with the ID of the new event
2. Add the information of the new event to the corresponding rows

## 4. Limitations
While the Excel file can be modified to add new resources and events, PyECOM has some limitations.

Data parser built does not consider more than two events per vehicle
- If more than two events are added, the first two will be used.

While owner and manager information are presented in the Excel file, PyECOM does not consider them
- Data parser considers and retrieves this information
- All resources are considered to be owned and managed by the same entity
- To make use of this information, the user must extend the resource classes and provide the information given by the parser
