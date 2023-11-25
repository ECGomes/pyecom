# Excel Parser for Hugo Morais' Excel files

import pandas as pd
from src.parsers.base_parser import BaseParser
import numpy as np


class HMParser(BaseParser):
    """
    Inherits from BaseParser.
    Excel reader for the energy community scenario.
    Has as arguments:
    - file_path: str
    - ec_id: int
    - vals: int -> Not used in current version

    When initialized will have the following information:
    - Generators (.generator)
    - Loads (.load)
    - Storage (.storage)
    - Charging stations (.charging_station)
    - Peers (.peers)
    - Vehicles (.vehicle)
    """

    def __init__(self, file_path: str, ec_id: int, vals: int = None):
        super().__init__(file_path)
        self.data = None

        # Energy Community ID assignment
        self.ec_id = ec_id

        # File path
        self.file_path = file_path

        # Prepare the variables
        self.generator = None
        self.load = None
        self.storage = None
        self.charging_station = None
        self.peers = None
        self.vehicle = None

        return

    def parse(self):
        # self.read_general_info()
        # self.read_branch_data()
        self.read_generator_data()
        self.read_load_data()
        self.read_storage_data()
        self.read_charging_station_data()
        self.read_peers_data()
        self.read_vehicle_data()

        return

    @staticmethod
    def get_timeseries(values, component: str):

        temp_data_idx = np.where(values == component)
        temp_data = values.iloc[temp_data_idx[0]].copy(deep=True).to_numpy()
        for i in np.arange(temp_data.shape[0]):
            temp_data[i, :] = pd.to_numeric(temp_data[i, :], errors='coerce')
        temp_data = temp_data[:, temp_data_idx[1][0] + 1:]

        return temp_data

    @staticmethod
    def get_characteristic(values, component: str, keep_string: bool = False):

        temp_data_idx = np.where(values == component)
        temp_data = values.iloc[temp_data_idx[0]].copy(deep=True).to_numpy()
        if not keep_string:
            for i in np.arange(temp_data.shape[0]):
                temp_data[i, :] = pd.to_numeric(temp_data[i, :], errors='coerce')
            temp_data = temp_data[:, temp_data_idx[1][0] + 1]
        else:
            for i in np.arange(temp_data.shape[0]):
                temp_data[i, :] = temp_data[i, :].astype(str)
            temp_data = temp_data[:, temp_data_idx[1][0] + 1]

        return temp_data

    @staticmethod
    def get_events(values, component: str, keep_string: bool = False):

        temp_data_idx = np.where(values == component)
        temp_data = values.iloc[temp_data_idx[0]].copy(deep=True).to_numpy()
        if not keep_string:
            for i in np.arange(temp_data.shape[0]):
                temp_data[i, :] = pd.to_numeric(temp_data[i, :], errors='coerce')
            temp_data = temp_data[:, temp_data_idx[1][0] + 1:temp_data_idx[1][0] + 3]
        else:
            for i in np.arange(temp_data.shape[0]):
                temp_data[i, :] = temp_data[i, :].astype(str)
            temp_data = temp_data[:, temp_data_idx[1][0] + 1:temp_data_idx[1][0] + 3]

        return temp_data

    def read_generator_data(self):
        # Read the Excel
        sheet_name_gen = 'Generator_EC{}'.format(self.ec_id)
        data_gen = pd.read_excel(self.file_path, sheet_name=sheet_name_gen, header=None)

        gen = {'p_forecast': self.get_timeseries(data_gen, 'P Forecast (kW)') * 0.5,
               'cost_parameter_a': self.get_timeseries(data_gen, 'Cost Parameter A (m.u.)'),
               'cost_parameter_b': self.get_timeseries(data_gen, 'Cost Parameter B (m.u.)'),
               'cost_parameter_c': self.get_timeseries(data_gen, 'Cost Parameter C (m.u.)'),
               'cost_nde': self.get_timeseries(data_gen, 'Cost NDE (m.u.)'),
               'ghg_cof_a': self.get_timeseries(data_gen, 'GHG Cof A (m.u.)'),
               'ghg_cof_b': self.get_timeseries(data_gen, 'GHG Cof B (m.u.)'),
               'ghg_cof_c': self.get_timeseries(data_gen, 'GHG Cof C (m.u.)'),
               'internal_bus_location': self.get_characteristic(data_gen, 'Internal Bus Location'),
               'type_generator': self.get_characteristic(data_gen, 'Generator Type'),
               'owner': self.get_characteristic(data_gen, 'Owner'),
               'manager': self.get_characteristic(data_gen, 'Manager'),
               'type_contract': self.get_characteristic(data_gen, 'Type of Contract'),
               'p_max': self.get_characteristic(data_gen, 'P Max. (kW)'),
               'p_min': self.get_characteristic(data_gen, 'P Min. (kW)'),
               'q_max': self.get_characteristic(data_gen, 'Q Max. (kW)'),
               'q_min': self.get_characteristic(data_gen, 'Q Min. (kW)')}

        # Correct the values of the forecast according to generator type
        for i in np.arange(gen['p_forecast'].shape[0]):
            if gen['type_generator'][i] == 1.0:
                gen['p_forecast'][i, :] = gen['p_max'][i]

        self.generator = gen
        return

    def read_load_data(self):

        # Read the Excel
        sheet_name_load = 'Load_EC{}'.format(self.ec_id)
        data_load = pd.read_excel(self.file_path, sheet_name=sheet_name_load, header=None)

        # Dictionary to place the values
        load = {'p_forecast': self.get_timeseries(data_load, 'P Forecast (kW)'),
                'q_forecast': self.get_timeseries(data_load, 'Q Forecast (kVAr)'),
                'p_reduce': self.get_timeseries(data_load, 'P Reduce (kW)'),
                'p_move': self.get_timeseries(data_load, 'P Move (kW)'),
                'p_in_move': self.get_timeseries(data_load, 'P In Move (kW)'),
                'cost_reduce': self.get_timeseries(data_load, 'Cost Reduce (m.u.)'),
                'cost_cut': self.get_timeseries(data_load, 'Cost Cut (m.u.)'),
                'cost_mov': self.get_timeseries(data_load, 'Cost Mov (m.u.)'),
                'cost_ens': self.get_timeseries(data_load, 'Cost ENS (m.u.)'),
                'internal_bus_location': self.get_characteristic(data_load, 'Internal Bus Location'),
                'charge_type': self.get_characteristic(data_load, 'Charge Type', keep_string=True),
                'owner_id': self.get_characteristic(data_load, 'Owner (ID)'),
                'manager_id': self.get_characteristic(data_load, 'Manager (ID)'),
                'type_contract': self.get_characteristic(data_load, 'Type of Contract'),
                'p_contracted': self.get_characteristic(data_load, 'P Contracted (kW)'),
                'tg_phi': self.get_characteristic(data_load, 'Tg phi')}

        self.load = load
        return

    def read_storage_data(self):
        # Read the Excel
        sheet_name_stor = 'Storage_EC{}'.format(self.ec_id)
        data_storage = pd.read_excel(self.file_path,
                                     sheet_name=sheet_name_stor,
                                     header=None)

        storage = {'p_charge_limit': self.get_timeseries(data_storage,
                                                         'P Charge Limit (kW)'),
                   'p_discharge_limit': self.get_timeseries(data_storage,
                                                            'P Discharge Limit (kW)'),
                   'charge_price': self.get_timeseries(data_storage,
                                                       'Charge price (m.u)'),
                   'discharge_price': self.get_timeseries(data_storage,
                                                          'Discharge price (m.u.)'),
                   'internal_bus_location': self.get_characteristic(data_storage,
                                                                    'Internal Bus Location'),
                   'battery_type': self.get_characteristic(data_storage,
                                                           'Battery Type',
                                                           keep_string=True),
                   'owner': self.get_characteristic(data_storage,
                                                    'Owner'),
                   'manager': self.get_characteristic(data_storage,
                                                      'Manager'),
                   'type_contract': self.get_characteristic(data_storage,
                                                            'Type of Contract'),
                   'energy_capacity': self.get_characteristic(data_storage,
                                                              'Energy Capacity (kVAh)'),
                   'energy_min_percentage': self.get_characteristic(data_storage,
                                                                    'Energy Min (%)') / 100.0,
                   'charge_efficiency': self.get_characteristic(data_storage,
                                                                'Charge Efficiency (%)') / 100.0,
                   'discharge_efficiency': self.get_characteristic(data_storage,
                                                                   'Discharge Efficiency (%)') / 100.0,
                   'initial_state': self.get_characteristic(data_storage,
                                                            'Initial State (%)') / 100.0,
                   'p_charge_max': self.get_characteristic(data_storage,
                                                           'P Charge Max (kW)'),
                   'p_discharge_max': self.get_characteristic(data_storage,
                                                              'P Discharge Max (kW)')}

        self.storage = storage
        return

    def read_charging_station_data(self):
        # Read the Excel
        sheet_name_charging_station = 'CStation_EC{}'.format(self.ec_id)
        data_charging_station = pd.read_excel(self.file_path,
                                              sheet_name=sheet_name_charging_station,
                                              header=None)

        charging_station = {'p_charge_limit': self.get_timeseries(data_charging_station,
                                                                  'P Charge Limit (kW)'),
                            'p_discharge_limit': self.get_timeseries(data_charging_station,
                                                                     'P Discharge Limit (kW)'),
                            'internal_bus_location': self.get_characteristic(data_charging_station,
                                                                             'Internal Bus Location'),
                            'owner': self.get_characteristic(data_charging_station,
                                                             'Owner'),
                            'manager': self.get_characteristic(data_charging_station,
                                                               'Manager'),
                            'type_contract': self.get_characteristic(data_charging_station,
                                                                     'Type of Contract'),
                            'p_charge_max': self.get_characteristic(data_charging_station,
                                                                    'P Charge Max (kW)'),
                            'p_discharge_max': self.get_characteristic(data_charging_station,
                                                                       'P Discharge Max (kW)'),
                            'charge_efficiency': self.get_characteristic(data_charging_station,
                                                                         'Charge Efficiency (%)'),
                            'discharge_efficiency': self.get_characteristic(data_charging_station,
                                                                            'Discharge Efficiency (%)'),
                            'e_capacity_max': self.get_characteristic(data_charging_station,
                                                                      'E Capacity Max (kWh)'),
                            'place_start': self.get_characteristic(data_charging_station,
                                                                   'Place Start'),
                            'place_end': self.get_characteristic(data_charging_station,
                                                                 'Place End')}

        self.charging_station = charging_station
        return

    def read_peers_data(self):
        # Read the Excel sheet
        sheet_name_peers = 'Peers_Info_EC{}'.format(self.ec_id)
        data_peers = pd.read_excel(self.file_path, sheet_name=sheet_name_peers, header=None)

        peers = {'p_forecast': self.get_timeseries(data_peers,
                                                   'P Forecast (kW)'),
                 'buy_price': self.get_timeseries(data_peers,
                                                  'Buy Price (m.u.)'),
                 'sell_price': self.get_timeseries(data_peers,
                                                   'Sell Price (m.u.)'),
                 'import_contracted_p_max': self.get_timeseries(data_peers,
                                                                'Import Contracted P Max (p.u)'),
                 'export_contracted_p_max': self.get_timeseries(data_peers,
                                                                'Export Contracted P Max'),
                 'type_peer': self.get_characteristic(data_peers,
                                                      'Type of Peer', keep_string=True),
                 'type_contract': self.get_characteristic(data_peers,
                                                          'Type of Contract'),
                 'owner_id': self.get_characteristic(data_peers,
                                                     'Owner ID')}

        self.peers = peers
        return

    def read_vehicle_data(self):
        # Read the Excel sheet
        sheet_name_v2g = 'Vehicle_EC{}'.format(self.ec_id)
        data_v2g = pd.read_excel(self.file_path, sheet_name=sheet_name_v2g, header=None)

        vehicle = {'arrive_time_period': self.get_events(data_v2g,
                                                         'Arrive time period'),
                   'departure_time_period': self.get_events(data_v2g,
                                                            'Departure time period'),
                   'place': self.get_events(data_v2g,
                                            'Place'),
                   'used_soc_percentage_arriving': self.get_events(data_v2g,
                                                                   'Used SOC (%) Arriving'),
                   'soc_percentage_arriving': self.get_events(data_v2g,
                                                              'SOC (%) Arriving') / 100.0,
                   'soc_required_exit': self.get_events(data_v2g,
                                                        'SOC Required (%) Exit') / 100.0,
                   'p_charge_max_contracted': self.get_events(data_v2g,
                                                              'Pcharge Max contracted [kW]'),
                   'p_discharge_max_contracted': self.get_events(data_v2g,
                                                                 'PDcharge Max contracted [kW]'),
                   'charge_price': self.get_events(data_v2g,
                                                   'Charge Price'),
                   'discharge_price': self.get_events(data_v2g,
                                                      'Disharge Price'),
                   'type_vehicle': self.get_characteristic(data_v2g,
                                                           'Type of Vehicle', keep_string=True),
                   'owner': self.get_characteristic(data_v2g,
                                                    'Owner'),
                   'manager': self.get_characteristic(data_v2g,
                                                      'Manager'),
                   'type_contract': self.get_characteristic(data_v2g,
                                                            'Type of Contract'),
                   'e_capacity_max': self.get_characteristic(data_v2g,
                                                             'E Capacity Max (kWh)'),
                   'p_charge_max': self.get_characteristic(data_v2g,
                                                           'P Charge Max (kW)'),
                   'p_discharge_max': self.get_characteristic(data_v2g,
                                                              'P Discharge Max (kW)'),
                   'charge_efficiency': self.get_characteristic(data_v2g,
                                                                'Charge Efficiency (%)'),
                   'discharge_efficiency': self.get_characteristic(data_v2g,
                                                                   'Discharge Efficiency (%)'),
                   'initial_soc_percentage': self.get_characteristic(data_v2g,
                                                                     'Initial State SOC (%)'),
                   'min_technical_soc': self.get_characteristic(data_v2g,
                                                                'Minimun Technical SOC (%)') / 100.0}

        # Calculates the schedule for arrivals and departures
        schedule = np.zeros((vehicle['p_charge_max'].shape[0], self.generator['p_forecast'].shape[1]))
        schedule_charge = np.zeros((vehicle['p_charge_max'].shape[0], self.generator['p_forecast'].shape[1]))
        schedule_discharge = np.zeros((vehicle['p_charge_max'].shape[0], self.generator['p_forecast'].shape[1]))
        schedule_arrival_soc = np.zeros((vehicle['p_charge_max'].shape[0], self.generator['p_forecast'].shape[1]))
        schedule_departure_soc = np.zeros((vehicle['p_charge_max'].shape[0], self.generator['p_forecast'].shape[1]))
        schedule_cs_usage = np.zeros((self.charging_station['p_charge_limit'].shape[0],
                                      vehicle['p_charge_max'].shape[0], self.generator['p_forecast'].shape[1]))
        for v in range(vehicle['p_charge_max'].shape[0]):

            # Check the trips
            for t in range(vehicle['soc_required_exit'].shape[1]):
                schedule[v, int(vehicle['arrive_time_period'][v, t]) - 1:
                            int(vehicle['departure_time_period'][v, t])] = 1.0

                current_place = int(vehicle['place'][v, t]) - 1

                # Get the charging station usage
                schedule_cs_usage[current_place, v, int(vehicle['arrive_time_period'][v, t]) - 1:
                                                    int(vehicle['departure_time_period'][v, t])] += 1.0

                # get the maximum allowed charging and discharging
                charge_max = min(vehicle['p_charge_max'][v],
                                 self.charging_station['p_charge_max'][current_place])

                discharge_max = min(vehicle['p_discharge_max'][v],
                                    self.charging_station['p_discharge_max'][current_place])

                # build the schedule
                schedule_charge[v, int(vehicle['arrive_time_period'][v, t]) - 1:
                                   int(vehicle['departure_time_period'][v, t])] = charge_max

                schedule_discharge[v, int(vehicle['arrive_time_period'][v, t]) - 1:
                                      int(vehicle['departure_time_period'][v, t])] = discharge_max

                schedule_arrival_soc[v, int(vehicle['arrive_time_period'][v, t]) - 1] = vehicle['soc_percentage_arriving'][v, t] * \
                                                                                        vehicle['e_capacity_max'][v]

                schedule_departure_soc[v, int(vehicle['departure_time_period'][v, t]) - 1] = vehicle['soc_required_exit'][v, t] * \
                                                                                             vehicle['e_capacity_max'][v]

        vehicle['schedule'] = schedule
        vehicle['schedule_charge'] = schedule_charge
        vehicle['schedule_discharge'] = schedule_discharge
        vehicle['schedule_arrival_soc'] = schedule_arrival_soc
        vehicle['schedule_departure_soc'] = schedule_departure_soc
        vehicle['schedule_cs_usage'] = schedule_cs_usage

        self.vehicle = vehicle
        return
