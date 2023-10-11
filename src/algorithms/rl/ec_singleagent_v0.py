from collections import OrderedDict

import gymnasium as gym
import numpy as np
import random
from copy import copy


class EnergyCommunitySingleEnv_v0(gym.Env):
    """
    Energy Community Single Agent Environment v0

    The environment is a simple transition of states of the several components of an Energy Community.
    The agent's objective is to control the several components in order to minimize the cost of the energy.

    Rewards are attributed in a similar way as the Energy Community Single Agent Environment v0. However, when
    an illegal action is taken, the reward is -1000 and the environment does not transition to a new state.
    This forces the agent to learn the correct actions to take given the observations.
    """

    def __init__(self, env_config=None):

        # Define illegal costs for components
        self.illegal_ev_cost = 0.0
        self.illegal_battery_cost = 0.0
        self.illegal_ev_request_cost = 20.0
        self.illegal_import_cost = 50.0
        self.illegal_export_cost = 50.0

        self.gen_production = OrderedDict({
            'gen_01': np.array([0, 0, 0, 0, 0, 1.012226775, 2.882248633, 6.788123906,
                                11.48394025, 15.14569613, 17.5214277, 18.83098385,
                                18.74890841, 17.24180717, 14.89047338, 11.25796131,
                                6.331765741, 2.653460017, 0.935303552, 0, 0, 0, 0, 0]),
            'gen_02': np.ones(24) * 30.0,
            'gen_03': np.array([0, 0, 0, 0, 0, 0.506113387, 1.441124317, 3.394061953, 5.741970126, 7.572848065,
                                8.760713848, 9.415491926, 9.374454204, 8.620903584, 7.445236692, 5.628980657,
                                3.165882871, 1.326730009, 0.467651776, 0, 0, 0, 0, 0]),
            'gen_04': np.ones(24) * 10.0,
            'gen_05': np.array([0, 0, 0, 0, 0, 0.253056694, 0.720562158, 1.697030976, 2.870985063, 3.786424033,
                                4.380356924, 4.707745963, 4.687227102, 4.310451792, 3.722618346, 2.814490329,
                                1.582941435, 0.663365004, 0.233825888, 0, 0, 0, 0, 0]),
            'gen_06': np.ones(24) * 18.0,
            'gen_07': np.array([0, 0, 0, 0, 0, 0.151834016, 0.432337295, 1.018218586, 1.722591038, 2.27185442,
                                2.628214154, 2.824647578, 2.812336261, 2.586271075, 2.233571008, 1.688694197,
                                0.949764861, 0.398019003, 0.140295533, 0, 0, 0, 0, 0])
        })

        self.gen_max = sum(self.gen_production.values())

        # Loads
        load_factor = 5.0
        self.load_consumption = OrderedDict({
            'load_01': np.array([5.525, 5.119, 4.812, 4.710, 4.702, 4.753, 4.943, 5.276, 5.711, 6.417,
                                 6.934, 7.271, 7.222, 6.926, 6.733, 6.485, 6.425, 6.887, 7.133, 7.137,
                                 6.980, 6.603, 6.124, 5.658]) * load_factor,
            'load_02': np.array([0.808084, 0.755402, 0.724999, 0.719373, 0.736302, 0.776178, 0.894726, 1.154072,
                                 1.530171, 1.775261, 1.848323, 1.890054, 1.727325, 1.707589, 1.751228, 1.702357,
                                 1.673395, 1.639703, 1.523902, 1.383727, 1.271996, 1.164104, 1.055458, 0.916462
                                 ]) * load_factor,
            'load_03': np.array([1.07745, 1.0072026, 0.9666648, 0.9591642, 0.9817362, 1.0349046, 1.192968, 1.5387624,
                                 2.040228, 2.3670144, 2.4644304, 2.520072, 2.3031, 2.2767858, 2.3349708, 2.269809,
                                 2.2311936, 2.186271, 2.0318688, 1.8449694, 1.6959942, 1.5521382, 1.4072778,
                                 1.2219498]) * load_factor,
            'load_04': np.array([0.335024, 0.3079808, 0.2911168, 0.2851216, 0.2889824, 0.300784, 0.3307568, 0.3762144,
                                 0.434216, 0.5067616, 0.5350448, 0.5515104, 0.5300192, 0.503088, 0.4898752, 0.4743296,
                                 0.4711488, 0.505296, 0.5257456, 0.5096496, 0.485736, 0.4503008, 0.4152176, 0.36872
                                 ]) * load_factor,
            'load_05': np.array([0.3997, 0.373688, 0.3587, 0.355944, 0.364354, 0.384254, 0.442554, 0.571136, 0.75789,
                                 0.879138, 0.91485, 0.935172, 0.85459, 0.84456, 0.8659, 0.841922, 0.827102, 0.80984,
                                 0.753012, 0.683546, 0.628622, 0.575062, 0.521104, 0.461556]) * load_factor,
            'load_06': np.array([3.91636, 3.00622, 2.67526, 2.377396, 2.239496, 2.283624, 2.42704, 2.8959, 3.39234,
                                 3.64056, 3.64056, 3.651592, 3.673656, 3.61298, 3.47508, 3.436468, 4.186644, 5.516,
                                 6.45372, 7.14322, 6.64678, 6.3434, 5.95728, 5.01956]) * load_factor
        })

        self.import_max = 50
        self.export_max = 50

        self.grid_buys_price = np.array([0.0643, 0.0598, 0.0598, 0.0598, 0.0598, 0.0643, 0.0643, 0.0975, 0.0975,
                                         0.0975, 0.1186, 0.1186, 0.0975, 0.0975, 0.0975, 0.0975, 0.0975, 0.0975,
                                         0.0975, 0.1186, 0.0975, 0.0643, 0.0643, 0.0643])
        self.grid_sells_price = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
                                          0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])

        self.current_timestep: int = 0

        self.storage = {'storage_01': {'max': 10.0, 'current': 0.8 * 10.0, 'discharge_cost': 0.4},
                        'storage_02': {'max': 20.0, 'current': 0.8 * 20.0, 'discharge_cost': 0.08},
                        'storage_03': {'max': 4.0, 'current': 0.8 * 4.0, 'discharge_cost': 0.4}}

        self.ev = {'ev_01': {'max': 40.0, 'min': 0.2, 'current': 0.5, 'discharge_cost': 0.05,
                             'connected': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                                                    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64),
                             'soc_arrival': np.array([(8, 0.5), (16, 0.4)]),
                             'soc_required': np.array([(12, 0.7), (18, 0.5)]),
                             'max_charge': 120.0},
                   'ev_02': {'max': 60.0, 'min': 0.2, 'current': 0.5, 'discharge_cost': 0.06,
                             'connected': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.int64),
                             'soc_arrival': np.array([(1, 0.4), (13, 0.3)]),
                             'soc_required': np.array([(2, 0.5), (23, 1.0)]),
                             'max_charge': 120.0},
                   'ev_03': {'max': 40.0, 'min': 0.2, 'current': 0.5, 'discharge_cost': 0.05,
                             'connected': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64),
                             'soc_arrival': np.array([(12, 0.4), (17, 0.3)]),
                             'soc_required': np.array([(14, 0.5), (20, 0.45)]),
                             'max_charge': 120.0},
                   'ev_04': {'max': 40.0, 'min': 0.2, 'current': 0.5, 'discharge_cost': 0.05,
                             'connected': np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                                                    0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
                             'soc_arrival': np.array([(1, 0.4), (13, 0.5)]),
                             'soc_required': np.array([(7, 0.7), (16, 0.65)]),
                             'max_charge': 120.0},
                   'ev_05': {'max': 60.0, 'min': 0.2, 'current': 0.5, 'discharge_cost': 0.06,
                             'connected': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                                                    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], dtype=np.int64),
                             'soc_arrival': np.array([(9, 0.4), (19, 0.3)]),
                             'soc_required': np.array([(11, 0.5), (22, 0.45)]),
                             'max_charge': 120.0}}

        # Overall values
        self.energy_requirements = sum(self.load_consumption.values())

        # Agent observation space dictionary (note the Dict usage from the gym.spaces module)
        self._obs_space_in_preferred_format = True
        temp_observation_space = {}
        for key in self.gen_production.keys():
            temp_observation_space[key] = gym.spaces.Box(low=0.0, high=300.0, shape=(1,))
            temp_observation_space[key + '_next'] = gym.spaces.Box(low=0.0, high=300.0, shape=(1,))
        temp_observation_space['energy_requirements'] = gym.spaces.Box(low=0.0, high=500.0, shape=(1,))
        temp_observation_space['storage_01'] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        temp_observation_space['storage_02'] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        temp_observation_space['storage_03'] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))

        for key in self.ev.keys():
            temp_observation_space[key] = gym.spaces.Dict({
                'evConnected': gym.spaces.Discrete(2),
                'evSOC': gym.spaces.Box(low=0.0, high=self.ev[key]['max'], shape=(1,)),
                'evSOCRequired': gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
                'evSOCRequiredDelta': gym.spaces.Box(low=0.0, high=24.0, shape=(1,))})
        temp_observation_space['price_import'] = gym.spaces.Box(low=0.0, high=300.0, shape=(1,))
        temp_observation_space['price_export'] = gym.spaces.Box(low=0.0, high=300.0, shape=(1,))

        self.observation_space = gym.spaces.Dict(temp_observation_space)

        # Agent action space dictionary (note the Dict usage from the gym.spaces module)
        self._action_space_in_preferred_format = True
        temp_action_space = {}
        """temp_action_space = {'gen_01': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2),
                                                        'genVal': gym.spaces.Box(low=0.0, high=1.0, shape=(1,))}),
                             'gen_02': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2),
                                                        'genVal': gym.spaces.Box(low=1.0, high=1.0, shape=(1,))}),
                             'gen_03': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2),
                                                        'genVal': gym.spaces.Box(low=0.0, high=1.0, shape=(1,))}),
                             'gen_04': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2),
                                                        'genVal': gym.spaces.Box(low=1.0, high=1.0, shape=(1,))}),
                             'gen_05': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2),
                                                        'genVal': gym.spaces.Box(low=0.0, high=1.0, shape=(1,))}),
                             'gen_06': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2),
                                                        'genVal': gym.spaces.Box(low=1.0, high=1.0, shape=(1,))}),
                             'gen_07': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2),
                                                        'genVal': gym.spaces.Box(low=0.0, high=1.0, shape=(1,))}),
                             'storage_01': gym.spaces.Dict({'chargeCtl': gym.spaces.Discrete(3),
                                                            'chargeVal': gym.spaces.Box(low=0.0, high=1.0,
                                                                                        shape=(1,))}),
                             'storage_02': gym.spaces.Dict({'chargeCtl': gym.spaces.Discrete(3),
                                                            'chargeVal': gym.spaces.Box(low=0.0, high=1.0,
                                                                                        shape=(1,))}),
                             'storage_03': gym.spaces.Dict({'chargeCtl': gym.spaces.Discrete(3),
                                                            'chargeVal': gym.spaces.Box(low=0.0, high=1.0,
                                                                                        shape=(1,))})}"""

        temp_action_space = {'gen_01': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2)}),
                             'gen_02': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2)}),
                             'gen_03': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2)}),
                             'gen_04': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2)}),
                             'gen_05': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2)}),
                             'gen_06': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2)}),
                             'gen_07': gym.spaces.Dict({'genCtl': gym.spaces.Discrete(2)}),
                             'storage_01': gym.spaces.Dict({'chargeCtl': gym.spaces.Discrete(3),
                                                            'chargeVal': gym.spaces.Box(low=0.0, high=1.0,
                                                                                        shape=(1,))}),
                             'storage_02': gym.spaces.Dict({'chargeCtl': gym.spaces.Discrete(3),
                                                            'chargeVal': gym.spaces.Box(low=0.0, high=1.0,
                                                                                        shape=(1,))}),
                             'storage_03': gym.spaces.Dict({'chargeCtl': gym.spaces.Discrete(3),
                                                            'chargeVal': gym.spaces.Box(low=0.0, high=1.0,
                                                                                        shape=(1,))})}

        for key in self.ev.keys():
            temp_action_space[key] = gym.spaces.Dict({
                'evCtl': gym.spaces.Discrete(3),
                'evVal': gym.spaces.Box(low=0.0, high=1.0, shape=(1,))})

        self.action_space = gym.spaces.Dict(temp_action_space)

        # Agent initialization
        self.done = False

    def reset(self, *, seed=None, options=None):
        # Reset the timestep
        self.current_timestep = 0

        # Reset battery
        for storage in self.storage.keys():
            self.storage[storage]['current'] = 0.8

        # Reset EV
        for ev in self.ev.keys():
            self.ev[ev]['current'] = 0.5

        # Set the done flag to False
        self.done = False

        return self._get_obs()

    def step(self, action):

        # Check if done
        if self.current_timestep >= 24:
            return self._get_obs(), 0.0, True, {}

        # Production
        production = self._get_production(action)
        production_sum = sum(production.values())

        # Battery
        batteries_charge, batteries_discharge, batteries_cost = self._get_batteries(action)
        batteries_charge_sum = sum(batteries_charge.values())
        batteries_discharge_sum = sum(batteries_discharge.values())
        batteries_cost = sum(batteries_cost.values())

        # EV
        evs_charge, evs_discharge, evs_cost, ev_reward = self._get_evs(action)
        evs_charge_sum = sum(evs_charge.values())
        evs_discharge_sum = sum(evs_discharge.values())
        evs_cost_sum = sum(evs_cost.values())
        evs_reward_sum = sum(ev_reward.values())

        # Calculate the total energy requirements
        total_energy_requirements = sum([self.energy_requirements[self.current_timestep],
                                         -production_sum, batteries_charge_sum, -batteries_discharge_sum,
                                         -evs_discharge_sum, evs_charge_sum])

        # Consider the imports and exports into the total energy requirements
        imported = 0.0
        exported = 0.0
        if total_energy_requirements > 0.0:
            imported = total_energy_requirements
        elif total_energy_requirements < 0.0:
            exported = np.abs(total_energy_requirements)

        # Calculate the reward
        reward = self._get_reward(production_sum, batteries_cost,
                                  imported, exported, evs_reward_sum)

        # Update the info dictionary
        info = {'total_production': production_sum,
                'energy_requirements': self.energy_requirements[self.current_timestep],
                'calculated_requirements': total_energy_requirements,
                'import': imported,
                'export': exported,
                'batteries_charge': batteries_charge_sum,
                'batteries_discharge': batteries_discharge_sum,
                'evs_charge': evs_charge_sum,
                'evs_discharge': evs_discharge_sum}
        for i in self.storage.keys():
            info[i] = self.storage[i]['current'] * self.storage[i]['max']
        for i in self.ev.keys():
            info[i] = self.ev[i]['current'] * self.ev[i]['max']

        # Calculate the done flag
        done = self.current_timestep == 24

        # Calculate the next observation
        self.current_timestep += 1
        next_obs = self._get_obs()

        return next_obs, reward, done, info

    def _get_obs(self):

        if self.current_timestep == 24:
            return self.observation_space.sample()

        temp_obs = {}
        for gen in self.gen_production.keys():
            temp_obs[gen] = np.array([self.gen_production[gen][self.current_timestep]], dtype=np.float32)
            temp_obs[gen + '_next'] = np.array([self.gen_production[gen][self.current_timestep + 1]
                                                if self.current_timestep < 23 else 0.0],
                                               dtype=np.float32)

        temp_obs['energy_requirements'] = np.array([self.energy_requirements[self.current_timestep]],
                                                   dtype=np.float32)

        for battery in self.storage.keys():
            temp_obs[battery] = np.array([self.storage[battery]['current']], dtype=np.float32)

        for ev in self.ev.keys():
            next_required = np.where(self.ev[ev]['soc_required'][:, 0] > self.current_timestep)[0]
            next_required = self.ev[ev]['soc_required'][next_required[0]] if len(next_required) > 0 else (24.0, 0.0)
            temp_obs[ev] = {'evConnected': self.ev[ev]['connected'][self.current_timestep],
                            'evSOC': np.array([self.ev[ev]['current']],
                                              dtype=np.float32),
                            'evSOCRequired': np.array([next_required[1]],
                                                      dtype=np.float32),
                            'evSOCRequiredDelta': np.array([next_required[0] - self.current_timestep],
                                                           dtype=np.float32)}

        temp_obs['price_import'] = np.array([self.grid_sells_price[self.current_timestep]], dtype=np.float32)
        temp_obs['price_export'] = np.array([self.grid_buys_price[self.current_timestep]], dtype=np.float32)

        return temp_obs

    def _get_production(self, action):
        temp_gens = {}
        for gen in self.gen_production.keys():
            # temp_val = action[gen]['genCtl'] * action[gen]['genVal'][0]
            # temp_val *= self.gen_production[gen][self.current_timestep]
            # temp_gens[gen] = np.clip(temp_val, 0.0, self.gen_production[gen][self.current_timestep])
            temp_gens[gen] = self.gen_production[gen][self.current_timestep] * action[gen]['genCtl']

        return temp_gens

    def _get_batteries(self, action) -> (dict, dict, dict):
        """
        Takes the storage actions and returns dictionaries of charge, discharge and costs
        :param action:
        :return: charge, discharge, costs, illegal
        """

        temp_charge = {}
        temp_discharge = {}
        temp_cost = {}

        for battery in self.storage.keys():
            charge, discharge, illegal = self._get_battery(action[battery],
                                                           self.storage[battery]['current'])

            # Update the storage current SOC
            self.storage[battery]['current'] += (charge - discharge)

            charge = charge * self.storage[battery]['max']
            discharge = discharge * self.storage[battery]['max']

            temp_charge[battery] = charge
            temp_discharge[battery] = discharge
            temp_cost[battery] = sum([discharge * self.storage[battery]['discharge_cost'],
                                      illegal * self.illegal_battery_cost])

        return temp_charge, temp_discharge, temp_cost

    def _get_battery(self, action, battery_current) -> (float, float, bool):
        """
        Takes the battery action and returns the charge and discharge values, as well as the illegal flag.
        :param action: Dictionary of battery actions
        :param battery_current: Current battery SOC percentage
        :return: charge, discharge, illegal
        """

        # chargeCtl = 0: no charge
        # chargeCtl = 1: charge
        # chargeCtl = 2: discharge
        # chargeVal: percentage of battery to charge/discharge

        battery_current = battery_current

        if action['chargeCtl'] == 0:
            return 0.0, 0.0, False

        # Need to check if we can actually charge/discharge the battery
        if action['chargeCtl'] == 1:
            if battery_current >= 1.0:
                return 0.0, 0.0, True
            else:
                charge_val = action['chargeVal'][0]  # * battery_max
                if battery_current + charge_val > 1.0:
                    return 1.0 - battery_current, 0.0, True
                return charge_val, 0.0, False

        if action['chargeCtl'] == 2:
            if battery_current <= 0:
                return 0.0, 0.0, True
            else:
                discharge_val = action['chargeVal'][0]  # * battery_max
                if battery_current - discharge_val <= 0:
                    return 0.0, battery_current, True
                return 0.0, discharge_val, False

    def _get_evs(self, action) -> (dict, dict, dict, dict):
        """
        Returns the connected state of the EV, the charge and discharge values, as well as the discharge cost.
        :param action:
        :return: connected, charge, discharge, cost
        """

        temp_connected = {}
        temp_charge = {}
        temp_discharge = {}
        temp_cost = {}
        temp_reward = {}
        for ev in self.ev.keys():
            is_connected = self.ev[ev]['connected'][self.current_timestep]
            charge, discharge, cost, illegal = self._get_ev(action[ev], ev, is_connected)
            # cost = 0.0
            ev_reward = 0.0

            # Check if the current timestep is the departure timestep
            ev_departures = np.array([self.ev[ev]['soc_required'][i][0]
                                      for i in range(len(self.ev[ev]['soc_required']))])
            ev_departure_req = np.array([self.ev[ev]['soc_required'][i][1]
                                         for i in range(len(self.ev[ev]['soc_required']))])

            ev_arrivals = np.array([self.ev[ev]['soc_arrival'][i][0]
                                    for i in range(len(self.ev[ev]['soc_arrival']))])
            ev_arrival_req = np.array([self.ev[ev]['soc_arrival'][i][1]
                                       for i in range(len(self.ev[ev]['soc_arrival']))])

            # Check if the EV is arriving -> no charging or discharging at this point
            if self.current_timestep in ev_arrivals:
                current_index = np.where(ev_arrivals == self.current_timestep)[0][0]
                self.ev[ev]['current'] = ev_arrival_req[current_index]
                charge = 0.0
                discharge = 0.0

            # Update the EV current SOC
            self.ev[ev]['current'] += (charge - discharge)

            if self.current_timestep in ev_departures:
                current_index = np.where(ev_departures == self.current_timestep)[0][0]
                if self.ev[ev]['current'] < ev_departure_req[current_index]:
                    ev_reward = -self.illegal_ev_request_cost
                    charge = 0.0
                    discharge = 0.0
                    self.ev[ev]['current'] = 0.0
                else:
                    ev_reward = self.illegal_ev_request_cost
                    self.ev[ev]['current'] -= ev_departure_req[current_index]

            # Assign values to the dictionary
            temp_charge[ev] = charge
            temp_discharge[ev] = discharge
            temp_cost[ev] = cost + illegal * self.illegal_ev_cost
            temp_reward[ev] = ev_reward

        return temp_charge, temp_discharge, temp_cost, temp_reward

    def _get_ev(self, action, ev, is_connected) -> (float, float, float, bool):
        """
        Takes the EV action and returns the charge and discharge values, as well as the illegal flag.
        :param action: Dictionary of EV actions
        :param ev: EV name
        :param is_connected: Boolean of whether the EV is connected
        :return: charge, discharge, cost, illegal
        """

        # evCtl = 0: no charge
        # evCtl = 1: charge
        # evCtl = 2: discharge
        # evVal: percentage of EV to charge/discharge

        if not is_connected:
            return 0.0, 0.0, 0.0, False
        else:
            if action['evCtl'] == 0:
                return 0.0, 0.0, 0.0, False
            elif action['evCtl'] == 1:
                charge = np.round(action['evVal'][0], 1)
                if charge + self.ev[ev]['current'] >= 1.0:
                    charge = 1.0 - self.ev[ev]['current']
                    return charge, 0.0, 0.0, True
                return charge, 0.0, 0.0, False
            elif action['evCtl'] == 2:
                discharge = np.round(action['evVal'][0], 1)
                if discharge > self.ev[ev]['current']:
                    discharge = self.ev[ev]['current']
                    return 0.0, discharge, discharge * self.ev[ev]['discharge_cost'] * self.ev[ev]['max'], True
                return 0.0, discharge, discharge * self.ev[ev]['discharge_cost'] * self.ev[ev]['max'], False

    def _get_grid(self, action) -> (float, float):
        # gridCtl = 0: no import/export
        # gridCtl = 1: import
        # gridCtl = 2: export
        # gridVal: percentage of energy to import/export

        imports: float = 0.0
        exports: float = 0.0

        if action['gridCtl'] == 0:
            return imports, exports
        elif action['gridCtl'] == 1:
            imports = action['gridVal'][0]
            return imports, exports
        elif action['gridCtl'] == 2:
            exports = action['gridVal'][0]
            return imports, exports

    def _get_reward(self, productions: float, batteries: float, imports: float, exports: float,
                    evs: float) -> float:
        import_costs = imports * self.grid_sells_price[self.current_timestep] \
            if imports <= 50.0 else self.illegal_import_cost
        export_costs = exports * self.grid_buys_price[self.current_timestep] \
            if exports <= 50.0 else -self.illegal_export_cost

        # EVs are summed, as we have a positive reward whenever the required SOC is met and negative otherwise.
        return -productions * 0.08 - batteries - import_costs + export_costs + evs
