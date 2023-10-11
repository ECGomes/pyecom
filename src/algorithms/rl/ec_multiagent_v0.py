import gymnasium as gym
import numpy as np
import random
from copy import copy
from collections import OrderedDict
from ...resources.base_resource import BaseResource

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class EnergyCommunityMultiEnv_v0(MultiAgentEnv):
    metadata = {'name': 'energy_community_multi_v0'}

    def __init__(self, resources: list[BaseResource]):
        super().__init__()

        # Define the resources
        self.resources = resources
        self.generators = [resource for resource in self.resources
                           if resource.name.startswith('gen')]
        self.storages = [resource for resource in self.resources if resource.name == 'storage']
        self.loads = [resource for resource in self.resources
                      if resource.name.startswith('load')]
        self.evs = [resource for resource in self.resources if resource.name == 'ev']
        #self.__separate_resources__()

        # Define illegal costs for components
        self.illegal_ev_cost = 0.0
        self.illegal_battery_cost = 0.0
        self.illegal_ev_request_cost = 20.0
        self.illegal_import_cost = 50.0
        self.illegal_export_cost = 50.0

        # Production specification
        self.gen_production = OrderedDict([(self.generators[i].name,
                                            self.generators[i].upper_bound)
                                           for i in range(len(self.generators))])

        self.gen_max = sum(self.gen_production.values())

        # Loads
        load_factor = 5.0
        self.load_consumption = OrderedDict([(self.loads[i].name,
                                              self.loads[i].value)
                                             for i in range(len(self.loads))])

        self.import_max = 100.0
        self.export_max = 100.0

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
                             # 'soc_arrival': np.array([(8, 0.5), (16, 0.4)]),
                             # 'soc_required': np.array([(12, 0.7), (18, 0.5)]),
                             'soc_arrival': np.array([(8, 20.0), (16, 16.0)]),
                             'soc_required': np.array([(12, 28.0), (18, 20.0)]),
                             'max_charge': 120.0},
                   'ev_02': {'max': 60.0, 'min': 0.2, 'current': 0.5, 'discharge_cost': 0.06,
                             'connected': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.int64),
                             # 'soc_arrival': np.array([(1, 0.4 * 60.0), (13, 0.3 * 60.0)]),
                             # 'soc_required': np.array([(2, 0.5 * 60.0), (23, 1.0 * 60.0)]),
                             'soc_arrival': np.array([(1, 24.0), (13, 18.0)]),
                             'soc_required': np.array([(2, 30.0), (23, 60.0)]),
                             'max_charge': 120.0},
                   'ev_03': {'max': 40.0, 'min': 0.2, 'current': 0.5, 'discharge_cost': 0.05,
                             'connected': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64),
                             # 'soc_arrival': np.array([(12, 0.4 * 40.0), (17, 0.3 * 40.0)]),
                             # 'soc_required': np.array([(14, 0.5 * 40.0), (20, 0.45 * 40.0)]),
                             'soc_arrival': np.array([(12, 16.0), (17, 12.0)]),
                             'soc_required': np.array([(14, 20.0), (20, 18.0)]),
                             'max_charge': 120.0},
                   'ev_04': {'max': 40.0, 'min': 0.2, 'current': 0.5, 'discharge_cost': 0.05,
                             'connected': np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                                                    0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
                             # 'soc_arrival': np.array([(1, 0.4 * 40.0), (13, 0.5 * 40.0)]),
                             # 'soc_required': np.array([(7, 0.7 * 40.0), (16, 0.65 * 40.0)]),
                             'soc_arrival': np.array([(1, 16.0), (13, 20.0)]),
                             'soc_required': np.array([(7, 28.0), (16, 26.0)]),
                             'max_charge': 120.0},
                   'ev_05': {'max': 60.0, 'min': 0.2, 'current': 0.5, 'discharge_cost': 0.06,
                             'connected': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                                                    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], dtype=np.int64),
                             # 'soc_arrival': np.array([(9, 0.4 * 60.0), (19, 0.3 * 60.0)]),
                             # 'soc_required': np.array([(11, 0.5 * 60.0), (22, 0.45 * 60.0)]),
                             'soc_arrival': np.array([(9, 24.0), (19, 18.0)]),
                             'soc_required': np.array([(11, 30.0), (22, 27.0)]),
                             'max_charge': 120.0}}

        # Overall values
        self.energy_requirements = sum(self.load_consumption.values())

        # Create agents
        self.possible_agents = ['gen', 'storage', 'ev']
        self.agents = self.__create_agents__()
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

        # Observation space
        self._obs_space_in_preferred_format = True
        temp_observation_space = {}
        for agent in self.agents:
            if agent.startswith('gen'):
                # Generators can see their current max power output;
                # As well as each storage and EV current state

                temp_observation = {'genVal': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
                                    'price_import': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
                                    'price_export': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32)}
                for storage in self.storage:
                    temp_observation[storage] = gym.spaces.Box(low=0.0, high=self.storage[storage]['max'],
                                                               shape=(1,),
                                                               dtype=np.float32)
                for ev in self.ev:
                    temp_observation[ev] = gym.spaces.Box(low=0.0, high=self.ev[ev]['max'],
                                                          shape=(1,),
                                                          dtype=np.float32)
                temp_observation_space[agent] = gym.spaces.Dict(temp_observation)

            elif agent.startswith('storage'):
                # Storage can see their current state of charge;
                # As well energy prices and maximum possible production

                temp_observation = {'storageVal': gym.spaces.Box(low=0.0,
                                                                 high=self.storage[agent]['max'],
                                                                 shape=(1,), dtype=np.float32),
                                    'price_import': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
                                    'price_export': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
                                    'genMax': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32)}

                temp_observation_space[agent] = gym.spaces.Dict(temp_observation)

            elif agent.startswith('ev'):
                # Similar to storage, EVs can see their current state of charge;
                # As well energy prices and maximum possible production
                # This information is complemented with the current EV request and time delta

                temp_observation = {'evVal': gym.spaces.Box(low=0.0,
                                                            high=self.ev[agent]['max'],
                                                            shape=(1,), dtype=np.float32),
                                    'evReq': gym.spaces.Box(low=0.0,
                                                            high=self.ev[agent]['max'],
                                                            shape=(1,), dtype=np.float32),
                                    'evDelta': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
                                    'evConnected': gym.spaces.Discrete(2),
                                    'price_import': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
                                    'price_export': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32),
                                    'genMax': gym.spaces.Box(low=0.0, high=300.0, shape=(1,), dtype=np.float32)}

                temp_observation_space[agent] = gym.spaces.Dict(temp_observation)
        self.observation_space = gym.spaces.Dict(temp_observation_space)

        # Action space
        self._action_space_in_preferred_format = True
        temp_action_space = {}
        for agent in self.agents:
            if agent.startswith('gen'):
                temp_action_space[agent] = gym.spaces.Discrete(2)
            elif agent.startswith('storage'):
                temp_action_space[agent] = gym.spaces.Dict(
                    {'storageCtl': gym.spaces.Discrete(3),
                     'storageVal': gym.spaces.Box(low=0.0,
                                                  high=self.storage[agent]['max'],
                                                  shape=(1,),
                                                  dtype=np.float32)}
                )
            elif agent.startswith('ev'):
                temp_action_space[agent] = gym.spaces.Dict(
                    {'evCtl': gym.spaces.Discrete(3),
                     'evVal': gym.spaces.Box(low=0.0,
                                             high=self.ev[agent]['max'],
                                             shape=(1,),
                                             dtype=np.float32)})
        self.action_space = gym.spaces.Dict(temp_action_space)

    # Separate resouces
    def __separate_resources__(self) -> None:
        """
        Separates the resources into their respective categories
        :return: None
        """

        generators = [resource for resource in self.resources if resource.name.startswith('gen')]
        loads = [resource for resource in self.resources if resource.name.startswith('load')]
        storage = [resource for resource in self.resources if resource.name.startswith('storage')]
        ev = [resource for resource in self.resources if resource.name.startswith('ev')]

        self.gen_production = {agent: self.gen_production[agent] for agent in self.gen_production if
                               agent.startswith('gen')}
        self.storage = {agent: self.storage[agent] for agent in self.storage if agent.startswith('storage')}
        self.ev = {agent: self.ev[agent] for agent in self.ev if agent.startswith('ev')}

    def __create_agents__(self) -> list:

        temp_agents = ['gen_{:02d}'.format(i + 1) for i in range(len(self.gen_production))] + \
                      ['storage_{:02d}'.format(i + 1) for i in range(len(self.storage))] + \
                      ['ev_{:02d}'.format(i + 1) for i in range(len(self.ev))]

        return temp_agents

    def _get_observations(self) -> dict:
        """
        Returns the observations of all agents
        :return: dict
        """

        if self.current_timestep == 24:
            return self.observation_space.sample()

        observations = {}
        for agent in self.agents:
            if agent.startswith('gen'):
                # Generators can see their current max power output;
                # As well as each storage and EV current state

                temp_observation = {'genVal': np.array([self.gen_production[agent][self.current_timestep]],
                                                       dtype=np.float32),
                                    'price_import': np.array([self.grid_sells_price[self.current_timestep]],
                                                             dtype=np.float32),
                                    'price_export': np.array([self.grid_buys_price[self.current_timestep]],
                                                             dtype=np.float32)}
                for storage in self.storage:
                    temp_observation[storage] = np.array([self.storage[storage]['current']],
                                                         dtype=np.float32)
                for ev in self.ev:
                    temp_observation[ev] = np.array([self.ev[ev]['current']],
                                                    dtype=np.float32)
                observations[agent] = temp_observation

            elif agent.startswith('storage'):
                # Storage can see their current state of charge;
                # As well energy prices and maximum possible production

                temp_observation = {'storageVal': np.array([self.storage[agent]['current']],
                                                           dtype=np.float32),
                                    'price_import': np.array([self.grid_sells_price[self.current_timestep]],
                                                             dtype=np.float32),
                                    'price_export': np.array([self.grid_buys_price[self.current_timestep]],
                                                             dtype=np.float32),
                                    'genMax': np.array([self.gen_max[self.current_timestep]],
                                                       dtype=np.float32)}

                observations[agent] = temp_observation

            elif agent.startswith('ev'):
                # Similar to storage, EVs can see their current state of charge;
                # As well energy prices and maximum possible production
                # This information is complemented with the current EV request and time delta

                next_required = np.where(self.ev[agent]['soc_required'][:, 0] > self.current_timestep)[0]
                next_required = self.ev[agent]['soc_required'][next_required[0]] \
                    if len(next_required) > 0 else (24.0, 0.0)

                temp_observation = {'evVal': np.array([self.ev[agent]['current']],
                                                      dtype=np.float32),
                                    'evReq': np.array([next_required[1]],
                                                      dtype=np.float32),
                                    'evDelta': np.array([next_required[0] - self.current_timestep],
                                                        dtype=np.float32),
                                    'evConnected': self.ev[agent]['connected'][self.current_timestep],
                                    'price_import': np.array([self.grid_sells_price[self.current_timestep]],
                                                             dtype=np.float32),
                                    'price_export': np.array([self.grid_buys_price[self.current_timestep]],
                                                             dtype=np.float32),
                                    'genMax': np.array([self.gen_max[self.current_timestep]],
                                                       dtype=np.float32)}

                observations[agent] = temp_observation

        return observations

    def reset(self, *, seed=None, options=None):
        self.agents = self.__create_agents__()
        self.terminateds = set()
        self.truncateds = set()

        # Reset the environment
        self.current_timestep = 0

        # Reset the storage
        for storage in self.storage.keys():
            self.storage[storage]['current'] = 0.8 * self.storage[storage]['max']

        # Reset the EVs
        for ev in self.ev.keys():
            self.ev[ev]['current'] = 0.5 * self.ev[ev]['max']

        observations = self._get_observations()

        return observations, {}

    def step(self, action_dict):

        if self.current_timestep >= 24:
            terminations = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}
            terminations['__all__'] = True
            truncations['__all__'] = True

            observations = self._get_observations()
            reward = {a: 0.0 for a in self.agents}
            info = {a: {} for a in self.agents}

            return observations, reward, terminations, truncations, info

        exists_actions = len(action_dict) > 0

        # Execute actions if they exist
        production = {'total': 0.0}
        storage_charge, storage_discharge, storage_cost = {'total': 0.0}, {'total': 0.0}, {'total': 0.0}
        ev_charge, ev_discharge, ev_cost, ev_reward = {'total': 0.0}, {'total': 0.0}, {'total': 0.0}, {'total': 0.0}
        if exists_actions:
            production = self._do_action_gen(action_dict)

            temp_storage = self._do_action_storage(action_dict)
            storage_charge = temp_storage[0]
            storage_discharge = temp_storage[1]
            storage_cost = temp_storage[2]

            temp_ev = self._do_action_ev(action_dict)
            ev_charge = temp_ev[0]
            ev_discharge = temp_ev[1]
            ev_cost = temp_ev[2]
            ev_reward = temp_ev[3]

        # Sums of values
        production_sum = sum(production.values())

        storage_charge_sum = sum(storage_charge.values())
        storage_discharge_sum = sum(storage_discharge.values())

        ev_charge_sum = sum(ev_charge.values())
        ev_discharge_sum = sum(ev_discharge.values())

        # Calculate energy requirements
        total_energy_requirements = sum([self.energy_requirements[self.current_timestep],
                                         -production_sum, storage_charge_sum, -storage_discharge_sum,
                                         ev_charge_sum, -ev_discharge_sum])

        # Consider the imports and exports into the total energy requirements
        imported = 0.0
        exported = 0.0
        if total_energy_requirements > 0.0:
            imported = total_energy_requirements
        elif total_energy_requirements < 0.0:
            exported = np.abs(total_energy_requirements)

        # Calculate shared cost of the community
        shared_cost = sum([imported * self.grid_sells_price[self.current_timestep]
                           if imported < self.import_max else self.illegal_import_cost,
                           -exported * self.grid_buys_price[self.current_timestep]
                           if exported < self.export_max else self.illegal_export_cost]) / len(self.agents)

        # Calculate the rewards
        rewards = {}

        # Rewards - Production
        for key in self.gen_production.keys():
            rewards[key] = self._get_rewards(production[key], 0.08, shared_cost)

        # Rewards - Storage
        for key in self.storage.keys():
            rewards[key] = self._get_rewards(storage_discharge[key],
                                             self.storage[key]['discharge_cost'],
                                             shared_cost)  # + storage_cost[key])

        # Rewards - EV
        for key in self.ev.keys():
            rewards[key] = self._get_rewards(ev_discharge[key],
                                             self.ev[key]['discharge_cost'],
                                             shared_cost + ev_cost[key])

        # Update the info dictionary
        info = {}
        for agent in self.agents:
            if agent.startswith('gen'):
                info[agent] = {'production': production[agent],
                               'reward': rewards[agent],
                               'imports': imported,
                               'exports': exported}
            elif agent.startswith('storage'):
                info[agent] = {'current': self.storage[agent]['current'],
                               'charge': storage_charge[agent],
                               'discharge': storage_discharge[agent],
                               'reward': rewards[agent]}
            elif agent.startswith('ev'):
                info[agent] = {'current': self.ev[agent]['current'],
                               'charge': ev_charge[agent],
                               'discharge': ev_discharge[agent],
                               'reward': rewards[agent]}

        # Termination and truncation
        terminations = {a: False for a in self.agents}
        terminations['__all__'] = False
        truncations = {a: False for a in self.agents}
        truncations['__all__'] = False

        if self.current_timestep >= 24:
            terminations = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}
            terminations['__all__'] = True
            truncations['__all__'] = True
            self.agents = []

        # Update the current timestep
        self.current_timestep += 1

        # Get the new observations
        observations = self._get_observations()

        return observations, rewards, terminations, truncations, info

    def _do_action_gen(self, action_dict) -> dict:
        """
        Executes the actions of the generators
        :param action_dict: dict
        :return: float
        """

        production = {}
        for agent in self.gen_production.keys():
            production[agent] = self.gen_production[agent][self.current_timestep] * action_dict[agent]

        return production

    def _do_action_storage(self, action) -> (dict, dict, dict):
        """
        Takes the storage actions and returns dictionaries of charge, discharge and costs
        :param action:
        :return: charge, discharge, costs, illegal
        """

        temp_charge = {}
        temp_discharge = {}
        temp_cost = {}

        for storage in self.storage.keys():
            charge, discharge, illegal = self._get_storage(action, storage)

            # Update the storage current SOC
            self.storage[storage]['current'] += (charge - discharge)

            temp_charge[storage] = charge
            temp_discharge[storage] = discharge
            temp_cost[storage] = sum([discharge * self.storage[storage]['discharge_cost'],
                                      illegal * self.illegal_battery_cost])

        return temp_charge, temp_discharge, temp_cost

    def _get_storage(self, action, storage) -> (float, float, bool):
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

        battery_current = self.storage[storage]['current']

        if action[storage]['storageCtl'] == 0:
            return 0.0, 0.0, False

        # Need to check if we can actually charge/discharge the battery
        if action[storage]['storageCtl'] == 1:
            if battery_current >= 1.0:
                return 0.0, 0.0, True
            else:
                charge_val = np.round(action[storage]['storageVal'][0], 1)
                if battery_current + charge_val > self.storage[storage]['max']:
                    return self.storage[storage]['max'] - battery_current, 0.0, True
                return charge_val, 0.0, False

        if action[storage]['storageCtl'] == 2:
            if battery_current <= 0:
                return 0.0, 0.0, True
            else:
                discharge_val = np.round(action[storage]['storageVal'][0], 1)
                if battery_current - discharge_val <= 0:
                    return 0.0, battery_current, True
                return 0.0, discharge_val, False

    def _do_action_ev(self, action) -> (dict, dict, dict, dict):
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
                if charge + self.ev[ev]['current'] >= self.ev[ev]['max']:
                    charge = self.ev[ev]['max'] - self.ev[ev]['current']
                    return charge, 0.0, 0.0, True
                return charge, 0.0, 0.0, False
            elif action['evCtl'] == 2:
                discharge = np.round(action['evVal'][0], 1)
                if discharge > self.ev[ev]['current']:
                    discharge = self.ev[ev]['current']
                    return 0.0, discharge, discharge * self.ev[ev]['discharge_cost'], True
                return 0.0, discharge, discharge * self.ev[ev]['discharge_cost'], False

    def _get_rewards(self, val, cost, shared_cost) -> float:
        """
        Returns the reward for the current timestep.
        :return: reward according to the excess or deficit of the energy for the community
        """

        return -np.abs(val) * cost - shared_cost
