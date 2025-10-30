import random
from copy import deepcopy
from typing import Union

import gymnasium as gym
import numpy as np
from functools import cached_property
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.resources import Generator, Load, Storage, Vehicle, Aggregator
from src.priorities import ContributionPriority

from numba import njit
import torch


@njit
def update_soc(soc, charge_amt, discharge_amt, eff_c, eff_d) -> float:
    return soc + charge_amt * eff_c - discharge_amt / eff_d


def set_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)


class EnergyCommunityContributionPriorityV6(MultiAgentEnv):
    """
    Based on the contribution priority, this environment is designed to
    simulate an energy community with multiple agents.
    Reward is given only when a day ends
    """

    metadata = {'name': 'EnergyCommunityContribPriority-v6'}

    @cached_property
    def ren_gen_actions(self):
        return np.round(np.arange(0.0, 1.1, 0.1), 1)

    @cached_property
    def battery_actions(self):
        return np.round(np.arange(-1.0, 1.1, 0.1), 1)

    @cached_property
    def ev_actions(self):
        return np.round(np.arange(-1.0, 1.1, 0.1), 1)

    @cached_property
    def gen_actions(self):
        return np.round(np.arange(0.0, 1.1, 0.1), 1)

    def __init__(self,
                 ren_generators: list[Generator],
                 loads: list[Load],
                 storages: list[Storage],
                 evs: list[Vehicle],
                 generators: list[Generator],
                 aggregator: Aggregator,
                 storage_penalty: float = 1.0,
                 ev_penalty: float = 1.0,
                 balance_penalty: float = 1.0,
                 look_ahead: int = 3,
                 max_episode_length: int = 24,
                 seed: int | None = None,
                 is_training: bool = True
                 ):
        super().__init__()

        # Set the seed
        self.seed = seed
        set_seed(seed)

        self.is_training = is_training

        # Initialize the resources and the environment
        self.original_resources = {'ren_generators': ren_generators,
                                   'loads': loads,
                                   'storages': storages,
                                   'evs': evs,
                                   'generators': generators,
                                   'aggregator': aggregator}

        # Look-ahead settings for agents
        self.look_ahead = look_ahead if look_ahead > 1 else 1
        self.max_episode_length = max_episode_length if (
                max_episode_length <= len(self.original_resources['ren_generators'][0].value)) else (
            len(self.original_resources['ren_generators'][0].value))

        # Initialize timestep
        self.timestep: int = 0

        # Maximum allowed timestep
        self.max_timestep = min(self.max_episode_length - 1,
                                len(self.original_resources['ren_generators'][0].value) - 1)
        self.time_of_day = 0

        # Initialize the environment
        self._reset()

        # Create indexes for resources to make it easier to lookup
        self.ev_index = {ev.name: i for i, ev in enumerate(self.evs)}
        self.storage_index = {storage.name: i for i, storage in enumerate(self.storages)}

        # Possible penalties
        self.storage_penalty = storage_penalty
        self.high_storage_penalty = storage_penalty ** 2  # Penalty for exchange of energy between storages and EVs
        self.ev_penalty = ev_penalty
        self.balance_penalty = balance_penalty

        # Handle observation and action spaces
        self._handle_observation_space()
        self._handle_action_space()

        # Reward accumulation
        self.global_rewards = []
        self.global_actions = {}

        # Import cost average
        self.import_cost_mean = np.round(np.mean(self.aggregator.import_cost), 2)

    # Initialize the environment
    def _reset(self):

        # Reset the timestep and resources if timestep reached max
        if self.timestep >= self.max_timestep:
            self.timestep = 0

        self.time_of_day = self.timestep % 96

        if self.timestep == 0:
            # Reset the resources
            self.resources = {
                k: [deepcopy(r) for r in v] if isinstance(v, list) else deepcopy(v)
                for k, v in self.original_resources.items()
            }
            self.ren_generators: list[Generator] = self.resources['ren_generators']
            self.loads: list[Load] = self.resources['loads']
            self.storages: list[Storage] = self.resources['storages']
            self.evs: list[Vehicle] = self.resources['evs']
            self.generators: list[Generator] = self.resources['generators']
            self.aggregator: Aggregator = self.resources['aggregator']

            # Define the execution order
            self.priority_system = ContributionPriority(self.storages + self.evs)
            self.priority_system.initialize_priority()

            # Set the global rewards to an empty list
            self.global_rewards = []

        # Sum of loads
        self.load_consumption: np.array = np.sum([load.value for load in self.loads], axis=0)
        self.gen_production: np.array = np.sum([gen.upper_bound for gen in self.ren_generators], axis=0)

        # Set the renewable generators to the maximum possible
        for ren_gen in self.ren_generators:
            ren_gen.value = ren_gen.upper_bound

        # Available overall and renewable energy for current timestep
        self.available_energy: float = self.gen_production[self.timestep] - self.load_consumption[self.timestep]

        priorities_for_timestep = self.priority_system.priorities.iloc[self.timestep]
        self.execution_order = np.append(priorities_for_timestep.sort_values(ascending=False).index.values,
                                         'aggregator')

        self.executed_agents = [False for _ in range(len(self.execution_order))]

        # Create the agents
        self.possible_agents = ['storage', 'ev', 'aggregator']
        self.agents = self.__create_agents__()
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

        # Current rewards
        self.current_rewards = {agent: 0.0 for agent in self.agents}

        # Agent execution variables
        self._current_agent_idx: int = 0
        self._previous_agent_idx: int = 0

        # Energy vector
        self.energy_history = []

    # Create agents
    def __create_agents__(self) -> dict:
        agents = {}
        for agent in np.arange(len(self.evs)):
            agents[str(self.evs[agent].name)] = self.evs[agent]

        for agent in np.arange(len(self.storages)):
            agents[str(self.storages[agent].name)] = self.storages[agent]

        agents['aggregator'] = self.aggregator

        agents_copy = deepcopy(agents)
        return agents_copy

    # Handle observation space
    def _handle_observation_space(self) -> None:

        self._obs_space_in_preferred_format = True
        temp_observation_space = {}

        # Storages
        storages = self.__create_storage_obs__()
        for storage in storages:
            temp_observation_space[storage] = storages[storage]

        # EVs
        evs = self.__create_ev_obs__()
        for ev in evs:
            temp_observation_space[ev] = evs[ev]

        # Aggregator
        temp_observation_space['aggregator'] = gym.spaces.Dict({
            'available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
            'import_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            'export_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        })

        self.observation_space = gym.spaces.Dict(temp_observation_space)

        return

    # Handle action space
    def _handle_action_space(self) -> None:

        self._act_space_in_preferred_format = True
        temp_action_space = {}

        # Storages
        storages = self.__create_storage_actions__()
        for storage in storages:
            temp_action_space[storage] = storages[storage]

        # EVs
        evs = self.__create_ev_actions__()
        for ev in evs:
            temp_action_space[ev] = evs[ev]

        temp_action_space['aggregator'] = gym.spaces.Discrete(1)

        self.action_space = gym.spaces.Dict(temp_action_space)

        return

    # Create Storage Observation Space
    def __create_storage_obs__(self) -> dict:
        """
        Create the observation space for the storages
        Each storage will have the following observations:
        - Current state of charge (float)
        - Current available energy (float)
        - Current available renewable energy (float)
        - Current buy price (float)
        - Current sell price (float)
        """
        storage_observations = {}
        for storage in self.storages:
            storage_observations[storage.name] = gym.spaces.Dict({
                'soc': gym.spaces.Box(low=0.0, high=storage.capacity_max, shape=(1,), dtype=np.float32),
                'capacity_max': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
                'maximum_charge': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'maximum_discharge': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'import_prices': gym.spaces.Box(low=0, high=1.0, shape=(self.look_ahead,), dtype=np.float32),
                'export_prices': gym.spaces.Box(low=0, high=1.0, shape=(self.look_ahead,), dtype=np.float32),
                'time_of_day': gym.spaces.Box(low=0, high=95, shape=(1,), dtype=np.int32),
                'predicted_production': gym.spaces.Box(low=0, high=99999.0, shape=(self.look_ahead,), dtype=np.float32),
                'predicted_consumption': gym.spaces.Box(low=0, high=99999.0, shape=(self.look_ahead,),
                                                        dtype=np.float32),
                'priority_score': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32)
            })

        return storage_observations

    # Get current Storage Observation
    def __get_storage_observations__(self, storage) -> dict:
        """
        Get the observations for the storages
        :param storage: storage resource to get the observations
        :return: dict
        """

        predicted_production = np.zeros(self.look_ahead, dtype=np.float32)
        predicted_consumption = np.zeros(self.look_ahead, dtype=np.float32)
        import_prices = np.zeros(self.look_ahead, dtype=np.float32)
        export_prices = np.zeros(self.look_ahead, dtype=np.float32)

        # Check for the lookahead
        start = self.timestep
        end = start + self.look_ahead
        actual_end = min(end, len(self.load_consumption))

        predicted_production[:actual_end - start] = self.gen_production[start:actual_end]
        predicted_consumption[:actual_end - start] = self.load_consumption[start:actual_end]
        import_prices[:actual_end - start] = self.aggregator.import_cost[start:actual_end]
        export_prices[:actual_end - start] = self.aggregator.export_cost[start:actual_end]

        current_soc = storage.value[self.timestep - 1] if self.timestep > 0 else storage.initial_charge
        current_soc = np.clip(current_soc, 0.0, storage.capacity_max)

        current_priority = self.priority_system.priorities.loc[self.timestep, storage.name]

        storage_observations: dict = {
            'soc': np.array([current_soc],
                            dtype=np.float32),
            'capacity_max': np.array([storage.capacity_max],
                                     dtype=np.float32),
            'available_energy': np.array([self.available_energy],
                                         dtype=np.float32),
            'maximum_charge': np.array([storage.charge_max[self.timestep]],
                                       dtype=np.float32),
            'maximum_discharge': np.array([storage.discharge_max[self.timestep]],
                                          dtype=np.float32),
            'import_prices': np.array(import_prices,
                                      dtype=np.float32),
            'export_prices': np.array(export_prices,
                                      dtype=np.float32),
            'time_of_day': np.array([self.time_of_day],
                                    dtype=np.int32),
            'predicted_production': np.array(predicted_production,
                                             dtype=np.float32),
            'predicted_consumption': np.array(predicted_consumption,
                                              dtype=np.float32),
            'priority_score': np.array([current_priority],
                                       dtype=np.float32)
        }

        return storage_observations

    # Create Storage Action Space
    def __create_storage_actions__(self) -> dict:
        """
        Create the action space for the storages
        Will have the following actions:
        - ctl: discrete values corresponding to the percentage of charge/discharge
        [-1.0, -0.9, -0.8, ..., 0.0, ..., 0.8, 0.9, 1.0]
        a total of 21 actions
        :return: dict
        """

        storage_actions = {}
        for storage in self.storages:
            #storage_actions[storage.name] = gym.spaces.Discrete(self.battery_actions.shape[0])
            storage_actions[storage.name] = gym.spaces.Box(low=-storage.discharge_max[0],
                                                           high=storage.charge_max[0],
                                                           dtype=np.float32,
                                                           shape=(1,))

        return storage_actions

    # Storage behaviour
    def __execute_storage_actions__(self, storage, actions) -> tuple[float, float]:
        """
        Execute the actions for the storages
        :param storage: storage resource
        :param actions: actions to be executed
        :return: reward to be used as penalty
        """

        # Initialize costs and penalties
        cost: float = 0.0
        penalty: float = 0.0

        # Initialize charge and discharge
        to_charge: float = 0.0
        to_discharge: float = 0.0

        # Get the index of the storage
        idx = self.storage_index.get(storage.name)

        # Check if it is the first timestep
        if self.timestep == 0:
            storage.value[self.timestep] = storage.initial_charge
            self.storages[idx].value[self.timestep] = storage.initial_charge
        else:
            storage.value[self.timestep] = storage.value[self.timestep - 1]
            self.storages[idx].value[self.timestep] = storage.value[self.timestep - 1]

        #storage_action = self.battery_actions[actions]
        storage_action = np.round(actions[0], 2)

        # Idle state
        if storage_action == 0.0:
            storage.charge[self.timestep] = 0.0
            storage.discharge[self.timestep] = 0.0

            # Update the resource values
            self.storages[idx].charge[self.timestep] = 0.0
            self.storages[idx].discharge[self.timestep] = 0.0

            if storage.value[self.timestep] < storage.capacity_min:
                penalty += (storage.capacity_min - storage.value[self.timestep]) * self.storage_penalty

        # Charge state
        elif storage_action > 0.0:
            # Check if we can charge
            to_charge: float = storage_action #* storage.charge_max[self.timestep]

            # Check if we can charge
            if storage.value[self.timestep] >= storage.capacity_max:
                # We are already at 100%
                penalty += to_charge * self.storage_penalty
                to_charge = 0.0

            elif storage.value[self.timestep] + to_charge > storage.capacity_max:
                # If we cannot charge, charge the maximum possible
                penalty += (storage.value[self.timestep] + to_charge - storage.capacity_max) * self.storage_penalty
                to_charge = abs(storage.capacity_max - storage.value[self.timestep])

            to_charge = np.round(to_charge, 4)

        # Discharge state
        elif storage_action < 0.0:

            to_discharge: float = abs(storage_action) #* storage.discharge_max[self.timestep]

            # Check if we can discharge
            if storage.value[self.timestep] <= storage.capacity_min:
                # If we are already at the minimum charge, we cannot discharge
                penalty += to_discharge * self.storage_penalty
                to_discharge = 0.0

            elif storage.value[self.timestep] - to_discharge \
                    < storage.capacity_min:
                # If we cannot discharge, discharge the maximum possible
                penalty += abs(storage.capacity_min - abs(storage.value[self.timestep] - to_discharge))
                to_discharge = storage.value[self.timestep] - storage.capacity_min

            to_discharge = np.round(to_discharge, 4)

        # Calculate the cost before energy pool update
        cost_before = (self.aggregator.import_cost[self.timestep] * max(-self.available_energy, 0.0) +
                       self.aggregator.export_cost[self.timestep] * max(self.available_energy, 0.0))

        # Update the available energy
        self.available_energy = self.available_energy - to_charge + to_discharge

        # Calculate the cost
        cost_action = (to_charge * storage.cost_charge[self.timestep]
                       + to_discharge * storage.cost_discharge[self.timestep])

        cost_after = (self.aggregator.import_cost[self.timestep] * max(-self.available_energy, 0.0) +
                      self.aggregator.export_cost[self.timestep] * max(self.available_energy, 0.0))

        # cost = cost_after - cost_before + 0.01 * self.import_cost_mean * to_discharge
        cost = cost_after - cost_before + cost_action

    # Update resource charge and discharge values
        storage.charge[self.timestep] = to_charge  # * storage.charge_efficiency
        storage.discharge[self.timestep] = to_discharge  # / storage.discharge_efficiency

        # Update the storage value
        new_storage_value = update_soc(storage.value[self.timestep],
                                       to_charge,
                                       to_discharge,
                                       storage.charge_efficiency,
                                       storage.discharge_efficiency)
        new_storage_value = np.round(new_storage_value, 4)
        new_storage_value = np.clip(new_storage_value, 0.0, storage.capacity_max)

        storage.value[self.timestep] = new_storage_value
        self.storages[idx].value[self.timestep] = new_storage_value
        self.storages[idx].charge[self.timestep] = to_charge
        self.storages[idx].discharge[self.timestep] = to_discharge

        return cost, penalty

    # Create EV Observation Space
    def __create_ev_obs__(self) -> dict:
        """
        Create the observation space for the EVs
        Each EV will have the following observations:
        - current_soc (float): Current SoC
        - current_available_energy (float): Current available energy
        - grid_connection (bool): Grid connection status
        - next_departure_time (int): Next departure time
        - time_until_next_departure (int): Time until next departure
        - next_departure_energy_requirement (float): Energy requirement for next departure
        - current_buy_price (float): Current buy price
        - current_sell_price (float): Current sell price
        - current_time (int): Current time
        :return: dict
        """

        ev_observations = {}
        for ev in self.evs:
            ev_observations[ev.name] = gym.spaces.Dict({
                'soc': gym.spaces.Box(low=0.0, high=ev.capacity_max, shape=(1,), dtype=np.float32),
                'capacity_max': gym.spaces.Box(low=0.0, high=ev.capacity_max, shape=(1,), dtype=np.float32),
                'available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
                'maximum_charge': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'maximum_discharge': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'grid_connections': gym.spaces.Box(low=0, high=1, shape=(self.look_ahead,), dtype=np.int32),
                'next_departure_energy_requirement': gym.spaces.Box(low=0, high=3.0, shape=(1,), dtype=np.float32),
                'import_prices': gym.spaces.Box(low=0, high=1.0, shape=(self.look_ahead,), dtype=np.float32),
                'export_prices': gym.spaces.Box(low=0, high=1.0, shape=(self.look_ahead,), dtype=np.float32),
                'current_time': gym.spaces.Box(low=0, high=95, shape=(1,), dtype=np.int32),
                'predicted_consumption': gym.spaces.Box(low=0, high=99999.0,
                                                        shape=(self.look_ahead,), dtype=np.float32),
                'predicted_production': gym.spaces.Box(low=0, high=99999.0, shape=(self.look_ahead,), dtype=np.float32),
                'priority_score': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32)

            })

        return ev_observations

    # Create EV Action Space
    def __create_ev_actions__(self) -> dict:
        """
        Create the action space for the EVs, same as storage
        Will have the following actions:
        - ctl: discrete values corresponding to the percentage of charge/discharge
        [-1.0, -0.9, -0.8, ..., 0.0, ..., 0.8, 0.9, 1.0]
        a total of 21 actions
        :return: dict
        """

        ev_actions = {}
        for ev in self.evs:
            ev_actions[ev.name] = gym.spaces.Discrete(self.ev_actions.shape[0])
            #ev_actions[ev.name] = gym.spaces.Box(low=-ev.schedule_discharge[0],
            #                                     high=ev.schedule_charge[0],
            #                                     shape=(1,),
            #                                     dtype=np.float32)

        return ev_actions

    # Get current EV Observation
    def __get_ev_observations__(self, ev) -> dict:
        """
        Get the observations for the EVs
        :param ev: EV resource
        :return: dict
        """

        # Get the next departure time and energy requirement
        '''
        next_departure = np.where(ev.schedule_requirement_soc > 0)[0]
        next_departure = next_departure[next_departure >= self.timestep]

        remains_trips = len(next_departure) > 0
        next_departure_soc = ev.schedule_requirement_soc[next_departure[0]] \
            if remains_trips else ev.min_charge
        '''
        future_soc = ev.schedule_requirement_soc[self.timestep:]
        relative_indices = np.where(future_soc > 0)[0]

        if relative_indices.size > 0:
            next_departure_soc = future_soc[relative_indices[0]]
        else:
            next_departure_soc = ev.min_charge

        grid_connections = np.zeros(self.look_ahead, dtype=np.int32)
        predicted_production = np.zeros(self.look_ahead, dtype=np.float32)
        predicted_consumption = np.zeros(self.look_ahead, dtype=np.float32)
        import_prices = np.zeros(self.look_ahead, dtype=np.float32)
        export_prices = np.zeros(self.look_ahead, dtype=np.float32)

        # Check if there are more than n steps remaining
        start = self.timestep
        end = start + self.look_ahead
        actual_end = min(end, len(self.load_consumption))

        grid_connections[:actual_end - start] = ev.schedule_connected[start:actual_end]
        predicted_production[:actual_end - start] = self.gen_production[start:actual_end]
        predicted_consumption[:actual_end - start] = self.load_consumption[start:actual_end]
        import_prices[:actual_end - start] = self.aggregator.import_cost[start:actual_end]
        export_prices[:actual_end - start] = self.aggregator.export_cost[start:actual_end]

        current_soc = ev.value[self.timestep - 1] if self.timestep > 0 else ev.initial_charge
        current_soc = np.clip(current_soc, 0.0, ev.capacity_max)

        current_priority = self.priority_system.priorities.loc[self.timestep, ev.name]

        ev_observations: dict = {
            'soc': np.array([current_soc],
                            dtype=np.float32),
            'capacity_max': np.array([ev.capacity_max],
                                     dtype=np.float32),
            'available_energy': np.array([self.available_energy],
                                         dtype=np.float32),
            'maximum_charge': np.array([ev.schedule_charge[self.timestep]],
                                       dtype=np.float32),
            'maximum_discharge': np.array([ev.schedule_discharge[self.timestep]],
                                          dtype=np.float32),
            'grid_connections': grid_connections,
            'next_departure_energy_requirement': np.array([next_departure_soc / ev.capacity_max],
                                                          dtype=np.float32),
            'import_prices': import_prices,
            'export_prices': export_prices,
            'current_time': np.array([self.time_of_day],
                                     dtype=np.int32),
            'predicted_production': predicted_production,
            'predicted_consumption': predicted_consumption,
            'priority_score': np.array([current_priority],
                                       dtype=np.float32)
        }

        return ev_observations

    # Execute EV Actions
    def __execute_ev_actions__(self, ev: Vehicle, actions) -> tuple[float, float]:
        """
        Execute the actions for the EVs
        :param ev: EV resource
        :param actions: actions to be executed
        :return: cost and penalty
        """

        # Initialize costs and penalties
        cost: float = 0.0
        penalty: float = 0.0

        # Initialize charge and discharge
        to_charge: float = 0.0
        to_discharge: float = 0.0

        # Get the index of the EV
        idx = self.ev_index.get(ev.name)

        # Check if it is the first timestep
        if self.timestep == 0:
            ev.value[self.timestep] = ev.initial_charge * ev.capacity_max
            self.evs[idx].value[self.timestep] = ev.initial_charge * self.evs[idx].capacity_max

        else:
            ev.value[self.timestep] = ev.value[self.timestep - 1]
            self.evs[idx].value[self.timestep] = self.evs[idx].value[self.timestep - 1]

        # First, check if the EV is not connected to the grid
        if ev.schedule_connected[self.timestep] == 0:
            # ev.value[self.current_timestep] = 0.0
            ev.charge[self.timestep] = 0.0
            ev.discharge[self.timestep] = 0.0

            # self.evs[idx].value[self.current_timestep] = 0.0
            self.evs[idx].charge[self.timestep] = 0.0
            self.evs[idx].discharge[self.timestep] = 0.0

            return cost, penalty

        # Else the EV is connected
        elif ev.schedule_connected[self.timestep] == 1:
            # Get the EV action
            ev_action = self.ev_actions[actions]
            #  ev_action = np.round(actions[0], 4)

            # Idle state
            if abs(ev_action) == 0.0:
                to_charge = 0.0
                to_discharge = 0.0

                # Update values
                ev.charge[self.timestep] = to_charge
                ev.discharge[self.timestep] = to_discharge

                # Update resource values
                self.evs[idx].charge[self.timestep] = to_charge
                self.evs[idx].discharge[self.timestep] = to_discharge

            # Charge state
            elif ev_action > 0.0:
                # Get the charge value
                to_charge: float = abs(ev_action) * ev.schedule_charge[self.timestep]
                to_discharge: float = 0.0

                # Check if we can charge
                if ev.value[self.timestep] >= ev.capacity_max:
                    # We are already at 100%
                    to_charge = 0.0

                elif ev.value[self.timestep] + to_charge > ev.capacity_max:
                    # If we cannot charge fully, charge the maximum possible
                    to_charge = np.round(abs(ev.capacity_max - ev.value[self.timestep]), 4)

            # Discharge state
            elif ev_action < 0.0:
                # Get the discharge value
                to_discharge: float = abs(ev_action) * ev.schedule_discharge[self.timestep]
                to_charge: float = 0.0

                # Check if we can discharge
                if ev.value[self.timestep] <= ev.min_charge:
                    # If we are already at the minimum charge, we cannot discharge
                    to_discharge = 0.0

                elif ev.value[self.timestep] - to_discharge < ev.min_charge:
                    # If we cannot discharge, discharge the maximum possible
                    to_discharge = abs(ev.value[self.timestep] - ev.min_charge)

            # Check if the there is a trip and if EV meets the energy requirement for the departure
            if self.evs[idx].schedule_requirement_soc[self.timestep] > 0.0:

                next_departure_soc = ev.schedule_requirement_soc[self.timestep]

                if (ev.value[self.timestep] - ev.min_charge) < next_departure_soc:
                    # Attribute penalty
                    penalty += self.ev_penalty

                    # Discharge the EV with the possible energy
                    ev.value[self.timestep] = ev.min_charge
                    self.evs[idx].value[self.timestep] = ev.min_charge

                else:
                    new_ev_val = ev.value[self.timestep] - next_departure_soc
                    ev.value[self.timestep] = new_ev_val
                    self.evs[idx].value[self.timestep] = new_ev_val

        # Update the EV costs
        cost = to_charge * ev.cost_charge[self.timestep] + to_discharge * ev.cost_discharge[self.timestep]

        self.available_energy = self.available_energy - to_charge + to_discharge

        new_ev_value = update_soc(ev.value[self.timestep],
                                  to_charge,
                                  to_discharge,
                                  ev.charge_efficiency,
                                  ev.discharge_efficiency)

        ev.value[self.timestep] = new_ev_value
        self.evs[idx].value[self.timestep] = new_ev_value
        self.evs[idx].charge[self.timestep] = to_charge
        self.evs[idx].discharge[self.timestep] = to_discharge

        # Check if the EV is under 0.2 SoC and penalize if so
        if ev.value[self.timestep] < ev.min_charge:
            penalty += self.ev_penalty

        return cost, penalty

    # Get the aggregator observations
    def __get_aggregator_observations__(self) -> dict:
        """
        Get the observations for the aggregator
        :return: dict
        """

        aggregator_observations: dict = {
            'available_energy': np.array([self.available_energy],
                                         dtype=np.float32),
            'import_price': np.array([self.aggregator.import_cost[self.timestep]],
                                     dtype=np.float32),
            'export_price': np.array([self.aggregator.export_cost[self.timestep]],
                                     dtype=np.float32)
        }

        return aggregator_observations

    # Execute the aggregator
    def __execute_aggregator__(self) -> tuple[float, float]:
        """
        Execute the aggregator. The aggregator will:
        - Buy energy from the grid if there is not enough provided
        - Sell energy to the grid if there is too much provided
        - Final objective is to minimize the costs
        :return: tuple
        """
        # Initialize costs and penalties
        # cost: float = 0.0
        penalty: float = 0.0
        energy_to_export: float = 0.0
        energy_to_import: float = 0.0

        # Check the current energy balance
        if self.available_energy > 0.0:
            # Then we have too much energy and need to export

            # Calculate the energy to be exported
            energy_to_export = deepcopy(self.available_energy)

            if energy_to_export > self.aggregator.export_max[self.timestep]:
                # Attribute penalty
                penalty += self.balance_penalty * (energy_to_export - self.aggregator.export_max[self.timestep])

                energy_to_export = self.aggregator.export_max[self.timestep]

        elif self.available_energy < 0.0:
            # Then we need to import energy

            # Calculate the energy to be imported
            energy_to_import = deepcopy(abs(self.available_energy))

            if energy_to_import > self.aggregator.import_max[self.timestep]:
                # Attribute penalty
                penalty += self.balance_penalty * (energy_to_import - self.aggregator.import_max[self.timestep])

                energy_to_import = self.aggregator.import_max[self.timestep]

        #cost = (energy_to_import * self.aggregator.import_cost[self.timestep] -
        #        energy_to_export * self.aggregator.export_cost[self.timestep])
        cost = energy_to_import * self.aggregator.import_cost[self.timestep]

        # Update resource values
        self.aggregator.imports[self.timestep] = energy_to_import
        self.aggregator.exports[self.timestep] = energy_to_export

        # Update the energy pool
        self.available_energy = self.available_energy - energy_to_export + energy_to_import

        # Track the energy balance as the aggregator value
        self.aggregator.value[self.timestep] = self.available_energy

        return cost, penalty

    # Reset the environment
    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed, options=options)

        # Reset the environment variables
        self._reset()

        observations = self._get_observations()
        info = self._log_info()

        return observations, info

    # Step function
    def step(self, action_dict: dict) -> tuple:

        # Initialize the observations, info and rewards
        reward = {}

        # Check for actions
        if len(action_dict) > 0:

            # Get the action of the current agent if it exists
            if self.execution_order[self._current_agent_idx] not in action_dict:
                observations = self._get_observations()
                info = self._log_info()
                terminateds, truncateds = self._log_ending(False)
                return observations, reward, terminateds, truncateds, info

            else:
                agent_name = self.execution_order[self._current_agent_idx]
                actions = action_dict[agent_name]

                # Dispatch the actions
                if agent_name.startswith('storage'):
                    current_res = self.storages[self.storage_index[agent_name]]
                    cost, penalty = self.__execute_storage_actions__(current_res, actions)
                    reward[agent_name] = - cost - penalty

                elif agent_name.startswith('ev'):
                    current_res = self.evs[self.ev_index[agent_name]]
                    cost, penalty = self.__execute_ev_actions__(current_res, actions)
                    reward[agent_name] = - cost - penalty

                elif agent_name.startswith('aggregator'):
                    # Add the current energy balance to the history for debug
                    self.energy_history.append(self.available_energy)
                    cost, penalty = self.__execute_aggregator__()
                    reward[agent_name] = - cost - penalty

                # Update the agent execution
                self.executed_agents[self._current_agent_idx] = True

                # Point to the next agent
                self._current_agent_idx = (self._current_agent_idx + 1) % len(self.execution_order)

                # Check if all agents have been executed
                if all(self.executed_agents):
                    # Reset the execution order
                    self.executed_agents = [False for _ in range(len(self.execution_order))]

                    # Check for episode end
                    if self.timestep >= self.max_timestep:
                        observations = self._get_observations()
                        info = self._log_info()
                        terminateds, truncateds = self._log_ending(True)
                        return observations, reward, terminateds, truncateds, info
                    else:
                        # Update the timestep
                        self.timestep += 1
                        self.time_of_day = self.timestep % 96

                        # Update the available energy
                        self.available_energy = self.gen_production[self.timestep] - self.load_consumption[
                            self.timestep]

                        # Calculate the contributions
                        contributions = self.create_contribution_dict()
                        self.priority_system.update_resources(contributions, self.timestep)
                        priorities_for_timestep = self.priority_system.priorities.iloc[self.timestep]
                        self.execution_order = np.append(priorities_for_timestep.sort_values(ascending=False).index.values,
                                                         'aggregator')

                        end_episode = False
                        # if self.timestep % 96 == 95 and self.is_training:
                        #     end_episode = True

                        observations = self._get_observations() if not end_episode else {}
                        info = self._log_info() if not end_episode else {}
                        terminateds, truncateds = self._log_ending(end_episode)
                        return observations, reward, terminateds, truncateds, info

                # Next observation
                observations = self._get_observations()
                info = self._log_info()
                terminateds, truncateds = self._log_ending(False)
                return observations, reward, terminateds, truncateds, info

        else:
            terminateds, truncateds = self._log_ending(True)
            return {}, {}, terminateds, truncateds, {}

    def _get_observations(self):
        current_agent_name = self.execution_order[self._current_agent_idx]
        observations = {}

        agent_type = current_agent_name.split('_')[0]  # assumes naming like 'storage_X', 'ev_Y'

        dispatch = {
            'storage': lambda name: self.__get_storage_observations__(
                next((s for s in self.storages if s.name == name), None)
            ),
            'ev': lambda name: self.__get_ev_observations__(
                next((ev for ev in self.evs if ev.name == name), None)
            ),
            'aggregator': lambda name: self.__get_aggregator_observations__()
        }

        if agent_type in dispatch:
            obs = dispatch[agent_type](current_agent_name)
            if obs is not None:
                observations[current_agent_name] = obs

        return observations

    def create_contribution_dict(self):
        """
        Create the contribution dictionary for the environment
        :return: dict
        Example dictionary:
        EV = {'discharge_cost': 0.2,
              'charge_cost': 0.2,
              'soc': 0.4,
              'connected': 1.0,
              'max_discharge_power': 20,
              'max_charge_power': 10,
              'grid_balance': -50},
        """

        contributions = {}
        for res in self.storages:
            contributions[res.name] = {'generation': np.sum(res.discharge[:self.timestep]),
                                       'consumption': np.sum(res.charge[:self.timestep])}
        for res in self.evs:
            contributions[res.name] = {'generation': np.sum(res.discharge[:self.timestep]),
                                       'consumption': np.sum(res.charge[:self.timestep])}



        return contributions

    # Log the episode truncations and terminations
    def _log_ending(self, flag: bool) -> tuple[dict, dict]:
        terminateds = {a: flag for a in self.agents}
        terminateds['__all__'] = flag
        truncateds = {a: flag for a in self.agents}
        truncateds['__all__'] = flag

        return terminateds, truncateds

    def _log_info(self) -> dict:

        # Check if there are keys on the reward
        return {'{}'.format(self.execution_order[self._current_agent_idx]): {}}
