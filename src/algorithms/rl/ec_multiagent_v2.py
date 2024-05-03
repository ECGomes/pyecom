from typing import Union

import gymnasium as gym
import numpy as np
from copy import deepcopy
from src.resources.base_resource import BaseResource
from src.algorithms.rl.utils import separate_resources

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class EnergyCommunityMultiEnv_v2(MultiAgentEnv):
    """
    Energy Community Environment for multi-agent reinforcement learning
    Generators can be renewable or non-renewable:
    - Renewable generators can be controlled, but not switched on/off
    - Non-renewable generators can be switched on/off, but not controlled
    Loads are fixed
    Storages can be controlled
    - Storages can be idle, charged or discharged
    - No efficiencies are considered (100% efficiency)
    EVs can be controlled
    - EVs can be charged or discharged
    - EVs can be connected or disconnected
    - EVs will charge/discharge depending on energy price
    Import/Export can be controlled with an input price

    Rewards are attributed to the community as a whole
    Rewards are based on the following:
    - Total energy consumption
    - Total energy production
    - Total energy storage
    - Total energy import
    - Total energy export
    """

    metadata = {'name': 'EnergyCommunitySequential-v0'}

    def __init__(self, resources: list[BaseResource],
                 import_penalty,
                 export_penalty,
                 storage_action_reward,
                 storage_action_penalty,
                 ev_action_reward,
                 ev_action_penalty,
                 ev_requirement_penalty,
                 balance_penalty):
        super().__init__()

        # Define the resources
        self.resources = deepcopy(resources)

        # Split the incoming resources
        temp_resources = separate_resources(self.resources)
        self.generators = temp_resources['generators']
        self.loads = temp_resources['loads']
        self.storages = temp_resources['storages']
        self.evs = temp_resources['evs']
        self.aggregator = temp_resources['aggregator'][0]

        # Calculate the sum of loads for each timestep
        self.load_consumption = np.sum([load.value for load in self.loads], axis=0)

        # Initialize time counter
        self.current_timestep: int = 0

        # Initialize variables for production and consumption
        self.current_production: float = 0
        self.current_consumption: float = 0

        # Initialize variables for currently available energy
        self.current_available_energy: float = 0

        # Create agents
        self.possible_agents = ['gen', 'storage', 'ev', 'aggregator']
        self.agents = self.__create_agents__()
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

        # Observation space
        self._handle_observation_space()

        # Action space
        self._handle_action_space()

        # Costs for each resource
        self.accumulated_generator_cost: float = 0.0
        self.accumulated_storage_cost: float = 0.0
        self.accumulated_ev_cost: float = 0.0
        self.accumulated_import_cost: float = 0.0
        self.accumulated_export_cost: float = 0.0

        # Rewards for each resource
        self.storage_action_reward: float = storage_action_reward
        self.ev_action_reward: float = ev_action_reward

        # Penalties
        self.storage_action_penalty = storage_action_penalty
        self.ev_action_penalty = ev_action_penalty
        self.ev_requirement_penalty = ev_requirement_penalty
        self.import_penalty = import_penalty
        self.export_penalty = export_penalty
        self.balance_penalty = balance_penalty

        # Set a penalty for each resource
        self.accumulated_generator_penalty: float = 0.0
        self.accumulated_storage_penalty: float = 0.0
        self.accumulated_ev_penalty: float = 0.0
        self.accumulated_ev_penalty_trip: float = 0.0
        self.accumulated_import_penalty: float = 0.0
        self.accumulated_export_penalty: float = 0.0

        # Balance history
        self.balance_history = []

        # History of the environment
        self.history = []
        self.history_dictionary = {}

        # Current real reward
        self.current_real_reward = {}

    # Create agents
    def __create_agents__(self) -> dict:
        agents = {}
        for agent in np.arange(len(self.generators)):
            agents[str(self.generators[agent].name)] = self.generators[agent]

        for agent in np.arange(len(self.evs)):
            agents[str(self.evs[agent].name)] = self.evs[agent]

        for agent in np.arange(len(self.storages)):
            agents[str(self.storages[agent].name)] = self.storages[agent]

        agents['aggregator'] = self.aggregator

        agents_copy = deepcopy(agents)
        return agents_copy

    # Handle Observation Space
    def _handle_observation_space(self) -> None:
        # Observation space
        self._obs_space_in_preferred_format = True
        temp_observation_space = {}

        # Generator observation space
        gens = self.__create_generator_observations__()
        for gen in gens.keys():
            temp_observation_space[gen] = gens[gen]

        # Storage observation space
        storages = self.__create_storage_observations__()
        for storage in storages.keys():
            temp_observation_space[storage] = storages[storage]

        # EV observation space
        evs = self.__create_ev_observations__()
        for ev in evs.keys():
            temp_observation_space[ev] = evs[ev]

        # Aggregator observation space
        temp_observation_space['aggregator'] = self.__create_aggregator_observations__()

        # Set the observation space
        self.observation_space = gym.spaces.Dict(temp_observation_space)

    # Handle Action Space
    def _handle_action_space(self) -> None:
        # Action space
        self._action_space_in_preferred_format = True
        temp_action_space = {}

        # Generator action space
        gens = self.__create_generator_actions__()
        for gen in gens.keys():
            temp_action_space[gen] = gens[gen]

        # Storage action space
        storages = self.__create_storage_actions__()
        for storage in storages.keys():
            temp_action_space[storage] = storages[storage]

        # EV action space
        evs = self.__create_ev_actions__()
        for ev in evs.keys():
            temp_action_space[ev] = evs[ev]

        # Aggregator action space
        temp_action_space['aggregator'] = self.__create_aggregator_actions__()

        # Set the action space
        self.action_space = gym.spaces.Dict(temp_action_space)

    # Create Generator Observation Space
    def __create_generator_observations__(self) -> dict:
        """
        Create the observation space for the generators
        Each generator will have the following observations:
        - Available Energy pool (float)
        - Current buy price (float)
        - Current sell price (float)
        - Current loads (float)
        :return: dict
        """

        generator_observations = {}
        for gen in self.generators:
            generator_observations[gen.name] = gym.spaces.Dict({
                'current_available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
                'current_buy_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_sell_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_loads': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32)
            })

        return generator_observations

    # Handle generator observations
    def __get_generator_observations__(self) -> dict:
        """
        Get the observations for one generator
        :return: dict
        """

        generator_observations: dict = {
            'current_available_energy': np.array([self.current_available_energy],
                                                 dtype=np.float32),
            'current_buy_price': np.array([self.aggregator.import_cost[self.current_timestep]],
                                          dtype=np.float32),
            'current_sell_price': np.array([self.aggregator.export_cost[self.current_timestep]],
                                           dtype=np.float32),
            'current_loads': np.array([self.load_consumption[self.current_timestep]],
                                      dtype=np.float32)
        }

        return generator_observations

    # Create Generator Action Space
    def __create_generator_actions__(self) -> dict:
        """
        Create the action space for the generators
        Varies according to the resource's renewable variable
        Renewable generators will have the following actions:
        - production (float) -> Renewable generators can control their production
        Non-renewable generators will have the following actions:
        - active (bool)

        :return: dict
        """

        generator_actions = {}
        for gen in self.generators:
            # renewable = False
            # if isinstance(gen.is_renewable, bool):
            #     renewable = gen.is_renewable

            if gen.is_renewable == 2.0 or gen.is_renewable is True:  # Hack for the Excel file
                generator_actions[gen.name] = gym.spaces.Dict({
                    'production': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
                })
            else:
                generator_actions[gen.name] = gym.spaces.Dict({
                    'active': gym.spaces.Discrete(2)
                })

        return generator_actions

    # Execute generator actions
    def __execute_generator_actions__(self, gen, actions) -> tuple[float, float]:
        """
        Execute the actions for the generators
        :param gen: generator resource
        :param actions: actions to be executed
        :return: float
        """

        # Calculate the cost of the generator
        penalty: float = 0.0

        # Placeholder for produced energy
        produced_energy: float = 0.0

        # Check if actions has active or production
        if 'active' in actions.keys():
            produced_energy = (actions['active'] *
                               gen.upper_bound[self.current_timestep])
        elif 'production' in actions.keys():
            produced_energy = (actions['production'][0] *
                               gen.upper_bound[self.current_timestep])

        # Attribute the produced energy to the generator
        gen.value[self.current_timestep] = produced_energy
        self.current_production += produced_energy
        self.current_available_energy += produced_energy

        # Update on the resource
        # Get the index of the corresponding name
        idx = [i for i, resource in enumerate(self.generators) if resource.name == gen.name][0]
        self.generators[idx].value[self.current_timestep] = produced_energy

        cost: float = gen.upper_bound[self.current_timestep] - produced_energy

        return cost, penalty

    # Create Storage Observation Space
    def __create_storage_observations__(self) -> dict:
        """
        Create the observation space for the storages
        Each storage will have the following observations:
        - current_soc (float): Current SoC
        - current_available_energy (float): Current available energy
        - current_loads (float): Current loads
        - current_buy_price (float): Current buy price
        - current_sell_price (float): Current sell price
        :return: dict
        """

        storage_observations = {}
        for storage in self.storages:
            storage_observations[storage.name] = gym.spaces.Dict({
                'current_soc': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
                'current_loads': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'current_buy_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_sell_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })

        return storage_observations

    # Handle storage observations
    def __get_storage_observations__(self, storage) -> dict:
        """
        Get the observations for the storages
        :param storage: storage resource to get the observations
        :return: dict
        """

        storage_observations: dict = {
            'current_soc': np.array([storage.value[self.current_timestep - 1] if self.current_timestep > 0
                                     else storage.initial_charge],
                                    dtype=np.float32),
            'current_available_energy': np.array([self.current_available_energy],
                                                 dtype=np.float32),
            'current_loads': np.array([self.load_consumption[self.current_timestep]],
                                      dtype=np.float32),
            'current_buy_price': np.array([self.aggregator.import_cost[self.current_timestep]],
                                          dtype=np.float32),
            'current_sell_price': np.array([self.aggregator.export_cost[self.current_timestep]],
                                           dtype=np.float32)
        }

        return storage_observations

    # Create Storage Action Space
    def __create_storage_actions__(self) -> dict:
        """
        Create the action space for the storages
        Will have the following actions:
        - ctl: control the storage (bool) -> 0/1/2 for none/charge/discharge
        - value: value to be charged or discharged (float)
        :return: dict
        """

        storage_actions = {}
        for storage in self.storages:
            storage_actions[storage.name] = gym.spaces.Dict({
                'ctl': gym.spaces.Discrete(3),
                'value': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })

        return storage_actions

    # Execute storage actions
    def __execute_storage_actions__(self, storage, actions) -> tuple[float, float]:
        """
        Execute the actions for the storages
        :param storage: storage resource
        :param actions: actions to be executed
        :return: reward to be used as penalty
        """

        # Calculate the cost of the storage
        cost: float = 0.0

        # Set up the penalty to be returned in case of illegal storage actions
        # Such as overcharging or discharging beyond the available energy.
        # The reward will be the deviation from the bounds, to be later used as a penalty
        penalty: float = 0.0

        idx = [i for i, resource in enumerate(self.storages) if resource.name == storage.name][0]

        # Check if it is the first timestep
        if self.current_timestep == 0:
            storage.value[self.current_timestep] = storage.initial_charge
            self.storages[idx].value[self.current_timestep] = storage.initial_charge
        else:
            storage.value[self.current_timestep] = storage.value[self.current_timestep - 1]
            self.storages[idx].value[self.current_timestep] = storage.value[self.current_timestep - 1]

        # Idle state
        if actions['ctl'] == 0:
            storage.charge[self.current_timestep] = 0.0
            storage.discharge[self.current_timestep] = 0.0

            return cost, penalty

        # Charge state
        elif actions['ctl'] == 1:
            charge = actions['value'][0]

            # Set the charge as a percentage of the maximum charge allowed
            charge = charge * storage.charge_max[self.current_timestep] / storage.capacity_max

            if storage.value[self.current_timestep] + charge > 1.0:
                # Calculate the deviation from the bounds
                deviation = storage.value[self.current_timestep] + charge - 1.0
                charge = 1.0 - storage.value[self.current_timestep]
                penalty = deviation

            # Get the cost of the energy
            # cost = charge * storage.cost_charge[self.current_timestep]

            # Heavily penalize the storage action if it requires importing energy
            if self.current_available_energy - charge * storage.capacity_max < 0:
                penalty += self.storage_action_penalty
            elif self.current_available_energy - charge * storage.capacity_max > 0:
                penalty -= self.storage_action_reward

            # Remove energy from the pool
            self.current_available_energy -= charge * storage.capacity_max

            # Assign resource charge and discharge variables
            storage.value[self.current_timestep] += charge
            storage.charge[self.current_timestep] = charge
            storage.discharge[self.current_timestep] = 0.0

            # Update as well on the resource
            self.storages[idx].value[self.current_timestep] += charge
            self.storages[idx].charge[self.current_timestep] = charge
            self.storages[idx].discharge[self.current_timestep] = 0.0

            return cost, penalty

        # Discharge state
        elif actions['ctl'] == 2:
            discharge = actions['value'][0]

            # Set discharge as a percentage of the maximum discharge allowed
            discharge = discharge * storage.discharge_max[self.current_timestep] / storage.capacity_max

            if storage.value[self.current_timestep] - discharge < 0.0:
                # Calculate the deviation from the bounds
                deviation = abs(storage.value[self.current_timestep] - discharge)
                discharge = storage.value[self.current_timestep]
                penalty = deviation

            # Get the cost of the energy
            # cost = discharge * storage.cost_discharge[self.current_timestep]
            cost = storage.discharge_max[self.current_timestep] - discharge * storage.capacity_max

            # Assign resource charge and discharge variables
            storage.value[self.current_timestep] -= discharge
            storage.charge[self.current_timestep] = 0.0
            storage.discharge[self.current_timestep] = discharge

            # Update as well on the resource
            self.storages[idx].value[self.current_timestep] -= discharge
            self.storages[idx].charge[self.current_timestep] = 0.0
            self.storages[idx].discharge[self.current_timestep] = discharge

            # Add the energy to the pool
            self.current_available_energy += discharge * storage.capacity_max

            return cost, penalty

        return cost, penalty

    # Create EV Observation Space
    def __create_ev_observations__(self) -> dict:
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
                'current_soc': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
                'grid_connection': gym.spaces.Discrete(2),
                'next_departure_time': gym.spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32),
                'time_until_next_departure': gym.spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32),
                'next_departure_energy_requirement': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_buy_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_sell_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_time': gym.spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32)
            })

        return ev_observations

    # Handle EV observations
    def __get_ev_observations__(self, ev) -> dict:
        """
        Get the observations for the EVs
        :param ev: EV resource
        :return: dict
        """

        # Get the next departure time and energy requirement
        next_departure = np.where(ev.schedule_requirement_soc > 0)[0]
        # print('Next departure', next_departure)
        next_departure = next_departure[next_departure >= self.current_timestep]

        remains_trips = len(next_departure) > 0
        next_departure_soc = ev.schedule_requirement_soc[next_departure[0]] \
            if remains_trips else ev.min_charge

        time_until_departure = abs(next_departure[0] - self.current_timestep) \
            if remains_trips else ev.schedule_connected.shape[0] - 1

        next_departure = next_departure[0] if remains_trips else 9999

        ev_observations: dict = {
            'current_soc': np.array([ev.value[self.current_timestep - 1] if self.current_timestep > 0
                                     else ev.initial_charge],
                                    dtype=np.float32),
            'current_available_energy': np.array([self.current_available_energy],
                                                 dtype=np.float32),
            'grid_connection': int(ev.schedule_connected[self.current_timestep]),
            'next_departure_time': np.array([next_departure],
                                            dtype=np.int32),
            'time_until_next_departure': np.array([time_until_departure],
                                                  dtype=np.int32),
            'next_departure_energy_requirement': np.array([next_departure_soc / ev.capacity_max],
                                                          dtype=np.float32),
            'current_buy_price': np.array([self.aggregator.import_cost[self.current_timestep]],
                                          dtype=np.float32),
            'current_sell_price': np.array([self.aggregator.export_cost[self.current_timestep]],
                                           dtype=np.float32),
            'current_time': np.array([self.current_timestep],
                                     dtype=np.int32)
        }

        return ev_observations

    # Create EV Action Space
    def __create_ev_actions__(self) -> dict:
        """
        Create the action space for the EVs
        Will have the following actions:
        - ctl: control the EV -> 0/1/2 for none/charge/discharge
        - value: value to be charged or discharged (float)

        :return: dict
        """

        ev_actions = {}
        for ev in self.evs:
            ev_actions[ev.name] = gym.spaces.Dict({
                'ctl': gym.spaces.Discrete(3),
                'value': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })

        return ev_actions

    # Execute EV actions
    def __execute_ev_actions__(self, ev, actions) -> tuple[float, float, float]:
        """
        Execute the actions for the EVs
        :param ev: EV resource
        :param actions: actions to be executed
        :return: cost and penalty to be used
        """

        # Set up the cost of the EVs to return
        cost: float = 0.0

        # Set up the reward to be returned in case of illegal EV actions
        # Such as overcharging or discharging beyond the available energy.
        # The reward will be the deviation from the bounds, to be later used as a penalty
        # Also includes the penalty for not meeting the energy requirement for the next departure
        penalty: float = 0.0

        # Set up the penalty for not meeting the energy requirement for the next departure
        penalty_trip: float = 0.0

        idx = [i for i, resource in enumerate(self.evs) if resource.name == ev.name][0]

        # Check if it's the first timestep
        if self.current_timestep == 0:
            ev.value[self.current_timestep] = ev.initial_charge
            self.evs[idx].value[self.current_timestep] = ev.initial_charge
        else:
            ev.value[self.current_timestep] = ev.value[self.current_timestep - 1]
            self.evs[idx].value[self.current_timestep] = ev.value[self.current_timestep - 1]

        # First, check if the EV is connected to the grid
        if ev.schedule_connected[self.current_timestep] == 0:
            # ev.value[self.current_timestep] = 0.0
            ev.charge[self.current_timestep] = 0.0
            ev.discharge[self.current_timestep] = 0.0

            # self.evs[idx].value[self.current_timestep] = 0.0
            self.evs[idx].charge[self.current_timestep] = 0.0
            self.evs[idx].discharge[self.current_timestep] = 0.0

            # Check if the there is a trip and if EV meets the energy requirement for the departure
            if self.evs[idx].schedule_requirement_soc[self.current_timestep] > 0:
                next_departure_soc = ev.schedule_requirement_soc[self.current_timestep]

                # print('EV departure')

                if ev.value[self.current_timestep] < next_departure_soc / ev.capacity_max:
                    # Calculate the deviation from the value
                    deviation = next_departure_soc - ev.value[self.current_timestep]
                    penalty_trip = deviation

                    # Discharge the EV with the possible energy
                    ev.value[self.current_timestep] = 0.0
                    self.evs[idx].value[self.current_timestep] = 0.0

                else:
                    ev.value[self.current_timestep] = ((ev.value[self.current_timestep] * ev.capacity_max -
                                                        next_departure_soc) / ev.capacity_max)
                    self.evs[idx].value[self.current_timestep] = ev.value[self.current_timestep]

            return cost, penalty, penalty_trip

        # Check if EV is arriving and there is a SOC on arrival
        if self.evs[idx].schedule_arrival_soc is not None:
            if self.evs[idx].schedule_arrival_soc[self.current_timestep] > 0.0:
                ev.value[self.current_timestep] = (self.evs[idx].schedule_arrival_soc[self.current_timestep]
                                                   / ev.capacity_max)

                self.evs[idx].value[self.current_timestep] = ev.value[self.current_timestep]
        else:
            self.evs[idx].schedule_arrival_soc[self.current_timestep] = ev.capacity_max * 0.2 # Arrives with minimum SOC

        # Idle state
        if actions['ctl'] == 0:
            ev.charge[self.current_timestep] = 0.0
            ev.discharge[self.current_timestep] = 0.0

        # Charge state
        elif actions['ctl'] == 1:
            charge = actions['value'][0]

            # Set charge to a percentage of the maximum charge allowed
            charge = charge * ev.schedule_charge[self.current_timestep] / ev.capacity_max

            if ev.value[self.current_timestep] + charge > 1.0:
                # Calculate the deviation from the bounds
                deviation = ev.value[self.current_timestep] + charge - 1.0
                charge = 1.0 - ev.value[self.current_timestep]
                penalty = deviation

            # Calculate the cost of charging
            # cost = charge * ev.cost_charge[self.current_timestep]

            if self.current_available_energy - charge * ev.capacity_max > 0:
                penalty -= self.ev_action_reward

            # Remove energy from the pool
            self.current_available_energy -= charge * ev.capacity_max

            # Assign resource charge and discharge variables
            ev.charge[self.current_timestep] = charge
            ev.discharge[self.current_timestep] = 0.0

        # Discharge state
        elif actions['ctl'] == 2:
            discharge = actions['value'][0]

            # Set discharge to a percentage of the maximum discharge allowed
            discharge = discharge * ev.schedule_discharge[self.current_timestep] / ev.capacity_max

            if ev.value[self.current_timestep] - discharge < 0.0:
                # Calculate the deviation from the bounds
                deviation = abs(ev.value[self.current_timestep] - discharge)
                discharge = ev.value[self.current_timestep]
                penalty = deviation

            # Calculate the cost of discharging
            # cost = discharge * ev.cost_discharge[self.current_timestep]

            # Add discharged energy to the pool
            self.current_available_energy += discharge * ev.capacity_max

            # Assign resource charge and discharge variables
            ev.charge[self.current_timestep] = 0.0
            ev.discharge[self.current_timestep] = discharge

        # Update EV storage
        ev.value[self.current_timestep] = (ev.value[self.current_timestep] +
                                           ev.charge[self.current_timestep] -
                                           ev.discharge[self.current_timestep])

        # Update the resource
        self.evs[idx].value[self.current_timestep] = ev.value[self.current_timestep]
        self.evs[idx].charge[self.current_timestep] = ev.charge[self.current_timestep]
        self.evs[idx].discharge[self.current_timestep] = ev.discharge[self.current_timestep]

        return cost, penalty, penalty_trip

    # Create Aggregator Observation Space
    def __create_aggregator_observations__(self) -> gym.spaces.Dict:
        """
        Create the observation space for the aggregator
        Will have the following observations:
        - current_buy_price (float): Current buy price
        - current_sell_price (float): Current sell price
        - current_available_energy (float): Current available energy
        :return: dict
        """

        return gym.spaces.Dict({
            'current_buy_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            'current_sell_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            'current_available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32)
        })

    # Handle aggregator observations
    def __get_aggregator_observations__(self) -> dict:
        """
        Get the observations for the aggregator
        :return: dict
        """

        return {
            'current_buy_price': np.array([self.aggregator.import_cost[self.current_timestep]],
                                          dtype=np.float32),
            'current_sell_price': np.array([self.aggregator.export_cost[self.current_timestep]],
                                           dtype=np.float32),
            'current_available_energy': np.array([self.current_available_energy],
                                                 dtype=np.float32)
        }

    # Create Aggregator Action Space
    def __create_aggregator_actions__(self) -> gym.spaces.Dict:
        """
        Create the action space for the aggregator
        Will have the following actions:
        - ctl: action to take 0/1/2 for none/import/export
        - value: value to be imported or exported (float)

        :return: dict
        """

        return gym.spaces.Dict({
            'ctl': gym.spaces.Discrete(3),
            'value': gym.spaces.Box(low=0, high=max(self.aggregator.import_max),
                                    shape=(1,), dtype=np.float32)
        })

    # Execute aggregator actions
    def __execute_aggregator_actions__(self, agent, actions) -> tuple[float, float]:
        """
        Execute the actions for the aggregator
        :param actions: actions to be executed
        :return: aggregator costs and penalty
        """

        # Set up the aggregator's costs
        cost: float = 0.0

        # Set up the reward to be returned in case of illegal aggregator actions
        # Such as importing or exporting beyond the allowed limits.
        # The reward will be the deviation from the bounds, to be later used as a penalty
        penalty: float = 0.0

        # If we still have energy left, there is no need to import extra energy
        if self.current_available_energy > 0:

            # Force to export
            # 1 - Get the deviation from the bounds
            to_export = self.current_available_energy
            if to_export > agent.export_max[self.current_timestep]:
                deviation = to_export - agent.export_max[self.current_timestep]
                to_export = agent.export_max[self.current_timestep]
                penalty = deviation

            # 2 - Set the exports for agent and resource
            agent.imports[self.current_timestep] = 0.0
            agent.exports[self.current_timestep] = to_export

            self.aggregator.imports[self.current_timestep] = 0.0
            self.aggregator.exports[self.current_timestep] = to_export

            # 3 - Update the available energy pool
            self.current_available_energy -= to_export

            # Update the cost of the export
            cost = to_export * self.aggregator.export_cost[self.current_timestep]

            return -cost, penalty

        # Check if there is a defect of energy that needs to be imported
        if self.current_available_energy < 0:

            # If not, we are forced to import
            to_import = abs(self.current_available_energy)
            if to_import > agent.import_max[self.current_timestep]:
                # Calculate the deviation from the bounds
                deviation = to_import - agent.import_max[self.current_timestep]
                to_import = agent.import_max[self.current_timestep]
                penalty = deviation

            # Set the imports for agent and resource
            agent.imports[self.current_timestep] = to_import
            agent.exports[self.current_timestep] = 0.0

            self.aggregator.imports[self.current_timestep] = to_import
            self.aggregator.exports[self.current_timestep] = 0.0

            # Update the available energy pool
            self.current_available_energy += to_import

            # Get the associated costs of importation
            cost = to_import * agent.import_cost[self.current_timestep]

            return cost, penalty

        return cost, penalty

    # Get observations
    def __get_next_observations__(self) -> dict:
        """
        Get the observations for the environment
        :return: dict
        """

        # Get the observation for the next resource
        observations = {}

        if self.current_timestep >= self.loads[0].value.shape[0]:
            return observations

        for agent in self.agents.keys():
            if agent.startswith('generator'):
                observations[agent] = self.__get_generator_observations__()
            elif agent.startswith('storage'):
                observations[agent] = self.__get_storage_observations__(self.agents[agent])
            elif agent.startswith('ev'):
                observations[agent] = self.__get_ev_observations__(self.agents[agent])
            elif agent.startswith('aggregator'):
                observations[agent] = self.__get_aggregator_observations__()

        return observations

    # Reset environment
    def reset(self, *, seed=None, options=None):

        # Define the resources
        temp_resources = deepcopy(self.resources)

        # Split the incoming resources
        temp_resources = separate_resources(temp_resources)
        self.generators = temp_resources['generators']
        self.loads = temp_resources['loads']
        self.storages = temp_resources['storages']
        self.evs = temp_resources['evs']
        self.aggregator = temp_resources['aggregator'][0]

        # Create new agents
        self.agents = self.__create_agents__()

        # Set the initial pool of available energy to load consumption at the first timestep
        self.current_available_energy = -self.load_consumption[0]

        # Set the termination and truncation flags
        self.terminateds = set()
        self.truncateds = set()

        # Reset environment variables
        self.current_timestep = 0
        self.current_production = 0
        self.current_consumption = 0
        self.current_available_energy = 0

        # Reset penalties and costs
        self.accumulated_generator_cost = 0.0
        self.accumulated_storage_cost = 0.0
        self.accumulated_ev_cost = 0.0
        self.accumulated_import_cost = 0.0
        self.accumulated_export_cost = 0.0

        self.accumulated_generator_penalty = 0.0
        self.accumulated_storage_penalty = 0.0
        self.accumulated_ev_penalty = 0.0
        self.accumulated_ev_penalty_trip = 0.0
        self.accumulated_import_penalty = 0.0
        self.accumulated_export_penalty = 0.0

        # Clear the history
        self.balance_history = []
        self.history = []
        self.history_dictionary = {}

        observations = self.__get_next_observations__()

        return observations, {}

    # Step function for environment transitions
    def step(self, action_dict: dict) -> tuple:
        """
        Step function for environment transitions
        Agents will act in the following order:
        1. Generators
        2. EVs
        3. Storages
        4. Aggregator

        :param action_dict: dict
        :return: tuple
        """

        # Check for completion of the episode
        if self.current_timestep >= self.loads[0].value.shape[0]:
            terminateds, truncateds = self._log_ending(True)
            observations, reward, info = {}, {}, {}

            return observations, reward, terminateds, truncateds, info

        # Check for existing actions
        exists_actions = len(action_dict) > 0

        # Observations
        observations = {}

        # Reward
        reward: dict = {}

        # Info dictionary
        info: dict = {}

        # Execute the actions
        if exists_actions:

            # Do the actions
            for action in action_dict.keys():
                # Execute the actions
                action_result_00, action_result_01, action_result_02 = self._execute_action(self.agents[action],
                                                                                            action_dict[action])

                # Log the agent
                self._log_agents(deepcopy(self.agents[action]))

                # Calculate the true reward
                self.current_real_reward[self.agents[action].name] = self._calculate_reward(self.agents[action],
                                                                                            action_result_00,
                                                                                            action_result_01,
                                                                                            action_result_02)

            # Update the timestep
            self.current_timestep += 1

            self.balance_history.append(self.current_available_energy)
            self.history.append(self.history_dictionary)

            # Terminations and truncations
            terminateds, truncateds = self._log_ending(False)

            # Check for end of episode
            if self.current_timestep >= self.loads[0].value.shape[0]:
                terminateds, truncateds = self._log_ending(True)
                self.agents = []

            # Update the timestep
            self.current_timestep += 1

            next_obs = self.__get_next_observations__()

            return next_obs, self.current_real_reward, terminateds, truncateds, info

        # Terminations and truncations
        terminateds = {a: False for a in self.agents}
        terminateds['__all__'] = False
        truncateds = {a: False for a in self.agents}
        truncateds['__all__'] = False

        return observations, reward, terminateds, truncateds, info

    # Handle action execution
    def _execute_action(self, agent, actions) -> tuple[float, float, float]:
        # Execute the actions for the generators
        if agent.name.startswith('generator'):
            generator_cost, generator_penalty = self.__execute_generator_actions__(agent,
                                                                                   actions)
            self.accumulated_generator_cost += generator_cost
            self.accumulated_generator_penalty += generator_penalty

            return generator_cost, generator_penalty, 0.0

        # Execute the actions for the storages
        elif agent.name.startswith('storage'):
            storage_cost, storage_penalty = self.__execute_storage_actions__(agent,
                                                                             actions)
            self.accumulated_storage_cost += storage_cost
            self.accumulated_storage_penalty += storage_penalty

            return storage_cost, storage_penalty, 0.0

        elif agent.name.startswith('ev'):
            ev_cost, ev_penalty, ev_penalty_trip = self.__execute_ev_actions__(agent,
                                                                               actions)
            self.accumulated_ev_cost += ev_cost
            self.accumulated_ev_penalty += ev_penalty
            self.accumulated_ev_penalty_trip += ev_penalty_trip

            return ev_cost, ev_penalty, ev_penalty_trip

        elif agent.name.startswith('aggregator'):
            if actions['ctl'] == 1:
                import_cost, import_penalty = self.__execute_aggregator_actions__(agent,
                                                                                  actions)
                self.accumulated_import_cost += import_cost
                self.accumulated_import_penalty += import_penalty

                import_penalty *= self.import_penalty

                return import_cost, import_penalty, 0.0

            elif actions['ctl'] == 2:
                export_cost, export_penalty = self.__execute_aggregator_actions__(agent,
                                                                                  actions)
                self.accumulated_export_cost -= export_cost
                self.accumulated_export_penalty += export_penalty

                return -export_cost, export_penalty, 0.0

        return 0.0, 0.0, 0.0

    def _calculate_reward(self, agent, cost: float, penalty: float, additional_penalty: float) -> float:

        # Calculate the penalties to use
        if agent.name.startswith('generator'):
            penalty *= 0.0
        elif agent.name.startswith('storage'):
            penalty *= self.storage_action_penalty
        elif agent.name.startswith('ev'):
            penalty *= self.ev_action_penalty
            penalty += additional_penalty * self.ev_requirement_penalty
        elif agent.name.startswith('aggregator'):
            penalty *= self.import_penalty
            penalty += self.balance_penalty * abs(self.current_available_energy)

        # Calculate the reward
        reward = - cost - penalty

        return reward

    # Build the info logs
    def _log_info(self):

        info = {}

        # Generators
        for gen in self.generators:
            info[gen.name] = {
                'cost': self.accumulated_generator_cost,
                'penalty': self.accumulated_generator_penalty
            }

        # Storages
        for storage in self.storages:
            info[storage.name] = {
                'cost': self.accumulated_storage_cost,
                'penalty': self.accumulated_storage_penalty
            }

        # EVs
        for ev in self.evs:
            info[ev.name] = {
                'cost': self.accumulated_ev_cost,
                'penalty': self.accumulated_ev_penalty,
                'penalty_trip': self.accumulated_ev_penalty_trip
            }

        # Aggregator
        info[self.aggregator.name] = {
            'cost': self.accumulated_import_cost + self.accumulated_export_cost,
            'penalty': self.accumulated_import_penalty + self.accumulated_export_penalty
        }

        return info

    # Log ending of episode
    def _log_ending(self, flag: bool) -> tuple[dict, dict]:
        terminateds = {a: flag for a in self.agents}
        terminateds['__all__'] = flag
        truncateds = {a: flag for a in self.agents}
        truncateds['__all__'] = flag

        return terminateds, truncateds

    # Log actions
    def _log_agents(self, agent):

        self.history_dictionary[agent.name] = agent

        return
