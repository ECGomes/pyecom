from copy import deepcopy

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.resources import Generator, Load, Storage, Vehicle, Aggregator


class EnergyCommunityBaselineV2(MultiAgentEnv):
    """
    Energy Community Environment for multi-agent reinforcement learning
    Generators can be renewable or non-renewable:
    - Renewable generators can be controlled, but not switched on/off
    - Non-renewable generators can be switched on/off, but not controlled
    Loads are fixed
    Storages can be controlled
    - Storages can be idle, charged or discharged
    EVs can be controlled
    - EVs can be charged or discharged
    - EVs can be connected or disconnected
    Import/Export can be controlled with an input price

    Rewards are attributed to the community as a whole
    Rewards are based on the following:
    - Total energy consumption
    - Total energy production
    - Total energy storage
    - Total energy import
    - Total energy export

    Removed the following:
    - Generators are always on
    - Available renewable energy from the observation space
    """

    metadata = {'name': 'EnergyCommunitySequential-v4'}

    def __init__(self,
                 ren_generators: list[Generator],
                 loads: list[Load],
                 storages: list[Storage],
                 evs: list[Vehicle],
                 generators: list[Generator],
                 aggregator: Aggregator,
                 ev_penalty: float = 1.0,
                 balance_penalty: float = 1.0,
                 ):
        super().__init__()

        # Initialize the resources and the environment
        self.original_resources = {'ren_generators': ren_generators,
                                   'loads': loads,
                                   'storages': storages,
                                   'evs': evs,
                                   'generators': generators,
                                   'aggregator': aggregator}

        # Initialize the environment
        self._reset(ren_generators, loads, storages, evs, generators, aggregator)

        # Possible penalties
        self.ev_penalty = ev_penalty
        self.balance_penalty = balance_penalty

        # Handle observation and action spaces
        self._handle_observation_space()
        self._handle_action_space()

        # Initialize the available energy vector
        self.energy_history = np.zeros(ren_generators[0].value.shape)

    # Initialize the environment
    def _reset(self,
               ren_generators: list[Generator],
               loads: list[Load],
               storages: list[Storage],
               evs: list[Vehicle],
               generators: list[Generator],
               aggregator: Aggregator):

        # Define the resources
        self.resources = deepcopy(self.original_resources)

        self.ren_generators = self.resources['ren_generators']
        self.loads = self.resources['loads']
        self.storages = self.resources['storages']
        self.evs = self.resources['evs']
        self.generators = self.resources['generators']
        self.aggregator = self.resources['aggregator']

        # Sum of loads
        self.load_consumption: float = np.sum([load.value for load in self.loads], axis=0)

        # Timestep counter
        self.timestep: int = 0

        # Available overall and renewable energy for current timestep
        self.available_energy: float = -self.load_consumption[self.timestep]

        # Create the agents
        self.possible_agents = ['ren_gen', 'storage', 'ev', 'gen']
        self.agents = self.__create_agents__()
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

    # Create agents
    def __create_agents__(self) -> dict:
        agents = {}
        for agent in np.arange(len(self.ren_generators)):
            agents[self.ren_generators[agent].name] = self.ren_generators[agent]

        for agent in np.arange(len(self.evs)):
            agents[str(self.evs[agent].name)] = self.evs[agent]

        for agent in np.arange(len(self.storages)):
            agents[str(self.storages[agent].name)] = self.storages[agent]

        for agent in np.arange(len(self.generators)):
            agents[str(self.generators[agent].name)] = self.generators[agent]

        agents_copy = deepcopy(agents)
        return agents_copy

    # Handle observation space
    def _handle_observation_space(self) -> None:

        self._obs_space_in_preferred_format = True
        temp_observation_space = {}

        # Renewable Generators
        ren_gens = self.__create_ren_gen_obs__()
        for ren_gen in ren_gens:
            temp_observation_space[ren_gen] = ren_gens[ren_gen]

        # Storages
        storages = self.__create_storage_obs__()
        for storage in storages:
            temp_observation_space[storage] = storages[storage]

        # EVs
        evs = self.__create_ev_obs__()
        for ev in evs:
            temp_observation_space[ev] = evs[ev]

        # Generators
        gens = self.__create_gen_obs__()
        for gen in gens:
            temp_observation_space[gen] = gens[gen]

        self.observation_space = gym.spaces.Dict(temp_observation_space)
        self._obs_space_in_preferred_format = True

        return

    # Handle action space
    def _handle_action_space(self) -> None:

        self._act_space_in_preferred_format = True
        temp_action_space = {}

        # Renewable Generators
        ren_gens = self.__create_ren_gen_actions__()
        for ren_gen in ren_gens:
            temp_action_space[ren_gen] = ren_gens[ren_gen]

        # Storages
        storages = self.__create_storage_actions__()
        for storage in storages:
            temp_action_space[storage] = storages[storage]

        # EVs
        evs = self.__create_ev_actions__()
        for ev in evs:
            temp_action_space[ev] = evs[ev]

        # Generators
        gens = self.__create_gen_actions__()
        for gen in gens:
            temp_action_space[gen] = gens[gen]

        self.action_space = gym.spaces.Dict(temp_action_space)
        self._act_space_in_preferred_format = True

        return

    # Create Generator Observation Space
    def __create_ren_gen_obs__(self) -> dict:
        """
        Create the observation space for the generators
        Each generator will have the following observations:
        - Available Energy pool (float)
        - Current buy price (float)
        - Current sell price (float)
        :return: dict
        """

        ren_gen_obs = {}
        for gen in self.ren_generators:
            ren_gen_obs[gen.name] = gym.spaces.Dict({
                'available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
                'expected_production': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'import_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'export_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            })

        return ren_gen_obs

    # Get current Renewable Generator Observation
    # Handle generator observations
    def __get_ren_gen_observations__(self, gen) -> dict:
        """
        Get the observations for one generator
        :return: dict
        """

        generator_observations: dict = {
            'available_energy': np.array([self.available_energy],
                                         dtype=np.float32),
            'expected_production': np.array([gen.upper_bound[self.timestep]],
                                            dtype=np.float32),
            'import_price': np.array([self.aggregator.import_cost[self.timestep]],
                                     dtype=np.float32),
            'export_price': np.array([self.aggregator.export_cost[self.timestep]],
                                     dtype=np.float32)
        }

        return generator_observations

    # Create Generator Action Space
    def __create_ren_gen_actions__(self) -> dict:
        """
        Create the action space for the generators
        """

        generator_actions = {}
        for gen in self.ren_generators:
            generator_actions[gen.name] = gym.spaces.Dict({
                'production': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            })

        return generator_actions

    # Generator Behaviour
    def __execute_ren_gen_actions__(self, gen, actions) -> tuple[float, float]:
        """
        Execute the actions for the renewable generators
        :param gen: Generator
        :param actions: dict
        :return: tuple
        """

        # Initialize costs and penalties
        cost: float = 0.0
        penalty: float = 0.0

        production_coefficient: float = np.round(actions['production'][0], 0)
        production: float = production_coefficient * gen.upper_bound[self.timestep]

        cost = production * gen.cost[self.timestep]

        # If there is available energy, attribute to the available renewable pool
        self.available_energy += production

        # Set the production values on the resources
        idx = [i for i in range(len(self.ren_generators)) if self.ren_generators[i].name == gen.name][0]
        gen.value[self.timestep] = production
        self.ren_generators[idx].value[self.timestep] = production

        return abs(cost), abs(penalty)

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
                'soc': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
                'maximum_charge': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'maximum_discharge': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'import_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'export_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })

        return storage_observations

    # Get current Storage Observation
    def __get_storage_observations__(self, storage) -> dict:
        """
        Get the observations for the storages
        :param storage: storage resource to get the observations
        :return: dict
        """

        storage_observations: dict = {
            'soc': np.array([storage.value[self.timestep - 1] if self.timestep > 0
                             else storage.initial_charge],
                            dtype=np.float32),
            'available_energy': np.array([self.available_energy],
                                         dtype=np.float32),
            'maximum_charge': np.array([storage.charge_max[self.timestep]],
                                       dtype=np.float32),
            'maximum_discharge': np.array([storage.discharge_max[self.timestep]],
                                          dtype=np.float32),
            'import_price': np.array([self.aggregator.import_cost[self.timestep]],
                                     dtype=np.float32),
            'export_price': np.array([self.aggregator.export_cost[self.timestep]],
                                     dtype=np.float32)
        }

        return storage_observations

    # Create Storage Action Space
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
        charge: float = 0.0
        to_charge: float = 0.0
        discharge: float = 0.0
        to_discharge: float = 0.0

        # Get the index of the storage
        idx = [i for i, resource in enumerate(self.storages) if resource.name == storage.name][0]

        # Check if it is the first timestep
        if self.timestep == 0:
            storage.value[self.timestep] = storage.initial_charge
            self.storages[idx].value[self.timestep] = storage.initial_charge
        else:
            storage.value[self.timestep] = storage.value[self.timestep - 1]
            self.storages[idx].value[self.timestep] = storage.value[self.timestep - 1]

        # Idle state
        if actions['ctl'] == 0:
            storage.charge[self.timestep] = 0.0
            storage.discharge[self.timestep] = 0.0

            # Update the resource values
            self.storages[idx].charge[self.timestep] = 0.0
            self.storages[idx].discharge[self.timestep] = 0.0

            return cost, penalty

        # Charge state
        elif actions['ctl'] == 1:
            # Check if we can charge
            charge: float = actions['value'][0]
            to_charge: float = charge * storage.charge_max[self.timestep]

            # Check if we can charge
            if storage.value[self.timestep] >= 0.9:
                # We are already at 90%
                charge = 0.0
                to_charge = 0.0

            elif storage.value[self.timestep] + to_charge / storage.capacity_max > 0.9:
                # If we cannot charge, charge the maximum possible
                to_charge = (0.9 - storage.value[self.timestep]) * storage.capacity_max
                charge = to_charge / storage.charge_max[self.timestep]

            # The cost is the charge * cost_charge, as efficiency should not be considered
            cost = to_charge * storage.cost_charge[self.timestep]

            # Update the available energy
            self.available_energy -= to_charge

        # Discharge state
        elif actions['ctl'] == 2:

            discharge: float = actions['value'][0]
            to_discharge: float = discharge * storage.discharge_max[self.timestep]

            # Check if we can discharge
            if storage.value[self.timestep] < storage.capacity_min:
                # If we are already at the minimum charge, we cannot discharge
                discharge = 0.0
                to_discharge = 0.0

            elif storage.value[self.timestep] * storage.capacity_max - to_discharge \
                    < storage.capacity_min * storage.capacity_max:
                # If we cannot discharge, discharge the maximum possible
                to_discharge = (storage.value[self.timestep] - storage.capacity_min) * storage.capacity_max
                discharge = to_discharge / storage.discharge_max[self.timestep]

            # Update the available energy
            self.available_energy += to_discharge

            # Calculate the cost
            cost = to_discharge * storage.cost_discharge[self.timestep]

        # Update resource charge and discharge values
        storage.charge[self.timestep] = to_charge * storage.charge_efficiency
        storage.discharge[self.timestep] = to_discharge / storage.discharge_efficiency

        # Update the storage value
        new_storage_value = storage.value[self.timestep] \
                            + ((to_charge * storage.charge_efficiency) \
                               - (to_discharge / storage.discharge_efficiency)) / storage.capacity_max
        storage.value[self.timestep] = new_storage_value
        self.storages[idx].value[self.timestep] = new_storage_value
        self.storages[idx].charge[self.timestep] = to_charge * storage.charge_efficiency
        self.storages[idx].discharge[self.timestep] = to_discharge / storage.discharge_efficiency

        return abs(cost), abs(penalty)

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
                'soc': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
                'maximum_charge': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'maximum_discharge': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'grid_connection': gym.spaces.Discrete(2),
                'next_departure_time': gym.spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32),
                'time_until_next_departure': gym.spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32),
                'next_departure_energy_requirement': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'import_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'export_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_time': gym.spaces.Box(low=0, high=9999, shape=(1,), dtype=np.int32)
            })

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
                'value': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            })

        return ev_actions

    # Get current EV Observation
    def __get_ev_observations__(self, ev) -> dict:
        """
        Get the observations for the EVs
        :param ev: EV resource
        :return: dict
        """

        # Get the next departure time and energy requirement
        next_departure = np.where(ev.schedule_requirement_soc > 0)[0]
        next_departure = next_departure[next_departure >= self.timestep]

        remains_trips = len(next_departure) > 0
        next_departure_soc = ev.schedule_requirement_soc[next_departure[0]] \
            if remains_trips else ev.min_charge

        time_until_departure = abs(next_departure[0] - self.timestep) \
            if remains_trips else ev.schedule_connected.shape[0] - 1

        next_departure = next_departure[0] if remains_trips else 9999

        ev_observations: dict = {
            'soc': np.array([ev.value[self.timestep - 1] if self.timestep > 0
                             else ev.initial_charge],
                            dtype=np.float32),
            'available_energy': np.array([self.available_energy],
                                         dtype=np.float32),
            'maximum_charge': np.array([ev.schedule_charge[self.timestep]],
                                       dtype=np.float32),
            'maximum_discharge': np.array([ev.schedule_discharge[self.timestep]],
                                          dtype=np.float32),
            'grid_connection': int(ev.schedule_connected[self.timestep]),
            'next_departure_time': np.array([next_departure],
                                            dtype=np.int32),
            'time_until_next_departure': np.array([time_until_departure],
                                                  dtype=np.int32),
            'next_departure_energy_requirement': np.array([next_departure_soc / ev.capacity_max],
                                                          dtype=np.float32),
            'import_price': np.array([self.aggregator.import_cost[self.timestep]],
                                     dtype=np.float32),
            'export_price': np.array([self.aggregator.export_cost[self.timestep]],
                                     dtype=np.float32),
            'current_time': np.array([self.timestep],
                                     dtype=np.int32)
        }

        return ev_observations

    # Execute EV Actions
    def __execute_ev_actions__(self, ev, actions) -> tuple[float, float]:
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
        charge: float = 0.0
        to_charge: float = 0.0
        discharge: float = 0.0
        to_discharge: float = 0.0

        # Get the index of the EV
        idx = [i for i, resource in enumerate(self.evs) if resource.name == ev.name][0]

        # Check if it is the first timestep
        if self.timestep == 0:
            ev.value[self.timestep] = ev.initial_charge
            self.evs[idx].value[self.timestep] = ev.initial_charge
        else:
            ev.value[self.timestep] = ev.value[self.timestep - 1]
            self.evs[idx].value[self.timestep] = ev.value[self.timestep - 1]

        # First, check if the EV is not connected to the grid
        if ev.schedule_connected[self.timestep] == 0:
            # ev.value[self.current_timestep] = 0.0
            ev.charge[self.timestep] = 0.0
            ev.discharge[self.timestep] = 0.0

            # self.evs[idx].value[self.current_timestep] = 0.0
            self.evs[idx].charge[self.timestep] = 0.0
            self.evs[idx].discharge[self.timestep] = 0.0

            # Check if the there is a trip and if EV meets the energy requirement for the departure
            if self.evs[idx].schedule_requirement_soc[self.timestep] > 0:
                next_departure_soc = ev.schedule_requirement_soc[self.timestep]

                if ev.value[self.timestep] < next_departure_soc / ev.capacity_max:
                    # Calculate the deviation from the value
                    # deviation = abs(next_departure_soc - ev.value[self.timestep])
                    # penalty = deviation * self.ev_penalty

                    penalty = self.ev_penalty

                    # Discharge the EV with the possible energy
                    ev.value[self.timestep] = 0.0
                    self.evs[idx].value[self.timestep] = 0.0

                else:
                    ev.value[self.timestep] = ((ev.value[self.timestep] * ev.capacity_max -
                                                next_departure_soc) / ev.capacity_max)
                    self.evs[idx].value[self.timestep] = ev.value[self.timestep]

            return abs(cost), abs(penalty)

        # Else the EV is connected
        elif ev.schedule_connected[self.timestep] == 1:
            # Check if there is a trip on the current timestep
            if ev.schedule_requirement_soc[self.timestep] > 0:
                # Check if the EV has enough energy
                if ev.value[self.timestep] < ev.schedule_requirement_soc[self.timestep]:
                    # Calculate the difference in energy
                    # difference = abs(ev.schedule_requirement_soc[self.timestep] - ev.value[self.timestep])

                    # If not, add the difference between the required and the current energy as penalty
                    # penalty = difference * self.ev_penalty
                    penalty = self.ev_penalty

                    # Empty the EV
                    ev.value[self.timestep] = 0.0
                    self.evs[idx].value[self.timestep] = 0.0

                else:
                    # Just deduct the required energy
                    ev.value[self.timestep] -= ev.schedule_requirement_soc[self.timestep]
                    self.evs[idx].value[self.timestep] -= ev.schedule_requirement_soc[self.timestep]

            # If not, we can execute actions
            # Idle state
            if actions['ctl'] == 0:
                charge = 0.0
                discharge = 0.0

                # Update values
                ev.charge[self.timestep] = charge
                ev.discharge[self.timestep] = discharge

                # Update resource values
                self.evs[idx].charge[self.timestep] = charge
                self.evs[idx].discharge[self.timestep] = discharge

            # Charge state
            elif actions['ctl'] == 1:
                # Get the charge value
                charge = np.round(actions['value'][0],2)
                to_charge = charge * ev.schedule_charge[self.timestep]
                discharge = 0.0

                # Check if we can charge
                if ev.value[self.timestep] >= 0.9:
                    # We are already at 90%
                    charge = 0.0
                    to_charge = 0.0

                elif ev.value[self.timestep] + to_charge / ev.capacity_max >= 0.9:
                    # If we cannot charge fully, charge the maximum possible
                    to_charge = (0.9 - ev.value[self.timestep]) * ev.capacity_max
                    charge = to_charge / ev.schedule_charge[self.timestep]

                # Calculate the cost
                cost = charge * ev.schedule_charge[self.timestep] * ev.cost_charge[self.timestep]

                # Update the available energy
                self.available_energy -= to_charge

                # Update the resource values
                ev.charge[self.timestep] = to_charge * ev.charge_efficiency

                # Update the value of the EV
                ev.value[self.timestep] = ev.value[self.timestep] + to_charge / ev.capacity_max * ev.charge_efficiency
                self.evs[idx].value[self.timestep] = ev.value[self.timestep] + to_charge / ev.capacity_max * \
                                                     ev.charge_efficiency
                self.evs[idx].charge[self.timestep] = to_charge * ev.charge_efficiency
                self.evs[idx].discharge[self.timestep] = discharge

            # Discharge state
            elif actions['ctl'] == 2:
                # Get the discharge value
                discharge = np.round(actions['value'][0], 2)
                to_discharge = discharge * ev.schedule_discharge[self.timestep]
                charge = 0.0

                # Check if we can discharge
                if ev.value[self.timestep] > ev.min_charge:
                    if ev.value[self.timestep] - to_discharge / ev.capacity_max < ev.min_charge:
                        # If we cannot discharge, discharge the maximum possible
                        to_discharge = (ev.value[self.timestep] - ev.min_charge) * ev.capacity_max
                        discharge = to_discharge / ev.schedule_discharge[self.timestep]
                else:
                    # If we are already at the minimum charge, we cannot discharge
                    discharge = 0.0
                    to_discharge = 0.0

                # Update the available energy
                self.available_energy += to_discharge

                # Calculate the cost
                cost = discharge * ev.schedule_discharge[self.timestep] * ev.cost_discharge[self.timestep]

                # Update the resource values
                ev.discharge[self.timestep] = to_discharge / ev.discharge_efficiency
                ev.charge[self.timestep] = 0.0

                # Update the value of the EV
                ev.value[self.timestep] = ev.value[self.timestep] - to_discharge / ev.capacity_max / \
                                          ev.discharge_efficiency
                self.evs[idx].value[self.timestep] = ev.value[self.timestep] - to_discharge / ev.capacity_max / \
                                                     ev.discharge_efficiency

                self.evs[idx].charge[self.timestep] = 0.0
                self.evs[idx].discharge[self.timestep] = to_discharge / ev.discharge_efficiency

        return abs(cost), abs(penalty)

    # Create Generator Observation Space
    def __create_gen_obs__(self) -> dict:
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
                'available_energy': gym.spaces.Box(low=-99999.0, high=99999.0, shape=(1,), dtype=np.float32),
                'max_production': gym.spaces.Box(low=0, high=99999.0, shape=(1,), dtype=np.float32),
                'cost_production': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'import_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'export_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })

        return generator_observations

    # Handle generator observations
    def __get_gen_observations__(self, gen) -> dict:
        """
        Get the observations for one generator
        :return: dict
        """

        generator_observations: dict = {
            'available_energy': np.array([self.timestep],
                                         dtype=np.float32),
            'max_production': np.array([gen.upper_bound[self.timestep]],
                                       dtype=np.float32),
            'cost_production': np.array([gen.cost[self.timestep]],
                                        dtype=np.float32),
            'import_price': np.array([self.aggregator.import_cost[self.timestep]],
                                     dtype=np.float32),
            'export_price': np.array([self.aggregator.export_cost[self.timestep]],
                                     dtype=np.float32)
        }

        return generator_observations

    # Create Generator Action Space
    def __create_gen_actions__(self) -> dict:
        """
        Create the action space for the generators
        """

        generator_actions = {}
        for gen in self.generators:
            generator_actions[gen.name] = gym.spaces.Dict({
                'active': gym.spaces.Discrete(2)
            })

        return generator_actions

    # Execute Generator Actions
    def __execute_gen_actions__(self, gen, actions) -> tuple[float, float]:
        """
        Execute the actions for the generators
        :return: tuple
        """

        # Initialize costs and penalties
        cost: float = 0.0
        penalty: float = 0.0

        # Calculate production and production costs
        production = actions['active'] * gen.upper_bound[self.timestep]

        cost = production * gen.cost[self.timestep]

        # Update energy pool
        self.available_energy += production

        # Set the production values on the resources
        idx = [i for i in range(len(self.generators)) if self.generators[i].name == gen.name][0]
        gen.value[self.timestep] = production
        self.generators[idx].value[self.timestep] = production

        return cost, penalty

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
        cost: float = 0.0
        penalty: float = 0.0
        energy_to_export: float = 0.0
        energy_to_import: float = 0.0

        # Add the current energy balance to the history for debug
        self.energy_history[self.timestep] = self.available_energy

        # Check the current energy balance
        if self.available_energy > 0:
            # Then we have too much energy and need to export

            # Calculate the energy to be exported
            energy_to_export = deepcopy(self.available_energy)

            if energy_to_export > self.aggregator.export_max[self.timestep]:
                energy_to_export = self.aggregator.export_max[self.timestep]

                # Calculate the deviation
                deviation = self.available_energy - energy_to_export

                # Calculate the penalty
                penalty = abs(deviation * self.balance_penalty)

            # Calculate the cost
            cost = - abs(energy_to_export * self.aggregator.export_cost[self.timestep])

        elif self.available_energy < 0:
            # Then we need to import energy

            # Calculate the energy to be imported
            energy_to_import = deepcopy(abs(self.available_energy))

            if energy_to_import > self.aggregator.import_max[self.timestep]:
                energy_to_import = self.aggregator.import_max[self.timestep]

                # Calculate the deviation
                deviation = abs(self.available_energy) - energy_to_import

                # Calculate the penalty
                penalty = deviation * self.balance_penalty

            # Calculate the cost
            cost = abs(energy_to_import * self.aggregator.import_cost[self.timestep])

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

        # Reset the environment variables
        self._reset(ren_generators=self.resources['ren_generators'],
                    loads=self.resources['loads'],
                    storages=self.resources['storages'],
                    evs=self.resources['evs'],
                    generators=self.resources['generators'],
                    aggregator=self.resources['aggregator'])

        observations = self._get_observations()

        return observations, {}

    # Step function
    def step(self, action_dict: dict) -> tuple:

        # Initialize costs and penalties
        cost: float = 0.0
        penalty: float = 0.0

        # Initialize the observations, info and rewards
        observations = {}
        info = {}
        reward = {}

        # Truncations and terminations
        terminateds, truncateds = self._log_ending(False)

        # Check for actions
        if len(action_dict) > 0:

            for key in action_dict.keys():

                actions = action_dict[key]

                # Dispatch the actions
                if key.startswith('gen'):
                    current_res = [res for res in self.generators if res.name == key][0]
                    cost, penalty = self.__execute_gen_actions__(current_res, actions)
                elif key.startswith('storage'):
                    current_res = [res for res in self.storages if res.name == key][0]
                    cost, penalty = self.__execute_storage_actions__(current_res, actions)
                elif key.startswith('ev'):
                    current_res = [res for res in self.evs if res.name == key][0]
                    cost, penalty = self.__execute_ev_actions__(current_res, actions)
                elif key.startswith('ren_gen'):
                    current_res = [res for res in self.ren_generators if res.name == key][0]
                    cost, penalty = self.__execute_ren_gen_actions__(current_res, actions)

                # Calculate the reward
                reward[key] = - np.round(cost,4) - np.round(penalty, 4)

            # Execute the aggregator
            aggregator_cost, aggregator_penalty = self.__execute_aggregator__()

            # Split the earnings between the agents
            # So we don't have agents with huge rewards
            # We leave the penalty the same, so we have a common goal
            split_agg_reward = aggregator_cost / len(self.agents)

            # Create a reward dictionary with all agents
            for _ag in self.agents:
                reward[_ag] = 0.0 if _ag not in reward else reward[_ag]
                reward[_ag] = reward[_ag] - split_agg_reward - abs(aggregator_penalty)

            # print(reward)

            # Check for episode end
            if self.timestep == self.loads[0].value.shape[0] - 1:
                terminateds, truncateds = self._log_ending(True)
                return {}, reward, terminateds, truncateds, {}
            else:
                # Update the timestep
                self.timestep += 1

                # Reset energy pools
                self.available_energy = 0.0

                # Update the pool with the sum of loads
                self.available_energy -= self.load_consumption[self.timestep]

                # Next observation
                observations = self._get_observations()
                terminateds, truncateds = self._log_ending(False)
                return observations, reward, terminateds, truncateds, info

        else:
            terminateds, truncateds = self._log_ending(True)
            observations = self._get_observations()
            reward = {}
            info = {}

        return observations, reward, terminateds, truncateds, info

    def _get_observations(self):

        observations = {}

        for ren_gen in self.ren_generators:
            observations[ren_gen.name] = self.__get_ren_gen_observations__(ren_gen)

        for storage in self.storages:
            observations[storage.name] = self.__get_storage_observations__(storage)

        for ev in self.evs:
            observations[ev.name] = self.__get_ev_observations__(ev)

        for gen in self.generators:
            observations[gen.name] = self.__get_gen_observations__(gen)

        return observations

    # Log the episode truncations and terminations
    def _log_ending(self, flag: bool) -> tuple[dict, dict]:
        terminateds = {a: flag for a in self.agents}
        terminateds['__all__'] = flag
        truncateds = {a: flag for a in self.agents}
        truncateds['__all__'] = flag

        return terminateds, truncateds

    def _log_info(self) -> dict:

        # Check if there are keys on the reward
        return {'': {}}
