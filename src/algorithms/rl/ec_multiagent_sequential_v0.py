import gymnasium as gym
import numpy as np
from copy import deepcopy
from src.resources.base_resource import BaseResource
from src.algorithms.rl.utils import separate_resources

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class EnergyCommunitySequentialV0(MultiAgentEnv):
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

    def __init__(self, resources: list[BaseResource],
                 import_penalty,
                 export_penalty,
                 storage_action_penalty,
                 ev_action_penalty,
                 ev_requirement_penalty):
        super().__init__()

        # Define the resources
        self.resources = deepcopy(resources)

        # Split the incoming resources
        temp_resources = separate_resources(self.resources)
        self.generators = temp_resources['generators']
        self.loads = temp_resources['loads']
        self.storages = temp_resources['storages']
        self.evs = temp_resources['evs']
        self.imports = temp_resources['imports']
        self.exports = temp_resources['exports']

        # Initialize time counter
        self.current_timestep: int = 0

        # Initialize variables for production and consumption
        self.current_production: float = 0
        self.current_consumption: float = 0

        # Initialize variables for currently available energy
        self.current_available_energy: float = 0

        # Create agents
        self.possible_agents = ['gen', 'storage', 'ev', 'agg']
        self.agents = self.__create_agents__()
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

        # Index of executed agents, to be used in the step function
        self._agent_sequence = [resource.name for resource in self.generators] + \
                               [resource.name for resource in self.evs] + \
                               [resource.name for resource in self.storages] + \
                               ['aggregator']
        self._executed_agent: int = 0
        self._current_agent = self._agent_sequence[self._executed_agent]

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

        # Penalties
        self.storage_action_penalty = storage_action_penalty
        self.ev_action_penalty = ev_action_penalty
        self.ev_requirement_penalty = ev_requirement_penalty
        self.import_penalty = import_penalty
        self.export_penalty = export_penalty

        # Set a penalty for each resource
        self.accumulated_generator_penalty: float = 0.0
        self.accumulated_storage_penalty: float = 0.0
        self.accumulated_ev_penalty: float = 0.0
        self.accumulated_import_penalty: float = 0.0
        self.accumulated_export_penalty: float = 0.0

    # Create agents
    def __create_agents__(self) -> dict:
        agents = {}
        for agent in np.arange(len(self.generators)):
            agents[str(self.generators[agent].name)] = self.generators[agent]

        for agent in np.arange(len(self.storages)):
            agents[str(self.storages[agent].name)] = self.storages[agent]

        for agent in np.arange(len(self.evs)):
            agents[str(self.evs[agent].name)] = self.evs[agent]

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
        temp_observation_space['agg'] = self.__create_aggregator_observations__()

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
        temp_action_space['agg'] = self.__create_aggregator_actions__()

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
                'current_available_energy': gym.spaces.Box(low=0, high=9999.0, shape=(1,), dtype=np.float32),
                'current_buy_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_sell_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_loads': gym.spaces.Box(low=0, high=max(self.loads), shape=(1,), dtype=np.float32)
            })

        return generator_observations

    # Handle generator observations
    def __get_generator_observations__(self) -> dict:
        """
        Get the observations for the generators
        :return: dict
        """

        generator_observations: dict = {}
        for gen in self.generators:
            generator_observations[gen.name] = {
                'current_available_energy': np.array([self.current_available_energy],
                                                     dtype=np.float32),
                'current_buy_price': np.array([self.imports[0].cost[self.current_timestep]],
                                              dtype=np.float32),
                'current_sell_price': np.array([self.exports[0].cost[self.current_timestep]],
                                               dtype=np.float32),
                'current_loads': np.array([self.loads[0].value[self.current_timestep]],
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
            renewable = False
            if isinstance(gen.renewable, bool):
                renewable = gen.renewable

            if renewable | (gen.is_renewable == 2):  # Hack for the Excel file
                generator_actions[gen.name] = gym.spaces.Dict({
                    'production': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
                })
            else:
                generator_actions[gen.name] = gym.spaces.Dict({
                    'active': gym.spaces.Discrete(2)
                })

        return generator_actions

    # Execute generator actions
    def __execute_generator_actions__(self, gen, actions) -> float:
        """
        Execute the actions for the generators
        :param gen: generator resource
        :param actions: actions to be executed
        :return: float
        """

        produced_energy: float = 0.0

        # Check if actions has active or production
        if 'active' in actions.keys():
            produced_energy = (actions['active'] *
                               max(gen.upper_bound[self.current_timestep]))
        elif 'production' in actions.keys():
            produced_energy = (actions['production'][0] *
                               max(gen.upper_bound[self.current_timestep]))

        # Attribute the produced energy to the generator
        gen.value[self.current_timestep] = produced_energy
        self.current_production += gen.value[self.current_timestep]
        self.current_available_energy += gen.value[self.current_timestep]

        return 0.0

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
                'current_available_energy': gym.spaces.Box(low=0, high=9999.0, shape=(1,), dtype=np.float32),
                'current_loads': gym.spaces.Box(low=0, high=max(self.loads), shape=(1,), dtype=np.float32),
                'current_buy_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                'current_sell_price': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })

        return storage_observations

    # Handle storage observations
    def __get_storage_observations__(self) -> dict:
        """
        Get the observations for the storages
        :return: dict
        """

        storage_observations: dict = {}
        for storage in self.storages:
            storage_observations[storage.name] = {
                'current_soc': np.array([storage.value / storage.capacity_max],
                                        dtype=np.float32),
                'current_available_energy': np.array([self.current_available_energy],
                                                     dtype=np.float32),
                'current_loads': np.array([self.loads[0].value[self.current_timestep]],
                                          dtype=np.float32),
                'current_buy_price': np.array([self.imports[0].cost[self.current_timestep]],
                                              dtype=np.float32),
                'current_sell_price': np.array([self.exports[0].cost[self.current_timestep]],
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

        # Idle state
        if actions['ctl'] == 0:
            storage.charge = 0.0
            storage.discharge = 0.0
            penalty = 0.0

        # Charge state
        elif actions['ctl'] == 1:
            charge = actions['value'][0]
            if storage.value + charge > 1.0:
                # Calculate the deviation from the bounds
                deviation = storage.value + charge - 1.0
                charge = 1.0 - storage.value
                penalty = deviation

            if self.current_available_energy < charge:
                # Calculate if there is any remaining energy
                deviation = charge - self.current_available_energy

                # Get the cost of additional energy
                cost = (charge *
                        storage.cost_charge[self.current_timestep] +
                        deviation *
                        self.imports[0].cost[self.current_timestep])

            else:
                # Get the cost of the energy
                cost = charge * storage.cost_charge[self.current_timestep]

            storage.charge = charge
            storage.discharge = 0.0

        # Discharge state
        elif actions['ctl'] == 2:
            discharge = actions['value'][0]
            if storage.value - discharge < 0.0:
                # Calculate the deviation from the bounds
                deviation = abs(storage.value - discharge)
                discharge = storage.value
                penalty = deviation

            storage.charge = 0.0
            storage.discharge = discharge

            # Add the energy to the pool
            self.current_available_energy += discharge

            # Get the cost of the energy
            cost = discharge * storage.cost_discharge[self.current_timestep]

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
                'current_available_energy': gym.spaces.Box(low=0, high=9999.0, shape=(1,), dtype=np.float32),
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
    def __get_ev_observations__(self) -> dict:
        """
        Get the observations for the EVs
        :return: dict
        """

        ev_observations: dict = {}
        for ev in self.evs:
            # Get the next departure time and energy requirement
            next_departure = np.where(ev.schedule_requirement_soc > self.current_timestep)[0]
            remains_trips = len(next_departure) > 0
            next_departure_soc = ev.schedule_requirement_soc[next_departure[0]] \
                if remains_trips else ev.min_charge

            time_until_departure = abs(next_departure[0] - self.current_timestep) \
                if remains_trips else ev.schedule_connected.shape[0] - 1

            ev_observations[ev.name] = {
                'current_soc': np.array([ev.value / ev.capacity_max],
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
                'current_buy_price': np.array([self.imports[0].cost[self.current_timestep]],
                                              dtype=np.float32),
                'current_sell_price': np.array([self.exports[0].cost[self.current_timestep]],
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
    def __execute_ev_actions__(self, ev, actions) -> float:
        """
        Execute the actions for the EVs
        :param ev: EV resource
        :param actions: actions to be executed
        :return: reward to be used as penalty
        """

        # Set up the reward to be returned in case of illegal EV actions
        # Such as overcharging or discharging beyond the available energy.
        # The reward will be the deviation from the bounds, to be later used as a penalty
        # Also includes the penalty for not meeting the energy requirement for the next departure
        reward: float = 0.0

        # Idle state
        if actions['ctl'] == 0:
            ev.charge = 0.0
            ev.discharge = 0.0
            reward = 0.0

        # Charge state
        elif actions['ctl'] == 1:
            charge = actions['value'][0]
            if ev.value + charge > 1.0:
                # Calculate the deviation from the bounds
                deviation = ev.value + charge - 1.0
                charge = 1.0 - ev.value
                reward = deviation

            ev.charge = charge
            ev.discharge = 0.0

        # Discharge state
        elif actions['ctl'] == 2:
            discharge = actions['value'][0]
            if ev.value - discharge < 0.0:
                # Calculate the deviation from the bounds
                deviation = abs(ev.value - discharge)
                discharge = ev.value
                reward = deviation

            ev.charge = 0.0
            ev.discharge = discharge

        # Check if the EV meets the energy requirement for the next departure
        next_departure = np.where(ev.schedule_requirement_soc > self.current_timestep)[0]
        remains_trips = len(next_departure) > 0

        if remains_trips:
            next_departure_soc = ev.schedule_requirement_soc[next_departure[0]]
            if ev.value < next_departure_soc / ev.capacity_max:
                # Calculate the deviation from the value
                deviation = next_departure_soc - ev.value
                reward += deviation

        return reward

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
            'current_available_energy': gym.spaces.Box(low=0, high=9999.0, shape=(1,), dtype=np.float32)
        })

    # Handle aggregator observations
    def __get_aggregator_observations__(self) -> dict:
        """
        Get the observations for the aggregator
        :return: dict
        """

        return {
            'current_buy_price': np.array([self.imports[0].cost[self.current_timestep]],
                                          dtype=np.float32),
            'current_sell_price': np.array([self.exports[0].cost[self.current_timestep]],
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
            'value': gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        })

    # Execute aggregator actions
    def __execute_aggregator_actions__(self, actions) -> float:
        """
        Execute the actions for the aggregator
        :param actions: actions to be executed
        :return: reward to be used as penalty
        """

        # Set up the reward to be returned in case of illegal aggregator actions
        # Such as importing or exporting beyond the allowed limits.
        # The reward will be the deviation from the bounds, to be later used as a penalty
        reward: float = 0.0

        # Idle state
        if actions['ctl'] == 0:
            self.imports[0].value = 0.0
            self.exports[0].value = 0.0

        # Import state
        elif actions['ctl'] == 1:
            imports = actions['value'][0]
            if imports > self.imports[0].upper_bound:
                # Calculate the deviation from the bounds
                deviation = imports - self.imports[0].upper_bound
                imports = self.imports[0].upper_bound
                reward = deviation

            self.imports[0].value = imports
            self.exports[0].value = 0.0

            return reward

        # Export state
        elif actions['ctl'] == 2:
            exports = actions['value'][0]
            if exports > self.exports[0].upper_bound:
                # Calculate the deviation from the bounds
                deviation = exports - self.exports[0].upper_bound
                exports = self.exports[0].upper_bound
                reward = deviation

            self.imports[0].value = 0.0
            self.exports[0].value = exports

            return reward

        return reward

    # Get next agent to act
    def __get_next_agent__(self):
        """
        Get the next agent to act
        :return: None
        """

        # Check the executed agents
        next_agent = self._agent_sequence[self._executed_agent]

        # Increment the executed agent
        self._executed_agent += 1

        return next_agent

    # Get observations
    def __get_next_observations__(self) -> dict:
        """
        Get the observations for the environment
        :return: dict
        """

        # Get the observation for the next resource
        observations = {}

        return observations

    # Reset environment
    def reset(self, *, seed=None, options=None):

        # Create new agents
        self.agents = self.__create_agents__()

        # Set the flag of executed agents
        self._executed_agent = 0

        # Set the termination and truncation flags
        self.terminateds = set()
        self.truncateds = set()

        # Reset environment variables
        self.current_timestep = 0
        self.current_production = 0
        self.current_consumption = 0
        self.current_available_energy = 0

        # Get the new observations
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
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: True for agent in self.agents}
            terminations['__all__'] = True
            truncations['__all__'] = True

            observations = self.__get_next_observations__()
            reward = {agent: 0 for agent in self.agents}
            info = {agent: {} for agent in self.agents}

        # Check for existing actions
        exists_actions = len(action_dict) > 0

        # Choose the agent to act
        agent = self.__get_next_agent__()

        # Execute the actions
        if exists_actions:
            if self._executed_agent < len(self._agent_sequence):
                if agent in action_dict.keys():
                    actions = action_dict[agent]
                    reward = 0.0

                    # Execute the actions for the generators
                    if agent in [resource.name for resource in self.generators]:
                        self.accumulated_generator_penalty += self.__execute_generator_actions__(self.agents[agent],
                                                                                                 actions)

                    # Execute the actions for the storages
                    elif agent in [resource.name for resource in self.storages]:
                        storage_cost, storage_penalty = self.__execute_storage_actions__(self.agents[agent],
                                                                                         actions)
                        self.accumulated_storage_cost += storage_cost
                        self.accumulated_storage_penalty += storage_penalty

                    # Execute the actions for the EVs
                    elif agent in [resource.name for resource in self.evs]:
                        self.accumulated_ev_penalty += self.__execute_ev_actions__(self.agents[agent],
                                                                                   actions)

                    # Execute the actions for the aggregator
                    elif agent == 'aggregator':
                        if actions['ctl'] == 1:
                            self.accumulated_import_penalty += self.__execute_aggregator_actions__(actions)
                        elif actions['ctl'] == 2:
                            self.accumulated_export_penalty += self.__execute_aggregator_actions__(actions)

                    return self.__get_next_observations__(), 0.0, False, {}

            else:
                real_reward = (self.accumulated_generator_penalty +
                               self.accumulated_storage_penalty +
                               self.accumulated_ev_penalty +
                               self.accumulated_import_penalty +
                               self.accumulated_export_penalty)
                return self.__get_next_observations__(), 0.0, False, {}

        # Get the new observations
        # This needs to be done sequentially
        # Possibly through mixed action/observation moments
        observations = self.__get_observations__()

        # Get the reward
        reward = self.__get_reward__()

        # Check if the environment is done
        done = self.__is_done__()

        # Set info
        info = {}

        return observations, reward, done, info