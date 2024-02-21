import gymnasium as gym
import numpy as np
import random
from copy import copy, deepcopy
from collections import OrderedDict
from ...resources.base_resource import BaseResource

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class EnergyCommunityMultiEnv_v1(MultiAgentEnv):
    """
    Energy community environment for multi-agent reinforcement learning
    Generators can be renewable or non-renewable:
    - Renewable generators can be controlled, but not switched on/off
    - Non-renewable generators can be switched on/off, but not controlled
    Loads are fixed
    Storages can be controlled
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
    metadata = {'name': 'energy_community_multi_v1'}

    def __init__(self, resources: list[BaseResource],
                 storage_action_penalty,
                 ev_action_penalty,
                 ev_requirement_penalty
                 ):
        super().__init__()

        # Define the resources
        self.resources = deepcopy(resources)
        self.generators = []
        self.loads = []
        self.batteries = []
        self.evs = []
        self.imports = []
        self.exports = []

        # Separate the resources
        self.__separate_resources__()

        # Initialize time counter
        self.current_timestep: int = 0

        # Initialize variables for production and consumption
        self.current_production: float = 0.0
        self.current_consumption: float = 0.0

        self.sampled_days = 1

        # Create agents
        self.possible_agents = ['gen', 'storage', 'ev']
        self.agents = self.__create_agents__()
        self._agent_ids = set(self.agents)
        self.terminateds = set()
        self.truncateds = set()

        # Observation space
        self._obs_space_in_preferred_format = True

        temp_observation_space = {}
        gens = self.__create_gen_observations__()
        for gen in gens.keys():
            temp_observation_space[gen] = gens[gen]

        stors = self.__create_battery_observations__()
        for stor in stors.keys():
            temp_observation_space[stor] = stors[stor]

        evs = self.__create_ev_observations__()
        for ev in evs.keys():
            temp_observation_space[ev] = evs[ev]
        self.observation_space = gym.spaces.Dict(temp_observation_space)

        # Action space
        self._action_space_in_preferred_format = True
        temp_action_space = {}
        gens = self.__create_gen_actions__()
        for gen in gens.keys():
            temp_action_space[gen] = gens[gen]

        stors = self.__create_battery_actions__()
        for stor in stors.keys():
            temp_action_space[stor] = stors[stor]

        evs = self.__create_ev_actions__()
        for ev in evs.keys():
            temp_action_space[ev] = evs[ev]
        self.action_space = gym.spaces.Dict(temp_action_space)

        # Penalties
        # Storage illegal action penalty
        self.storage_action_penalty = storage_action_penalty

        # EV illegal action and not accomplishing requirements penalty
        self.ev_action_penalty = ev_action_penalty
        self.ev_requirement_penalty = ev_requirement_penalty

        self.illegal_import_cost = 100.0
        self.illegal_export_cost = 100.0

    # Separate resources
    def __separate_resources__(self) -> None:
        """
        Separates the resources into their respective categories
        :return: None
        """

        self.generators = [resource for resource in self.resources if resource.name.startswith('gen')]
        self.loads = [resource for resource in self.resources if resource.name.startswith('load')]
        self.batteries = [resource for resource in self.resources if resource.name.startswith('stor')]
        self.evs = [resource for resource in self.resources if resource.name.startswith('ev')]
        self.imports = [resource for resource in self.resources if resource.name.startswith('import')]
        self.exports = [resource for resource in self.resources if resource.name.startswith('export')]

        return

    def __create_agents__(self) -> dict:
        agents = {}
        for agent in np.arange(len(self.generators)):
            agents[str(self.generators[agent].name)] = self.generators[agent]

        for agent in np.arange(len(self.batteries)):
            agents[str(self.batteries[agent].name)] = self.batteries[agent]

        for agent in np.arange(len(self.evs)):
            agents[str(self.evs[agent].name)] = self.evs[agent]

        agents_copy = deepcopy(agents)
        return agents_copy

    def __create_gen_observations__(self):
        """
        Create the observation space for the generators
        Can observe:
        - Grid buy price
        - Grid sell price
        - Percentage of possible production relative to consumption
        :return:
        """

        agents_generators = {}
        for i in self.generators:
            agents_generators[i.name] = gym.spaces.Dict({
                'buy_price': gym.spaces.Box(low=0.0, high=1.0,
                                            shape=(1,),
                                            dtype=np.float32),
                'sell_price': gym.spaces.Box(low=0.0, high=1.0,
                                             shape=(1,),
                                             dtype=np.float32),
                'production_vs_consumption': gym.spaces.Box(low=0.0, high=1.0,
                                                            shape=(1, ),
                                                            dtype=np.float32)})
        return agents_generators

    def __get_gen_observations__(self):
        """
        Get the observation for generators
        :return:
        """

        agents_generators = {}
        for i in self.generators:

            power_calc: float = i.value / self.current_consumption \
                if self.current_consumption > 0.0 else 0.0
            if power_calc > 1:
                power_calc = 0.0

            agents_generators[i.name] = {
                'buy_price': np.array([self.imports[0].cost[self.current_timestep]],
                                      dtype=np.float32),
                'sell_price': np.array([self.exports[0].cost[self.current_timestep]],
                                       dtype=np.float32),
                'production_vs_consumption': np.array([power_calc],
                                                      dtype=np.float32)
            }

        return agents_generators

    def __create_gen_actions__(self):
        """
        Creates the action space for the generators according to being renewable or not
        :return: None
        """

        agents_generators = {}
        for i in self.generators:
            renewable = False
            if isinstance(i.is_renewable, bool):
                renewable = i.is_renewable

            if renewable | (i.is_renewable == 2):
                agents_generators[i.name] = gym.spaces.Dict({'action': gym.spaces.Box(low=0.0, high=1.0,
                                                                                      shape=(1,),
                                                                                      dtype=np.float32)})
            else:
                agents_generators[i.name] = gym.spaces.Dict({'action': gym.spaces.Discrete(1)})

        return agents_generators

    def __get_gen_actions__(self, gen, actions):
        """
        Executes the actions of the generators
        :param actions:
        :return:
        """
        if (actions['action'] == 0) | (actions['action'] == 1):
            gen.value = actions['action'] * max(gen.upper_bound)
            return actions['action']
        else:
            gen.value = actions['action'][0] * max(gen.upper_bound)
            return actions['action'][0]

    def __create_battery_observations__(self):
        """
        Create observation space for batteries
        Can observe:
        - Current SOC
        - Grid buy price
        - Grid sell price
        - Power from generators
        - Forecasted required power
        :return:
        """

        agents_batteries = {}
        for i in self.batteries:
            agents_batteries[i.name] = gym.spaces.Dict({
                'soc': gym.spaces.Box(low=0.0, high=1.0,
                                      shape=(1,),
                                      dtype=np.float32),
                'buy_price': gym.spaces.Box(low=0.0, high=1.0,
                                            shape=(1,),
                                            dtype=np.float32),
                'sell_price': gym.spaces.Box(low=0.0, high=1.0,
                                             shape=(1,)),
                'available_power': gym.spaces.Box(low=0.0, high=50000.0,
                                                  shape=(1,),
                                                  dtype=np.float32),
                'required_power': gym.spaces.Box(low=0.0, high=50000.0,
                                                 shape=(1,),
                                                 dtype=np.float32)})
        return agents_batteries

    def __get_battery_observations__(self):

        agents_batteries = {}
        for i in self.batteries:
            agents_batteries[i.name] = {
                'soc': np.array([i.value / i.capacity_max],
                                dtype=np.float32),
                'buy_price': np.array([self.imports[0].cost[self.current_timestep]],
                                      dtype=np.float32),
                'sell_price': np.array([self.exports[0].cost[self.current_timestep]],
                                       dtype=np.float32),
                'available_power': np.array([self.current_production],
                                            dtype=np.float32),
                'required_power': np.array([self.current_consumption],
                                           dtype=np.float32)
            }

        return agents_batteries

    def __create_battery_actions__(self):
        """
        Create the storage actions. Composed by:
        - ctl: 0/1/2 for idle/charge/discharge
        - val: amount to charge/discharge relative to capacity
        :return:
        """

        agent_batteries = {}
        for i in self.batteries:
            agent_batteries[i.name] = gym.spaces.Dict({'ctl': gym.spaces.Discrete(2),
                                                       'val': gym.spaces.Box(low=0.0, high=1.0,
                                                                             shape=(1,),
                                                                             dtype=np.float32)})
        return agent_batteries

    def __get_battery_actions__(self, stor, actions):

        reward = 0.0
        if actions['ctl'] == 0:
            stor.charge = 0.0
            stor.discharge = 0.0
            reward = 0.0

        elif actions['ctl'] == 1:
            charge = actions['val'][0]
            if stor.value + charge > 1.0:
                charge_delta = stor.value + charge - 1.0
                charge = 1.0 - stor.value
                reward = charge_delta

            stor.charge = charge
            stor.discharge = 0.0

        elif actions['ctl'] == 2:
            stor.charge = 0.0

            discharge = actions['val'][0]
            if stor.value - discharge / stor.discharge_efficiency < 0.0:
                discharge_delta = abs(stor.value - discharge)
                discharge = stor.value
                reward = discharge_delta

            stor.discharge = discharge
        stor.value = stor.value + \
                     stor.charge * stor.charge_efficiency - \
                     stor.discharge / stor.discharge_efficiency
        return reward

    def __create_ev_observations__(self):
        """
        EVs have access to:
        - current SOC
        -
        :return:
        """

        agent_evs = {}
        for i in self.evs:
            agent_evs[i.name] = gym.spaces.Dict({
                'state': gym.spaces.Box(low=0.0,
                                        high=1.0,
                                        shape=(1,), dtype=np.float32),
                'next_departure': gym.spaces.Box(low=0,
                                                 high=24,
                                                 shape=(1,), dtype=np.int64),
                'time_until_departure': gym.spaces.Box(low=0,
                                                       high=24,
                                                       shape=(1,), dtype=np.int64),
                'required_soc': gym.spaces.Box(low=0.0,
                                               high=1.0,
                                               shape=(1,), dtype=np.float32),
                'remaining_soc': gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(1,), dtype=np.float32),
                'connected': gym.spaces.Discrete(2),
                'energy_price': gym.spaces.Box(low=0.0,
                                               high=self.imports[0].cost.max(),
                                               shape=(1,), dtype=np.float32),
                'time': gym.spaces.Box(low=0,
                                       high=24,
                                       shape=(1,), dtype=np.int64),
            })

        return agent_evs

    def __get_ev_observations__(self):

        obs = {}

        for agent in self.agents.keys():

            ev = self.agents[agent] if self.agents[agent].name.startswith('ev') else None

            if ev is not None:
                next_departure = np.where(ev.schedule_requirement_soc > self.current_timestep)[0]
                remains_trips = len(next_departure) > 0
                next_departure_soc = ev.schedule_requirement_soc[next_departure[0]] \
                    if remains_trips else ev.min_charge

                time_until_departure = abs(next_departure[0] - self.current_timestep) \
                    if remains_trips else ev.schedule_connected.shape[0] - 1

                observation = {'state': np.array([ev.value / ev.capacity_max],
                                                 dtype=np.float32),
                               'next_departure': np.array([next_departure[0]],
                                                          dtype=np.int64) if remains_trips else np.array([0],
                                                                                                         dtype=np.int64),
                               'time_until_departure': np.array([time_until_departure],
                                                                dtype=np.int64),
                               'required_soc': np.array([next_departure_soc / ev.capacity_max],
                                                        dtype=np.float32),
                               'remaining_soc': np.array([(next_departure_soc - ev.value) / ev.capacity_max],
                                                         dtype=np.float32) if ev.value < next_departure_soc
                               else np.array([0], dtype=np.float32),
                               'connected': int(ev.schedule_connected[self.current_timestep]),
                               'energy_price': np.array([self.imports[0].cost[self.current_timestep]],
                                                        dtype=np.float32),
                               'time': np.array([self.current_timestep],
                                                dtype=np.int64)}
                obs[ev.name] = observation
        return obs

    def __create_ev_actions__(self):
        """
        Similar to batteries, EVs can be idle/charge/discharge. Action space is the same
        :return:
        """

        agent_evs = {}
        for i in self.evs:
            agent_evs[i.name] = gym.spaces.Dict({'ctl': gym.spaces.Discrete(2),
                                                 'val': gym.spaces.Box(low=0.0, high=1.0,
                                                                       shape=(1,),
                                                                       dtype=np.float32)})
        return agent_evs

    def __get_ev_actions__(self, ev, actions):

        # Initialize penalty and costs
        penalty_action = 0.0
        penalty_charge = 0.0

        # Get the action
        action_value = actions['val'][0]
        action_indicator = actions['ctl']

        # Get the current state
        connected = ev.schedule_connected[self.current_timestep]
        charge = 0.0
        discharge = 0.0

        # Check if EV is connected
        if connected == 0:
            ev.value = 0.0

        # Check if EV is arriving
        if ev.schedule_arrival_soc[self.current_timestep] > 0:
            ev.value = ev.schedule_arrival_soc[self.current_timestep]

        # Get the next state
        if action_indicator == 0:
            charge = 0.0
            discharge = 0.0
        elif action_indicator == 1:
            # Charge
            charge = action_value * ev.charge_efficiency
        elif action_indicator == 2:
            # Discharge
            discharge = action_value / ev.discharge_efficiency
        else:
            raise ValueError('Action indicator should be 0, 1 or 2')

        # Check if the next state is within bounds
        if ev.value - discharge < ev.min_charge:
            # If not, set the discharge to reach the lower bound
            discharge = ev.value - ev.min_charge

            # Difference between next state and min charge
            penalty_action += abs(ev.min_charge - ev.value - discharge)

        if ev.value + charge > ev.capacity_max:
            # If not, set the next state to the max charge
            charge = ev.capacity_max - ev.value

            # Difference between next state and max charge
            penalty_action += ev.value + charge - ev.capacity_max

        # Update the EV
        ev.value = ev.value + charge - discharge
        ev.charge = charge
        ev.discharge = discharge

        # Check if EV is departing
        if ev.schedule_requirement_soc[self.current_timestep] > 0:
            if ev.value < ev.schedule_requirement_soc[self.current_timestep]:
                penalty_charge += self.ev_requirement_penalty

        return penalty_charge + penalty_action ** 2

    def _get_observations(self) -> dict:
        """
        Returns the observations of all agents
        :return: dict
        """

        if self.current_timestep == 24:
            return self.observation_space.sample()

        observations = {}
        gens_obs = self.__get_gen_observations__()
        stor_obs = self.__get_battery_observations__()
        ev_obs = self.__get_ev_observations__()

        for gen in gens_obs.keys():
            observations[gen] = gens_obs[gen]
        for stor in stor_obs.keys():
            observations[stor] = stor_obs[stor]
        for ev in ev_obs.keys():
            observations[ev] = ev_obs[ev]

        return observations

    def reset(self, *, seed=None, options=None):
        self.agents = self.__create_agents__()
        self.terminateds = set()
        self.truncateds = set()

        # Reset the environment
        self.current_timestep = 0
        self.current_production = 0.0
        self.current_consumption = 0.0

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

        self.current_production = sum([i.upper_bound[self.current_timestep]
                                       for i in self.generators])
        self.current_consumption = sum([i.upper_bound[self.current_timestep]
                                        for i in self.loads])

        updated_generation = 0.0
        total_stor_discharge = 0.0
        total_stor_charge = 0.0
        total_ev_discharge = 0.0
        total_ev_charge = 0.0

        gen_rewards = {}
        battery_rewards = {}
        ev_rewards = {}

        # Do actions
        if exists_actions:
            # Do generator action
            for action in action_dict.keys():
                if action.startswith('gen'):
                    gen_rewards[action] = self.__get_gen_actions__(self.agents[action], action_dict[action])
                elif action.startswith('stor'):
                    battery_rewards[action] = self.__get_battery_actions__(self.agents[action], action_dict[action])
                elif action.startswith('ev'):
                    ev_rewards[action] = self.__get_ev_actions__(self.agents[action], action_dict[action])

            # Generators
            updated_generation = sum([self.agents[agent].value if agent.startswith('gen') else 0.0
                                      for agent in self.agents.keys()])

            # Battery
            total_stor_discharge = sum([self.agents[agent].discharge / \
                                        self.agents[agent].discharge_efficiency
                                        if agent.startswith('stor') else 0.0
                                        for agent in self.agents.keys()])

            total_stor_charge = sum([self.agents[agent].charge * self.agents[agent].charge_efficiency
                                     if agent.startswith('stor') else 0.0
                                     for agent in self.agents.keys()])

            # EVs
            total_ev_discharge = sum([self.agents[agent].discharge / \
                                      self.agents[agent].discharge_efficiency
                                      if agent.startswith('ev') else 0.0
                                      for agent in self.agents.keys()])
            total_ev_charge = sum([self.agents[agent].charge * self.agents[agent].charge_efficiency
                                   if agent.startswith('ev') else 0.0
                                   for agent in self.agents.keys()])

        # Calculate energy requirements
        total_energy_requirements = sum([self.current_consumption,
                                         -1.0 * updated_generation,
                                         total_stor_charge, -1.0 * total_stor_discharge,
                                         total_ev_charge, -1.0 * total_ev_discharge])

        # Consider the imports and exports into the total energy requirements
        imported = 0.0
        exported = 0.0
        if total_energy_requirements > 0.0:
            imported = total_energy_requirements
        elif total_energy_requirements < 0.0:
            exported = np.abs(total_energy_requirements)

        # Calculate shared cost of the community
        shared_cost = sum([imported * self.imports[0].cost[self.current_timestep]
                           if imported < self.imports[0].upper_bound[0]
                           else self.illegal_import_cost,
                           -exported * self.exports[0].cost[self.current_timestep]
                           if exported < self.exports[0].upper_bound[0]
                           else self.illegal_export_cost]) / len(self.agents)

        # Calculate the rewards
        rewards = {}
        for key in gen_rewards.keys():
            rewards[key] = -1.0 * gen_rewards[key] - shared_cost
        for key in battery_rewards.keys():
            rewards[key] = -1.0 * battery_rewards[key] - shared_cost
        for key in ev_rewards.keys():
            rewards[key] = -1.0 * ev_rewards[key] - shared_cost

        # Update the info dictionary
        info = {}
        for key in self.agents.keys():
            agent = self.agents[key]

            if agent.name in rewards.keys():
                if agent.name.startswith('gen'):
                    info[agent.name] = {'production': agent.value,
                                        'reward': rewards[agent.name],
                                        'imports': imported,
                                        'exports': exported}
                elif agent.name.startswith('stor'):
                    info[agent.name] = {'current': agent.value,
                                        'charge': agent.charge,
                                        'discharge': agent.discharge,
                                        'reward': rewards[agent.name]}
                elif agent.name.startswith('ev'):
                    info[agent.name] = {'current': agent.value,
                                        'charge': agent.charge,
                                        'discharge': agent.discharge,
                                        'reward': rewards[agent.name]}

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

    def _get_rewards(self, val, cost, shared_cost) -> float:
        """
        Returns the reward for the current timestep.
        :return: reward according to the excess or deficit of the energy for the community
        """

        return -np.abs(val) * cost - shared_cost
