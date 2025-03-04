# Auxiliar function to create the policies for the training environment.

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig


def create_ppo_policies(env, gammas: dict) -> dict:
    """
    Create the policies for the training environment.
    """

    policies = {}

    for agent in env.agents:
        gamma = gammas[str(type(env.agents[agent])).split('.')[-1].split("'")[0]]

        policies['{}'.format(agent)] = (None,
                                        env.observation_space[agent],
                                        env.action_space[agent],
                                        PPOConfig.overrides(gamma=gamma))

    return policies


def create_hierarchical_ppo_policies(env, gammas: dict) -> dict:
    policies = {}

    for agent in env.possible_agents:

        if agent != 'aggregator':
            gamma = gammas[str(type(env.agents[f'{agent}_01'])).split('.')[-1].split("'")[0]]

            policies['{}'.format(agent)] = (None,
                                            env.observation_space[f'{agent}_01'],
                                            env.action_space[f'{agent}_01'],
                                            PPOConfig.overrides(gamma=gamma))
        else:
            gamma = gammas[str(type(env.agents[agent])).split('.')[-1].split("'")[0]]

            policies['{}'.format(agent)] = (None,
                                            env.observation_space[agent],
                                            env.action_space[agent],
                                            PPOConfig.overrides(gamma=gamma))

    return policies


def create_sac_policies(env, gammas: dict) -> dict:
    """
    Create the policies for the training environment.
    """

    policies = {}

    for agent in env.agents:
        gamma = gammas[str(type(env.agents[agent])).split('.')[-1].split("'")[0]]

        policies['{}'.format(agent)] = (None,
                                        env.observation_space[agent],
                                        env.action_space[agent],
                                        SACConfig.overrides(gamma=gamma))

    return policies
