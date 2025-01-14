# Auxiliar function to create the policies for the training environment.

from ray.rllib.algorithms.ppo import PPOConfig


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
