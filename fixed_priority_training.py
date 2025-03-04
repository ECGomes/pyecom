# Imports

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from src.parsers import HMParser, CotevParser
from src.algorithms.rl import EnergyCommunitySequentialV12

from src.utils import load_multiple_upacs_pv, iterate_resources, create_ppo_policies
from src.priorities import EmpiricalPriority

import warnings
warnings.filterwarnings('ignore')


# Data parsing

# EC data for non-renewable generators and batteries
data_ec = HMParser(file_path='data/EC_V4.xlsx', ec_id=1)
data_ec.parse()

# EV data from the EV4EU simulator
data_ev = CotevParser(population_path=
                      'data/simulation_dataframes_2years/population_731.csv',
                      driving_history_path=
                      'data/simulation_dataframes_2years/ev_driving_history_731.csv',
                      assigned_segments_path='data/simulation_dataframes_2years/assigned_segments_731.csv',
                      parse_date_start='2019',
                      parse_date_end='2020')
data_ev.parse()




# UPAC Data load
data_upacs = load_multiple_upacs_pv('data/upac_data/upac*_pv.csv', resample='H')




# Create resources for the training environment

dataset_resources = iterate_resources(u=data_upacs, c=data_ec, e=data_ev, mode='monthly')

# Get the execution order

execution_order = EmpiricalPriority(dataset_resources['2019-01'])
execution_order = execution_order.calculate_priority()

order = [x for x in execution_order['name'] if 'load' not in x and 'ren_gen' not in x]

# Create the environment and check if everything is ok

temp_env = EnergyCommunitySequentialV12(ren_generators=dataset_resources[list(dataset_resources.keys())[0]][:5],
                                        generators=[],
                                        loads=dataset_resources[list(dataset_resources.keys())[0]][5:10],
                                        storages=dataset_resources[list(dataset_resources.keys())[0]][10:13],
                                        evs=dataset_resources[list(dataset_resources.keys())[0]][13:-1],
                                        aggregator=dataset_resources[list(dataset_resources.keys())[0]][-1],
                                        storage_penalty=1,
                                        ev_penalty=1,
                                        balance_penalty=1,
                                        execution_order=order,
                                        look_ahead=12)
temp_env.reset()
terminations = truncations = {a: False for a in temp_env.agents}
terminations['__all__'] = False
truncations['__all__'] = False
while not terminations['__all__'] and not truncations['__all__']:

    actions = temp_env.action_space_sample()
    next_obs, rewards, terminations, truncations, infos = temp_env.step(actions)

print('Terminated: {}'.format(terminations['__all__']))

# Create the policies to train

# The keys of the dictionary respect the class names of the agents
gammas = {'Generator': 0.0, 'Storage': 0.9, 'Vehicle': 0.9, 'Aggregator': 0.9}

# Create the policies, one for each agent. Each policy has the name of the agent.
policies = create_ppo_policies(temp_env, gammas)

# Create individual networks for each agent

model_cfgs = {}
for agent in temp_env.agents:

    if agent.startswith('storage'):
        model_cfg = {'use_lstm': True,
                     'lstm_cell_size': 128,
                     'fcnet_hiddens': [32, 32],
                     'fcnet_activation': 'relu',
                     'lstm_use_prev_action': True,
                     'lstm_use_prev_reward': True,
                     'vf_share_layers': False}

    elif agent.startswith('ev'):
        model_cfg = {'use_lstm': True,
                     'lstm_cell_size': 128,
                     'fcnet_hiddens': [32, 32],
                     'fcnet_activation': 'relu',
                     'lstm_use_prev_action': True,
                     'lstm_use_prev_reward': True,
                     'vf_share_layers': False}

    else:
        model_cfg = {'fcnet_hiddens': [1],}

    model_cfgs[agent] = model_cfg

    # Create an RLlib Algorithm instance from a PPOConfig to learn how to
# act in the above environment.

from ray.tune import register_env
from ray import tune, train
from ray.air import CheckpointConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper

ray.shutdown()
ray.init()

IMPORT_PENALTY = 1 #100
EXPORT_PENALTY = 1 #10
STORAGE_ACTION_PENALTY = 50 #100
STORAGE_ACTION_REWARD = 5 #10
EV_ACTION_PENALTY = 1 #1000
EV_ACTION_REWARD = 5 #10
EV_REQUIREMENT_PENALTY = 50 #3000
BALANCE_PENALTY = 5000 #20000

MAX_ITER = 500

checkpoint = None
checkpoint_path = None
algo = None
current_best = None

# Build a loop for using separate resources on a daily basis

temp_resources = dataset_resources['2019-01']

env = EnergyCommunitySequentialV12(ren_generators=temp_resources[:5],
                                   generators=[],
                                   loads=temp_resources[5:10],
                                   storages=temp_resources[10:13],
                                   evs=temp_resources[13:-1],
                                   aggregator=temp_resources[-1],
                                   storage_penalty=STORAGE_ACTION_PENALTY,
                                   ev_penalty=EV_REQUIREMENT_PENALTY,
                                   balance_penalty=BALANCE_PENALTY,
                                   execution_order=order,
                                   look_ahead=12)
register_env("EC_Seq_V2", lambda config: env)

# Define the PPOConfig
_config = (PPOConfig()
           .environment(env="EC_Seq_V2", disable_env_checking=False)
           .training(train_batch_size=256,
                     lr=5e-5,
                     gamma=0.99,
                     use_gae=True,
                     use_critic=True,
                     use_kl_loss=True,
                     clip_param=0.1,
                     grad_clip=4,
                     )
           .exploration(exploration_config={})
           .framework('torch')
           .multi_agent(policies=policies,
                        policies_to_train=list(policies.keys())[:-1],
                        policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs:
                                           agent_id),
                        algorithm_config_overrides_per_module=model_cfgs)
           .rollouts(batch_mode='complete_episodes',
                     num_rollout_workers=10,
                     rollout_fragment_length='auto'))


scheduler = AsyncHyperBandScheduler(time_attr="training_iteration",
                                    max_t=MAX_ITER,
                                    metric="episode_reward_mean",
                                    mode="max",
                                    grace_period=10)

# Train the algorithm with Tuner
tuner = tune.Tuner(
    "PPO",
    param_space=_config,
    run_config=train.RunConfig(stop={'training_iteration': MAX_ITER, 'episode_reward_mean': -50.0},
                               checkpoint_config=CheckpointConfig(checkpoint_frequency=10,
                                                                  checkpoint_at_end=True)),
    tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=1)
)

results = tuner.fit()

print(results.get_best_result('episode_reward_mean',
                              'max').get_best_checkpoint('episode_reward_mean', 'max').path)
