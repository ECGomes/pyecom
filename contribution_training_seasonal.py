# Imports

import ray
import pandas as pd
import numpy as np

import IPython.core.display_functions

from src.parsers import HMParser, CotevParser
from src.resources import Generator, Load, Storage, Aggregator, Vehicle
from src.algorithms.rl import EnergyCommunityContributionPriorityV6

import torch
from ray.tune import register_env
from ray import tune, train
from ray.air import CheckpointConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.rllib.algorithms.ppo import PPOConfig

import warnings
warnings.filterwarnings('ignore')

SEASON = 'summer'  # 'winter', 'spring', 'summer', 'autumn'
SEASON_START = 0
SEASON_END = 0

# General data
N_HOUSES = 20
N_STEPS = 24 * 366 * 4  # 14496  # 6576  # 24 * 366
# 1 year - 2020 (leap year), 4 because we are using 15 minutes intervals

# EC data for mostly cost values
data_ec = HMParser(file_path='data/EC_V4.xlsx', ec_id=1)
data_ec.parse()

# EV data from the EV4EU simulator
data_ev = CotevParser(population_path=
                      'data/simulation_20evs_1year_15t/population_366.csv',
                      driving_history_path=
                      'data/simulation_20evs_1year_15t/ev_driving_history_366.csv',
                      assigned_segments_path=
                      'data/simulation_20evs_1year_15t/assigned_segments_366.csv',
                      parse_date_start='2020',
                      parse_date_end='2020')
data_ev.parse()

# Create the generator and load lists to add on to
generators = []
loads = []
used_idx = []

temp_costs_df = pd.DataFrame({'cost_parameter_b': data_ec.generator['cost_parameter_b'][0, 0],
                              'cost_nde': data_ec.generator['cost_nde'][0],
                              'cost_cut': data_ec.load['cost_cut'][0],
                              'cost_reduce': data_ec.load['cost_reduce'][0],
                              'cost_ens': data_ec.load['cost_ens'][0],
                              'discharge_price': np.array(data_ec.storage['discharge_price'][2]),
                              'charge_price': np.array(data_ec.storage['charge_price'][0]),
                              'import_contracted_p_max': 200,
                              'export_contracted_p_max': 200,
                              'buy_price': data_ec.peers['buy_price'][0],
                              'sell_price': data_ec.peers['sell_price'][0]},
                             index=pd.date_range(start='2020-01-01', freq='H', periods=24))
# Resample to 15T and forward fill the values
temp_costs_df = temp_costs_df.resample('15T').ffill()
# print('Initial costs DataFrame:\n{}'.format(temp_costs_df))

# Now fill the remaining 3 timestamps
temp_costs_df = temp_costs_df.reindex(pd.date_range(start='2020-01-01', freq='15T', periods=96), method='ffill')

aux_date_range = pd.date_range(start='2020-01-01', freq='15T', periods=N_STEPS)
winter_start = 0
winter_end = aux_date_range.get_loc(aux_date_range[aux_date_range.month == 2].max())
spring_start = aux_date_range.get_loc(aux_date_range[aux_date_range.month == 3].min())
spring_end = aux_date_range.get_loc(aux_date_range[aux_date_range.month == 5].max())
summer_start = aux_date_range.get_loc(aux_date_range[aux_date_range.month == 6].min())
summer_end = aux_date_range.get_loc(aux_date_range[aux_date_range.month == 8].max())
autumn_start = aux_date_range.get_loc(aux_date_range[aux_date_range.month == 9].min())
autumn_end = aux_date_range.get_loc(aux_date_range[aux_date_range.month == 11].max())

if SEASON == 'winter':
    N_STEPS = winter_end - winter_start - 1344
    SEASON_START = winter_start
    SEASON_END = winter_end - 1344
elif SEASON == 'spring':
    N_STEPS = spring_end - spring_start - 1344
    SEASON_START = spring_start
    SEASON_END = spring_end - 1344
elif SEASON == 'summer':
    N_STEPS = summer_end - summer_start - 1344
    SEASON_START = summer_start
    SEASON_END = summer_end - 1344
elif SEASON == 'autumn':
    N_STEPS = autumn_end - autumn_start - 1344
    SEASON_START = autumn_start
    SEASON_END = autumn_end - 1344

for i in range(1, 21):
    temp_data = pd.read_csv('data/housedata/W/H{}_W.csv'.format(i))

    # Fill the generator and load missing data with zeros
    temp_data = temp_data.fillna(0)

    # Resample to 1H intervals
    temp_data['date'] = pd.to_datetime(temp_data['date'])
    temp_data = temp_data.set_index('date', drop=True)
    temp_data = temp_data.resample('15T').mean()

    # Convert to kWh
    temp_data[' Production(kW)'] = temp_data[' Production(W)'] / 1000
    temp_data[' Consumption(kW)'] = temp_data[' Consumption(W)'] / 1000

    # Check if there are enough data points
    if temp_data.shape[0] < 366 * 96:
        print(f'Not enough data for house {i}, skipping...')
        continue

    # Create the generator
    generators.append(Generator(name='ren_generator_{:02d}'.format(i),
                                value=np.zeros(N_STEPS),
                                lower_bound=np.zeros(N_STEPS),
                                upper_bound=temp_data[' Production(kW)'][SEASON_START:SEASON_END],
                                cost=np.tile(temp_costs_df['cost_parameter_b'], (int(N_STEPS / 95)))[:-1],
                                cost_nde=np.tile(temp_costs_df['cost_nde'], (int(N_STEPS / 95)))[:-1],
                                is_renewable=True))

    # Create the load
    loads.append(Load(name='load_{:02d}'.format(i),
                      value=temp_data[' Consumption(kW)'][SEASON_START:SEASON_END],
                      lower_bound=np.zeros(N_STEPS),
                      upper_bound=temp_data[' Consumption(kW)'][SEASON_START:SEASON_END],
                      cost=np.ones(N_STEPS),
                      cost_cut=np.tile(temp_costs_df['cost_cut'], (int(N_STEPS / 95)))[:-1],
                      cost_reduce=np.tile(temp_costs_df['cost_reduce'], (int(N_STEPS / 95)))[:-1],
                      cost_ens=np.tile(temp_costs_df['cost_ens'], (int(N_STEPS / 95)))[:-1]))

    used_idx.append(i - 1)  # Store the index of the house used

# Create the storages
storages = []
for i in used_idx:
    storages.append(Storage(name='storage_{:02d}'.format(i + 1),
                            value=np.tile(8.0, N_STEPS),
                            lower_bound=np.tile(2.0, N_STEPS),
                            upper_bound=np.tile(10.0, N_STEPS),
                            cost=np.zeros(N_STEPS),
                            cost_discharge=np.tile(temp_costs_df['discharge_price'], (int(N_STEPS / 95)))[:-1],
                            cost_charge=np.tile(temp_costs_df['charge_price'], (int(N_STEPS / 95)))[:-1],
                            capacity_max=10.0,
                            capacity_min=2.0,  # 0.2 * capacity_max
                            initial_charge=8.0,
                            discharge_efficiency=0.95,
                            discharge_max=np.tile(3.3, N_STEPS),
                            charge_efficiency=0.95,
                            charge_max=np.tile(3.3, N_STEPS),
                            capital_cost=np.array([0.05250, 0.10500, 0.01575])))

# EVs are already in the data_ev.resources, but we'll trim the excess data
evs = []
for i in used_idx:
    temp_res = data_ev.resources[i]
    evs.append(Vehicle(name='ev_{:02d}'.format(i + 1),
                       value=temp_res.value[SEASON_START:SEASON_END],
                       lower_bound=temp_res.lower_bound[SEASON_START:SEASON_END],
                       upper_bound=temp_res.upper_bound[SEASON_START:SEASON_END],
                       cost=temp_res.cost[SEASON_START:SEASON_END],
                       cost_discharge=temp_res.cost_discharge[SEASON_START:SEASON_END],
                       cost_charge=temp_res.cost_charge[SEASON_START:SEASON_END],
                       capacity_max=temp_res.capacity_max,
                       initial_charge=temp_res.initial_charge,
                       min_charge=temp_res.min_charge,
                       discharge_efficiency=temp_res.discharge_efficiency,
                       charge_efficiency=temp_res.charge_efficiency,
                       schedule_connected=temp_res.schedule_connected[SEASON_START:SEASON_END],
                       schedule_discharge=temp_res.schedule_discharge[SEASON_START:SEASON_END],
                       schedule_charge=temp_res.schedule_charge[SEASON_START:SEASON_END],
                       schedule_requirement_soc=temp_res.schedule_requirement_soc[SEASON_START:SEASON_END],
                       schedule_arrival_soc=temp_res.schedule_arrival_soc[SEASON_START:SEASON_END]))

# Create the aggregator
aggregator = Aggregator(name='aggregator',
                        value=np.zeros(N_STEPS),
                        lower_bound=np.zeros(N_STEPS),
                        upper_bound=np.tile(temp_costs_df['import_contracted_p_max'], (int(N_STEPS / 95)))[:-1],
                        cost=np.tile(data_ec.peers['buy_price'], (int(N_STEPS / 95)))[:-1],
                        imports=np.zeros(N_STEPS),
                        exports=np.zeros(N_STEPS),
                        import_cost=np.tile(temp_costs_df['buy_price'] * 2, (int(N_STEPS / 95)))[:-1],
                        export_cost=np.tile(temp_costs_df['sell_price'], (int(N_STEPS / 95)))[:-1],
                        import_max=np.tile(temp_costs_df['import_contracted_p_max'], (int(N_STEPS / 95)))[:-1],
                        export_max=np.tile(temp_costs_df['export_contracted_p_max'], (int(N_STEPS / 95)))[:-1])

# Create the environment and check if everything is ok
temp_env = EnergyCommunityContributionPriorityV6(ren_generators=generators,
                                                 generators=[],
                                                 loads=loads,
                                                 storages=storages,
                                                 evs=evs,
                                                 aggregator=aggregator,
                                                 storage_penalty=1,
                                                 ev_penalty=1,
                                                 balance_penalty=1,
                                                 look_ahead=12,
                                                 max_episode_length=N_STEPS - 1,
                                                 seed=42)
temp_env.reset()
terminations = truncations = {a: False for a in temp_env.agents}
terminations['__all__'] = False
truncations['__all__'] = False
while not terminations['__all__'] and not truncations['__all__']:
    actions = temp_env.action_space.sample()
    next_obs, rewards, terminations, truncations, infos = temp_env.step(actions)

print('Terminated: {}'.format(terminations['__all__']))

# Create the policies to train
# The keys of the dictionary respect the class names of the agents
gammas = {'Storage': 0.99, 'Vehicle': 0.99, 'Aggregator': 0.99}

policies = {}
for ev in temp_env.evs:
    policies[ev.name] = (None,
                         temp_env.observation_space[ev.name],
                         temp_env.action_space[ev.name],
                         PPOConfig.overrides(gamma=gammas['Vehicle']))

for storage in temp_env.storages:
    policies[storage.name] = (None,
                              temp_env.observation_space[temp_env.storages[0].name],
                              temp_env.action_space[temp_env.storages[0].name],
                              PPOConfig.overrides(gamma=gammas['Storage']))

policies['aggregator'] = (None,
                          temp_env.observation_space['aggregator'],
                          temp_env.action_space['aggregator'],
                          PPOConfig.overrides(gamma=gammas['Aggregator']))

# Create an RLlib Algorithm instance from a PPOConfig to learn how to
# act in the above environment.

num_gpus = int(torch.cuda.is_available())

ray.shutdown()
ray.init(num_gpus=num_gpus)

IMPORT_PENALTY = 1  # 100
EXPORT_PENALTY = 1  # 10
STORAGE_ACTION_PENALTY = 50  # 100
STORAGE_ACTION_REWARD = 5  # 10
EV_ACTION_PENALTY = 1  # 1000
EV_ACTION_REWARD = 5  # 10
EV_REQUIREMENT_PENALTY = 50  # 3000
BALANCE_PENALTY = 5000  # 20000

MAX_ITER = 3000

checkpoint = None
checkpoint_path = None
algo = None
current_best = None

# Create the environment to train on
env = EnergyCommunityContributionPriorityV6(ren_generators=generators,
                                            generators=[],
                                            loads=loads,
                                            storages=storages,
                                            evs=evs,
                                            aggregator=aggregator,
                                            storage_penalty=STORAGE_ACTION_PENALTY,
                                            ev_penalty=EV_REQUIREMENT_PENALTY,
                                            balance_penalty=BALANCE_PENALTY,
                                            look_ahead=12,
                                            max_episode_length=N_STEPS - 1,
                                            seed=42,
                                            is_training=True)
register_env("EC_Contrib_V6", lambda config: env)

# Define the PPOConfig
_config = (PPOConfig()
           .environment(env="EC_Contrib_V6", disable_env_checking=False)
           .training(train_batch_size=12288,
                     sgd_minibatch_size=2048,
                     num_sgd_iter=5,
                     lr=1e-4,
                     gamma=0.99)
           .exploration(exploration_config={})
           .framework('torch')
           .resources(num_cpus_per_worker=num_gpus)
           .multi_agent(policies=policies,
                        policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: agent_id))
           .rollouts(batch_mode='truncate_episodes',
                     num_rollout_workers=6,
                     num_envs_per_worker=2,
                     rollout_fragment_length=1024))

# Clear the Jupyter cell output
IPython.core.display_functions.clear_output()

scheduler = AsyncHyperBandScheduler(time_attr="training_iteration",
                                    max_t=MAX_ITER,
                                    metric="episode_reward_mean",
                                    mode="max",
                                    grace_period=10)

# Train the algorithm with Tuner
tuner = tune.Tuner(
    "PPO",
    param_space=_config,
    run_config=train.RunConfig(stop={'training_iteration': MAX_ITER},
                               checkpoint_config=CheckpointConfig(checkpoint_frequency=100,
                                                                  checkpoint_at_end=True),
                               verbose=1),
    tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=1),
)

results = tuner.fit()

print(results.get_best_result('episode_reward_mean',
                              'max').get_best_checkpoint('episode_reward_mean', 'max').path)
