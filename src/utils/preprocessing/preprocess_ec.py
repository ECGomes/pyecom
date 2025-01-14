# Auxiliar methods to create the sample EC from UPAC, COTEV, and Excel data.

import numpy as np

from ...resources import Generator, Load, Storage, Vehicle, Aggregator


def create_resources(upacs, ec, ev):
    """
    Create the resources for the training environment.
    return a list of resources.
    :param upacs: dict with the UPAC data
    :param ec: dict with the EC data
    :param ev: dict with the EV data
    """

    resources = []
    # Add generators (from pv column from the UPAC data)
    for i in range(len(upacs)):
        current_name = list(upacs.keys())[i]
        resources.append(Generator(
            name='ren_generator_' + current_name,
            value=np.zeros(upacs[current_name]['pv'].shape),
            lower_bound=np.zeros(upacs[current_name]['pv'].shape),
            upper_bound=upacs[current_name]['pv'].values,
            cost=ec.generator['cost_parameter_b'][0, 0] * np.ones(upacs[current_name].shape[0]),
            cost_nde=np.tile(ec.generator['cost_nde'][0], (int(upacs[current_name].shape[0] / 24))),
            is_renewable=True))

    # Add loads (from load column from the UPAC data)
    for i in range(len(upacs)):
        current_name = list(upacs.keys())[i]
        resources.append(Load(
            name='load_' + current_name,
            value=upacs[current_name]['load'],
            lower_bound=np.zeros(upacs[current_name].shape),
            upper_bound=upacs[current_name]['load'].values,
            cost=np.ones(upacs[current_name].shape[0]),
            cost_cut=np.tile(ec.load['cost_cut'][0], (int(upacs[current_name].shape[0] / 24))),
            cost_reduce=np.tile(ec.load['cost_reduce'][0], (int(upacs[current_name].shape[0] / 24))),
            cost_ens=np.tile(ec.load['cost_ens'][0], (int(upacs[current_name].shape[0] / 24)))))

    # Add storage (from the EC data)
    for i in range(ec.storage['p_charge_limit'].shape[0]):
        resources.append(Storage(
            name='storage_{:02d}'.format(i + 1),
            value=ec.storage['initial_state'][i] * np.ones(upacs['02'].shape[0]),
            lower_bound=np.ones(upacs['02'].shape[0]) * ec.storage['energy_min_percentage'][i],
            upper_bound=(ec.storage['energy_capacity'][i] * np.ones(upacs['02'].shape[0])),
            cost=np.ones(upacs['02'].shape[0]) * 0,
            cost_discharge=np.tile(ec.storage['discharge_price'][i], (int(upacs['02'].shape[0] / 24))),
            cost_charge=np.tile(ec.storage['charge_price'][i], (int(upacs['02'].shape[0] / 24))),
            capacity_max=ec.storage['energy_capacity'][i],
            capacity_min=ec.storage['energy_min_percentage'][i],
            initial_charge=ec.storage['initial_state'][i],
            discharge_efficiency=ec.storage['discharge_efficiency'][i],
            discharge_max=np.tile(ec.storage['p_discharge_limit'][i], (int(upacs['02'].shape[0] / 24))),
            charge_efficiency=ec.storage['charge_efficiency'][i],
            charge_max=np.tile(ec.storage['p_charge_limit'][i], (int(upacs['02'].shape[0] / 24))),
            capital_cost=np.array([0.05250, 0.10500, 0.01575])))

    # Add vehicles (from the EV data)
    for i in np.arange(len(ev)):
        # Append to the list of resources
        resources.append(ev[i])

    # Append Aggregator
    resources.append(Aggregator(
        name='aggregator',
        value=np.zeros(upacs['02'].shape[0]),
        lower_bound=np.zeros(upacs['02'].shape[0]),
        upper_bound=np.tile(ec.peers['import_contracted_p_max'][0, 0], (upacs['02'].shape[0])),
        cost=np.tile(ec.peers['buy_price'][0, 0], (upacs['02'].shape[0])),
        imports=np.zeros(upacs['02'].shape[0]),
        exports=np.zeros(upacs['02'].shape[0]),
        import_cost=np.tile(ec.peers['buy_price'][0], (int(upacs['02'].shape[0] / 24))),
        export_cost=np.tile(ec.peers['sell_price'][0], (int(upacs['02'].shape[0] / 24))),
        import_max=np.tile(ec.peers['import_contracted_p_max'][0, 0], (int(upacs['02'].shape[0]))),
        export_max=np.tile(ec.peers['export_contracted_p_max'][0, 0], (int(upacs['02'].shape[0])))))

    return resources


def iterate_resources(u, c, e, mode='daily'):
    """
    Iterate over the datasets to create the resources for the training environment.
    :param u: dict with the UPAC data
    :param c: dict with the EC data
    :param e: dict with the EV data
    :param mode: str with the mode to iterate over the datasets. Options: 'daily', 'monthly', 'yearly'
    """

    temp = {}

    # Save first key of upac data
    first_key = list(u.keys())[0]

    if mode == 'daily':

        # Loop to iterate over days in the datasets
        for i in np.unique(u[first_key].index.date):
            # Create the resources for the training environment

            date = i.strftime('%Y-%m-%d')

            temp_u = {k: v.loc[date] for k, v in u.items()}
            temp_e = e.create_resources(e.population, e.trips_grid, e.assigned_segments, date)

            temp[date] = create_resources(upacs=temp_u,
                                          ec=c,
                                          ev=temp_e)

    elif mode == 'monthly':

        # Loop to iterate over months in the datasets
        # Need to be careful with different years
        unique_months = np.unique(u['02'].index.strftime('%Y-%m'))

        for i in unique_months:
            # Create the resources for the training environment
            date = i

            temp_u = {k: v.loc[date] for k, v in u.items()}
            temp_e = e.create_resources(e.population, e.trips_grid, e.assigned_segments, date)

            temp[date] = create_resources(upacs=temp_u,
                                          ec=c,
                                          ev=temp_e)

    elif mode == 'yearly':

        # Loop to iterate over years in the datasets
        unique_years = np.unique(u['02'].index.strftime('%Y'))

        for i in unique_years:
            # Create the resources for the training environment
            date = i

            temp_u = {k: v.loc[date] for k, v in u.items()}
            temp_e = e.create_resources(e.population, e.trips_grid, e.assigned_segments, date)

            temp[date] = create_resources(upacs=temp_u,
                                          ec=c,
                                          ev=temp_e)

    return temp
