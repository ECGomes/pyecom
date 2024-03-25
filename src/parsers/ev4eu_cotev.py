# Parser logic for the COTEV simulator outputs

import pandas as pd
import numpy as np

from src.parsers.base_parser import BaseParser
from src.resources.vehicle import Vehicle


class CotevParser(BaseParser):
    def __init__(self,
                 population_path: str,
                 driving_history_path: str,
                 assigned_segments_path: str,
                 return_resources: bool = True):
        super().__init__(population_path)

        self.population_path = population_path
        self.driving_history_path = driving_history_path
        self.assigned_segments_path = assigned_segments_path

        self.return_resources = return_resources

    def parse_driving_history(self):
        # Read the driving history CSV
        driving_history = pd.read_csv(self.driving_history_path)
        driving_history = driving_history.transpose()

        # Set the first row as headers
        driving_history.columns = driving_history.iloc[0]
        driving_history = driving_history.drop(driving_history.index[0])

        # Set the index as the datetime
        driving_history.index = pd.to_datetime(driving_history.index)

        # Convert the columns to numeric
        driving_history.dtype = 'bool'

        return driving_history

    def parse_population(self):

        # Read the population CSV
        population = pd.read_csv(self.population_path, index_col=0)

        return population

    def parse_assigned_segments(self, grid, population):
        # Get the trips per EV

        df_trips = pd.read_csv(self.assigned_segments_path, index_col=0)

        # Create an empty DataFrame with the same shape as the grid
        df_trips_grid = pd.DataFrame(np.zeros(grid.shape), index=grid.index, columns=grid.columns)

        # Fill the DataFrame with the trip requirements
        for ev in grid.columns:
            current_trips = df_trips[df_trips['ev_id'] == ev]

            for i in np.arange(current_trips.shape[0]):
                start = current_trips.iloc[i]['trip_start_time']

                # We're working with 1h intervals
                start = pd.Timestamp(start).replace(minute=0, second=0)

                df_trips_grid.loc[start, ev] = current_trips.iloc[i]['trip_required_soc'] * \
                                               population[population['ev_id'] == ev]['battery_size'].values[0]

        return df_trips_grid

    def parse(self):

        # Parse the driving history
        dh = self.parse_driving_history()

        # Build a schedule of grid connections
        # Will be the reverse of the driving history -> whenever it is stopped we assume it is connected
        # Create a new dataframe
        df_grid = {'{}'.format(col): [not x for x in dh[col]] for col in dh.columns}
        df_grid = pd.DataFrame(df_grid, index=dh.index)
        df_grid = df_grid.astype('int')

        # Parse the population
        population = self.parse_population()

        # Parse the assigned segments
        assigned_segments = self.parse_assigned_segments(df_grid, population)

        if self.return_resources:
            resources = self.create_resources(population, df_grid, assigned_segments)

            return resources

        return

    @staticmethod
    def create_resources(population, grid, segments):
        # Turn into a list of Vehicle objects
        resources = []
        for i in np.arange(grid.shape[1]):
            current_ev = grid.columns[i]
            vehicle = Vehicle(name='ev_{:02d}'.format(i + 1),
                              value=grid[current_ev].shape,
                              lower_bound=np.ones(grid[current_ev].shape) * population.iloc[i]['battery_size'] * 0.0,
                              upper_bound=np.ones(grid[current_ev].shape) * population.iloc[i]['battery_size'],
                              cost=np.zeros(grid[current_ev].shape),
                              cost_discharge=np.ones(grid[current_ev].shape) * 0.05,
                              cost_charge=np.ones(grid[current_ev].shape) * 0.0,
                              capacity_max=population.iloc[i]['battery_size'],
                              initial_charge=population.iloc[i]['battery_size'] * population.iloc[i]['soc_min'],
                              min_charge=population.iloc[i]['battery_size'] * 0.2,
                              discharge_efficiency=0.9,
                              charge_efficiency=0.9,
                              schedule_connected=grid[current_ev],
                              schedule_discharge=grid[current_ev] * 7.2,
                              schedule_charge=grid[current_ev] * 7.2,
                              schedule_requirement_soc=segments[current_ev],
                              schedule_arrival_soc=np.zeros(grid[current_ev].shape),
                              )
            resources.append(vehicle)

        return resources
