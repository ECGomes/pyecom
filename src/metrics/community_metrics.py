# Imports

import math

import numpy as np

from .metrics_base import BaseMetric
from ..resources.base_resource import BaseResource
from .utils import aux_get_size, aux_error_checking


class CommunityMetrics(BaseMetric):

    def __repr__(self):
        print('Community Metrics available: ')
        print('- self_consumption')
        print('- prosumer_consumption_community')
        print('- total_produced')
        print('- total_consumed')
        print('- import_export_balance')

        return ''

    def check_metrics_1input(self, components_a, metric_list):
        """
        Checks the metrics and returns the metrics that are valid for the given inputs.
        Only support metrics with one input.
        :param components_a: Ground truth dataframe
        :param metric_list: List of metrics to check
        :return: List of valid metrics
        """

        # Create a placeholder for the metrics
        metrics = {}
        for metric in metric_list:
            try:
                metrics[metric] = self.callFunction(metric, components_a, None)
            except Exception as e:
                print('The requested metric is not available with provided inputs: ', metric)
                metrics[metric] = None

        return metrics

    def check_metrics_2inputs(self, components_a, components_b, metric_list):
        """
        Checks the metrics and returns the metrics that are valid for the given inputs.
        Only support metrics with two inputs.
        :param components_a: Ground truth dataframe
        :param components_b: Predicted dataframe
        :param metric_list: List of metrics to check
        :return: List of valid metrics
        """

        # Create a placeholder for the metrics
        metrics = {}
        for metric in metric_list:
            try:
                metrics[metric] = self.callFunction(metric, components_a, components_b)
            except Exception as e:
                print('The requested metric is not available with provided inputs: ', metric)
                metrics[metric] = None

        return metrics

    @staticmethod
    def cmd_self_consumption(production: np.ndarray, consumption: np.ndarray) -> float:
        """
        Calculates the self-consumption metric. Values closer to 1.0 are better.
        :param production: List of production resources
        :param consumption: List of consumption resources
        :return: Self-consumption value
        """
        total_production = np.sum(production)
        total_consumption = np.sum(consumption)

        return total_consumption / total_production

    @staticmethod
    def cmd_prosumer_consumption_community(prosumer_consumption: BaseResource,
                                           community_consumption: list) -> float:
        """
        Calculates the consumed power of a prosumer against the total consumed by the community.
        :param prosumer_consumption: Consumption of the prosumer
        :param community_consumption: net consumption of the community
        :return: Consumption of the prosumer against the total consumed by the community
        """

        prosumer = np.sum(prosumer_consumption)
        total_consumption = np.sum(community_consumption)

        return prosumer / total_consumption

    @staticmethod
    def cmd_total_produced(production: np.ndarray) -> float:
        """
        Calculates the total production of the community.
        :param production: Production resources
        :return: Sum production of the community
        """
        return np.sum(production)

    @staticmethod
    def cmd_total_consumed(consumption: np.ndarray) -> float:
        """
        Calculates the total consumption of the community.
        :param consumption: Consumption resources
        :return: Sum consumption of the community
        """

        return np.sum(consumption)

    @staticmethod
    def cmd_import_export_balance(imports: np.ndarray, exports: np.ndarray) -> float:
        """
        Calculates the import/export balance of the community. Negative values indicate that the community is
        exporting more than it is importing.
        :param imports: Community imports
        :param exports: Community exports
        :return: 
        """

        return np.sum(imports) - np.sum(exports)
