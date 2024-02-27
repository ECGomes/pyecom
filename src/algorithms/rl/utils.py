# Contains definition of auxiliary functions for reinforcement learning algorithms

from src.resources.base_resource import BaseResource


# Resource separation
def separate_resources(resources: list[BaseResource]) -> dict:
    """
    Separate the resources into different categories:
    - Generators
    - Loads
    - Storages
    - EVs
    - Imports
    - Exports

    Expecting the resources to be named as follows:
    - Generators: generator_*
    - Loads: load_*
    - Storages: storage_*
    - EVs: ev_*
    - Imports: import_*
    - Exports: export_*

    :param resources: list of resources
    :return: dictionary with separated resources
    """

    separated_resources = {
        "generators": [resource
                       for resource in resources
                       if resource.name.startswith('generator')],
        "loads": [resource
                  for resource in resources
                  if resource.name.startswith('load')],
        "storages": [resource
                     for resource in resources
                     if resource.name.startswith('storage')],
        "evs": [resource
                for resource in resources
                if resource.name.startswith('ev')],
        "aggregator": [resource
                       for resource in resources
                       if resource.name.startswith('aggregator')],
    }

    return separated_resources
