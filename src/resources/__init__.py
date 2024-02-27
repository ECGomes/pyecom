# Add reference to the resources package

# Path: src\resources\__init__.py
# Add the resources package to __init__.py
from .base_resource import BaseResource
from .generator import Generator, GeneratorProbabilistic
from .storage import Storage
from .load import Load, LoadProbabilistic
from .vehicle import Vehicle
from .binary_resource import BinaryResource
from .aggregator import Aggregator
