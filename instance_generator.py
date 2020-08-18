# -*- coding: utf-8 -*-
"""Class for generating problem instances (in this case, weighted bipartite graphs).

Contains the InstanceGenerator class, which creates problem instances for uniform cardinal valuations under unit-sum and unit-range normalizations, in addition to risk-averse,
risk-neutral, and risk-loving utility functions. ALso logs all problem instances created by InstanceGenerator.

"""

import networkx as nx
import numpy as np
import pandas as pd