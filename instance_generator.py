# -*- coding: utf-8 -*-
"""Class for generating problem instances (in this case, weighted bipartite graphs).

Contains the InstanceGenerator class, which creates problem instances for uniform cardinal valuations under unit-sum and unit-range normalizations, in addition to risk-averse,
risk-neutral, and risk-loving utility functions. ALso logs all problem instances created by InstanceGenerator.

"""

import networkx as nx
import numpy as np
import pandas as pd

class InstanceGenerator:
    """Object that generates problem instances for the simulation.
    
    Attributes:
        logging (bool): Whether or not to log generated instances for this InstanceGenerator.
        history (dict): A dictionary containing all logged entries.
        index (int): an index representing the id of the next entry that will be logged.
        
        
    
    """

    def __init__(self, logging=False):
        """Initializes a new InstanceGenerator.
        
        Args:
            logging: Whether or not all instances generated by this InstanceGenerator should be stored.
        """
        self.logging = logging
        self.history = {}
        self.index = 1

    def matrix_to_graph(self, M):
        """Helper method that transforms a square matrix into a weighted bipartite graph.
        
        Args:
            M: two-dimensional square numpy array, where rows represent agent preferences over goods, of size n.

        Returns:
            Weighted bipartite Graph, with nodes 1 through n representing agents.    
        """
        n=M.shape[0]
        G = nx.Graph()
        G.add_nodes_from(list(range(1, 2*n+1)))

        goods_array = np.array(list(range(n+1, 2*n+1)))

        for i in range(n):
            agent_array = np.array([i+1]*n)
            value_array = M[i]
            weight_data = np.array([agent_array, goods_array, value_array]).T
            G.add_weighted_edges_from(weight_data)

        return G

    def generate_unit_range_unif(self, n):
        """Generates a new instance of a weighted bipartite graph with 2n nodes, where agents' valuations are drawn
        uniformly at random from the space of all unit-range valuations.
        
        Args:
            n: the number of agents in the bipartite graph. For unit-range to make sense, n must be 2 or greater
            
        Returns:
            Weighted bipartite Graph, with nodes 1 through n representing agents.
        """
        M = []
        for i in range(n):
            val = np.concatenate([np.array([0,1]), np.random.rand(n-2)], axis=0)
            np.random.shuffle(val)
            M.append(val)

        M = np.array(M)

        if self.logging:
            self.history[self.index] = M
            self.index += 1

        return self.matrix_to_graph(M)

    def generate_unit_sum_unif(self, n):
        """Generates a new instance of a weighted bipartite graph with 2n nodes, where agents' valuations are drawn
        uniformly at random from the space of all unit-sum valuations.
        
        Args:
            n: the number of agents in the bipartite graph
            
        Returns:
            Weighted bipartite Graph, with nodes 1 through n representing agents.
        """
        M = []

        for i in range(n):
            #code for uniformly generating a point from the n-dimensional unit simplex
            cut = np.random.rand(n)
            cut = np.sort(cut)
            cut = np.insert(cut, 0, 0)
            val = [cut[j+1] - cut[j] for j in range(n)]
            val = np.array(val)
            np.random.shuffle(val)

            M.append(val)

        M = np.array(M)

        if self.logging:
            self.history[self.index] = M
            self.index += 1

        return self.matrix_to_graph(M)
