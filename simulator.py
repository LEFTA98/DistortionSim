# -*- coding: utf-8 -*-
"""Overall class for running the simulations.

Contains the Simulator class, which holds all of the matching algorithms and calculates the distortion. Also logs the outcomes of the experiments for analysis and
visualization.

"""

import networkx as nx
import numpy as np
import pandas as pd
import instance_generator
from solver import serial_dictatorship, partial_max_matching, modified_max_matching, hybrid_max_matching, calculate_distortion


class Simulator:
    """Object that solves instances of matching problems given to it and aggregates the results in a pretty manner.

    Attributes:
        instance_generator (InstanceGenerator): InstanceGenerator for creating problem instances for simulation
        history (dict): history of all results, stored in a dictionary for later conversion to pd.DataFrame
    """

    def __init__(self, instance_generate):
        """Initializes a new Simulator.

        Args:
            instance_generate (InstanceGenerator): InstanceGenerator for creating problem instances for simulation
        """
        self.instance_generator = instance_generate
        self.history = {'id': [], 'val_index': [], 'size': [], 'valuation':[], 'algo': [], 'distortion': []}
        self.id = 1


    def serial_dictatorship_experiment(self, val_index, val_type, G, size=None, agent_cap=None):
        """Finds the distortion of running serial dictatorship on the given input.
        
        Args:
            val_index (int): The id of the valuation in self.instance_generator.
            val_type (string): The method by which the valuation was generated.
            G (nx.Graph): The actual matching input. Must be a weighted bipartite graph with numerical node labels.
            size (int): The number of agents in the input. Defaults to half the size of G.nodes if not given.
            agent_cap (int): An integer value i such that for all nodes with label <= i, those nodes are agents. Defaults to len(G.nodes//2) if not given.
            
        """
        if size is None:
            size = len(G.nodes)//2

        M = serial_dictatorship(G,agent_cap)

        self.history['id'].append(self.id)
        self.id += 1
        self.history['val_index'].append(val_index)
        self.history['size'].append(size)
        self.history['valuation'].append(val_type)
        self.history['algo'].append('serial_dictatorship')
        self.history['distortion'].append(calculate_distortion(G,M))


    def partial_max_matching_experiment(self, val_index, val_type, G, m, size=None, agent_cap=None):
        """Finds the distortion of running PartialMaxMatching on the given input.
        
        Args:
            val_index (int): The id of the valuation in self.instance_generator.
            val_type (string): The method by which the valuation was generated.
            G (nx.Graph): The actual matching input. Must be a weighted bipartite graph with numerical node labels.
            m (int): The number of buckets, used as input to PartialMaxMatching.
            size (int): The number of agents in the input. Defaults to half the size of G.nodes if not given.
            agent_cap (int): An integer value i such that for all nodes with label <= i, those nodes are agents. Defaults to len(G.nodes//2) if not given.
            
        """
        if size is None:
            size = len(G.nodes)//2

        M = partial_max_matching(G,m,agent_cap)

        self.history['id'].append(self.id)
        self.id += 1
        self.history['val_index'].append(val_index)
        self.history['size'].append(size)
        self.history['valuation'].append(val_type)
        self.history['algo'].append('partial_max_matching'+ '_' + str(m))
        self.history['distortion'].append(calculate_distortion(G,M))        


    def modified_max_matching_experiment(self, val_index, val_type, G, size=None, agent_cap=None):
        """Finds the distortion of running ModifiedMaxMatching on the given input.
        
        Args:
            val_index (int): The id of the valuation in self.instance_generator.
            val_type (string): The method by which the valuation was generated.
            G (nx.Graph): The actual matching input. Must be a weighted bipartite graph with numerical node labels.
            size (int): The number of agents in the input. Defaults to half the size of G.nodes if not given.
            agent_cap (int): An integer value i such that for all nodes with label <= i, those nodes are agents. Defaults to len(G.nodes//2) if not given.
            
        """
        if size is None:
            size = len(G.nodes)//2

        M = modified_max_matching(G,agent_cap)

        self.history['id'].append(self.id)
        self.id += 1
        self.history['val_index'].append(val_index)
        self.history['size'].append(size)
        self.history['valuation'].append(val_type)
        self.history['algo'].append('modified_max_matching')
        self.history['distortion'].append(calculate_distortion(G,M))


    def hybrid_max_matching_experiment(self, val_index, val_type, G, size=None, agent_cap=None):
        """Finds the distortion of running HybridMaxMatching on the given input.
        
        Args:
            val_index (int): The id of the valuation in self.instance_generator.
            val_type (string): The method by which the valuation was generated.
            G (nx.Graph): The actual matching input. Must be a weighted bipartite graph with numerical node labels.
            size (int): The number of agents in the input. Defaults to half the size of G.nodes if not given.
            agent_cap (int): An integer value i such that for all nodes with label <= i, those nodes are agents. Defaults to len(G.nodes//2) if not given.
            
        """
        if size is None:
            size = len(G.nodes)//2

        M = hybrid_max_matching(G,agent_cap)

        self.history['id'].append(self.id)
        self.id += 1
        self.history['val_index'].append(val_index)
        self.history['size'].append(size)
        self.history['valuation'].append(val_type)
        self.history['algo'].append('hybrid_max_matching')
        self.history['distortion'].append(calculate_distortion(G,M))


if __name__=='__main__':
    instantiator = instance_generator.InstanceGenerator(True)
    sim = Simulator(instantiator)
    for k in range(1,6):
        print('current batch is', k)
        n = 5*k

        for j in range(20):

            G = instantiator.generate_unit_range_unif(n)
            val_index = instantiator.index-1
            val_type = 'unit_range_unif'

            sim.serial_dictatorship_experiment(val_index,val_type,G)
            sim.partial_max_matching_experiment(val_index,val_type,G,np.floor(np.log(n)))
            sim.modified_max_matching_experiment(val_index,val_type,G)
            sim.hybrid_max_matching_experiment(val_index,val_type,G)

    
    df = pd.DataFrame(sim.history)
    df.to_csv("C:/Users/sqshy/Desktop/University/Fifth Year/research/DistortionSim/data/unit_range_unif_test.csv")

    df = pd.Series(sim.instance_generator.history)
    df.to_csv("C:/Users/sqshy/Desktop/University/Fifth Year/research/DistortionSim/data/unit_range_test_instances.csv")