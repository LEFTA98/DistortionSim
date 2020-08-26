# -*- coding: utf-8 -*-
"""Overall class for running the simulations.

Contains the Simulator class, which holds all of the matching algorithms and calculates the distortion. Also logs the outcomes of the experiments for analysis and
visualization.

"""

import networkx as nx
import numpy as np
import pandas as pd


class Simulator:
    """Object that solves instances of matching problems given to it and aggregates the results in a pretty manner.

    Attributes:
        instance_generator (InstanceGenerator): InstanceGenerator for creating problem instances for simulation
        history (dict): history of all results, stored in a dictionary for later conversion to pd.DataFrame
    """

    def __init__(self, instance_generator):
        """Initializes a new Simulator.

        Args:
            instance_generator (InstanceGenerator): InstanceGenerator for creating problem instances for simulation
        """
        self.instance_generator = instance_generator
        self.history = {'id': [], 'size': [], 'valuation':[], 'algo': [], 'distortion': []}


    def rankify_graph(self, G, agent_cap=None):
        """Augments a weighted bipartite graph with nodes 1 through n with ranks. For example, if agent 1 values good n-1 as their second-highest
        valued good, assign a rank of 2 to the edge (1, n-1).

        Args:
            G (nx.Graph): Weighted bipartite Graph with nodes named 1 through n and positive weights on each edge. Agents are assumed to be nodes 1 through
            i for some i <= n.
            agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
            to len(G.nodes)//2 if not given.

        Returns:
            G (nx.Graph): Weighted bipartite Graph with nodes named 1 through n and positive weights on each edge, augmented with the rank of each edge.
        """
        if agent_cap is None:
            agent_cap = len(G.nodes)//2

        for i in range(1, agent_cap+1):
            adj_list = [(key, G.adj[key][i]['weight']) for key in G.adj[i].keys() if G.adj[i][key] != {}]
            adj_list.sort(key = lambda tup: tup[1], reverse = True) #sort by second value in the tuple

            for j in range(len(adj_list)):
                G[i][adj_list[j][0]]['rank'] = j+1

        return G


    def bfs_even_odd(self,H,G,v):
        """Helper function that runs breadth-first search on a graph H, with source node v, augmenting every node with whether it is an even or odd distance away
        from the source node in graph G, of which H is a subset.

        Args:
            H (nx.Graph): undirected subgraph of G.
            G (nx.Graph): undirected graph.
            v (Object): the name of a source node that is a part of H.

        Returns:
            NoneType
        """
        nx.set_node_attributes(G, {v: 'E'}, 'decomp')
        queue = list(H.adj[v].keys())
        discovered = set().union(set('v'), H.adj[v].keys())
        for item in queue:
            nx.set_node_attributes(G, {item: 'O'}, 'decomp')

        while len(queue) != 0:
            curr = queue.pop(0)
            curr_parity = "E" if G.nodes[curr]['decomp'] == "O" else "O"
            for key in H.adj[curr].keys():
                if key not in discovered:
                    discovered.add(key)
                    queue.append(key)
                    nx.set_node_attributes(G, {key: curr_parity}, 'decomp')
                        

    def even_odd_unreachable_decomposition(self, G):
        """Decomposes a bipartite graph into even, odd, and unreachable vertices, based on the following decomposition: 
        https://en.wikipedia.org/wiki/Dulmage%E2%80%93Mendelsohn_decomposition

        Args:
            G (nx.Graph): Bipartite Graph with nodes named 1 through n. Agents are assumed to be nodes 1 through i for some i <= n.

        Returns:
            NoneType
        """
        S = [item[0] for item in nx.maximal_matching(G)] + [item[1] for item in nx.maximal_matching(G)]
        connected_components = nx.algorithms.components.connected_components(G)
        connected_subgraphs = [G.subgraph(c).copy() for c in connected_components]

        for component in connected_subgraphs:
            if set().union(set(S), set(component.nodes)) == set(S):
                for node in component.nodes:
                    nx.set_node_attributes(G, {node: 'U'}, 'decomp')

            #only reach here if there's at least one node in component not in max_matching
            else:
                T = list(S.difference(component.nodes))
                v = T[0]
                self.bfs_even_odd(component,G,v)

