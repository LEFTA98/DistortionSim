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
        self.history = {'id': [], 'val_index': [], 'size': [], 'valuation':[], 'algo': [], 'distortion': []}


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
                T = list(set(S).difference(set(component.nodes)))
                v = T[0]
                self.bfs_even_odd(component,G,v)


    def rank_maximal_allocation(self, G,agent_cap=None):
        """Returns a rank-maximal matching of G, which is assumed to be a weighted bipartite graph, using Irving's algorithm.

        Args:
            G (nx.Graph): Weighted bipartite Graph with nodes named 1 through n and positive weights on each edge. Agents are assumed to be nodes 1 through
            i for some i <= n.
            agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
            to len(G.nodes)//2 if not given.

        Returns:
            A set of 2-tuples representing a matching on G that has the rank-maximality property.
        """
        if agent_cap is None:
            agent_cap = len(G.nodes)//2

        H = G.copy()
        n = len(H.nodes)
        self.rankify_graph(H,agent_cap)

        edge_list = [set() for i in range(n)]

        for (agent, good) in H.edges:
            edge_list[H[agent][good]['rank']].add((agent, good))

        I = nx.Graph()
        I.add_nodes_from(H.nodes)

        for i in range(n):
            I.add_edges_from(edge_list[i])
            S = nx.maximal_matching(I)
            self.even_odd_unreachable_decomposition(I)

            for j in range(i+1,n):
                to_remove = set()
                for (u,v) in edge_list[j]:
                    if I.nodes[u]['decomp'] in ['O', 'U'] or I.nodes[v]['decomp'] in ['O','U']:
                        to_remove.add((u,v))
                        
                edge_list[j] = edge_list[j].difference(to_remove)
                        
                    

            for (x,y) in I.edges:
                if I.nodes[x]['decomp']+I.nodes[y]['decomp'] in ['OO', 'UO', 'OU']:
                    I.remove_edge((x,y))

        return S


    def serial_dictatorship(self, G, agent_cap=None, need_ranks=True):
        """ Returns a matching of G created by running serial dictatorship. G is assumed to be a weighted bipartite graph.

        Args:
            G (nx.Graph): Weighted bipartite Graph with nodes named after natural numbers and positive weights on each edge. Agents are assumed to be nodes of value up
            to i for some i.
            agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
            to len(G.nodes)//2 if not given.
            need_ranks (bool): whether or not the G already has ranks assigned to the edges. Defaults to True.

        Returns:
            A set of 2-tuples representing the matching found by serial dictatorship.
        """
        if agent_cap is None:
            agent_cap = len(G.nodes)//2

        H = G.copy()
        if need_ranks:
            self.rankify_graph(H, agent_cap)

        M = set()

        for i in range(1, agent_cap+1):
            if i in H.nodes and len(H.adj[i]) != 0 :
                most_preferred = min(H.adj[i].keys(), key = lambda k: H.adj[i][k]['rank'])
                M.add((i, most_preferred))
                H.remove_node(most_preferred)

        return M


    def compute_bucket(self, x, n, m):
        """Helper function for figuring out which bucket of form 1/n^(k/m) to put an edge of value x into, given n total agents and m buckets. Used in
        partial_max_matching. x must be in [0,1].
        
        Args:
            x (float): Edge value in [0,1].
            n (int): Number of agents.
            m (int): Number of buckets.

        Returns:
            float representing approximation of value x according to the buckets.
        """
        if x < 1/n:
            return 0
        
        k = np.ceil(m * np.log(1/x)/np.log(n))
        return 1/n**(k/m)


    def partial_max_matching(self, G, m, agent_cap=None):
        """Runs the PartialMaxMatching algorithm on weighted bipartite graph G with a total of m buckets. 

        Args:
            G (nx.Graph): Weighted bipartite Graph.
            m (int): Number of buckets.
            agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
            to len(G.nodes)//2 if not given.

        Returns:
             A set of 2-tuples representing a matching on G found by PartialMaxMatching.
        """
        if agent_cap is None:
            agent_cap = len(G.nodes)//2

        H = nx.Graph()
        H.add_nodes_from(G.nodes)

        for (u,v) in G.edges:
            new_weight = self.compute_bucket(G[u][v]['weight'], agent_cap, m)

            if new_weight != 0:
                H.add_weighted_edges_from([(u,v, new_weight)])

        first_matching = nx.algorithms.matching.max_weight_matching(H)

        if sum([[u,v] for (u,v) in first_matching], []) == len(G.nodes):
            return first_matching
        else:
            I = G.copy()
            self.rankify_graph(I,agent_cap)
            I.remove_nodes_from(sum([[u,v] for (u,v) in first_matching], []))

            second_matching = self.serial_dictatorship(I,agent_cap,False)

            return set().union(first_matching, second_matching)