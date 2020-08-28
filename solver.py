"""Module containing methods to compute matchings and calculate distortion."""

import networkx as nx
import numpy as np
import pandas as pd


def rankify_graph(G, agent_cap=None):
    """Augments a weighted bipartite graph containing nodes in {1...n} with ranks. For example, if agent 1 values good n-1 as their second-highest
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
        if i in G.nodes:
            adj_list = [(key, G.adj[key][i]['weight']) for key in G.adj[i].keys() if G.adj[i][key] != {}]
            adj_list.sort(key = lambda tup: tup[1], reverse = True) #sort by second value in the tuple

            for j in range(len(adj_list)):
                G[i][adj_list[j][0]]['rank'] = j+1

    return G


def bfs_even_odd(H,G,v):
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
                    

def even_odd_unreachable_decomposition(G):
    """Decomposes a bipartite graph into even, odd, and unreachable vertices, based on the following decomposition: 
    https://en.wikipedia.org/wiki/Dulmage%E2%80%93Mendelsohn_decomposition

    Args:
        G (nx.Graph): Bipartite Graph with nodes in {1...n}. Agents are assumed to be nodes in {1..n} that are less than or equal to i for some i <= n.

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
            bfs_even_odd(component,G,v)


def rank_maximal_allocation(G,agent_cap=None):
    """Returns a rank-maximal matching of G, which is assumed to be a weighted bipartite graph, using Irving's algorithm.

    Args:
        G (nx.Graph): Weighted bipartite Graph with nodes named after natural numbers and positive weights on each edge. Agents are assumed to be nodes 1 through
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
    rankify_graph(H,agent_cap)

    edge_list = [set() for i in range(n)]

    for (agent, good) in H.edges:
        edge_list[H[agent][good]['rank']].add((agent, good))

    I = nx.Graph()
    I.add_nodes_from(H.nodes)

    for i in range(n):
        I.add_edges_from(edge_list[i])
        S = nx.maximal_matching(I)
        even_odd_unreachable_decomposition(I)

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


def serial_dictatorship(G, agent_cap=None, need_ranks=True):
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
        rankify_graph(H, agent_cap)

    M = set()

    for i in range(1, agent_cap+1):
        if i in H.nodes and len(H.adj[i]) != 0 :
            most_preferred = min(H.adj[i].keys(), key = lambda k: H.adj[i][k]['rank'])
            M.add((i, most_preferred))
            H.remove_node(most_preferred)

    return M


def compute_bucket(x, n, m):
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


def partial_max_matching(G, m, agent_cap=None):
    """Runs the PartialMaxMatching algorithm on weighted bipartite graph G with a total of m buckets. Assumes the nodes of G are enumerated from 1 to n.

    Args:
        G (nx.Graph): Weighted bipartite Graph, with all edge weights assumed to be in [0,1].
        m (int): Number of buckets.
        agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
        to len(G.nodes)//2 if not given.

    Returns:
        A set of 2-tuples representing a matching on G found by PartialMaxMatching.
    """
    if agent_cap is None: # note in instances where nodes MUST be labelled 1...n, agent_cap=|agents|
        agent_cap = len(G.nodes)//2

    H = nx.Graph()
    H.add_nodes_from(G.nodes)

    for (u,v) in G.edges:
        new_weight = compute_bucket(G[u][v]['weight'], agent_cap, m)

        if new_weight != 0:
            H.add_weighted_edges_from([(u,v, new_weight)])

    first_matching = nx.algorithms.matching.max_weight_matching(H)

    if sum([[u,v] for (u,v) in first_matching], []) == len(G.nodes):
        return first_matching
    else:
        I = G.copy()
        rankify_graph(I,agent_cap)
        I.remove_nodes_from(sum([[u,v] for (u,v) in first_matching], []))

        second_matching = serial_dictatorship(I,agent_cap,False)

        return set().union(first_matching, second_matching)


def modified_max_matching(G,agent_cap=None):
    """Runs the ModifiedMaxMatching algorithm on a weighted bipartite Graph G that has been normalized to unit-range; that is, each agent's most valued good
    must have a value of 1. Assumes the nodes of G are enumerated from 1 to n.

    Args:
        G (nx.Graph): Weighted bipartite Graph, with all edge weights assumed to be in [0,1], and each agent's most preferred good has a value of 1.
        agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
        to len(G.nodes)//2 if not given.

    Returns:
        A set of 2-tuples representing a matching on G found by ModifiedMaxMatching.
    """
    if agent_cap is None: # note in instances where nodes MUST be labelled 1...n, agent_cap=|agents|
        agent_cap = len(G.nodes)//2

    H = nx.Graph()
    H.add_nodes_from(G.nodes)

    for (u,v) in G.edges:
        if G[u][v]['weight'] == 1:
            H.add_weighted_edges_from([(u,v,1)])
        elif G[u][v]['weight'] >= 1/np.sqrt(agent_cap):
            H.add_weighted_edges_from([(u,v,1/np.sqrt(agent_cap))])
        else:
            H.add_weighted_edges_from([(u,v,0)])

    first_matching = nx.algorithms.matching.max_weight_matching(H)

    I = G.copy()
    I.remove_nodes_from(sum([[u,v] for (u,v) in first_matching], []))
    agent_nodes = [node for node in I.nodes if node <= agent_cap]
    good_nodes = [node for node in I.nodes if node > agent_cap]
    second_matching = set()
    i=0
    while i < min(len(good_nodes), len(agent_nodes)):
        second_matching.add((agent_nodes[i], good_nodes[i]))
        i += 1

    return set().union(first_matching, second_matching)


def hybrid_max_matching(G, agent_cap=None):
    """Runs the HybridMaxMatching algorithm on a weighted bipartite graph G, whose nodes are enumerated from 1 to n. Notice the top-trading cycle step is not implemented.
    
    Args:
        G (nx.Graph): Weighted bipartite Graph, with all edge weights assumed to be in [0,1].
        agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
        to len(G.nodes)//2 if not given.

    Returns:
        A set of 2-tuples representing a matching on G found by HybridMaxMatching.
    """
    if agent_cap is None:
        agent_cap = len(G.nodes)//2

    H = G.copy() # make a copy of graph with ranks; this'll be useful for later
    rankify_graph(H, agent_cap)

    I = nx.Graph() #add weights from query to graph, compute max-matching
    I.add_nodes_from(G.nodes)
    for (u,v) in G.edges:
        if H[u][v]['rank'] == 1 and H[u][v]['weight'] >= 1/np.sqrt(agent_cap): # note in instances where nodes MUST be labelled 1...n, agent_cap=|agents|
            I.add_weighted_edges_from([(u,v,1/np.sqrt(agent_cap))])
        elif H[u][v]['rank'] > 1 and H[u][v]['weight'] >= np.reciprocal((agent_cap**0.75)*min(H[u][v]['rank'], agent_cap**0.25)):
            I.add_weighted_edges_from([(u,v,np.reciprocal((agent_cap**0.75)*min(H[u][v]['rank'], agent_cap**0.25)))])

    first_matching = nx.algorithms.matching.max_weight_matching(I)

    H.remove_nodes_from(sum([[u,v] for (u,v) in first_matching], []))
    rankify_graph(H,agent_cap)

    second_matching = rank_maximal_allocation(H,agent_cap) #compute rank-maximal alloc on remainder

    H.remove_nodes_from(sum([[u,v] for (u,v) in second_matching], []))

    if len(H.nodes) == 0: #arbitrarily match remainder
        return set().union(first_matching, second_matching)
    else:
        agent_nodes = [node for node in H.nodes if node <= agent_cap]
        good_nodes = [node for node in H.nodes if node > agent_cap]
        third_matching = set()
        i = 0         
        while i < min(len(good_nodes), len(agent_nodes)):
            third_matching.add((agent_nodes[i], good_nodes[i]))
            i += 1

        return set().union(third_matching, set().union(first_matching, second_matching))   


def calculate_distortion(G,M):
    """calculates the approximation ratio of the social welfare of the optimal matching on weighted complete bipartite graph G versus the social welfare
    accrued by the given matching M. Notice that this is not actually the definition of distortion, as we are not taking a supremum over all possible
    valuations.
    
    Args:
        G (nx.Graph): Weighted bipartite complete Graph with weights in [0,1].
        M (set((nx.Node,nx.Node))): A set of 2-tuples, with each 2-tuple representing a pairing of an agent in G to a good in G. 

    Returns:
        A float value >= 1 representing the approximation ratio of the optimal social welfare to the social welfare generated by M. 
    """
    algo_weight = sum([G[u][v]['weight'] for (u,v) in M], [])
    opt_weight = sum([G[u][v]['weight'] for (u,v) in nx.algorithms.matching.max_weight_matching(G)], [])

    return opt_weight/algo_weight