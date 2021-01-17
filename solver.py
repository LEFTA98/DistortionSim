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


def priority_augment(G, prio='pareto', agent_cap=None):
    """Augments a weighted bipartite graph G to generate a new weighted biparitite graph H, such that G and H have identical vertices,
    but solving the maximum weight matching problem on H will yield a matching in G that is maximum weight subject to either pareto optimality, rank maximality,
    max-cardinality rank maximality, or fairness. The augmentation is based on the paper associated with this experiment.

    Args:
        G (nx.Graph): Weighted bipartite Graph with nodes named 1 through n and positive weights on each edge. Agents are assumed to be nodes 1 through
        i for some i <= n.
        prio (String): some string in the list ['rank_maximal', 'max_cardinality_rank_maximal', 'fair'] that determines what property the maximum weight matching of
        the new Graph should have. If none of these options are given, rank-maximal will be assumed.
        agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
        to len(G.nodes)//2 if not given.

    Returns:
        H (nx.Graph): Weighted bipartite Graph with nodes named 1 through n and positive weights on each edge, augmented with the rank of each edge.
    """
    if agent_cap is None:
        agent_cap = len(G.nodes)//2

    H = G.copy()
    n = agent_cap
    H = rankify_graph(H, agent_cap = agent_cap)

    for (u,v) in H.edges:
        r = H[u][v]['rank']
        if prio=='fair':
            H[u][v]['weight'] += 4*np.power(n,2*n) - 2*np.power(n,r-1)
        elif prio=='max_cardinality_rank_maximal':
            H[u][v]['weight'] += np.power(n,2*n) + np.power(n,2*(n-r))
        elif prio=='rank_maximal':
            H[u][v]['weight'] += np.power(n,2*(n-r+1))

    return H


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
            T = list(set(component.nodes).difference(set(S)))
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

    S = set()

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


#TODO modify this to include priority-p
def modified_max_matching(G,prio='pareto',agent_cap=None):
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
    n = agent_cap

    for (u,v) in G.edges:
        if G[u][v]['weight'] == 1:
            H.add_weighted_edges_from([(u,v,1)])
        elif G[u][v]['weight'] >= 1/np.sqrt(agent_cap):
            H.add_weighted_edges_from([(u,v,1/np.sqrt(agent_cap))])
        else:
            H.add_weighted_edges_from([(u,v,0)])

    I = G.copy()
    I = rankify_graph(I)

    for (u,v) in H.edges:
        r = I[u][v]['rank']
        if prio=='fair':
            H[u][v]['weight'] += 4*np.power(n,2*n) - 2*np.power(n,r-1)
        elif prio=='max_cardinality_rank_maximal':
            H[u][v]['weight'] += np.power(n,2*n) + np.power(n,2*(n-r))
        elif prio=='rank_maximal':
            H[u][v]['weight'] += np.power(n,2*(n-r+1))

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

    match = set().union(first_matching, second_matching)
    match = set([(a,b) if a < b else (b,a) for (a,b) in match])
    if prio=='pareto':
        return top_trading_cycles(G,agent_cap,match)
    else:
        return match


#TODO remove this old version of ttc once new one is verified to work
# def top_trading_cycles(G, agent_cap=None):
#     """Runs the top-trading cycles algorithm on a weighted bipartite graph G, whose nodes are enumerated from 1 to n, assuming the initial 
#     allocation is agent i getting good i. Notice for this algorithm to work, agent_cap MUST be set at len(G.nodes)//2.
    
#     Args:
#         G (nx.Graph): Weighted bipartite Graph, with all edge weights assumed to be in [0,1].
#         agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
#         to len(G.nodes)//2 if not given.

#     Returns:
#         A set of 2-tuples representing a matching on G found by HybridMaxMatching.    
#     """
#     if agent_cap is None:
#         agent_cap = len(G.nodes)//2

#     H = G.copy()
#     M = set()
#     count = len(H.nodes)

#     while count > 0:
#         rankify_graph(H, agent_cap)

#         I = nx.DiGraph()
#         I.add_nodes_from(H.nodes)
#         for (u,v) in H.edges:
#             if H[u][v]['rank'] == 1:
#                 I.add_edge(u,int(v - len(G.nodes)//2)) #draw an edge between the agent u and the agent corresponding to good v

#         C = nx.algorithms.cycles.find_cycle(I)
#         for (u,v) in C:
#             M.add((u, int(v + len(G.nodes)//2)))
#             H.remove_nodes_from([u, int(v+len(G.nodes)//2)])
#             count -= 2

#     return M


def top_trading_cycles(G, agent_cap=None, initial_matching=None):
    """Runs the top-trading cycles algorithm on a weighted bipartite graph G, assuming the initial allocation is agent i getting good i. Notice for this
    algorithm to work, agent_cap MUST be set at len(G.nodes)//2.
    
    Args:
        G (nx.Graph): Weighted bipartite Graph, with all edge weights assumed to be in [0,1].
        agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
        to len(G.nodes)//2 if not given.

    Returns:
        A set of 2-tuples representing a matching on G found by HybridMaxMatching.    
    """
    if agent_cap is None:
        agent_cap = len(G.nodes)//2
        
    if initial_matching is None:
        initial_matching = set([(i+1, i+agent_cap+1) for i in range(agent_cap)])
        
    agents_to_goods_d = dict(initial_matching)
    goods_to_agents_d = dict([(b,a) for (a,b) in initial_matching])

    H = G.copy()
    M = set()

    while len(H.nodes) != 0:
        rankify_graph(H, agent_cap)
        

        I = nx.DiGraph()
        I.add_nodes_from(H.nodes)
        for (u,v) in H.edges:
            if H[u][v]['rank'] == 1:
                I.add_edge(u,goods_to_agents_d[v]) #draw an edge between the agent u and the agent corresponding to good v

        C = nx.algorithms.cycles.find_cycle(I)
        for (u,v) in C:
            M.add((u, agents_to_goods_d[v]))
            H.remove_nodes_from([u, agents_to_goods_d[v]])

    return M


def compute_epsilon_bucket(x,n,epsilon):
    """Helper function which computes which bucket of (2/2+epsilon)^i x should be rounded to, given n agents. x must be in [0,1]; if x
    is too small to be bucketed, -1 will be returned instead.
    
    Args:
        x (float): Edge value in [0,1].
        n (int): Number of agents.
        epsilon (float): epsilon where (1+epsilon) is the approximation ratio.
        
    Returns:
        float approximation of x according to the buckets."""

    cap = np.ceil(np.log(n**2/epsilon)/np.log(1+epsilon/2))
    threshold = np.power(2/(2+epsilon), cap)

    if x < threshold:
        return -1
    else:
        return np.ceil(np.log(x)/np.log(2/(2+epsilon)))


def epsilon_max_matching(G, epsilon, prio='pareto', agent_cap=None):
    """Runs Algorithm 2 from the write-up on weighted bipartite Graph G.
    
    Args:
        G (nx.Graph): Weighted bipartite Graph, with all edge weights assumed to be in [0,1].
        epsilon (float): the approximation ratio we wish to achieve; this algorithm guarantees a 1+epsilon approximation.
        prio (String): String in ['pareto', 'rank_maximal', 'max_cardinality_rank_maximal', 'fair'] that represents the priority vector used for this problem. Defaults to 'rank_maximal'.
        agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is 
        equal to i. Defaults to len(G.nodes)//2 if not given.

    Returns:
        A set of 2-tuples representing a matching on G found by HybridMaxMatching.  
    """
    if agent_cap is None: # note in instances where nodes MUST be labelled 1...n, agent_cap=|agents|
        agent_cap = len(G.nodes)//2

    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    n = agent_cap

    for (u,v) in G.edges:
        new_weight = compute_epsilon_bucket(G[u][v]['weight'], agent_cap, epsilon)

        if new_weight != -1:        
            H.add_weighted_edges_from([(u,v, np.power((2/(2+epsilon)),new_weight))])
        else:
            H.add_weighted_edges_from([(u,v,np.finfo(np.float).eps)]) #hack to still include this edge in matching

    I = G.copy()
    I = rankify_graph(I)

    for (u,v) in H.edges:
        r = I[u][v]['rank']
        if prio=='fair':
            H[u][v]['weight'] += 4*np.power(n,2*n) - 2*np.power(n,r-1)
        elif prio=='max_cardinality_rank_maximal':
            H[u][v]['weight'] += np.power(n,2*n) + np.power(n,2*(n-r))
        elif prio=='rank_maximal':
            H[u][v]['weight'] += np.power(n,2*(n-r+1))

    match = nx.algorithms.matching.max_weight_matching(H) # this technically should be TTC'ed afterwards
    match = set([(a,b) if a < b else (b,a) for (a,b) in match])
    if prio=='pareto':
        return top_trading_cycles(G,agent_cap,match)
    else:
        return match


def twothirds_max_matching(G,prio='rank_maximal',agent_cap=None):
    """Given a weighted bipartite Graph G and a priority prio, returns a priority-prio matching that is an O(n^2/3) approximation to the welfare-optimal
    priority-prio matching. Equivalent to Algorithm 3 in the final write-up.
    
    Args:
        G (nx.Graph): Weighted bipartite Graph assumed to satisfy either unit-sum or unit-range normalization.
        prio (String): String in ['rank_maximal', 'max_cardinality_rank_maximal', 'fair'] that represents the priority vector used for this problem. Defaults to 'rank_maximal'.
        agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is 
        equal to i. Defaults to len(G.nodes)//2 if not given.

    Returns:
        A set of 2-tuples representing a matching on G.
    """
    if agent_cap is None: # note in instances where nodes MUST be labelled 1...n, agent_cap=|agents|
        agent_cap = len(G.nodes)//2

    H = G.copy()
    H = rankify_graph(H)
    n = agent_cap

    for (u,v) in H.edges:
        r = H[u][v]['rank']

        if r == 1:
            H[u][v]['weight'] = np.reciprocal(np.power(n,1/3)) if H[u][v]['weight'] >= np.reciprocal(np.power(n,1/3)) else 0
        else:
            H[u][v]['weight'] = np.reciprocal(np.min([r, np.power(n, 1/3)])*np.power(n, 2/3)) if H[u][v]['weight'] >= np.reciprocal(np.min([r, np.power(n, 1/3)])*np.power(n, 2/3)) else 0
        
        if prio=='fair':
            H[u][v]['weight'] += 4*np.power(n,2*n) - 2*np.power(n,r-1)
        elif prio=='max_cardinality_rank_maximal':
            H[u][v]['weight'] += np.power(n,2*n) + np.power(n,2*(n-r))
        else:
            H[u][v]['weight'] += np.power(n,2*(n-r+1))

    return nx.algorithms.matching.max_weight_matching(H)
        

#TODO this is super messy, try to clean this up if you have time
def updated_hybrid_max_matching(G, agent_cap=None):
    """Runs the Algorithm 4 (an updated version of hybridMaxMatching) from the write-up on a weighted bipartite graph G, whose nodes are enumerated from 1 to n. Notice the top-trading cycle 
    step is not implemented.
    
    Args:
        G (nx.Graph): Weighted bipartite Graph, with all edge weights assumed to be in [0,1].
        agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is equal to i. Defaults
        to len(G.nodes)//2 if not given.

    Returns:
        A set of 2-tuples representing a matching on G found by HybridMaxMatching.
    """
    if agent_cap is None: # note in instances where nodes MUST be labelled 1...n, agent_cap=|agents|
        agent_cap = len(G.nodes)//2    

    H = G.copy() # make a copy of graph with ranks; this'll be useful for later
    H = rankify_graph(H, agent_cap)
    n = agent_cap

    for (u,v) in H.edges:
        r = H[u][v]['rank']

        if r == 1:
            H[u][v]['weight'] = np.reciprocal(np.power(n,1/3)) if H[u][v]['weight'] >= np.reciprocal(np.power(n,1/3)) else 0
        else:
            H[u][v]['weight'] = np.reciprocal(np.min([r, np.power(n, 1/3)])*np.power(n, 2/3)) if H[u][v]['weight'] >= np.reciprocal(np.min([r, np.power(n, 1/3)])*np.power(n, 2/3)) else 0

    M_mm = nx.algorithms.matching.max_weight_matching(H)

    to_remove = []
    for (u,v) in M_mm:
        if H[u][v]['weight'] == 0:
            to_remove.append((u,v))

    M_mm = M_mm.difference(set(to_remove))

    I = G.copy()
    I = rankify_graph(I, agent_cap)

    if len(M_mm) == 0:
        for (u,v) in I.edges:
            if I[u][v]['rank'] > np.floor(1/2 * np.power(n, 1/3)):
                I.remove_edge(u,v)

        M_aux = nx.algorithms.matching.max_weight_matching(I,maxcardinality=True) #compute a maximum cardinality matching

    else:
        for (u,v) in I.edges:
            if I[u][v]['rank'] > 1:
                I.remove_edge(u,v)


        M_aux = nx.algorithms.matching.max_weight_matching(I,maxcardinality=True)

    to_remove = []
    for (u,v) in M_aux:
        if u in set.union(*map(set,M_mm)) or v in set.union(*map(set,M_mm)):
            to_remove.append((u,v))
    
    M_aux = M_aux.difference(set(to_remove))

    J = G.copy()
    J.remove_nodes_from(list(set.union(*map(set, set().union(M_mm,M_aux)))))
    agent_nodes = [node for node in J.nodes if node <= agent_cap]
    good_nodes = [node for node in J.nodes if node > agent_cap]
    M_rest = set()
    i = 0         
    while i < min(len(good_nodes), len(agent_nodes)):
        M_rest.add((agent_nodes[i], good_nodes[i]))
        i += 1

    match = set().union(M_rest, set().union(M_mm,M_aux))
    match = set([(a,b) if a < b else (b,a) for (a,b) in match])
    return top_trading_cycles(G,agent_cap,match)


def reassign_labels(G,M,agent_cap=None):
    """Given a weighted bipartite Graph G and a matching M on G, where G is enumerated from 1 to n, returns a relabelled version of G such 
    that each agent i gets good i + n//2. This primes the graph for the top trading cycle algorithm.
    
    Args:
        G (nx.Graph): Weighted bipartite Graph, with all edge weights assumed to be in [0,1].
        M (set of (int,int)): the matching that we are trying to relabel the Graph to.
        agent_cap (int): The numerical label of the last agent; that is, if agents are enumerated by nodes 1 through i, then agent_cap is 
        equal to i. Defaults to len(G.nodes)//2 if not given.

    Returns:
        weighted bipartite Graph
    """
    if agent_cap is None: # note in instances where nodes MUST be labelled 1...n, agent_cap=|agents|
        agent_cap = len(G.nodes)//2

    li = list(M)
    li = [(j, i + len(G.nodes)//2) if i < j else (i, j+len(G.nodes)//2) for (i,j) in li]
    d = dict(li)

    return nx.relabel.relabel_nodes(G,d,True)


def calculate_distortion(G,M):
    """Calculates the approximation ratio of the social welfare of the optimal matching on weighted complete bipartite graph G versus the social welfare
    accrued by the given matching M. Notice that this is not actually the definition of distortion, as we are not taking a supremum over all possible
    valuations.
    
    Args:
        G (nx.Graph): Weighted bipartite complete Graph with weights in [0,1].
        M (set((nx.Node,nx.Node))): A set of 2-tuples, with each 2-tuple representing a pairing of an agent in G to a good in G. 

    Returns:
        A float value >= 1 representing the approximation ratio of the optimal social welfare to the social welfare generated by M. 
    """
    algo_weight = sum([G[u][v]['weight'] for (u,v) in M])
    opt_weight = sum([G[u][v]['weight'] for (u,v) in nx.algorithms.matching.max_weight_matching(G)])

    return opt_weight/algo_weight


def calculate_modified_distortion(G,M,prio='rank_maximal'):
    """Calculates the approximation of social welfare among allocations that satisfy the criterion prio by finding the social welfare of the
    optimal matching on weighted complete bipartite graph G subject to the criterion, then dividing it by the social welfare achieved by given
    matching M.
    
    Args:
        G (nx.Graph): Weighted bipartite complete Graph with weights in [0,1].
        M (set((nx.Node,nx.Node))): A set of 2-tuples, with each 2-tuple representing a pairing of an agent in G to a good in G.
        prio (String): A string from the list ['rank_maximal', 'max_cardinality_rank_maximal', 'fair'] which specifies which criterion we want to
        enforce on the matching.

    Returns:
        A float value representing the approximation ratio of the optimal social welfare to the social welfare generated by M. 
    """
    H = priority_augment(G, prio)
    max_weight_match = nx.algorithms.matching.max_weight_matching(H)

    algo_weight = sum([G[u][v]['weight'] for (u,v) in M])
    opt_weight = sum([G[u][v]['weight'] for (u,v) in max_weight_match])

    return opt_weight/algo_weight