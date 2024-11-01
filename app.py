import numpy as np
from typedefs import *
import random
import networkx as nx
import random
import numpy as np

# Define the topology as a directed graph
topology = nx.DiGraph()

# Add nodes
nodes = ['A', 'B', 'C', 'D']
topology.add_nodes_from(nodes)

# Add directed edges with updated link IDs (0 to 7)
edges = {
    ('A', 'B'): 0, ('B', 'A'): 1,
    ('B', 'C'): 2, ('C', 'B'): 3,
    ('C', 'D'): 4, ('D', 'C'): 5,
    ('D', 'A'): 6, ('A', 'D'): 7
}

# Add edges to the graph
topology.add_edges_from(edges.keys())

class App:
    def __init__(self, trial, id, links, T, max_n_states, max_n_flows, max_n_flow_hop):
        random.seed(123+trial+id)
        np.random.seed(123+trial+id)
        self.id = id
        self.links = links
        self.T = T
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.n_states = random.randint(2, self.max_n_states)
        self.states = self.generate_states()
        self.transitions = self.generate_transitions()
        self.k_max = self.determine_k_max()

    def generate_states(self):
        states = []
        for s in range(self.n_states):
            state = State(s)

            for f in range(random.randint(1, self.max_n_flows)):
                # Choose random source and destination nodes within the topology
                source = random.choice(list(topology.nodes))
                destination = random.choice([node for node in topology.nodes if node != source])
            
                # Find all paths within the max_n_flow_hop constraint
                all_paths = list(nx.all_simple_paths(topology, source=source, target=destination, cutoff=self.max_n_flow_hop))
            
                if all_paths:
                    # Choose a random path from the list of paths
                    path = random.choice(all_paths)

                    # Convert the path to links (edges) using the defined edges dictionary
                    links_in_path = [edges[(path[i], path[i+1])] for i in range(len(path) - 1)]
                
                    # Only create the flow if there's at least one link
                    if links_in_path:
                        period = int(random.choice([p for p in range(self.max_n_flow_hop*2, self.T + 1) if self.T % p == 0]))
                        flow = Flow(f, links_in_path, period)
                        state.flows.append(flow)

            states.append(state)
        return states


    def generate_transitions(self):
        n_states = len(self.states)
        transitions = np.zeros((n_states, n_states))
        for i in range(n_states):
            cut_points = np.sort(np.random.rand(n_states - 1))
            probs = np.diff(np.hstack(([0], cut_points, [1])))
            probs[-1] = 1 - probs[:-1].sum()
            transitions[i] = probs

        for row in transitions:
            assert sum(row) == 1
            for i in row:
                assert i > 0
        return transitions

    def determine_k_max(self):
        self.steady = self.steady_state_probabilities()
        self.M_k = [self.transitions[0]]
        for k in range(1, 100):
            p = np.linalg.matrix_power(self.transitions, k)[0]
            self.M_k.append(p)
            if np.abs(p[0] - self.steady[0]) <= 1e-4:
                return k

    def steady_state_probabilities(self):
        A = np.vstack((self.transitions.T - np.eye(self.n_states), np.ones(self.n_states)))
        b = np.vstack((np.zeros((self.n_states, 1)), np.ones((1, 1))))
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = pi / np.sum(pi)
        return pi.flatten()
