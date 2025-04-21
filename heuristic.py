from app import App
import copy
import numpy as np
import time
from typedefs import *
import pandas as pd

import random
from app import topology, edges


class Heuristic:
    def __init__(self, trial, apps, links, T, current_states, verbose=False):
        self.trial = trial
        self.apps = apps
        self.T = T
        self.links = links
        self.current_states = current_states
        self.verbose = verbose

        self.partition = [[Cell() for e in range(len(links))]
                          for t in range(T)]

        self.schedule = [[Schedule(self.count_txs(i, s)) for s in range(self.apps[i].n_states)]
                         for i in range(len(self.apps))]
        
        self.time_dependency = []

    def run(self):
        for t in range(self.T):
            for i in range(len(self.apps)):
                if t ==0:
                    self.time_dependency.append([])
                for s in range(self.apps[i].n_states):
                    if t == 0:
                        self.time_dependency[i].append([])
                    schedule = self.schedule[i][s]
                    schedule.current_used_links = {}
                    flows = self.apps[i].states[s].flows
                    for f in flows:
                        if t == 0:
                            self.time_dependency[i][s].append([])
                        if t % f.period == 0:
                            schedule.cur_hops[f.id] = 0
                            pkt = Packet(f.id, f.txs[schedule.cur_hops[f.id]], (t//f.period+1)*f.period, f.period)
                            schedule.packets_to_schedule.append(pkt)

                    schedule.packets_to_schedule.sort(key=lambda x: (x.deadline, x.flow_id)) 
            
            for e in self.links:
                if self.partition[t][e].app == -1:
                    gains: list[Gain] = []
                    for i in range(len(self.apps)):
                        gains.append(self.potenial_gain(i, e, t))
                    gains.sort(key=lambda g: (g.value, g.app), reverse=True)

                    if gains[0].value > 0:
                        app = gains[0].app

                        self.partition[t][e].app = app

                        for s in range(self.apps[app].n_states):
                            schedule: Schedule = self.schedule[app][s]
                            schedule.n_packets_scheduled += len(gains[0].pkts_per_state[s])
                            if len(gains[0].pkts_per_state[s]) > 0:
                                self.partition[t][e].states.append(s)
                            schedule.packets_to_schedule = list(
                                set(schedule.packets_to_schedule)-set(gains[0].pkts_per_state[s]))
                            schedule.packets_to_schedule.sort(key=lambda x: (x.deadline, x.flow_id))
                            for pkt in gains[0].pkts_per_state[s]:
                                schedule.current_used_links[pkt.link] = True
                                schedule.cur_hops[pkt.flow_id] += 1
                                self.time_dependency[app][s][pkt.flow_id].append(t)
                                f = self.apps[app].states[s].flows[pkt.flow_id]
                                if schedule.cur_hops[pkt.flow_id] < len(f.txs):
                                    pkt = Packet(f.id, f.txs[schedule.cur_hops[f.id]],
                                                 (t//f.period+1)*f.period, f.period)
                                    schedule.packets_to_schedule.append(pkt)


        flexibility = 0
        s0_feasibility = True
        for i in range(len(self.apps)):
            if self.schedule[i][self.current_states[i]].n_packets_scheduled != self.schedule[i][self.current_states[i]].n_total_packets:
                s0_feasibility = False
                break
        if s0_feasibility:
            flexibility = self.calculate_flexibility()

        return flexibility

    def count_txs(self, i, s):
        n = 0
        flows = self.apps[i].states[s].flows
        for f in flows:
            n += len(f.txs)*(self.T//f.period)
        return n

    def potenial_gain(self, i, e, t):
        self.partition[t][e].app = i
        gain = Gain(i)
        gamma = 0.9
        for s in range(self.apps[i].n_states):
            if (self.schedule[i][s].packets_to_schedule) == 0:
                continue
            gain.pkts_per_state += [self.check_new_scheduled_packets(t, i, s)]
            feasibility_gain = len(gain.pkts_per_state[s]) / self.schedule[i][s].n_total_packets
            weight = 0
            for k in range(1, self.apps[i].k_max+1):
                weight += gamma**k*self.apps[i].M_k[k][s]
            # higher weight for initial state
            if s == self.current_states[i]:
                weight = weight*20
            gain.value += feasibility_gain*weight

        self.partition[t][e].app = -1
        return gain

    def check_new_scheduled_packets(self, t, i, s):
        schedule = self.schedule[i][s]
        used_links = copy.deepcopy(schedule.current_used_links)
        
        new_scheduled_packets = []
        for pkt in schedule.packets_to_schedule:
            if self.partition[t][pkt.link].app == i and pkt.link not in used_links:
                if len(self.time_dependency[i][s][pkt.flow_id]) != 0 and self.time_dependency[i][s][pkt.flow_id][-1] < t:
                    used_links[pkt.link] = True
                    new_scheduled_packets += [pkt]
                if len(self.time_dependency[i][s][pkt.flow_id]) == 0:
                    used_links[pkt.link] = True
                    new_scheduled_packets += [pkt]

        return new_scheduled_packets

    def calculate_flexibility(self):
        total_flex = 0
        gamma = 0.9
        self.all_feasible_states = []
        self.all_infeasible_states = []
        for i, app in enumerate(self.apps):
            feasible_states = []
            infeasible_states = []
            for s, sch in enumerate(self.schedule[i]):
                if sch.n_packets_scheduled == sch.n_total_packets:
                    feasible_states.append(s)
                else:
                    infeasible_states.append(s)
            self.all_feasible_states.append(feasible_states)
            self.all_infeasible_states.append(infeasible_states)
            M_prime = copy.deepcopy(app.transitions)
            for s in range(len(M_prime)):
                if s not in feasible_states:
                    M_prime[s, :] = 0
                    M_prime[s, s] = 1
            flex = 0
            denominator = 0
            for k in range(1, app.k_max+1):
                k_step_matrix = np.linalg.matrix_power(M_prime, k)
                k_step_success_prob = sum(k_step_matrix[self.current_states[i], :][s] for s in feasible_states)
                flex += (gamma**k)*k_step_success_prob
                denominator += (gamma**k)
            flex = flex/denominator
            total_flex += flex
        return round(total_flex, 5)



if __name__ == "__main__":
    T = 50
    links = list(edges.values())

    # Step 1: Generate random apps as placeholders
    random_apps = [App(trial=0, id=i, links=links, T=T, max_n_states=4, max_n_flows=5, max_n_flow_hop=3) for i in range(5)]

    # Step 2: Replace the random content using spreadsheet
    df = pd.read_csv("State_flow_with_Path3.csv")

    app_name_map = {name: i for i, name in enumerate(df['Subsystem'].unique())}
    state_name_map = {}

    for i, app in enumerate(random_apps):
        app_name = df['Subsystem'].unique()[i]
        print(app_name)
        app_df = df[df['Subsystem'] == app_name]
        states = sorted(app_df['State'].unique())
        app.states = []
        for j, state_name in enumerate(states):
            print(j,state_name)
            state_df = app_df[app_df['State'] == state_name]
            state = State(id=j)
            for _, row in state_df.iterrows():
                tsn_path = str(row['TSN Path']).strip()
                print(tsn_path)
                path = list(map(int, tsn_path.split(',')))
                txs = [edges[(f"S{path[k]}", f"S{path[k+1]}")] for k in range(len(path)-1)]
                print(txs)
                flow_count = row['Flow Count every 5ms']
                if flow_count >= 10:
                    flow_count = 10
                period = int(50 / flow_count) if flow_count > 0 else T
                if 50%period != 0:
                    print('false')
                flow = Flow(id=len(state.flows), txs=txs, period=period)
                state.flows.append(flow)
            app.states.append(state)
        app.n_states = len(app.states)
        transitions = np.random.rand(app.n_states, app.n_states)
        app.transitions = transitions / transitions.sum(axis=1, keepdims=True)
        app.k_max = 10
        app.M_k = [app.transitions[0]] + [np.linalg.matrix_power(app.transitions, k)[0] for k in range(1, app.k_max+1)]

    h = Heuristic(0, random_apps, links=links, T=T, current_states=[0]*len(random_apps))
    h.run()
    print(h.calculate_flexibility())

# Function to export flows information of each app to a CSV file with node, slot, and delay information
def export_flows_information_with_nodes_to_csv(apps, reverse_edges, time_dependency, filename="flows_information_with_nodes.csv"):
    # Create a list of dictionaries to collect flows information
    flows_info = []
    for app in apps:
        for state in app.states:
            for flow in state.flows:
                # Reconstruct the node path from links
                node_path = [reverse_edges[flow.txs[0]][0]]
                for link_id in flow.txs:
                    node_path.append(reverse_edges[link_id][1])
                
                # Get the list of time slots from time_dependency for each link in the flow
                slots = time_dependency[app.id][state.id][flow.id]
                
                # Determine the number of times the flow was sent
                n_instances = len(slots) // len(flow.txs)
                
                # Calculate delays for each instance
                delays = []
                for i in range(n_instances):
                    instance_slots = slots[i * len(flow.txs):(i + 1) * len(flow.txs)]
                    delay = max(instance_slots) - min(instance_slots)
                    delays.append(delay)
                
                # Prepare the flow information with the delay
                flow_info = {
                    'App ID': app.id,
                    'State ID': state.id,
                    'Flow ID': flow.id,
                    'Links': ' -> '.join(str(link) for link in flow.txs),
                    'Node Path': ' -> '.join(node_path),
                    'Period': flow.period,
                    'Slot': ', '.join(str(slot) for slot in slots),
                    'Delay': ', '.join(str(delay) for delay in delays)
                }
                flows_info.append(flow_info)
    
    # Convert the list of dictionaries to a DataFrame
    df_flows = pd.DataFrame(flows_info, columns=['App ID', 'State ID', 'Flow ID', 'Links', 'Node Path', 'Period', 'Slot', 'Delay'])
    
    # Export to CSV
    df_flows.to_csv(filename, index=False)

# Function to export the partition schedule to a CSV file
def export_partition_schedule_to_csv(partition, T, links, filename="partition_schedule.csv"):
    # Create a DataFrame for the partition matrix
    partition_matrix = []
    for t in range(T):
        row = [partition[t][e].app for e in links]
        partition_matrix.append(row)
    
    df_partition = pd.DataFrame(partition_matrix, columns=[f"Link {e}" for e in links])
    df_partition.index.name = 'Time Slot'
    
    # Export to CSV
    df_partition.to_csv(filename, index=True)

# Example usage
edges = {
    ('A', 'B'): 0, ('B', 'A'): 1,
    ('B', 'C'): 2, ('C', 'B'): 3,
    ('C', 'D'): 4, ('D', 'C'): 5,
    ('D', 'A'): 6, ('A', 'D'): 7
}
reverse_edges = {v: k for k, v in edges.items()}
export_partition_schedule_to_csv(h.partition, T=h.T, links=h.links, filename="partition_schedule.csv")
export_flows_information_with_nodes_to_csv(h.apps, reverse_edges, time_dependency=h.time_dependency, filename="flows_information_with_nodes.csv")
