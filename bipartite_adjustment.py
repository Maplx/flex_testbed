import pulp
from typedefs import *
from app import App
from heuristic import Heuristic
import copy
import numpy as np
import random
import time
from itertools import groupby
from operator import itemgetter
from scipy.optimize import linear_sum_assignment
import numpy as np


class Adjustment:
    def __init__(self, trial, n_apps, T, links, max_n_states, max_n_flows, max_n_flow_hop, verbose):
        self.n_apps = n_apps
        self.T = T
        self.links = links
        self.max_n_states = max_n_states
        self.max_n_flows = max_n_flows
        self.max_n_flow_hop = max_n_flow_hop
        self.verbose = verbose
        self.apps = [App(trial, i, self.links, self.T,
                         max_n_states=self.max_n_states,
                         max_n_flows=self.max_n_flows,
                         max_n_flow_hop=self.max_n_flow_hop)
                     for i in range(self.n_apps)]
        self.reconfig_count = 0
        self.infeasible_times = 0
        self.transition_times = 0
        self.time = 0

    def run(self):
        self.current_states = [0]*self.n_apps
        self.heu = Heuristic(0, self.apps, self.links, self.T, self.current_states, verbose=False)
        flex = self.heu.run()
        self.flex = flex
        bt = time.time()
        if flex > 0 and flex < self.n_apps:
            print("init flex", flex)
            self.partition = self.heu.partition
            self.feasible_states = self.heu.all_feasible_states
            self.infeasible_states = self.heu.all_infeasible_states
            if self.verbose:
                print("infeasible states", self.infeasible_states)
            return self.transition()
        return []

    def transition(self):
        k = 500
        results = []
        for _ in range(k):
            for i in range(self.n_apps):
                if len(self.infeasible_states[i]) > 0:
                    next_state = np.random.choice(
                        range(self.apps[i].n_states), p=self.apps[i].transitions[self.current_states[i]])
                    self.current_states[i] = next_state
                    self.transition_times += 1
                    if next_state in self.infeasible_states[i]:
                        if self.verbose:
                            print(f"App {i} enters an infeasible state")
                            print("current states:", self.current_states)
                        
                        self.infeasible_times += 1
                        inf_app = i
                        request = self.find_resource_request(inf_app, self.current_states[inf_app])
                        
                        if self.verbose:
                            print('request:', request)
                        #print('request:', request)
                        request_sorted = sorted(request, key=itemgetter(0))
                        grouped_resource = [list(group) for key, group in groupby(request_sorted, key=itemgetter(0))]
                        #print(grouped_resource)
                        
                        all_covered  = True
                        total_loss = 0
                        bt = time.time()
                        n_app = 0
                        matched_apps = []
                        random.shuffle(grouped_resource)

                        for l in grouped_resource:
                            print(l)
                            provision, matching_request = self.find_provision(inf_app, single_link_resource_request=l)
                            matching, total_cost = self.hungarian_matching(provision,self.loss, matching_request)
                            total_loss += total_cost
                            #print(matching, total_cost)
                            if matching is None:
                                all_covered = False
                                break
                            
                            for m in matching:
                                if m[1] not in matched_apps:
                                    matched_apps.append(m[1])
                                    n_app += 1

                                a = m[1]
                                self.feasible_states[a] = [s for s in self.feasible_states[a] if s not in self.sacrificed_states[a] and s not in self.condi_states[a]]
                                self.infeasible_states[a] = self.infeasible_states[a]+self.sacrificed_states[a]
                                for r in provision[a]:
                                    self.partition[r[1]][r[0]].app = inf_app
                                    self.partition[r[1]][r[0]].states = [self.current_states[inf_app]]
                        
                        
                        
                      
                        if all_covered:
                            if self.verbose:
                                print( "total loss:", total_loss)

                            self.flex = self.flex-total_loss
                            self.infeasible_states[inf_app].remove(self.current_states[inf_app])
                            self.feasible_states[inf_app].append(self.current_states[inf_app])

                            print(n_app, 'apps engaged in matching adjustment')
                            self.time += time.time()-bt
                            print('Flexibility after matching:', self.flex)
                            res = {"method": "match","n_affected_apps": n_app, "flex": self.flex, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                            
                        if all_covered == False:
                            if self.verbose:
                                print("unable to satisfy resource request")
                            bt = time.time()
                            self.heu2 = Heuristic(0, self.apps, self.links, self.T, self.current_states, verbose=False)
                            self.flex = self.heu2.run()
                            if self.flex == 0:
                                # if self.verbose:
                                print("reconfig also failed")
                                return results
                            
                            else:
                                self.partition = self.heu2.partition
                                self.feasible_states = self.heu2.all_feasible_states
                                self.infeasible_states = self.heu2.all_infeasible_states
                                self.reconfig_count += 1
                                self.time += time.time()-bt

                                res = {"method": "reconfig", "n_affected_apps": self.n_apps - 10,
                                       "flex": self.flex, "time": time.time()-bt,  "reconfig_count":self.reconfig_count}
                                print("reconfig", end=",")
                        print('Flexibility after reconfiguration:', self.flex)
                        results.append(res)
                    else:
                        print("still feasible")
                        results.append({"method": "adjustment", "n_affected_apps": 0,
                                       "flex": self.flex, "time": 0,  "reconfig_count":self.reconfig_count})
                    
                    if self.transition_times >= 100:
                        return results

        print(f"Number of reconfigurations: {self.reconfig_count}")
        return results



        

    def find_resource_request(self, i, s):
        #rough calculation, to be improved
        #print('resource request start')
        
        flows = self.apps[i].states[s].flows

        resource_request = []
        cur_hops = {}
        for t in range(self.T):
            packets = []
            used_links = {}
            for f in flows:
                if t % f.period == 0:

                    if f.id in cur_hops and cur_hops[f.id] < len(f.txs):
                        '''
                        print('Current time: ',t, 'Flow period: ',f.period)
                        print('current hop of flow',f.id, 'is ',cur_hops[f.id], '. The len of links in flow is ', len(f.txs))
                        print('Flow links: ', f.txs)'''
                        
                        for i in range(cur_hops[f.id], len(f.txs)):
                            resource_request.append(
                            (f.txs[i], [(t//f.period)*f.period, (t//f.period+1)*f.period]))
                            #print('Append: ',f.txs[i])
                        
                           
 
                    cur_hops[f.id] = 0
                if cur_hops[f.id] < len(f.txs):
                    pkt = Packet(f.id, f.txs[cur_hops[f.id]], (t//f.period+1)*f.period, f.period)
                    packets.append(pkt)
            packets.sort(key=lambda x: x.deadline)

            for pkt in packets:
                if self.partition[t][pkt.link].app == i and pkt.link not in used_links:
                    used_links[pkt.link] = True
                    cur_hops[pkt.flow_id] += 1

        return resource_request


    def find_resource_provisions(self, inf_app, resource_request):
        safe_provisions = [[] for _ in range(self.n_apps)]
        unsafe_provisions = [[] for _ in range(self.n_apps)]

        self.sacrificed_states = [[] for _ in range(self.n_apps)]
        for r in resource_request:
            for t in [r[1][0], r[1][1]-1]:
                i = self.partition[t][r[0]].app
                if i != -1 and i != inf_app:
                    if self.current_states[i] not in self.partition[t][r[0]].states:
                        safe_provisions[i].append((r[0], t))
                        self.sacrificed_states[i] += self.partition[t][r[0]].states
                    else:
                        unsafe_provisions[i].append((r[0], t))

        self.loss = [0]*self.n_apps

        for i in range(self.n_apps):
            old_flex = self.calculate_flexibility(i, self.feasible_states[i], self.infeasible_states[i])
            self.sacrificed_states[i] = list(set(self.sacrificed_states[i]))
            new_flex = self.calculate_flexibility(
                i, [s for s in self.feasible_states[i] if s not in self.sacrificed_states[i]], self.infeasible_states[i]+self.sacrificed_states[i])
            self.loss[i] = old_flex-new_flex

        return safe_provisions, unsafe_provisions

            

    
    def find_provision(self, inf_app, single_link_resource_request):
        provision = [[] for _ in range(self.n_apps)]
        self.sacrificed_states = [[] for _ in range(self.n_apps)]
        self.condi_states = [[] for _ in range(self.n_apps)]

        matching_request = [[] for _ in range(self.n_apps)]
        matching_request[inf_app] = single_link_resource_request
        #print(single_link_resource_request)

        

        for r in single_link_resource_request:

            for t in [r[1][0], r[1][1]-1]:
                i = self.partition[t][r[0]].app
                if i != -1 and i != inf_app:
                    provision[i].append((r[0],t))

                    #rint(i)

                    if self.current_states[i] not in self.partition[t][r[0]].states:
        
                        self.sacrificed_states[i] += self.partition[t][r[0]].states
                    else:
                        #print('condi:',i)
                        s = self.current_states[i]
                        flow = self.apps[i].states[s].flows
                        a = t//flow[0].period
                        for f in flow:
                            if r[0] in f.txs:
                                a = t//f.period
                        matching_request[i].append((r[0],[a*f.period, (a+1)*f.period]))
                        self.condi_states[i] += self.partition[t][r[0]].states

        self.loss = [0]*self.n_apps
        
        for i in range(self.n_apps):
            old_flex = self.calculate_flexibility(i, self.feasible_states[i], self.infeasible_states[i])
            self.sacrificed_states[i] = list(set(self.sacrificed_states[i]))
            new_flex = self.calculate_flexibility(
                i, [s for s in self.feasible_states[i] if s not in self.sacrificed_states[i]]
                , self.infeasible_states[i]+self.sacrificed_states[i])
            self.loss[i] = old_flex-new_flex
        '''
        print(provision)
        print(self.loss)
        print(matching_request)
        '''
        return provision, matching_request
    
    def hungarian_matching(self, provision, loss, matching_request):
        # Prepare a list of valid requests and provisions
        valid_requests = []
        valid_provisions = []
        request_to_app = []
        provision_to_app = []

        # Collect all valid requests
        for req_app in range(len(matching_request)):
            for req in matching_request[req_app]:
                valid_requests.append(req)
                request_to_app.append(req_app)

        # Collect all valid provisions
        for prov_app in range(len(provision)):
            if provision[prov_app]:  # Only consider apps with provisions
                valid_provisions.append(provision[prov_app])
                provision_to_app.append(prov_app)

        # If there are no valid requests or provisions, return no matching
        if not valid_requests or not valid_provisions:
            return None, float('inf')  # Indicate no matching possible

        n_requests = len(valid_requests)
        n_provisions = len(valid_provisions)
        cost_matrix = np.full((n_requests, n_provisions), 9999.0)  # Large value for invalid matches

        # Populate the cost matrix
        for req_idx, (req_link, req_time_range) in enumerate(valid_requests):
            for prov_idx, prov_links in enumerate(valid_provisions):
                for prov_link, prov_time in prov_links:
                    if prov_link == req_link and req_time_range[0] <= prov_time <= req_time_range[1]:
                        # Set the cost as the loss of the provider
                        cost_matrix[req_idx][prov_idx] = loss[provision_to_app[prov_idx]]

        # Apply the Hungarian algorithm to the valid part of the cost matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Collect the matching result
        matching = []
        total_cost = 0
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < 9999:  # Only valid matches
                matching.append((request_to_app[row], provision_to_app[col]))
                total_cost += cost_matrix[row, col]
            else:
                return None, float('inf')  # Indicate that a valid matching was not found

        return matching, total_cost


    def calculate_flexibility(self, i, feasible_states, infeasible_states):
        gamma = 0.9

        M_prime = copy.deepcopy(self.apps[i].transitions)
        for s in range(len(M_prime)):
            if s in infeasible_states:
                M_prime[s, :] = 0
                M_prime[s, s] = 1
        flex = 0
        denominator = 0
        for k in range(1, self.apps[i].k_max+1):
            k_step_matrix = np.linalg.matrix_power(M_prime, k)
            k_step_success_prob = sum(k_step_matrix[0, :][s] for s in feasible_states)
            flex += (gamma**k)*k_step_success_prob
            denominator += (gamma**k)
        flex = flex/denominator
        return flex



if __name__ == "__main__":
    cnt_no_need = 0
    cnt_adj_success = 0
    cnt_adj_fail = 0
    avg_flex_loss = 0
    avg_affected_apps = 0
    n_trials = 1
    for i in range(n_trials):
        adj = Adjustment(trial=21, n_apps=10, T=50, links=range(30),
                         max_n_states=20, max_n_flows=8, max_n_flow_hop=5,
                         verbose=False)
        results = adj.run()
        

     

        print(len(results), "\n\n\n")
        if len(results) >= 10:

            n_apps = [res["n_affected_apps"] for res in results]
            flexs = [res["flex"] for res in results]
            times = [res["time"] for res in results]
            counts = [res["reconfig_count"] for res in results]

            print({
                "n_adjusted_apps": n_apps,
                "flex": flexs,
                "time": times,
                "reconfig_count": counts,
                "xAxis": [i for i in range(len(flexs))],
            })
        print(adj.infeasible_times)