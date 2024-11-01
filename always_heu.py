import pulp
from typedefs import *
from app import App
from heuristic import Heuristic
import copy
import numpy as np
import random
import time


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
                        bt = time.time()
                        all_covered = False
                        if not all_covered:
                          
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
                        print(self.flex)
                        results.append(res)
                    else:
                        print("still feasible")
                        results.append({"method": "adjustment", "n_affected_apps": 0,
                                       "flex": self.flex, "time": 0,  "reconfig_count":self.reconfig_count})
                    
                    if self.transition_times >= 100:
                        return results

        print(f"Number of reconfigurations: {self.reconfig_count}")
        return results
    



if __name__ == "__main__":
    cnt_no_need = 0
    cnt_adj_success = 0
    cnt_adj_fail = 0
    avg_flex_loss = 0
    avg_affected_apps = 0
    n_trials = 1
    for i in range(n_trials):
        adj = Adjustment(trial=21, n_apps=10, T=50, links=range(40),
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
