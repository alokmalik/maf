from macpp import MACPP, MACPPAgent
import networkx as nx
from collections import deque
import numpy as np

class MACPPOnline(MACPP):
    '''
    Online version of MACPP Algorithm
    '''
    def __init__(self, grid, num_directions):
        super().__init__(grid, num_directions)
        self.global_graph=self.graph
        #initialize local graph known by agent
        #it is empty initially as agent has no knowledge of the map in online case
        self.graph=nx.Graph()

    def graph_constructor(self,crds,agent,depth):
        '''
        constructs a graph of agent's field of view
        crds: oned coordinates of all agents
        agent: the id of agent or the index of agent's coordinates in crds
        depth: depth of bfs
        '''
        source=crds[agent]
        queue=deque()
        queue.append(source)
        distances={}
        distances[source]=0
        nodes=[source]
        while queue:
            node=queue.popleft()
            for neighbour in self.global_graph[node]:
                if neighbour not in distances.keys() and distances[node]<depth:
                    distances[neighbour]=distances[node]+1
                    queue.append(neighbour)
                    if neighbour not in nodes:
                        nodes.append(neighbour)
        for node in nodes:
            if node not in self.graph:
                self.graph.add_node(node,explored=0)
                for neighbour in self.global_graph[node]:
                    if neighbour in self.graph:
                        self.graph.add_edge(node,neighbour)

    def get_direction(self,crds,agent,depth):
        '''
        crds: a list containing coordinates of all agents
        agent: index of agent in list crds
        graph: the graph object of map
        depth: depth of bfs
        returns action preference scores of each direction
        '''
        #construct local graph of agent's field of view
        self.graph_constructor(crds,agent,depth)
        #get action preference scores of each direction from local graph
        return super().get_direction(crds,agent,depth)



    def mark(self,x,y,crds):
        '''
        input: cartesian coordinates of agent, and it's field of view
        marks a cell visited or explored
        '''
        m=super().mark(x,y,crds)
        x,y=self.convert(x,y)
        source=self.n*x+y
        if m:
            self.global_graph.remove_node(source)
        else:
            self.global_graph.nodes[source]['explored']=self.explored_cell
        return m