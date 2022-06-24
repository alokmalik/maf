from brick import GridToGraph, BrickMap
from maf import Map
import numpy as np
import networkx as nx
from collections import deque
import pdb
class MACPP(Map):
    def __init__(self, grid, num_directions):
        '''
        grid: numpy array of map
        num_directions: we can only have 4 connected grid
        '''
        assert num_directions==4, "Grid can be only 4 connected"
        super().__init__(grid, num_directions)
        self.wall_cell=-1
        self.explored_cell=1
        self.visited=0
        self.graph=self.makegraph()
        #self.graph_explored = nx.get_node_attributes(self.graph, "explored")
        self.visit_frequency=np.zeros_like(self.map)

    def makegraph(self,grid:np.ndarray=None):
        '''
        returns the graph object of numpy array
        '''
        if grid is None:
            grid=self.map
        #l is length, b is breadth
        l,b=grid.shape
        g = nx.Graph()
        #add one node for each cell in self.graph object
        for i in range(l*b):
            g.add_node(i,explored=0)
        for i in range(l*b):
            #convert 1d coordinate of graph to 2d coordinate of grid
            x=i//b
            y=i%b
            if grid[x,y]!=self.wall_cell:
                if x-1>=0 and grid[x-1,y]!=self.wall_cell:
                    g.add_edge(x*b+y,(x-1)*b+y)
                if x+1<l and grid[x+1,y]!=self.wall_cell:
                    g.add_edge(x*b+y,(x+1)*b+y)
                if y-1>=0 and grid[x,y-1]!=self.wall_cell:
                    g.add_edge(x*b+y,x*b+y-1)
                if y+1<b and grid[x,y+1]!=self.wall_cell:
                    g.add_edge(x*b+y,x*b+y+1)
        #remove all the node with degree 1
        for i in range(l*b):
            if g.degree(i)==0:
                g.remove_node(i)
        return g


    def bfs(self,source,depth):
        '''
        source: oned coordinate of graph
        depth: depth of bfs
        returns: all the valid paths from source
        '''
        paths={}
        #print(source)
        paths[source]=[source]
        queue=deque()
        queue.append(source)
        distances={}
        distances[source]=0
        while queue:
            node=queue.popleft()
            for neighbour in self.graph[node]:
                if neighbour not in distances.keys() and distances[node]<depth and self.graph.degree(neighbour)<4:
                    distances[neighbour]=distances[node]+1
                    paths[neighbour]=paths[node]+[neighbour]
                    queue.append(neighbour)
        return paths

    def articulation_points(self):
        return set(nx.articulation_points(self.graph))

    def availablemoves(self,crds,agent):
        '''
        crds: oned coordinate of all agents
        agent: the index of agent in crds
        '''
        source=crds[agent]
        c=0
        if source in self.graph:
            c=self.graph.degree[source]
        else:
            raise NameError('Source Node not in graph')
        return c

    
    def get_direction(self,crds,agent,depth):
        '''
        crds: a list containing coordinates of all agents
        agent: index of agent in list crds
        graph: the graph object of map
        depth: depth of bfs
        returns action preference scores of each direction
        '''
        l,b=self.m,self.n
        source=crds[agent]
        paths=self.bfs(source,depth)
        ap=self.articulation_points()
        self.ap=ap
        #relative oned coordinates of north and south nodes
        north=source-b
        south=source+b
        east=source+1
        west=source-1
        #initial ratings of all directions
        n,s,e,w=0,0,0,0
        pathrating={}
        #give each possible path rating
        for node in paths.keys():
            pathrating[node]=0
            for p in paths[node]:
                if self.graph.degree(p)==1:
                    pathrating[node]=+depth*2
                if p in ap:
                    pathrating[node]-=1
                if not self.graph.nodes[p]['explored']:
                    pathrating[node]+=1
                if p in crds and p!=source:
                    pathrating[node]-=1
        #for each direction choose path with highest rating
        for node in pathrating.keys():
            if north in paths[node] and n<pathrating[node]:
                n=pathrating[node]
            if south in paths[node] and s<pathrating[node]:
                s=pathrating[node]
            if east in paths[node] and e<pathrating[node]:
                e=pathrating[node]
            if west in paths[node] and w<pathrating[node]:
                w=pathrating[node]
        #deprioritize paths if other agents are in the way
        for crd in crds:
            for node in paths.keys():
                #print(paths[node],north,south,east,west)
                if north in paths[node] and crd in paths[node] and crd!=crds[agent]:
                    #print('hi1')
                    n-=depth*2
                    if south in self.graph and source in self.graph[south] and self.graph.nodes[south]['explored'] and south not in crds:
                        s+=1
                    if east in self.graph and source in self.graph[east] and self.graph.nodes[east]['explored'] and east not in crds:
                        e+=1
                    if west in self.graph and source in self.graph[west] and self.graph.nodes[west]['explored'] and west not in crds:
                        w+=1
                elif south in paths[node] and crd in paths[node] and crd!=crds[agent]:
                    #print('hi2')
                    s-=depth*2
                    if north in self.graph and source in self.graph[north] and self.graph.nodes[north]['explored'] and north not in crds:
                        n+=1
                    if east in self.graph and source in self.graph[east] and self.graph.nodes[east]['explored'] and east not in crds:
                        e+=1
                    if west in self.graph and source in self.graph[west] and self.graph.nodes[west]['explored'] and west not in crds:
                        w+=1
                elif east in paths[node] and crd in paths[node] and crd!=crds[agent]:
                    #print('hi3')
                    e-=depth*2
                    if north in self.graph and source in self.graph[north] and self.graph.nodes[north]['explored'] and north not in crds:
                        n+=1
                    if south in self.graph and source in self.graph[south] and self.graph.nodes[south]['explored'] and south not in crds:
                        s+=1
                    if west in self.graph and source in self.graph[west] and self.graph.nodes[west]['explored'] and west not in crds:
                        w+=1
                elif west in paths[node] and crd in paths[node] and crd!=crds[agent]:
                    #print('hi4')
                    w-=depth*2
                    if north in self.graph and source in self.graph[north] and self.graph.nodes[north]['explored'] and north not in crds:
                        n+=1
                    if south in self.graph and source in self.graph[south] and self.graph.nodes[south]['explored'] and south not in crds:
                        s+=1
                    if east in self.graph and source in self.graph[east] and self.graph.nodes[east]['explored'] and east not in crds:
                        e+=1
        return [n,e,s,w]



    def mark(self,x,y,crds):
        '''
        input: cartesian coordinates of agent, and it's field of view
        marks a cell visited or explored
        '''
        x,y=self.convert(x,y)
        self.visit_frequency[x,y]+=1
        source=self.n*x+y
        #print(source)
        if source not in self.graph:
            raise NameError('Trying to visit a marked cell')
        elif self.graph.nodes[source]['explored']!=self.explored_cell:
            self.explored+=1
            self.graph.nodes[source]['explored']=self.explored_cell
            self.map[x,y]=self.explored_cell
        if source not in self.ap and np.sum(crds==source)==1:
            self.map[x,y]=self.wall_cell
            self.visited+=1
            self.graph.remove_node(source)
            return True
        return False

    def marked_visited(self):
        return self.visited/self.explorable


class MACPPAgent:
    def __init__(self,x,y,id,map_object):
        self.startx=x
        self.starty=y
        self.x=x
        self.y=y
        self.last=[x,y]
        self.map=map_object
        self.id=id
        self.l,self.b=self.map.map.shape
        self.depth=5
        x,y=self.map.convert(x,y)
        self.last_cell=self.b*x+y
        self.terminate=False
        self.visited=0
        self.num_moves=0

    def move(self,direction,crds):
        '''
        move the agent in direction and update it's x,y coordinates in cartesian
        '''
        self.num_moves+=1
        mark=self.map.mark(self.x,self.y,crds)
        if mark:
            self.visited+=1

        if len(self.map.graph):
            #top
            if direction==0 and self.y+1<self.map.get_size()[1]:
                self.y+=1
            #right
            elif direction==1 and self.x+1<self.map.get_size()[0]:
                self.x+=1
            #bottom
            elif direction==2 and self.y-1>=0:
                self.y-=1
            #left
            elif direction==3 and self.x-1>=0:
                self.x-=1
            #error
            else:
                raise NameError('Invalid direction input!!')
            

        

    def next(self,crds):
        '''
        crds: coordinate of all agents
        '''
        l,b=self.l,self.b
        directions=np.array(self.map.get_direction(crds,self.id,self.depth),dtype=float)
        source=crds[self.id]
        am=self.map.availablemoves(crds,self.id)
        #actions=actiontuple.copy()
        pref=[]
        #print(am,actiontuple)
        i=0
        #get direction preference order in stack
        while i<4:
            m=np.argmax(directions)
            dp=[m]
            # if there are multiple direction with same rating, shuffle them
            while np.sum(directions==directions[m])>1:
                directions[m]=-np.inf
                m=np.argmax(directions)
                dp.append(m)
            directions[m]=-np.inf
            np.random.shuffle(dp)
            while dp:
                d=dp.pop()
                pref.append(d)
                i+=1
                
        #print(len(self.map.graph),np.sum(crds!=-1),end='|')
        if len(self.map.graph)<=np.sum(crds>0) and np.sum(crds==source)>1:
            self.terminate=True
            return -self.visited
        elif len(self.map.graph)==1:
            self.move(-1,crds)
            self.terminate=True
            return -self.visited
        if am==0 and len(self.map.graph)!=1:
            raise NameError('Available Actions zero')
        elif am==1:
            for neighbour in self.map.graph[source]:
                if neighbour==source-b:
                    self.last_cell=source
                    self.move(0,crds)
                    x,y=self.map.convert(self.x,self.y)
                    return self.b*x+y
                elif neighbour==source+b:
                    self.last_cell=source
                    self.move(2,crds)
                    x,y=self.map.convert(self.x,self.y)
                    return self.b*x+y
                elif neighbour==source+1:
                    self.last_cell=source
                    self.move(1,crds)
                    x,y=self.map.convert(self.x,self.y)
                    return self.b*x+y
                elif neighbour==source-1:
                    self.last_cell=source
                    self.move(3,crds)
                    x,y=self.map.convert(self.x,self.y)
                    return self.b*x+y
        else:
            while pref:
                dc=pref.pop(0)
                if dc==0 and source-b in self.map.graph and source in self.map.graph[source-b] and source-b!=self.last_cell:
                    self.last_cell=source
                    self.move(dc,crds)
                    x,y=self.map.convert(self.x,self.y)
                    return self.b*x+y
                elif dc==2 and source+b in self.map.graph and source in self.map.graph[source+b] and source+b!=self.last_cell:
                    self.last_cell=source
                    self.move(dc,crds)
                    x,y=self.map.convert(self.x,self.y)
                    return self.b*x+y
                elif dc==1 and source+1 in self.map.graph and source in self.map.graph[source+1] and source+1!=self.last_cell:
                    self.last_cell=source
                    self.move(dc,crds)
                    x,y=self.map.convert(self.x,self.y)
                    return self.b*x+y
                elif dc==3 and source-1 in self.map.graph and source in self.map.graph[source-1] and source-1!=self.last_cell:
                    self.last_cell=source
                    self.move(dc,crds)
                    x,y=self.map.convert(self.x,self.y)
                    return self.b*x+y
                else:
                    continue
            raise NameError('No action Selected,Source: {}, AM: {}, Original Action Preferences: {}, Pref: {}'.format(source,am,pref,directions))

    
    def state(self):
        return not self.terminate




            
if __name__=='__main__':

    l,b=50,50
    grid=np.zeros((l,b))
    grid[int(l/2),:]=-1
    grid[int(l/2),int(b/2)]=0
    grid[int(l/2),int(b/2)+1]=0
    
    m=MACPP(grid,4)
    agents=[]
    n=12
    x,y=m.convert(0,0)
    print(m.n*x+y)
    crds=np.ones(n)*(m.n*x+y)
    for i in range(n):
        agents.append(MACPPAgent(0,0,i,m))

    i=0
    while m.marked_visited()<1:
        if agents[i].state():
            crds[i]=agents[i].next(crds)
        i+=1
        i%=n
            #if i==0:
                #plt.figure(figsize=(25,25))
                #mask = np.ma.masked_greater(grid,1)
                #plt.imshow(grid)
                #plt.imshow(mask,cmap='rainbow')
                #print(i,m.marked_visited(),end=' | ')