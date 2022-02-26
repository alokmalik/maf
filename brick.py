import numpy as np
from maf import Map
import networkx as nx
import pdb
import matplotlib.pyplot as plt

class GridToGraph:
    '''
    converts the numpy matrix grid to a graph object
    where visited cells are not included and only 
    unvisited or explored cells in grid are converted to nodes
    '''
    def __init__(self) -> None:
        self.wall_cell=1

    
    def makegraph(self,array:np.ndarray):
        '''
        array: takes in the 2d array return nx.Graph object
        returns: nx.Graph() object of grid
        '''
        #l is length, b is breadth
        l,b=array.shape
        g = nx.Graph()
        #add one node for each cell in graph object
        for i in range(l*b):
            g.add_node(i,explored=0)
        for i in range(l*b):
            #convert 1d coordinate of graph to 2d coordinate of grid
            x=i//b
            y=i%b
            if array[x,y]!=self.wall_cell:
                if x-1>=0 and array[x-1,y]!=self.wall_cell:
                    g.add_edge(x*b+y,(x-1)*b+y)
                if x+1<l and array[x+1,y]!=self.wall_cell:
                    g.add_edge(x*b+y,(x+1)*b+y)
                if y-1>=0 and array[x,y-1]!=self.wall_cell:
                    g.add_edge(x*b+y,x*b+y-1)
                if y+1<b and array[x,y+1]!=self.wall_cell:
                    g.add_edge(x*b+y,x*b+y+1)
        #remove all the node with degree 1
        for i in range(l*b):
            if g.degree(i)==0:
                g.remove_node(i)
        return g



class BrickMap(Map):
    def __init__(self,grid:np.ndarray,num_directions:int):
        '''
        input: grid: numpy array
        num_directions: only 4
        '''
        assert num_directions==4, "Brick and Mortar is 4 connected only"
        super().__init__(grid,num_directions)
        #keeps track of which agent visited the cell last time
        self.visited_map=np.ones((self.m,self.n))*-1
        self.wall_cell=-1
        self.explored_cell=1
        #keeps count of how many cells have been marked visited
        self.visited=0
        #north,east,south,west
        self.default_neighbors=[[-1,0],[0,1],[1,0],[0,-1]]
        self.default_directions=[0,1,2,3]
        self.order_neighbor=self.perm_to_dict(self.permutations(self.default_neighbors))
        self.order_direction=self.perm_to_dict(self.permutations(self.default_directions))


    def permutations(self,x:list):
        '''
        x: a list
        return: all permutations of the elements inside dictionary
        '''
        if len(x)==1:
            return [x]
        else:
            ans=[]
            for i in range(len(x)):
                for p in self.permutations(x[:i]+x[i+1:]):
                    ans.append([x[i]]+p)
            return ans

    def perm_to_dict(self,x:list):
        '''
        x: a list
        return: list to dictionary 
        '''
        d={}
        for i,a in enumerate(x):
            d[i]=a[:]
        return d

    def fov(self,x:int,y:int,view:int):
        '''
        x: numpy x coordinate of agent
        y: numpy y coordinate of agent
        view: field of view
        return: the view of the agent located at x,y
        '''
        #x,y=self.convert(x,y)
        return self.map[max(0,x-view):min(self.m-1,x+view+1),max(0,y-view):min(self.n-1,y+view+1)]

    def makegraph(self,array:np.ndarray):
        '''
        arrey: 2d numpy array representation of map
        return: nx.Graph object
        '''
        #l is length, b is breadth
        l,b=array.shape
        g = nx.Graph()
        #add one node for each cell in graph object
        for i in range(l*b):
            g.add_node(i,explored=0)
        for i in range(l*b):
            #convert 1d coordinate of graph to 2d coordinate of grid
            x=i//b
            y=i%b
            if array[x,y]!=self.wall_cell:
                if x-1>=0 and array[x-1,y]!=self.wall_cell:
                    g.add_edge(x*b+y,(x-1)*b+y)
                if x+1<l and array[x+1,y]!=self.wall_cell:
                    g.add_edge(x*b+y,(x+1)*b+y)
                if y-1>=0 and array[x,y-1]!=self.wall_cell:
                    g.add_edge(x*b+y,x*b+y-1)
                if y+1<b and array[x,y+1]!=self.wall_cell:
                    g.add_edge(x*b+y,x*b+y+1)
        #remove all the node with degree 1
        for i in range(l*b):
            if g.degree(i)==0:
                g.remove_node(i)
        return g

    def visit_permission(self,crds:np.ndarray,agent:int,view:int):
        '''
        crds: cartesian coordinates of all agets
        agent: id of agent
        view: field of view of agent
        return: if we're allowed to mark cell as visited
        '''
        #l and b are length and breadth of grid
        x,y=crds[agent]
        mx,my=self.convert(x,y)
        grid=self.fov(mx,my,view)
        l,b=grid.shape
        #converting from global x and y to fov x and y
        lx=min(view,mx)
        ly=min(view,my)
        cellid=lx*b+ly
        graph=self.makegraph(grid)
        ap=set(nx.articulation_points(graph))

        return not cellid in ap and np.sum(crds==crds[agent])==1

    def mark(self,crds:np.ndarray,agent:int,view:int):
        '''
        crds: cartesian coordinates of all agents
        agent: id of agent
        view: field of view of agent
        returns: marks a cell visited or explored
        '''
        x,y=crds[agent]
        mx,my=self.convert(x,y)
        if self.map[mx,my]==self.wall_cell:
            raise NameError('Trying to visit a marked cell')
        elif self.map[mx,my]!=self.explored_cell:
            self.explored+=1
            self.map[mx,my]=self.explored_cell
        if self.visit_permission(crds,agent,view):
            self.map[mx,my]=self.wall_cell
            self.visited+=1
            return True
        return False
    
    def marked_visited(self):
        '''
        return: fraction of map marked visited
        '''
        return self.visited/self.explorable



    def get_direction(self,crds:np.ndarray,agent:int,view:int):
        '''
        crds: cartesian coordinates of all agents
        agent: id of agent
        view: fielf of view of agent
        return: the directions for the agent to move and and actions
        '''
        unexplored=[]
        unexplored_directions=[]
        explored=[]
        explored_directions=[]
        x,y=crds[agent]
        mx,my=self.convert(x,y)
        #american spelling, not british one
        #directions north, east,south, west
        neighbors=self.order_neighbor[agent%24]
        directions=self.order_direction[agent%24]
        for neighbor,d in zip(neighbors,directions):
            a,b=neighbor[0],neighbor[1]
            if mx+a>=0 and my+b>=0 and mx+a<self.m and my+b<self.n:
                #cell is unexplored
                if self.map[mx+a,my+b]==0:
                    fov_mat=self.fov(mx+a,my+b,view)
                    value=np.sum(fov_mat==-1)
                    unexplored_directions.append(d)
                    unexplored.append(value)
                elif self.map[mx+a,my+b]==self.explored_cell:
                    explored.append([mx+a,my+b])
                    explored_directions.append(d)
        if unexplored:
            return unexplored,unexplored_directions,'unexplored'
        elif explored:
            return explored,explored_directions,'explored'
        else:
            return [],[],'terminate'

    def get_order(self,id:int):
        '''
        id: id of agent
        returns: it's default order
        '''
        return self.order_direction[id%24]
    


class BNMAgent:
    def __init__(self,x:int,y:int,id:int,map_object:BrickMap):
        '''
        input: x and y coordinates of agents
        order: between 0 and 23 (including both) which defines default preferences of an agent
        map_object: The map agents will explore
        '''
        self.startx=x
        self.starty=y
        self.x=x
        self.y=y
        self.last=[x,y]
        self.id=id
        self.order=self.id%24
        self.terminate=False
        self.map=map_object
        self.fov=1
        sizey,sizex=self.map.get_size()
        self.indir=np.ones_like((self.map.map))*-1
        self.outdir=np.ones((sizex,sizey))*-1
        #four phases 0: loop detection, 1: loop control,
        # 2: loop closing, 3: loop cleaning
        self.phase=0
        self.loop_detected=False
        self.visited=0
        self.explored=0
        


    def move(self,crds:np.ndarray,direction:int):
        '''
        crds: coordinates of all the agents
        direction: direction to move into
        returns: move the agent in direction and update it's x,y coordinates in cartesian
        '''
        #keep track of cells marked explored and visited by agent
        if self.map.grid(self.x,self.y)!=self.map.explored_cell:
            self.explored+=1
        act=self.map.mark(crds,self.id,self.fov)
        if act:
            self.visited+=1

        #record outdirection
        mx,my=self.map.convert(self.x,self.y)
        self.last=[self.x,self.y]
        self.outdir[mx,my]=direction
        #top
        if direction==0 and self.y+1<self.map.get_size()[1]:
            self.y+=1
            if self.map.grid(self.x,self.y)==self.map.wall_cell:
                self.y-=1
                return False
            else:
                return True
        #right
        elif direction==1 and self.x+1<self.map.get_size()[0]:
            self.x+=1
            if self.map.grid(self.x,self.y)==self.map.wall_cell:
                self.x-=1
                return False
            else:
                return True
        #bottom
        elif direction==2 and self.y-1>=0:
            self.y-=1
            if self.map.grid(self.x,self.y)==self.map.wall_cell:
                self.y+=1
                return False
            else:
                return True
        #left
        elif direction==3 and self.x-1>=0:
            self.x-=1
            if self.map.grid(self.x,self.y)==self.map.wall_cell:
                self.x+=1
                return False
            else:
                return True
        #error
        else:
            #pdb.set_trace()
            raise NameError('Invalid direction input!!')


    def next(self,crds:np.ndarray):
        '''
        crds: cartesian coordinates of all the agents
        return: new coordinates of agent, 
        if the agent terminates it returns its count of cells it marked visited
        '''
        #loop detection phase
        if self.phase==0:
            values,directions,action=self.map.get_direction(crds,self.id,self.fov)

            if action=='unexplored':
                max_value=max(values)
                idx=values.index(max_value)
                d=directions[idx]
                if not self.move(crds,d):
                    raise NameError("Moved into wall cell!!")
            elif action=='explored':
                i=0
                moved=False
                while i<len(directions):
                    d=self.map.get_order(self.id)[i]
                    if d in directions and values[directions.index(d)]!=self.last:
                        moved=True
                        if not self.move(crds,d):
                            raise NameError("Moved into wall cell!!")
                        break
                    else:
                        i+=1
                if not moved:
                    d=directions[0]
                    if not self.move(crds,d):
                        raise NameError("Moved into wall cell!!")
            elif action=='terminate':
                self.terminate=True
                return self.visited
            #phase change
            mx,my=self.map.convert(self.x,self.y)
            if self.outdir[mx,my]!=-1 and self.outdir[mx,my]!=(d+2)%2:
                self.phase=1
            return self.x,self.y
        #below is the most complicated logic of the code
        #refer the the brick and mortar paper
        #loop control phase
        elif self.phase==1:
            mx,my=self.map.convert(self.x,self.y)
            d=self.outdir[mx,my]
            #if the agent bumps into a wall cell
            #it means that other agent was already in control of loop
            #and it has started marking cells visited and our agent
            #will start cleaning the loop
            if not self.move(crds,d):
                self.phase=3
            mx,my=self.map.convert(self.x,self.y)
            #if the currently marked id is less than agent's id
            #the the agent is free to take control of the cell
            if self.map.visited_map[mx,my]<self.id:
                self.map.visited_map[mx,my]==self.id
            #if the currently marked id of cell is equal to the agent's id
            #the agent has marked the entire loop with it's id successfully
            #and now it can move successfully to loop closing phase
            elif self.map.visited_map[mx,my]==self.id:
                self.phase=2
            #if the agent encounters a cell whose marked id greater than it's id
            #that means other agent is already working on marking the loop
            #current agent will leave it's effort and move to loop clearning phase
            elif self.map.visited_map[mx,my]>self.id:
                self.phase=3
            return self.x,self.y
        #loop closing phase
        elif self.phase==2:
            #if the agent can mark the cell visited that means loop has been closed
            #and we can move to loop cleaning phase
            #in loop cleaning phase we will mark cells visited if they're not blocking paths
            if self.map.visit_permission(crds,self.id,self.fov):
                d=self.outdir[self.x,self.y]
                if not self.move(crds,d):
                    raise NameError("Moved into wall cell")
                else:
                    self.phase=3
            #mark the current cell visited as it's part of loop
            else:
                mx,my=self.map.convert(self.x,self.y)
                # mark current cell visited only if no other agent
                # is not present on the current cell
                if np.sum(crds==crds[self.id])==1:
                    self.map.map[mx,my]=self.map.wall_cell
                    d=self.outdir[self.x,self.y]
                    if not self.move(crds,d):
                        raise NameError("Moved into wall cell")
                #skip turn and let other agent move i.e. standby
                else:
                    return self.x,self.y
        #loop cleaning
        elif self.phase==3:
            mx,my=self.map.convert(self.x,self.y)
            #if the current cell contains it's id remove it
            if self.map.visited_map[mx,my]==self.id:
                self.map.visited_map[mx,my]=-1
            #the agent has cleaned the loop and move to default loop detection phase
            else:
                self.phase=0
            #move in the direction of loop
            d=self.outdir[mx,my]
            #if the agent bumps into a wall then move to default loop detection phase
            #where agent will look for it's own path and mark cells visited
            #if they're not blocking any other path
            if not self.move(crds,d):
                self.phase=0
            return self.x,self.y

    def state(self):
        return not self.terminate


if __name__=='__main__':
    l,b=50,50
    grid=np.zeros((l,b))
    grid[int(l/2),:]=-1
    grid[int(l/2),int(b/2)]=0
    grid[int(l/2),int(b/2)+1]=0
    m=BrickMap(grid,4)
    agents=[]
    n=20
    crds=np.array([(0,0)]*n+[0,0,0])
    for i in range(n):
        agents.append(BNMAgent(0,0,i,m))
    coverage_percentage=[]
    i=0
    while m.marked_visited()<.5:
        if agents[i%n].state():
            x,y=agents[i%n].next(crds)
            crds[i%n]=[x,y]
        i+=1
        if i%n==0:
            print(i%n,m.covered(),end=' | ')
            coverage_percentage.append(m.covered())