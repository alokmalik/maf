from cmath import exp
from this import d
import numpy as np
from maf import Map
import networkx as nx



class GridToGraph:
    '''
    converts the numpy matrix grid to a graph object
    where visited cells are not included and only 
    unvisited or explored cells in grid are converted to nodes
    '''
    def __init__(self) -> None:
        self.wall_cell=1

    
    def makegraph(self,array):
        '''
        takes in the 2d array return nx.Graph object
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
    def __init__(self,grid,num_directions):
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


    def permutations(self,x):
        '''
        input: a list
        returns all permutations of the elements inside dictionary
        '''
        if len(x)==1:
            return [x]
        else:
            ans=[]
            for i in range(len(x)):
                for p in self.permutations(x[:i]+x[i+1:]):
                    ans.append([x[i]]+p)
            return ans

    def perm_to_dict(self,x):
        '''
        converts a list to dictionary 
        '''
        d={}
        for i,a in enumerate(x):
            d[i]=a[:]
        return d

    def fov(self,x,y,view):
        '''
        input: cartesian coordinates of agent and it's field of view
        return the view of the agent located at x,y
        '''
        x,y=self.convert(x,y)
        return self.map[max(0,x-view):min(self.m-1,x+view+1),max(0,y-view):min(self.n-1,y+view+1)]

    def makegraph(self,array):
        '''
        takes in the 2d array return nx.Graph object
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

    def visit_permission(self,x,y,view):
        '''
        input: cartesian coordinates of agent and it's field of view
        returns if we're allowed to mark cell as visited
        '''
        #l and b are length and breadth of grid
        grid=self.fov(x,y,view)
        l,b=grid.shape
        x,y=self.convert(x,y)
        #converting from global x and y to fov x and y
        x=min(view,x)
        y=min(view,y)
        cellid=x*b+y
        graph=self.makegraph()
        ap=set(nx.articulation_points(graph))
        return not cellid in ap

    def mark(self,x,y,view):
        '''
        input: cartesian coordinates of agent, and it's field of view
        marks a cell visited or explored
        '''
        x,y=self.convert(x,y)
        if self.map[x,y]==self.wall_cell:
            raise NameError('Trying to visit a marked cell')
        elif self.map[x,y]!=self.explored_cell:
            self.explored+=1
            self.map[x,y]=self.explored_cell
        if self.visit_permission(x,y,view):
            self.map[x,y]=self.wall_cell
            self.visited+=1
            return True
        return False



    def get_direction(self,x,y,view,order):
        '''
        input: cartesian coordinates of agent, it's fov, and it's default direction order
        returns the directions for the agent to move and and actions
        '''
        unexplored=[]
        unexplored_directions=[]
        explored=[]
        explored_directions=[]
        x,y=self.convert(x,y)
        #american spelling, not british one
        #directions north, east,south, west
        neighbors=self.order_neighbor[order]
        directions=self.order_direction[order]
        for d,a,b in enumerate(neighbors):
            if x+a>=0 and x+b>=0 and x+a<self.m and x+b<self.n:
                #cell is unexplored
                if self.mat[x+a,y+b]==0:
                    fov_mat=self.fov(x+a,y+b,view)
                    value=np.sum(fov_mat==-1)
                    unexplored_directions.append(directions[d])
                    unexplored.append(value)
                elif self.mat[x+a,y+b]==self.explored_cell:
                    explored.append([x+a,y+b])
                    explored_directions.append(directions[d])
        if unexplored:
            return unexplored,unexplored_directions,'unexplored'
        elif explored:
            return explored,explored_directions,'explored'
        else:
            return [],[],'terminate'
    



                    

class BNMAgent:
    def __init__(self,x,y,id,map_object):
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
        self.indir=np.ones((sizex,sizey))*-1
        self.outdir=np.ones((sizex,sizey))*-1
        #four phases 0: loop detection, 1: loop control, 2: loop closing, 3: loop cleaning
        self.phase=0
        self.loop_detected=False
        


    def move(self,direction):
        '''
        move the agent in direction and update it's x,y coordinates in cartesian
        '''
        mx,my=self.map.convert(self.x,self.y)
        
        
        self.last=[self.x,self.y]
        self.outdir[mx,my]=direction
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

        mx,my=self.map.convert(self.x,self.y)
        self.indir[mx,my]=direction
        
        act=self.map.mark(self.x,self.y,self.fov)
        
        if not act and self.phase==0 and self.outdir[mx,my]!=(direction+2)%2:
            self.loop_detected=True
        if self.loop_detected=True and self.map.visited_map[mx,my]==-1:
            self.phase=1
        if self.phase=1 and self.map.visited_map[mx,my]<self.id:
            self.map.visited_map[mx,my]=self.id

        
        

        

    def next(self):
        if self.phase==0:
            values,directions,action=self.map.get_direction(self.x,self.y,self.fov,self.order)

            if action=='unexplored':
                max_value=max(values)
                idx=values.index(max_value)
                self.move(directions[idx])
            elif action=='explored':
                i=0
                moved=False
                while i<len(directions):
                    d=self.map.order_direction[self.order][i]
                    if d in directions and values[directions.index(d)]!=self.last:
                        moved=True
                        self.move(d)
                        break
                    else:
                        i+=1
                if not moved:
                    self.move(directions[0])
            elif action=='terminate':
                self.terminate=True
        elif self.phase==1:


    def state(self):
        self.map.mark(self.x,self.y,self.fov)
        return not self.terminate


            


            











    

            
        
    






