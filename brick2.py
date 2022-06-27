import numpy as np
from maf import Map
import networkx as nx
import pdb
import matplotlib.pyplot as plt



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
        self.graph=self.makegraph(self.map)


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

    def grid(self,x:int,y:int):
        '''
        returns the value of x,y coordinate in grid
        '''
        if x>=0 and y>=0 and x<self.n and y<self.m:
            #x,y=self.convert(x,y)
            return self.map[x,y]
        else:
            return None

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
        oned=crds[agent]
        xl,yl=self.get_size()
        x,y=oned//xl,oned%xl
        grid=self.fov(x,y,view)
        l,b=grid.shape
        lx=min(view,x)
        ly=min(view,y)
        cellid=lx*b+ly
        graph=self.makegraph(grid)
        if np.sum(crds==crds[agent])==1:
            ap=set(nx.articulation_points(graph))
            return not cellid in ap
        return False

    def global_visit_permission(self,crds:np.ndarray,agent:int):
        '''
        crds: cartesian coordinates of all agets
        agent: id of agent
        return: if we're allowed to mark cell as visited
        '''
        #l and b are length and breadth of grid
        oned=crds[agent]
        if  np.sum(crds==crds[agent])==1:
            if self.graph.degree(oned)==1:
                return True
            else:
                #mx,my=self.convert(x,y)
                ap=set(nx.articulation_points(self.graph))
                return not oned in ap


    def mark(self,crds:np.ndarray,agent:int,view:int):
        '''
        crds: cartesian coordinates of all agents
        agent: id of agent
        view: field of view of agent
        returns: marks a cell visited or explored
        '''
        oned=crds[agent]
        xl,yl=self.get_size()
        x,y=oned//yl,oned%yl
        #mx,my=self.convert(x,y)
        if self.map[x,y]==self.wall_cell:
            raise NameError('Trying to visit a marked cell')
        elif self.map[x,y]!=self.explored_cell:
            self.explored+=1
            self.map[x,y]=self.explored_cell
        if self.visit_permission(crds,agent,view):
            self.map[x,y]=self.wall_cell
            self.graph.remove_node(oned)
            self.visited+=1
            return True
        return False
    
    def global_mark(self,crds:np.ndarray,agent:int):
        '''
        crds: cartesian coordinates of all agents
        agent: id of agent
        view: field of view of agent
        returns: marks a cell visited or explored
        '''
        oned=crds[agent]
        xl,yl=self.get_size()
        x,y=oned//xl,oned%xl
        #mx,my=self.convert(x,y)

        if self.global_visit_permission(crds,agent):
            self.map[x,y]=self.wall_cell
            self.graph.remove_node(oned)
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
        oned=crds[agent]
        xl,yl=self.get_size()
        x,y=oned//yl,oned%yl
        #mx,my=self.convert(x,y)
        #american spelling, not british one
        #directions north, east,south, west
        neighbors=self.order_neighbor[agent%24]
        directions=self.order_direction[agent%24]
        for neighbor,d in zip(neighbors,directions):
            a,b=neighbor[0],neighbor[1]
            if x+a>=0 and y+b>=0 and x+a<self.m and y+b<self.n:
                #cell is unexplored
                if self.map[x+a,y+b]==0:
                    value=3-self.graph.degree(xl*(x+a)+y+b)
                    unexplored_directions.append(d)
                    unexplored.append(value)
                elif self.map[x+a,y+b]==self.explored_cell and self.map[x+a,y+b]!=self.wall_cell:
                    explored.append([x+a,y+b])
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
        self.l,self.b=sizey,sizex
        self.visit=np.ones_like((self.map.map))*-1
        #self.indir=np.ones_like((self.map.map))*-1
        #self.outdir=np.ones((sizex,sizey))*-1
        #four phases 0: loop detection, 1: loop control,
        # 2: loop closing, 3: loop cleaning
        self.phase=0
        self.loop_detected=False
        self.visited=0
        self.explored=0
        


    def move(self,crds:np.ndarray,directions:list):
        '''
        crds: coordinates of all the agents
        direction: direction to move into
        returns: move the agent in direction and update it's x,y coordinates in cartesian
        '''
        #keep track of cells marked explored and visited by agent
        if self.map.grid(self.x,self.y)!=self.map.explored_cell:
            self.explored+=1
        if self.map.mark(crds,self.id,self.fov) or (self.visit[self.x,self.y]!=-1 and self.map.global_mark(crds,self.id)):
            self.visited+=1

        #record outdirection
        #mx,my=self.map.convert(self.x,self.y)
        self.visit[self.x,self.y]+=1
        #top
        #[[-1,0],[0,1],[1,0],[0,-1]]
        i=0
        while directions and i<len(directions):
            direction=directions[i]
            if direction==0 and self.x-1>=0 and ([self.x-1,self.y]!=self.last or len(directions)==1):
                self.last=[self.x,self.y]
                self.x-=1
                if self.map.grid(self.x,self.y)==self.map.wall_cell:
                    self.x+=1
                    return False
                else:
                    return True
            #right
            elif direction==1 and self.y+1<self.map.get_size()[1] and ([self.x,self.y+1]!=self.last or len(directions)==1):
                self.last=[self.x,self.y]
                self.y+=1
                if self.map.grid(self.x,self.y)==self.map.wall_cell:
                    self.y-=1
                    return False
                else:
                    return True
            #bottom
            elif direction==2 and self.x+1<self.map.get_size()[0] and ([self.x+1,self.y]!=self.last or len(directions)==1):
                self.last=[self.x,self.y]
                self.x+=1
                if self.map.grid(self.x,self.y)==self.map.wall_cell:
                    self.x-=1
                    return False
                else:
                    return True
            #left
            elif direction==3 and self.y-1>=0 and ([self.x,self.y-1]!=self.last or len(directions)==1):
                self.last=[self.x,self.y]
                self.y-=1
                if self.map.grid(self.x,self.y)==self.map.wall_cell:
                    self.y+=1
                    return False
                else:
                    return True
            i+=1
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
                dirs=[]
                while directions:
                    max_value=max(values)
                    idx=values.index(max_value)
                    values.pop(idx)
                    d=directions.pop(idx)
                    dirs.append(d)
                if not self.move(crds,dirs):
                    raise NameError("Moved into wall cell!!")
                    
            elif action=='explored':
                i=0
                moved=False
                dirs=[]
                while i<4:
                    d=self.map.get_order(self.id)[i]    
                    if d in directions:
                        dirs.append(d)
                    i+=1
                
                moved=self.move(crds,dirs)
                if not moved:
                    raise NameError("Moved into wall cell!!")
            elif action=='terminate':
                if len(self.map.graph)!=1:
                    raise NameError("trying to end early")
                self.terminate=True
                self.map.global_mark(crds,self.id)
                return -self.visited
            crds[self.id]=self.b*self.x+self.y
            '''if crds[self.id]==3861:
                print('stop')'''
            '''if self.visit[self.x,self.y]!=-1 and self.map.global_mark(crds,self.id):
                self.visited+=1'''
            #x,y=self.map.convert(self.x,self.y)
            return self.b*self.x+self.y
        
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
    crds=np.array([0]*n)
    for i in range(n):
        agents.append(BNMAgent(0,0,i,m))
    coverage_percentage=[]
    i=0
    while m.marked_visited()!=1:
        if agents[i%n].state():
            crds[i%n]=agents[i%n].next(crds)
        i+=1
        if i%n==0:
            print(i%n,m.covered(),end=' | ')
            coverage_percentage.append(m.covered())
    print(crds)