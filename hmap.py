#from maf import Map
import numpy as np
from macpp import MACPP, MACPPAgent
import networkx as nx
import json

class HGrid:
    def __init__(self,filename):
        self.room=1
        self.map=0
        self.plan=2
        with open(filename) as f:
            data=json.load(f)

        m,n=data['dimensions'][0],data['dimensions'][1]

        grid=np.zeros((3,n,m))
        values=0

        for d in data['map']:
            x=d['x']
            y=d['y']
            
            if d['isWall']==True:
                values+=1
                grid[self.map,x,y]=-1
                grid[self.plan,x,y]=-1

            grid[self.room,x,y]=d['room']
        self.grid=grid
        self.doors=data['doors']
        
    def return_grid(self):
        return self.grid,self.doors


class HMap:
    def __init__(self,grid:np.ndarray,num_directions:int,doors:dict):
        '''
        input:grid is a numpy array of map
        num_directions: defines if it's a 4 connected grid
        '''
        #layer numbers
        self.map=0
        self.room=1
        self.plan=2
        #the 3d map
        self.grid_map=grid
        #y size, x size
        self.m,self.n=self.grid_map[self.map,:,:].shape
        assert num_directions==4, "Grid can be 8 or 4 connected only"
        self.nd=num_directions
        self.explorable=np.sum(self.grid_map[self.room,:,:]!=0)
        #keep count of number of explored cells
        self.explored=0
        self.wall_cell=-1
        self.area={}
        #dict of doors
        self.doors=doors
        #print(self.doors)
        #Room numbers start from 1
        for i in range(1,len(doors)+1):
            area=np.sum(self.grid_map[self.map,:,:]==i)
            if area!=0:
                self.area[i]=area
            else:
                break
        self.grid_graph=self.makegraph(self.grid_map[self.plan,:,:])

    def convert(self,x:int,y:int):
        '''
        To convert cartesian coordinates to numpy matrix coordinates
        '''
        return self.m-y-1,x

    def mark(self,x:int,y:int,value:int):
        '''
        mark a point in map visited
        '''
        x,y=self.convert(x,y)
        if self.grid_map[self.map,x,y]==0:
            self.explored+=1
        self.grid_map[self.map,x,y]=value
    
    def grid(self,x:int,y:int):
        '''
        returns the value of x,y coordinate in grid
        '''
        if x>=0 and y>=0 and x<self.n and y<self.m:
            x,y=self.convert(x,y)
            return self.grid_map[self.map,x,y]
        else:
            return None
    
    def covered(self):
        '''
        returns the covered map fracting
        '''
        return self.explored/self.explorable
    
    def get_size(self):
        '''
        returns the size of grid (X axis length, Y axis length)
        '''
        return self.n,self.m
    
    def get_room(self,x:int,y:int):
        '''
        get the room number at a coordinate
        '''
        x,y=self.convert(x,y)
        return self.grid_map[self.room,x,y]

    def get_area(self):
        '''
        returns a dict of size of all rooms
        '''
        return self.area
    
    def makegraph(self,grid:np.ndarray=None):
        '''
        returns the graph object of numpy array
        '''
        if grid is None:
            grid=self.grid_map[self.map,:,:]
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

    def shortest_path(self,xs,ys,xd,yd):
        '''
        xs,ys: source x and y numpy coordinates
        xd,yd: destination x and y nnumpy coordinates
        to get shortest distance between source and destination
        returns: list of 1d coordinates
        '''
        #xs,ys = self.convert(xs,ys)
        #xd,yd = self.convert(xd,yd)

        source=self.n*xs+ys
        destination=self.n*xd+yd

        path=nx.shortest_path(self.grid_graph,source=source,target=destination)

        return path

    def room_map(self,room:int):
        '''
        room: the room number
        '''
        bl=self.grid_map[self.room,:,:]==int(room)
        g=np.ones((self.m,self.n))
        g=g*-1
        g[np.where(bl)]=0
        room_map=g
        #x,y=self.convert(x,y)
        #crds=np.ones(1)*(self.n*x+y)
        return MACPP(room_map,num_directions=4)
        


class HAgent:
    def __init__(self,x:int,y:int,speed:int,id:int,map_object:HMap,tasks:list):
        '''
        x: x cartesian coordinate of agent
        y: y cartesian coordinate of agent
        direction: the initial facing direction of agent
        map_object: map object which agent will explore
        tasks: the list of rooms assigned to the agent
        '''
        #start position of agent
        self.startx=x
        self.starty=y
        #current x and y coordinates
        self.x=x
        self.y=y
        self.last=[x,y]
        self.speed=speed
        
        #map object is constructed from map class
        self.grid_map=map_object
        self.l,self.b=self.grid_map.m,self.grid_map.n
        #step counter of every agent
        self.sc=1
        self.depth=5
        #0 for 'search' and 1 for 'return' and 2 for 'rest'
        self.mode=1
        self.lmap=None
        x,y=self.convert(x,y)
        self.last_cell=None
        self.id=0
        self.tasks=tasks
        self.visited=0
        self.num_moves=0
        self.path=[]

    def convert(self,x:int,y:int):
        '''
        To convert cartesian coordinates to numpy matrix coordinates
        '''
        return self.l-y-1,x

    def get_map(self,room:int):
        self.lmap=self.grid_map.room_map(room=room)

    def set_target(self,room:int):
        '''
        choses the nearest door of a room to navigate to,
        if a room as multiple door
        '''
        xs,ys=self.grid_map.convert(self.x,self.y)
        trgts=self.grid_map.doors[room]
        l=float('inf')
        path=None
        target=None
        for tgt in trgts:
            p=self.grid_map.shortest_path(xs,ys,tgt[0],tgt[1])
            if len(p)<l:
                path=p[:]
                target=tgt
                l=len(p)

        self.path=path
        self.target=target
    
    def move(self,direction,crds):
        '''
        move the agent in direction and update it's x,y coordinates in cartesian
        '''
        self.num_moves+=1
        mark=self.lmap.mark(self.x,self.y,crds)
        if mark:
            self.visited+=1
            #check this method
            self.grid_map.mark(self.x,self.y,crds)

        if len(self.lmap.graph):
            #top
            #change get size method
            if direction==0 and self.y+1<self.lmap.get_size()[1]:
                self.y+=1
            #right
            elif direction==1 and self.x+1<self.lmap.get_size()[0]:
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

    def next(self):
        '''
        move the agent to next state
        '''
        #coverage mode
        #self.move()
        if self.mode==0:
            l,b=self.l,self.b
            nx,ny=self.lmap.convert(self.x,self.y)
            crds=np.array([nx*b+ny])
            directions=np.array(self.lmap.get_direction(crds,self.id,self.depth),dtype=float)
            source=crds[self.id]
            am=self.lmap.availablemoves(crds,self.id)
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
            if len(self.lmap.graph)<=np.sum(crds>0) and np.sum(crds==source)>1:
                self.terminate=True
                self.mode=1
                x,y=self.lmap.convert(self.x,self.y)
                return self.b*x+y
            elif len(self.lmap.graph)==1:
                self.move(-1,crds)
                self.mode=1
                x,y=self.lmap.convert(self.x,self.y)
                return self.b*x+y
            if am==0 and len(self.map.graph)!=1:
                raise NameError('Available Actions zero')
            elif am==1:
                for neighbour in self.lmap.graph[source]:
                    if neighbour==source-b:
                        self.move(0,crds)
                        x,y=self.lmap.convert(self.x,self.y)
                        return self.b*x+y
                    elif neighbour==source+b:
                        self.move(2,crds)
                        x,y=self.lmap.convert(self.x,self.y)
                        return self.b*x+y
                    elif neighbour==source+1:
                        self.move(1,crds)
                        x,y=self.lmap.convert(self.x,self.y)
                        return self.b*x+y
                    elif neighbour==source-1:
                        self.move(3,crds)
                        x,y=self.lmap.convert(self.x,self.y)
                        return self.b*x+y
            else:
                while pref:
                    dc=pref.pop(0)
                    if dc==0 and source-b in self.lmap.graph and source in self.lmap.graph[source-b] and source-b!=self.last_cell:
                        self.move(dc,crds)
                        x,y=self.lmap.convert(self.x,self.y)
                        return self.b*x+y
                    elif dc==2 and source+b in self.lmap.graph and source in self.lmap.graph[source+b] and source+b!=self.last_cell:
                        self.move(dc,crds)
                        x,y=self.lmap.convert(self.x,self.y)
                        return self.b*x+y
                    elif dc==1 and source+1 in self.lmap.graph and source in self.lmap.graph[source+1] and source+1!=self.last_cell:
                        self.move(dc,crds)
                        x,y=self.lmap.convert(self.x,self.y)
                        return self.b*x+y
                    elif dc==3 and source-1 in self.lmap.graph and source in self.lmap.graph[source-1] and source-1!=self.last_cell:
                        self.move(dc,crds)
                        x,y=self.lmap.convert(self.x,self.y)
                        return self.b*x+y
                    else:
                        continue
                raise NameError('No action Selected,Source: {}, AM: {}, Original Action Preferences: {}, Pref: {}'.format(source,am,pref,directions))

            return self.b*x+y
        #travel mode
        elif self.mode==1:
            if self.path:
                n=self.path.pop(0)
                x,y=n//self.b,n%self.b
                xg,yg=y,self.l-x-1
                self.x,self.y=xg,yg
                #if it reaches destination it should 
                if not self.path:
                    self.mode=0
                x,y=self.lmap.convert(self.x,self.y)
                return self.b*x+y
            elif self.tasks:
                room=self.tasks.pop(0)
                self.get_map(room)
                d=float('inf')
                xr,yr=self.grid_map.convert(self.x,self.y)
                for xd,yd in self.grid_map.doors[room]:
                    p=self.grid_map.shortest_path(xs=xr,ys=yr,xd=xd,yd=yd)
                    if len(p)<d:
                        d=len(p)
                        self.path=p
                self.path.pop(0)
                n=self.path.pop(0)
                x,y=n//self.b,n%self.b
                xg,yg=y,self.l-x-1
                self.x,self.y=xg,yg
                return self.b*x+y
            else:
                self.mode=2
                x,y=self.grid_map.convert(self.x,self.y)
                #print('*'*30)
                #print('Agent {} is complete'.format(self.id))
                #print('*'*30)
                return self.b*x+y
        #rest mode: after an agent has finished all it's tasks
        elif self.mode==2:
            #do nothing
            return self.b*x+y