import numpy as np
import json

class Grid:
    '''
    This class loads a map from a JSON file and creates a 2D numpy array representing the map. Each cell of the array can have one of the following values:
    -1: represents a wall in the map
    0: represents an unexplored cell in the map
    1 or higher: represents a cell in the map that has been explored and its value indicates the order in which it was explored.
    '''
    def __init__(self,filename):
        with open(filename) as f:
            data=json.load(f)

        m,n=data['dimensions'][0],data['dimensions'][0]

        grid=np.zeros((n,m))

        for d in data['map']:
                x=d['x']
                y=d['y']
                if d['isWall']==True:
                    grid[x,y]=-1
        self.grid=grid
    
    def return_grid(self):
        return self.grid.astype('int')




class Map:
    '''
    Base class for map for all algorithms
    This class represents a map. It has the following attributes:
    map: a 2D numpy array representing the map
    m: the number of rows in the map
    n: the number of columns in the map
    nd: the number of directions in which the agent can move. It can be 4 or 8.
    explorable: the number of unexplored cells in the map at the beginning
    explored: the number of explored cells in the map
    '''
    def __init__(self,grid:np.ndarray,num_directions:int):
        '''
        input: grid is a numpy array of map
        num_directions: defines if it's a 4 connected grid or 8 connected grid
        '''
        self.map=grid
        #y size, x size
        self.m,self.n=self.map.shape
        assert num_directions==8 or num_directions==4, "Grid can be 8 or 4 connected only"
        self.nd=num_directions
        #keep count of number of unexplored cells
        self.explorable=np.sum(self.map==0)
        #keep count of number of explored cells
        self.explored=0
         
    def convert(self,x:int,y:int):
        '''
        input: x,y cartesian coordinates of a cell in the grid
        output: x,y numpy matrix coordinates of a cell in the grid
        '''
        return self.m-y-1,x
    
    def mark(self,x:int,y:int,value:int):
        '''
        input: x,y cartesian coordinates of a cell in the grid
        value: the counter of the agent
        mark a cell with a particular value
        '''
        x,y=self.convert(x,y)
        if self.map[x,y]==0:
            self.explored+=1
        self.map[x,y]=value
    
    def grid(self,x:int,y:int):
        '''
        input: x,y cartesian coordinates of a cell in the grid
        returns the current value of a cell in the grid
        '''
        if x>=0 and y>=0 and x<self.n and y<self.m:
            x,y=self.convert(x,y)
            return self.map[x,y]
        else:
            return None
    
    def covered(self):
        '''
        returns the covered map fraction
        '''
        return self.explored/self.explorable
    
    def get_size(self):
        '''
        returns the size of grid (X axis length, Y axis length)
        '''
        return self.n,self.m
    
        
class MAFMap(Map):
    '''
    This is class of Map for Multi-Agent Flodding algorithm
    This class represents a map with points of interest. It has the following attributes:
    map: a 2D numpy array representing the map
    m: the number of rows in the map
    n: the number of columns in the map
    nd: the number of directions in which the agent can move. It can be 4 or 8.
    explorable: the number of unexplored cells in the map at the beginning
    explored: the number of explored cells in the map
    num_poi: the number of points of interest in the map
    explored_poi: the number of explored points of interest in the map
    '''
    def __init__(self,grid:np.ndarray,num_directions:int,num_poi:int):
        '''
        grid: the numpy array of map
        num_directions: 4 connected or 8 connected grid
        num_poi: number of points of interest
        '''
        super().__init__(grid,num_directions)
        self.num_poi=num_poi
        if self.num_poi>0:
            self.explored_poi=0
            idxes=np.argwhere(grid == 0)
            poi=np.random.randint(len(idxes),size=self.num_poi)
            for n in range(len(poi)):
                self.map[idxes[poi[n]][0],idxes[poi[n]][1]]=-2
        
    def mark(self,x:int,y:int,value:int):
        '''
        x: x cartesian coordinate of array
        y: y cartesian coordinate of array
        value: the counter of the agent
        return: mark a cell with a particular value
        '''
        x,y=self.convert(x,y)
        if self.map[x,y]==0:
            self.explored+=1
        elif self.map[x,y]==-2:
            self.explored_poi+=1
        elif self.map[x,y]==-1:
            raise NameError('Trying to cover wall cell!!!')

        if self.map[x][y]>value or self.map[x,y]==0:
            self.map[x][y]=value
            return True
        return False

    def covered(self):
        '''
        return: fraction of map covered
        '''
        return self.explored/self.explorable
    
    def poi_covered(self):
        '''
        return: fraction of points of interest covered 
        '''
        return self.explored_poi/self.num_poi
        
    
    def get_direction(self,x:int,y:int):
        '''
        x: x cartesian coordinate of array
        y: y cartesian coordinate of array
        return: available directions and action
        '''
        unexplored=[]
        unexplored_directions=[]
        explored=[]
        explored_directions=[]
        if self.nd==8:
            d=0
            for xdir in [-1,0,1]:
                for ydir in [-1,0,1]:
                    value=self.grid(x+xdir,y+ydir)
                    if (xdir!=0 or ydir!=0):
                        #unexplored case
                        if value==0:
                            unexplored_directions.append(d)
                            unexplored.append(0)
                        #point of interest logic
                        elif value==-2:
                            return [-1],[d],'poi'
                        #marked cell logic
                        elif value!=None and value!=-1:
                            explored.append(value)
                            explored_directions.append(d)
                        d+=1
            if unexplored_directions:
                return unexplored,unexplored_directions,'unexplored'
            elif explored_directions:
                return explored,explored_directions,'explored'
            else:
                raise NameError('Agent is stuck, no cells to move around!!!')
        elif self.nd==4:
            '''
            ToDo: Test this part of code
            '''
            d=0
            for xdir,ydir in zip([0,1,0,-1],[1,0,-1,0]):
                value=self.grid(x+xdir,y+ydir)
                if value==0:
                    unexplored_directions.append(d)
                    unexplored.append(0)
                elif value==-2:
                    return [-1],[d],'poi'
                elif value!=None and value!=-1:
                    explored.append(value)
                    explored_directions.append(d)
                d+=1
            if unexplored_directions:
                return unexplored,unexplored_directions,'unexplored'
            elif explored_directions:
                return explored,explored_directions,'explored'
            else:
                raise NameError('Agent is stuck, no cells to move around!!!')
            
        
        
class Agent:
    '''
    This is class of Agent for Multi-Agent Flodding algorithm
    This class represents an agent. It has the following attributes:
    x: the x coordinate of the agent
    y: the y coordinate of the agent
    d: the direction in which the agent is moving
    sc: the step counter of the agent
    map: the map object which the agent will explore
    mode: the mode of the agent. 0 for 'search' and 1 for 'return'
    '''
    def __init__(self,x:int,y:int,direction:int,map_object:MAFMap):
        '''
        x: x cartesian coordinate of agent
        y: y cartesian coordinate of agent
        direction: the initial facing direction of agent
        map_object: map object which agent will explore
        '''
        #start position of agent
        self.startx=x
        self.starty=y
        #current x and y coordinates
        self.x=x
        self.y=y
        #current direction in which agent is moving
        if not direction:
            d=np.random.randint(8)
        self.d=direction
        #counter of every agent
        self.sc=1
        #map object is constructed from map class
        self.map=map_object
        #0 for 'search' and 1 for 'return'
        self.mode=0
    
    def move(self,move:int):
        '''
        move: direction agent will move to
        moves are {0:bottom left,1:left,2:top left,3:bottom
        4: top, 5: bottom right, 6: right, 7: top right}
        '''
        #top
        if self.d==4 and self.y+1<self.map.get_size()[1]:
            self.y+=1
        #top right
        elif self.d==7 and self.y+1<self.map.get_size()[1] and self.x+1<self.map.get_size()[0]:
            self.y+=1
            self.x+=1
        #right
        elif self.d==6 and self.x+1<self.map.get_size()[0]:
            self.x+=1
        #bottom right
        elif self.d==5 and self.y-1>=0 and self.x+1<self.map.get_size()[0]:
            self.y-=1
            self.x+=1
        #bottom
        elif self.d==3 and self.y-1>=0:
            self.y-=1
        #bottom left
        elif self.d==0 and self.y-1>=0 and self.x-1>=0:
            self.y-=1
            self.x-=1
        #left
        elif self.d==1 and self.x-1>=0:
            self.x-=1
        #top left
        elif self.d==2 and self.y+1<self.map.get_size()[1] and self.x-1>=0:
            self.y+=1
            self.x-=1
        #error
        else:
            raise NameError('Invalid direction input!!')

        if move and self.map.mark(self.x,self.y,self.sc):
            self.sc+=1


    def next(self):
        '''
        move the agent to next state
        '''
        #search mode
        if self.mode==0:
            values,directions,action=self.map.get_direction(self.x,self.y)
            if action=='poi':
                self.mode=1
                self.d=directions[0]
            
            elif action=='unexplored':
                if self.d in directions:
                    x=np.random.random()
                    if x<.05:
                        idx=np.random.randint(len(directions))
                        self.d=directions[idx]
                else:
                    idx=np.random.randint(len(directions))
                    self.d=directions[idx]
            
            elif action=='explored':
                x=np.random.random()
                #move to random direction 5% of time
                if x<.05:
                    idx=np.random.randint(len(directions))
                    self.d=directions[idx]
                #move to highest marked cell
                else:
                    max_value=max(values)
                    idx=values.index(max_value)
                    self.d=directions[idx]
            self.move(True)
            return self.x,self.y
        #return mode
        elif self.mode==1:
            values,directions,action=self.map.get_explored(self.x,self.y)
            self.d=directions[0]
            self.move(False)
            if self.x==self.startx and self.y==self.starty:
                self.mode=0
                self.sc=1
            return self.x,self.y