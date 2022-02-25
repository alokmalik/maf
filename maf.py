import numpy as np
import json

class Grid:
    def __init__(self,filename):
        with open(filename) as f:
            data=json.load(f)

        m,n=data['dimensions'][0]['rows'],data['dimensions'][0]['columns']

        grid=np.zeros((n,m))

        for d in data['map']:
                x=d['x']
                y=d['y']
                if d['isWall']=='true':
                    grid[x,y]=-1
        self.grid=grid
    
    def return_grid(self):
        return self.grid




class Map:
    def __init__(self,grid,num_directions):
        '''
        input:grid is a numpy array of map
        num_directions: defines if it's a 4 connected grid or 8 connected grid
        '''
        self.map=grid
        #y size, x size
        self.m,self.n=self.map.shape
        assert num_directions==8 , "Grid can be 8 connected only"
        self.nd=num_directions
        self.explorable=np.sum(self.map==0)
        self.explored=0
         
    def convert(self,x,y):
        '''
        To convert cartesian coordinates to numpy matrix coordinates
        '''
        return self.m-y-1,x
    
    def mark(self,x,y,value):
        x,y=self.convert(x,y)
        if self.map[x,y]==0:
            self.explored+=1
        self.map[x,y]=value
    
    def grid(self,x,y):
        '''
        returns the value of x,y coordinate in grid
        '''
        if x>=0 and y>=0 and x<self.n and y<self.m:
            x,y=self.convert(x,y)
            return self.map[x,y]
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
    
        
class MAFMap(Map):
    def __init__(self,grid,num_directions,num_poi):
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
        
    def mark(self,x,y,value):
        '''
        mark a cell with a particular value
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
        returns fraction of map covered
        '''
        return self.explored/self.explorable
    
    def poi_covered(self):
        '''
        returns fraction of points of interest covered 
        '''
        return self.explored_poi/self.num_poi
        
    
    def get_direction(self,x,y):
        '''
        input: x and y coordinate and current direction of agent
        output: available directions and action
        '''
        if self.nd==8:
            unexplored=[]
            unexplored_directions=[]
            explored=[]
            explored_directions=[]
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
            raise NotImplementedError("This method for 4 connected grid hasn't been implemented yet")

    def get_explored(self,x,y):
        '''
        input: x and y coordinate and current direction of agent
        output: direction to go to, value of cell, and action
        '''
        if self.nd==8:
            explored=[]
            explored_directions=[]
            d=0
            for xdir in [-1,0,1]:
                for ydir in [-1,0,1]:
                    value=self.grid(x+xdir,y+ydir)
                    if (xdir!=0 or ydir!=0):
                        #look at only explored cells
                        if value!=0 and value!=-1 and value!=None and value!=-2:
                            explored_directions.append(d)
                            explored.append(value)
                        d+=1
            if explored:
                min_value=min(explored)
                idx=explored.index(min_value)
                return [min_value],[explored_directions[idx]],'back'
            else:
                raise NameError("Impossible scenario, agent can't move anywhere!!!")
        elif self.nd==4:
            raise NotImplementedError("This method for 4 connected grid hasn't been implemented yet")

            
        
        
class Agent:
    def __init__(self,x,y,sc,direction,map_object):
        '''
        input:
        x and y: start position of agent
        direction: initial direction agent is facing
        map_object: The map with agent will explore created from Map class
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
        self.sc=sc
        #map object is constructed from map class
        self.map=map_object
        #0 for 'search' and 1 for 'return'
        self.mode=0
    
    def move(self,move):
        '''moves are {0:bottom left,1:left,2:top left,3:bottom
        4: top, 5: bottom right, 6: right, 7: top right}''' 
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







                    



                

                




            
    
            
    
    
            
            
            
            
        
        
        