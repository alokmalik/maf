from lib2to3.pytree import convert
from tkinter import Y
from macpp import MACPP
import numpy as np
import networkx as nx

class SpiralMap(MACPP):
    def __init__(self,grid:np.ndarray,num_directions:int):
        '''
        initialize spiral map class
        '''
        super().__init__(grid,num_directions)
        self.explorable_space = 0
        self.unexplorable_space = -1
        
        self.spiralmap=self.makeSpiralGrid()
        self.spiralgraph=self.makegraph(self.spiralmap)
        
        self.tree=self.spiralSTC(1,0,0)
        
        

    def getpath(self,sx:int,sy:int,dx:int,dy:int):
        '''
        sx,sy: source x and y coordinates in cartesian coordinates
        dx,dy: destination x and y coordinates in cartesian coordinates
        returns 1d coordinates of the shortest path
        '''
        sx,sy=self.convert(sx,sy)
        dx,dy=self.convert(dx,dy)

        scrd=self.n*sx+sy
        dcrd=self.n*dx+dy

        path=nx.shortest_path(self.graph,source=scrd,target=dcrd)

        return path

    def makeSpiralGrid(self):
        '''
        converts the original map to spiral grid map
        this uses the self.map grid which has cells of 1 to indicate passable terrain and -1 to indicate walls
        "filled" in this function checks if all 4 cell are passable. If they are passable it will add value of 1 (self.explorable) into the spiral grid
        '''
        grid = []

        grid = np.zeros((self.m//2,self.n//2))
        
        
        for ypoint in range(grid.shape[1]):
            for xpoint in range(grid.shape[0]):
                #specify in doc string
                filled = self.map[ypoint*2][xpoint*2] + self.map[ypoint*2][xpoint*2+1] + self.map[ypoint*2+1][xpoint*2] + self.map[ypoint*2+1][xpoint*2+1]
                if filled != 0:
                    grid[ypoint][xpoint] = self.unexplorable_space
        return grid





    def spiralSTC(self,direction:int,startx:int,starty:int):
        '''
        input:spiral graph
        direction == 1 means ccw
        direction == 0 means cw
        returns spiral tree(networkx object) on the spiral graph
        '''


        starty, startx = self.convert(startx,starty)

        assert direction == 1 or direction == 0,  "direction can only be 1 or 0"
        directionsccw = [[0,-1],[1,0],[0,1],[-1,0]]
        directionscw = [[0,1],[1,0],[0,-1],[-1,0]]
        pathdirection = directionsccw

        if direction == 0:
            pathdirection = directionscw
        
        self.spiralmovement(startx,starty,0,[],[],0,direction)

    
    def spiralmovement(self,i:int,j:int,numberpassed:int,path:list,fullpath:list,currentdirection:int,direction:list):
        
        #if numberpassed < self.spiralgraph.number_of_nodes:
        print(self.spiralgraph.number_of_nodes())
        #    pass

        






        


    





s=SpiralMap(np.zeros((100,100)),4)
