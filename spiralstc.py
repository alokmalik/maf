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
        '''
        grid = []

        grid = np.zeros((self.m//2,self.n//2))
        
        
        for ypoint in range(grid.shape[1]):
            for xpoint in range(grid.shape[0]):
                #specify in doc string
                filled = self.map[ypoint*2][xpoint*2] + self.map[ypoint*2][xpoint*2+1] + self.map[ypoint*2+1][xpoint*2] + self.map[ypoint*2+1][xpoint*2+1]
                if filled == 4 * self.explorable_space:
                    grid[ypoint][xpoint] = self.explorable_space
                else:
                    grid[ypoint][xpoint] = self.unexplorable_space
                
        return grid





    def spiralSTC(self,):
        '''
        input:spiral graph
        returns spiral tree(networkx object) on the spiral graph
        '''
        pass

    





s=SpiralMap(np.ones((100,100)),4)
