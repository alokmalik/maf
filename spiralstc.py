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
        this uses the self.map grid which has cells of 0 to indicate passable terrain and -1 to indicate walls
        "filled" in this function checks if all 0 cell are passable. If they are passable it will add value of 0 (self.explorable) into the spiral grid
        '''
        grid = []

        grid = np.zeros((self.m//2,self.n//2))
        self.map[0][6] = -1
        self.map[2][6] = -1
        self.map[4][6] = -1
        self.map[6][6] = -1
        
        
        for ypoint in range(grid.shape[1]):
            for xpoint in range(grid.shape[0]):
                #specify in doc string
                filled = self.map[ypoint*2][xpoint*2] + self.map[ypoint*2][xpoint*2+1] + self.map[ypoint*2+1][xpoint*2] + self.map[ypoint*2+1][xpoint*2+1]
                if filled != 0:
                    grid[ypoint][xpoint] = self.unexplorable_space
        print(grid)
        return grid





    def spiralSTC(self,direction:int,x:int,y:int):
        '''
        input:spiral graph
        direction == 1 means ccw
        direction == 0 means cw
        returns spiral tree(networkx object) on the spiral graph
        '''


        startx = self.spiralmap.shape[0]-y-1
        starty = x

        assert direction == 1 or direction == 0,  "direction can only be 1 or 0"
        directionsccw = [[0,-1],[1,0],[0,1],[-1,0]]
        directionscw = [[0,1],[1,0],[0,-1],[-1,0]]
        pathdirection = directionsccw

        if direction == 0:
            pathdirection = directionscw
        
        path,fullpath,numberpassed = self.spiralmovement(startx,starty,0,[],[],-1,pathdirection,[])
        print(path)
        print(len(path))
        print(self.spiralgraph.number_of_nodes())
        print(fullpath)
        return path,fullpath

    
    def spiralmovement(self,xpos:int,ypos:int,numberpassed:int,path:list,fullpath:list,currentdirection:int,direction:list,trypath:list):
        '''
        recursive functions to find the spiral tree path

        returns 3 values:
        path = non backtracking tree path 
        fullpath = backtracking tree path for when a second branch of the tree is needed
        numberpassed = value to indicate total number of unique nodes covered (used only to make sure all point in graph passed)
        
        '''
        
        
        node_number = xpos*self.spiralmap.shape[0]+ypos
        
        trypath.append(node_number)
        
        if numberpassed >= self.spiralgraph.number_of_nodes():
            return path,fullpath,numberpassed
        
        if xpos < self.spiralmap.shape[1] and xpos >= 0 and ypos < self.spiralmap.shape[0] and ypos >= 0:
            if self.spiralgraph.has_node(node_number):
                if self.spiralgraph.nodes[node_number]["explored"] == 0:
                    #print("to be passed")
                    numberpassed += 1

                    self.spiralgraph.nodes[node_number]["explored"] = 1
                    path.append(node_number)
                    fullpath.append(node_number)

                    currentdirection += 1
                    newx = xpos + direction[currentdirection%4][1]
                    newy = ypos + direction[currentdirection%4][0]
                    path,fullpath,numberpassed=self.spiralmovement(newx,newy,numberpassed,path,fullpath,currentdirection,direction,trypath)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)

                    currentdirection += 1
                    newx = xpos + direction[currentdirection%4][1]
                    newy = ypos + direction[currentdirection%4][0]
                    path,fullpath,numberpassed=self.spiralmovement(newx,newy,numberpassed,path,fullpath,currentdirection,direction,trypath)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)

                    currentdirection += 1
                    newx = xpos + direction[currentdirection%4][1]
                    newy = ypos + direction[currentdirection%4][0]
                    path,fullpath,numberpassed=self.spiralmovement(newx,newy,numberpassed,path,fullpath,currentdirection,direction,trypath)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)

                    currentdirection += 1
                    newx = xpos + direction[currentdirection%4][1]
                    newy = ypos + direction[currentdirection%4][0]
                    path,fullpath,numberpassed=self.spiralmovement(newx,newy,numberpassed,path,fullpath,currentdirection,direction,trypath)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes()-1:
                        fullpath.append(node_number)

        
        
        return path,fullpath,numberpassed
        

        






        


    





s=SpiralMap(np.zeros((10,10)),4)
