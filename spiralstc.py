from lib2to3.pytree import convert
from tkinter import Y
from macpp import MACPP
import numpy as np
import networkx as nx

class SpiralMap(MACPP):
    def __init__(self,grid:np.ndarray,num_directions:int,agent_count:int,startx:int,starty:int):
        '''
        initialize spiral map class
        '''
        super().__init__(grid,num_directions)
        self.agent_count = agent_count
        self.explorable_space = 0
        self.unexplorable_space = -1
        
        self.spiralmap=self.makeSpiralGrid()
        self.spiralgraph=self.makegraph(self.spiralmap)
        
        self.tree,self.fulltree,self.fulltree_points=self.spiralSTC(1,startx,starty)
        self.coverage = self.pathcoverage()
        print("full 1 agent path: ",self.coverage)
        brokendownpath = self.pathbreakdown(startx,starty)


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
        #self.map[0][6] = -1
        #self.map[2][6] = -1
        #self.map[4][6] = -1
        #self.map[6][6] = -1
        
        
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
        after getting paths back from sprialmovement this function generates the final return path and removed duplicates
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
        
        path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates = self.spiralmovement(startx,starty,0,[],[],[],-1,pathdirection,[],[],[])
        
        newreturnpath = []
        newreturnpathcoordinates = []
        previous = -1
        for i in returnpath:
            if i != previous:
                newreturnpath.append(i)
            previous = i
        previous = [-1,-1]
        for i in returnpathcoordinates:
            if i != previous:
                newreturnpathcoordinates.append(i)
            previous = i
        
        finalreturnpath = []
        finalreturncoordinates = []
        notremoved = 0
        for i in newreturnpath:
            if i not in finalreturnpath:
                finalreturnpath.append(i)
                notremoved = 1
            elif notremoved == 1:
                finalreturnpath.pop()
                notremoved = 0
        notremoved = 0
        for i in newreturnpathcoordinates:
            if i not in finalreturncoordinates:
                finalreturncoordinates.append(i)
                notremoved = 1
            elif notremoved == 1:
                finalreturncoordinates.pop()
                notremoved = 0
        
        fullpath = fullpath + finalreturnpath
        fullpathcoordinate = fullpathcoordinate + finalreturncoordinates
        return path,fullpath,fullpathcoordinate
    
    def spiralmovement(self,xpos:int,ypos:int,numberpassed:int,path:list,fullpath:list,fullpathcoordinate:list,currentdirection:int,direction:list,trypath:list,returnpath:list,returnpathcoordinates:list):
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
            return path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates
        
        if xpos < self.spiralmap.shape[1] and xpos >= 0 and ypos < self.spiralmap.shape[0] and ypos >= 0:
            if self.spiralgraph.has_node(node_number):
                if self.spiralgraph.nodes[node_number]["explored"] == 0:
                    numberpassed += 1

                    self.spiralgraph.nodes[node_number]["explored"] = 1
                    path.append(node_number)
                    fullpath.append(node_number)
                    fullpathcoordinate.append([xpos,ypos])

                    currentdirection += 1
                    newx = xpos + direction[currentdirection%4][1]
                    newy = ypos + direction[currentdirection%4][0]
                    path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates=self.spiralmovement(newx,newy,numberpassed,path,fullpath,fullpathcoordinate,currentdirection,direction,trypath,returnpath,returnpathcoordinates)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)
                        fullpathcoordinate.append([xpos,ypos])
                    elif fullpath[-1] != node_number:
                        returnpath.append(node_number)
                        returnpathcoordinates.append([xpos,ypos])

                    currentdirection += 1
                    newx = xpos + direction[currentdirection%4][1]
                    newy = ypos + direction[currentdirection%4][0]
                    path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates=self.spiralmovement(newx,newy,numberpassed,path,fullpath,fullpathcoordinate,currentdirection,direction,trypath,returnpath,returnpathcoordinates)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)
                        fullpathcoordinate.append([xpos,ypos])
                    elif fullpath[-1] != node_number:
                        returnpath.append(node_number)
                        returnpathcoordinates.append([xpos,ypos])

                    currentdirection += 1
                    newx = xpos + direction[currentdirection%4][1]
                    newy = ypos + direction[currentdirection%4][0]
                    path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates=self.spiralmovement(newx,newy,numberpassed,path,fullpath,fullpathcoordinate,currentdirection,direction,trypath,returnpath,returnpathcoordinates)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)
                        fullpathcoordinate.append([xpos,ypos])
                    elif fullpath[-1] != node_number:
                        returnpath.append(node_number)
                        returnpathcoordinates.append([xpos,ypos])

                    currentdirection += 1
                    newx = xpos + direction[currentdirection%4][1]
                    newy = ypos + direction[currentdirection%4][0]
                    path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates=self.spiralmovement(newx,newy,numberpassed,path,fullpath,fullpathcoordinate,currentdirection,direction,trypath,returnpath,returnpathcoordinates)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes()-1:
                        fullpath.append(node_number)
                        fullpathcoordinate.append([xpos,ypos])
                    elif fullpath[-1] != node_number:
                        returnpath.append(node_number)
                        returnpathcoordinates.append([xpos,ypos])


        
        
        return path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates

    def pathcoverage(self):
        
        


        prev = [0,0]
        current = [0,0]
        coveragepath = []
        prevdiff = [0,0]
        #print(self.spiralgraph)
        for i in range(len(self.fulltree)):
            nextpoint = i + 1
            prevpoint = i - 1
            if i == 0:
                #invert current[1] and fulltree
                current = self.fulltree_points[i]
                diff = [self.fulltree_points[nextpoint][0] - current[0],self.fulltree_points[nextpoint][1] - current[1]]
                

                if diff[0] == 1:
                    currentP = [self.fulltree_points[nextpoint][0] * 2,self.fulltree_points[nextpoint][1] * 2]
                    prevP = [self.fulltree_points[i][0] * 2,self.fulltree_points[i][1] * 2]
                    diffP = [currentP[0] - prevP[0],currentP[1] - prevP[1]]
                    coveragepath.append([prevP[0],prevP[1]])
                
                elif diff[0] == -1:
                    currentP = [self.fulltree_points[nextpoint][0] * 2 + 1,self.fulltree_points[nextpoint][1] * 2 + 1]
                    prevP = [self.fulltree_points[i][0] * 2 + 1,self.fulltree_points[i][1] * 2 + 1]
                    diffP = [currentP[0] - prevP[0],currentP[1] - prevP[1]]
                    coveragepath.append([prevP[0],prevP[1]])

                elif diff[1] == 1:
                    currentP = [self.fulltree_points[nextpoint][0] * 2 + 1,self.fulltree_points[nextpoint][1] * 2]
                    prevP = [self.fulltree_points[i][0] * 2 + 1,self.fulltree_points[i][1] * 2]
                    diffP = [currentP[0] - prevP[0],currentP[1] - prevP[1]]
                    coveragepath.append([prevP[0],prevP[1]])

                elif diff[1] == -1:
                    currentP = [self.fulltree_points[nextpoint][0] * 2,self.fulltree_points[nextpoint][1] * 2 + 1]
                    prevP = [self.fulltree_points[i][0] * 2,self.fulltree_points[i][1] * 2 + 1]
                    diffP = [currentP[0] - prevP[0],currentP[1] - prevP[1]]
                    coveragepath.append([prevP[0],prevP[1]])
                
            
            else:
                nextpoint = i + 1
                prevpoint = i - 1
                current = self.fulltree_points[i]
                diff = [current[0] - self.fulltree_points[prevpoint][0],current[1] - self.fulltree_points[prevpoint][1]]
                
                if diff[0] == 1:
                    currentP = [self.fulltree_points[i][0] * 2,self.fulltree_points[i][1] * 2]
                    prevP = coveragepath[-1]
                    diffP = [currentP[0] - prevP[0],currentP[1] - prevP[1]]
                    if prevdiff[0] == diff[0] * -1:

                        coveragepath.append([prevP[0]-1,prevP[1]])
                        coveragepath.append([prevP[0]-1,prevP[1]-1])
                        diffP = [currentP[0] - coveragepath[-1][0],currentP[1] - currentP[1]]
                        prevP = coveragepath[-1]
                    elif prevdiff[1] == -1:
                        coveragepath.append([prevP[0],prevP[1]-1])
                    for j in range(1, diffP[0] + 1):
                        coveragepath.append([prevP[0] + j,currentP[1]])

                elif diff[0] == -1:
                    currentP = [self.fulltree_points[i][0] * 2 + 1,self.fulltree_points[i][1] * 2 + 1]
                    prevP = coveragepath[-1]
                    diffP = [currentP[0] - prevP[0],currentP[1] - prevP[1]]
                    if prevdiff[0] == diff[0] * -1:

                        coveragepath.append([prevP[0]+1,prevP[1]])
                        coveragepath.append([prevP[0]+1,prevP[1]+1])
                        diffP = [currentP[0] - coveragepath[-1][0],currentP[1] - currentP[1]]
                        prevP = coveragepath[-1]
                    elif prevdiff[1] == 1:
                        coveragepath.append([prevP[0],prevP[1]+1])
                    for j in range(abs(diffP[0])):
                        coveragepath.append([prevP[0] - j -1,currentP[1]])
                        
                elif diff[1] == 1:
                    currentP = [self.fulltree_points[i][0] * 2 + 1,self.fulltree_points[i][1] * 2]
                    prevP = coveragepath[-1]
                    diffP = [currentP[0] - prevP[0],currentP[1] - prevP[1]]
                    if prevdiff[1] == diff[1] * -1:
                        coveragepath.append([prevP[0],prevP[1]-1])
                        coveragepath.append([prevP[0]+1,prevP[1]-1])
                        diffP = [currentP[0] - currentP[0],currentP[1] - coveragepath[-1][1]]
                        prevP = coveragepath[-1]
                    elif prevdiff[0] == 1:
                        coveragepath.append([prevP[0]+1,prevP[1]])
                    for j in range(1, diffP[1] + 1):
                        coveragepath.append([currentP[0],prevP[1]+j])

                elif diff[1] == -1:
                    currentP = [self.fulltree_points[i][0] * 2,self.fulltree_points[i][1] * 2 + 1]
                    prevP = coveragepath[-1]
                    diffP = [currentP[0] - prevP[0],currentP[1] - prevP[1]]
                    if prevdiff[1] == diff[1] * -1:
                        coveragepath.append([prevP[0],prevP[1]+1])
                        coveragepath.append([prevP[0]-1,prevP[1]+1])
                        diffP = [currentP[0] - currentP[0],currentP[1] - coveragepath[-1][1]]
                        prevP = coveragepath[-1]
                    elif prevdiff[0] == -1:
                        coveragepath.append([prevP[0]-1,prevP[1]])
                    for j in range(abs(diffP[1])):
                        coveragepath.append([currentP[0],prevP[1]-j-1])
            prev = current
            prevP = currentP
            prevdiff = diff
        coveragepath.append([currentP[0]+diff[0],currentP[1]+diff[1]])
        return coveragepath
                
    def pathbreakdown(self,startx:int,starty:int):
        agentbasic = self.agentbase()
        fullagent,direction = self.agentstart(agentbasic,startx,starty)
        
        finalpath = []
        for i in range(len(agentbasic)):
            if direction[i] == 0:
                agentbasic[i].reverse()
            fullagent[i].pop()
            finalpath.append(fullagent[i]+agentbasic[i])
        print("done")
    def agentbase(self):
        pathlist = []
        
        peragentpoints = []
        for i in range(self.agent_count):
            peragentpoints.append(len(self.coverage)//self.agent_count)
        for i in range(len(self.coverage)%self.agent_count):
            peragentpoints[i] += 1
        tmp_coverage = self.coverage
        for i in peragentpoints:
            j = i
            cumul = []
            while j > 0:
                cumul.append(tmp_coverage[0])
                tmp_coverage.pop(0)
                j -= 1
            pathlist.append(cumul)
        
        
        
        return pathlist
    
    def agentstart(self,agentbasic:list,x:int,y:int):
        print(agentbasic)
        nodepathlist = []
        pathlist = []
        direction = []
        print(self.graph)
        
        speed2start = []
        
        startx = self.spiralmap.shape[0]-y-1
        starty = x
        start_node = agentbasic[0][0][1] + agentbasic[0][0][0] * (self.spiralmap.shape[1] * 2)
        
        for i in range(self.agent_count):
            forward_node = agentbasic[i][0][1] + agentbasic[i][0][0] * self.spiralmap.shape[1] * 2
            back_node = agentbasic[i][-1][1] + agentbasic[i][-1][0] * self.spiralmap.shape[1] * 2
            forward_s = nx.dijkstra_path(self.graph,start_node,forward_node)
            back_s = nx.dijkstra_path(self.graph,start_node,back_node)
            if len(forward_s) <= len(back_s):
                nodepathlist.append(forward_s)
                direction.append(1)
            else:
                nodepathlist.append(back_s)
                direction.append(0)
        
        for i in nodepathlist:
            tmplist = []
            for j in i:
                tmplist.append([j//(self.spiralmap.shape[1] * 2),j%(self.spiralmap.shape[1] * 2)])
            pathlist.append(tmplist)
        
        print(pathlist)
                
        
        
        return pathlist,direction
                

        

        






        


    





s=SpiralMap(np.zeros((8,8)),4,3,1,3)
