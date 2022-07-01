import numpy as np
#from maf import Grid
import matplotlib.pyplot as plt
import json
from matplotlib import colors
import time
from lib2to3.pytree import convert
from tkinter import Y
from macpp import MACPP
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import random
print(sys.setrecursionlimit(10000000))

class SpiralMap(MACPP):
    def __init__(self,grid:np.ndarray,num_directions:int,agent_count:int,point:int,startx:int,starty:int):
        '''
        initialize spiral map class
        '''
        super().__init__(grid,num_directions)
        
        self.agent_count = agent_count
        self.explorable_space = 0
        self.unexplorable_space = -1
        self.point = point
        self.spiralmap=self.makeSpiralGrid()
        self.spiralgraph=self.makegraph(self.spiralmap)
        
        val = self.spiralmap[startx][starty]
        while val == -1:
            print("invalid start point generating new one")
            startx = random.randint(0,self.map.shape[0]-1)
            starty = random.randint(0,self.map.shape[0]-1)
            val = self.spiralmap[startx][starty]
        self.point = [startx,starty]
        
        
        self.tree,self.fulltree,self.fulltree_points=self.spiralSTC(1,startx,starty)
        
        self.coverage = self.pathcoverage()
        tmpx = []
        tmpy = []
        for i in self.coverage:
            tmpx.append(i[1])
            tmpy.append(16-i[0])
        plt.plot(tmpx,tmpy)
        plt.show()
        plt.savefig('./data/coverage'+str(self.point)+"_agent_"+str(self.agent_count)+'.png')
        
        self.brokendownpath = self.pathbreakdown(startx,starty)
        self.Plengths = []
        self.agent_travel_length = []
        self.agent_visit_length = []
        self.longest = 0
        self.savejson()
        self.plot_paths()
        self.agent_generation()
        


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
        
        
        for ypoint in range(grid.shape[1]):
            for xpoint in range(grid.shape[0]):
                #specify in doc string
                filled = self.map[ypoint*2][xpoint*2] + self.map[ypoint*2][xpoint*2+1] + self.map[ypoint*2+1][xpoint*2] + self.map[ypoint*2+1][xpoint*2+1]
                if filled != 0:
                    grid[ypoint][xpoint] = self.unexplorable_space
        return grid

    def spiralSTC(self,direction:int,x:int,y:int):
        '''
        input:spiral graph
        direction == 1 means ccw
        direction == 0 means cw
        after getting paths back from sprialmovement this function generates the final return path and removed duplicates
        returns spiral tree(networkx object) on the spiral graph
        '''

        startx = x
        starty = y
        
        assert direction == 1 or direction == 0,  "direction can only be 1 or 0"
        directionsccw = [[0,-1],[1,0],[0,1],[-1,0]]
        npdirectionsccw = [[-1,0],[0,1],[1,0],[0,-1]]
        alternate = [[0,1],[-1,0],[0,-1],[1,0]]
        directionscw = [[0,1],[1,0],[0,-1],[-1,0]]
        npdirectionscw = [[1,0],[0,1],[-1,0],[0,-1]]
        npcw = [[0,1],[1,0],[0,-1],[-1,0]]
        paper = [[0,-1],[-1,0],[0,1],[1,0]]
        invertpaper = [[0,1],[-1,0],[0,-1],[1,0]]
        
        
        
        pathdirection = paper

        if direction == 0:
            pathdirection = directionscw
        
        path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates = self.spiralmovement(startx,starty,0,[],[],[],0,pathdirection,[],[],[])
        
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
        tmpx = []
        tmpy = []
        for i in fullpathcoordinate:
            tmpx.append(i[1])
            tmpy.append(8-i[0])
        plt.plot(tmpx,tmpy)
        
        plt.show()
        plt.savefig('./data/fullpathcoordinate'+str(self.point)+"_agent_"+str(self.agent_count)+'.png')
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

                    newx = xpos + direction[(currentdirection+1)%4][0]
                    newy = ypos + direction[(currentdirection+1)%4][1]
                    path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates=self.spiralmovement(newx,newy,numberpassed,path,fullpath,fullpathcoordinate,currentdirection+1,direction,trypath,returnpath,returnpathcoordinates)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)
                        fullpathcoordinate.append([xpos,ypos])
                    elif fullpath[-1] != node_number:
                        returnpath.append(node_number)
                        returnpathcoordinates.append([xpos,ypos])

                    newx = xpos + direction[(currentdirection-1)%4][0]
                    newy = ypos + direction[(currentdirection-1)%4][1]
                    path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates=self.spiralmovement(newx,newy,numberpassed,path,fullpath,fullpathcoordinate,currentdirection-1,direction,trypath,returnpath,returnpathcoordinates)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)
                        fullpathcoordinate.append([xpos,ypos])
                    elif fullpath[-1] != node_number:
                        returnpath.append(node_number)
                        returnpathcoordinates.append([xpos,ypos])

                    newx = xpos + direction[(currentdirection-2)%4][0]
                    newy = ypos + direction[(currentdirection-2)%4][1]
                    path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates=self.spiralmovement(newx,newy,numberpassed,path,fullpath,fullpathcoordinate,currentdirection-2,direction,trypath,returnpath,returnpathcoordinates)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)
                        fullpathcoordinate.append([xpos,ypos])
                    elif fullpath[-1] != node_number:
                        returnpath.append(node_number)
                        returnpathcoordinates.append([xpos,ypos])

                    newx = xpos + direction[(currentdirection-3)%4][0]
                    newy = ypos + direction[(currentdirection-3)%4][1]
                    path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates=self.spiralmovement(newx,newy,numberpassed,path,fullpath,fullpathcoordinate,currentdirection-3,direction,trypath,returnpath,returnpathcoordinates)
                    if fullpath[-1] != node_number and numberpassed < self.spiralgraph.number_of_nodes():
                        fullpath.append(node_number)
                        fullpathcoordinate.append([xpos,ypos])
                    elif fullpath[-1] != node_number:
                        returnpath.append(node_number)
                        returnpathcoordinates.append([xpos,ypos])


        
        
        return path,fullpath,numberpassed,fullpathcoordinate,returnpath,returnpathcoordinates

    def pathcoverage(self):
        '''
        takes the tree path and generates the coverage path based on that
        returns 1 agent coverage path
        '''
        


        prev = [0,0]
        current = [0,0]
        coveragepath = []
        prevdiff = [0,0]
        for i in range(len(self.fulltree)):
            nextpoint = i + 1
            prevpoint = i - 1
            if i == 0:
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
        if coveragepath[1] == [-1]:
            coveragepath.pop()
        if coveragepath[0] == coveragepath[-1]:
            coveragepath.pop()
        return coveragepath
                
    def pathbreakdown(self,startx:int,starty:int):
        '''
        take the start point of the agents and with the tree generates the path that each agent must take for its coverage
        returns path of each agent for both the its travel and visit
        '''
        agentbasic = self.agentbase()
        fullagent,direction = self.agentstart(agentbasic,startx,starty)
        
        finalpath = []
        for i in range(len(agentbasic)):
            
            if direction[i] == 0:
                agentbasic[i].reverse()
            if agentbasic[i][1] == agentbasic[i][-1]:
                agentbasic[i].pop()
            if agentbasic[i][0] == agentbasic[i][-1]:
                agentbasic[i].pop()
            finalpath.append([fullagent[i],agentbasic[i]])
        return finalpath
    
    def agentbase(self):
        '''
        breaks down the path of each agent
        returns pathlist of the agents 
        '''
        
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
        
        '''
        calculates djikstras shortest path to the start point of each agent (wether it be start or end)
        returns path to agent start points
        '''
        
        nodepathlist = []
        pathlist = []
        direction = []
        
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
        
        
        return pathlist,direction
                
    
    def plot_paths(self):
        '''
        plotting and saving paths
        no returns
        '''
        
        for i in range(self.agent_count):
            tmpx = []
            tmpy = []
            for j in self.brokendownpath[i][1]:
                tmpx.append(j[0])
                tmpy.append(j[1])
            plt.plot(tmpx,tmpy)
        plt.show()
        plt.savefig('./data/brokendownpath'+str(self.point)+"_agent_"+str(self.agent_count)+'.png')
        
        for i in range(self.agent_count):
            tmpx = []
            tmpy = []
            for j in self.brokendownpath[i][1]:
                tmpx.append(j[1])
                tmpy.append(self.map.shape[0]-j[0])
            plt.plot(tmpx,tmpy)
            tmpx = []
            tmpy = []
            for j in self.brokendownpath[i][0]:
                tmpx.append(j[1])
                tmpy.append(self.map.shape[0]-j[0])
            plt.plot(tmpx,tmpy,color = 'red')
        plt.savefig('./data/allbrokenpaths'+str(self.point)+"_agent_"+str(self.agent_count)+'.png')
        plt.show()
        
    def agent_generation(self):
        
        '''
        generates agents and moves across their paths to generate heatmap
        '''
        agents = []
        agentsteps = []
        steppoints = []
        for i in range(self.agent_count):
            instance = Agent(i,self.map,self.brokendownpath)
            agents.append(instance)
        
        coverage = 0
        while coverage != self.agent_count:
            coverage = 0
            for i in range(self.agent_count):
                agents[i].next()
                coverage += agents[i].percent_covered
            
        for i in range(self.agent_count):
            steppoints = steppoints + agents[i].travelled + agents[i].covered
            steps = len(agents[i].travelled) + len(agents[i].covered)
            agentsteps.append(steps)
        
        stepmap = np.zeros((self.map.shape[0],self.map.shape[1]))
        for i in steppoints:
            stepmap[i[0]][i[1]] += 1
        
        df = pd.DataFrame(stepmap)
        
        p1 = sns.heatmap(stepmap)
        plt.savefig('./data/heatmaps'+str(self.point)+"_agent_"+str(self.agent_count)+'.png')
        plt.show()
        
    def savejson(self):
        
        '''
        json data for path generation
        return json data
        '''
        dictionarypaths = {}
        
        for i in range(self.agent_count):
            lenval = len(self.brokendownpath[i][0]) + len(self.brokendownpath[i][1])
            self.agent_travel_length.append(len(self.brokendownpath[i][0]))
            self.agent_visit_length.append(len(self.brokendownpath[i][1]))
            self.Plengths.append(lenval)
            if lenval > self.longest:
                self.longest = lenval
            dictionarypaths.update({str(i):self.brokendownpath[i]})
        with open("./data/json"+str(self.point)+"_agent_"+str(self.agent_count)+".json", "w") as outfile:
            json.dump(dictionarypaths, outfile)
        return dictionarypaths
        
    def returndata(self):
        '''
        returns data give travel data per agent and total
        '''
        return self.Plengths,self.longest,self.agent_travel_length,self.agent_visit_length
    
    
class Agent():
    def __init__(self,id:int,map,path:list):
        self.id = id
        self.map = map
        self.mode = 0
        self.path = path[self.id]
        self.travelled = []
        self.covered = []
        self.percent_covered = 0
        self.travel_len = len(self.path[1])
        self.x = self.path[self.mode][0][0]
        self.y = self.path[self.mode][0][1]
    
    def next(self):
        '''
        moves across agent paths to the next phase
        '''
        if self.mode == 0:
            if len(self.path[self.mode]) <= 1:
                self.mode = 1
                
            else:
                self.travelled.append([self.x,self.y])
                self.path[self.mode].pop(0)
                self.x = self.path[self.mode][0][0]
                self.y = self.path[self.mode][0][1]
                
                
        if self.mode == 1:
            if len(self.path[self.mode]) == 1:
                self.mode = 2
                self.covered.append([self.x,self.y])
                self.percent_covered = len(self.covered)/self.travel_len
            
            else:
                self.covered.append([self.x,self.y])
                self.percent_covered = len(self.covered)/self.travel_len
                self.path[self.mode].pop(0)
                self.x = self.path[self.mode][0][0]
                self.y = self.path[self.mode][0][1]
                
                      
class HGrid:
    def __init__(self,filename):
        
        '''
        pulls data from json file
        '''
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
        plt.plot(grid[0,:,:])
        self.grid=grid
        self.doors=data['doors']
        
    def return_grid(self):
        return self.grid,self.doors
    
g=HGrid('room_map.json')


grid,doors=g.return_grid()
values = []
ticki = 0
tickj = 0
print(grid)
print(grid[0])
print(grid[0][0])
print(len(grid[0]))
for i in grid[0]:
    tickj = 0
    for j in i:
        if j != -1:
            values.append([100-tickj-1,ticki])
        tickj += 1
    ticki += 1

plt.imshow(grid[0,:,:])
tick = 0
total = []
run_data = []
final_run_data = []
final_travel_data = []
final_visit_data = []
   

map = "new"
size = "50"


for i in range(10):
    current = []
    averaged = 0
    final_run_data = []
    final_travel_data = []
    final_visit_data = []
    rand = random.choice(values)
    for j in range(1,21):
        
        s=SpiralMap(grid[0,:,:],4,j,tick,20,18)
        lens,longest,travel,visit = s.returndata()
        final_run_data.append(lens)
        final_travel_data.append(travel)
        final_visit_data.append(visit)
        current.append(longest)
        averaged += longest
    for k in final_run_data:
        while len(k) < 20:
            k.append(0)
    for k in final_run_data:
        while len(k) < 20:
            final_travel_data.append(0)
    for k in final_run_data:
        while len(k) < 20:
            final_visit_data.append(0)
    data = pd.DataFrame(final_run_data)
    data.to_csv('./data/map_'+map+'_size_'+size+'_run_'+str(i)+'_total.csv', index=False)
    data = pd.DataFrame(final_visit_data)
    data.to_csv('./data/map_'+map+'_size_'+size+'_run_'+str(i)+'_explored.csv', index=False)
    data = pd.DataFrame(final_travel_data)
    data.to_csv('./data/map_'+map+'_size_'+size+'_run_'+str(i)+'_visit.csv', index=False)
    averaged = averaged/len(current)
    current.append(averaged)
    total.append(current)
data = pd.DataFrame(total)
data.to_csv('./data/longest.csv', index=False)
print("end")
