import numpy as np
import matplotlib.pyplot as plt
from hmap import HAgent, HGrid, HMap
import cv2
from tqdm import tqdm
from task import Baseline, GeneticAlgorithm
from collections import defaultdict
from macpp import MACPPAgent, MACPP
from maf import Grid
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

if not os.path.isdir('hmap_experiments'):
    os.mkdir('hmap_experiments')

size=100
filename=0
fname={}
runs=10
num_agents=5
#edit the dictionary for file name of both maps
fname[0]='room_map_{}.json'.format(size)
fname[1]='new_room_map_{}.json'.format(size)

data=np.zeros((num_agents,runs))
hdata=np.zeros((num_agents,runs))
mutation_probability=.1
population=5


for r in tqdm(range(runs)):
    visited=np.zeros(shape=(num_agents,num_agents))
    hvisited=np.zeros(shape=(num_agents,num_agents))
    for n in tqdm(range(1,num_agents+1)):
        speeds=np.random.randint(low=1,high=10,size=n)
        #hmap part
        speeds=np.flip(np.sort(speeds))
        g=HGrid(fname[filename])
        grid,doors=g.return_grid()
        #plt.imshow(grid[0,:,:])
        

        rooms=np.zeros(len(doors))

        new_doors=defaultdict(list)
        for room in doors.keys():
            rooms[int(room)-1]=np.sum(grid[1,:,:]==int(room))
            new_doors[int(room)]=doors[room].copy()
        doors=new_doors

        g=GeneticAlgorithm(rooms,speeds,population,mutation_probability)

        t,cost,costs=g.optimize(50)

        for i,assignment in enumerate(t):
            for j,_ in enumerate(assignment):
                t[i][j]+=1
        m=HMap(grid,4,doors)
        agents=[]
        x,y=None,None
        while x==None or grid[0,x,y]!=0:
            xc,yc=np.random.randint(0,m.n,size=2)
            x,y=m.convert(xc,yc)

        

        #print(m.n*x+y)
        crds=np.ones(n)*(m.n*x+y)
        for i in range(n):
            agents.append(HAgent(xc,yc,speeds[i],i,m,t[i]))
        
        i=0
        count=np.zeros(num_agents)

        modes=[1]*n
        while m.covered()!=1:
            v=0
            while v<agents[i].speed and agents[i].mode!=2:
                v+=1
                agents[i].next()
                hvisited[n-1,i]=agents[i].visited
            if agents[i].mode!=2:
                count[i]+=1
                hvisited[n-1,i]=agents[i].visited
            i+=1
            i=i%len(agents)

        
        hdata[n-1,r]=max(count)
        rcols = ['agent_{}'.format(i) for i  in range(num_agents)]
        rundf = pd.DataFrame(hvisited, columns = rcols)
        rundf.to_csv('hmap_experiments/hmap_map_{}_size_{}_run_{}_visited.csv'.format(filename,size,r),index=False)
        cols=['Run {}'.format(i) for i in range(runs)]
        df = pd.DataFrame(hdata, columns = cols)
        df.to_csv('hmap_experiments/results_hmap_map_{}_size_{}.csv'.format(filename,size),index=False)


        #macpp part
        g=Grid(fname[filename])
        grid=g.return_grid()
        m=MACPP(grid,4)
        agents=[]
        x,y=None,None
        while x==None or grid[x,y]!=0:
            xc,yc=np.random.randint(0,m.n,size=2)
            x,y=m.convert(xc,yc)

        #print(m.n*x+y)
        crds=np.ones(n)*(m.n*x+y)
        for i in range(n):
            agents.append(MACPPAgent(xc,yc,i,m))
        #print('agents made')
        i=0
        count=np.zeros(num_agents)

        while m.marked_visited()<1:
            s=0
            while agents[i].state() and s<speeds[i]:
                crds[i]=int(agents[i].next(crds))
                visited[n-1,i]=agents[i].visited
                s+=1
            if not agents[i].state:
                visited[n-1,i]=agents[i].visited
            else:
                count[i]+=1
            i+=1
            i%=n
            #if i==0:
            #print(m.marked_visited(),end=' | ')
        data[n-1,r]=max(count)
        rcols = ['agent_{}'.format(i) for i  in range(num_agents)]
        rundf = pd.DataFrame(visited, columns = rcols)
        rundf.to_csv('hmap_experiments/macpp_map_{}_size_{}_run_{}_visited.csv'.format(filename,size,r),index=False)
        cols=['Run {}'.format(i) for i in range(runs)]
        df = pd.DataFrame(data, columns = cols)
        df.to_csv('hmap_experiments/results_macpp_map_{}_size_{}.csv'.format(filename,size),index=False)
