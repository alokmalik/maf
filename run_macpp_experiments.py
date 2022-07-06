import numpy as np
from macpp import MACPPAgent, MACPP
from maf import Grid
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

if not os.path.isdir('offline_experiments'):
    os.mkdir('offline_experiments')

size=50
filename=0
fname={}
runs=10
num_agents=20
#edit the dictionary for file name of both maps
fname[0]='room_map_{}.json'.format(size)
fname[1]='new_room_map_{}.json'.format(size)

data=np.zeros((num_agents,runs))
g=Grid(fname[filename])
grid=g.return_grid()
plt.imshow(grid)

for r in tqdm(range(runs)):
    visited=np.zeros(shape=(num_agents,num_agents))
    for n in tqdm(range(1,num_agents+1)):
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
            if agents[i].state():
                crds[i]=int(agents[i].next(crds))
                count[i]+=1
                visited[n-1,i]=agents[i].visited
            else:
                visited[n-1,i]=agents[i].visited
            i+=1
            i%=n
            #if i==0:
            #print(m.marked_visited(),end=' | ')
        data[n-1,r]=max(count)
        rcols = ['agent_{}'.format(i) for i  in range(num_agents)]
        rundf = pd.DataFrame(visited, columns = rcols)
        rundf.to_csv('offline_experiments/offline_map_{}_size_{}_run_{}_visited.csv'.format(filename,size,r),index=False)
        cols=['Run {}'.format(i) for i in range(runs)]
        df = pd.DataFrame(data, columns = cols)
        df.to_csv('offline_experiments/results_offline_map_{}_size_{}.csv'.format(filename,size),index=False)