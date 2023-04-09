import numpy as np
from maf import MAFMap, Agent,Grid
from brick2 import BNMAgent, BrickMap
import matplotlib.pyplot as plt
import json
from matplotlib import colors
import time
from tqdm import tqdm
import pandas as pd
import os


if not os.path.isdir('bnm_experiments'):
    os.mkdir('bnm_experiments')

size=50
filename=1
fname={}
runs=10
num_agents=20
#edit the dictionary for file name of both maps
fname[0]='maps/room_map_{}.json'.format(size)
fname[1]='maps/new_room_map_{}.json'.format(size)


data=np.zeros((num_agents,runs))
g=Grid(fname[filename])
grid=g.return_grid()
plt.imshow(grid)

plt.imshow(grid)

for r in tqdm(range(runs)):
    visited=np.zeros(shape=(num_agents,num_agents))
    
    for n in tqdm(range(1,num_agents+1)):
        c=True
        while c:
            try:
                g=Grid(fname[filename])
                grid=g.return_grid()
                m=BrickMap(grid,4)
                agents=[]
                x,y=None,None
                while x==None or grid[x,y]!=0:
                    x,y=np.random.randint(0,m.n,size=2)
                for i in range(n):
                    agents.append(BNMAgent(x,y,i,m))
                #print(m.n*x+y)
                crds=[m.n*x+y]*n
                crds=np.array(crds)
                
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
                #print('Run {}, Agents {}, Count {}'.format(r,n,count))
                rcols = ['agent_{}'.format(i) for i  in range(num_agents)]
                rundf = pd.DataFrame(visited, columns = rcols)
                rundf.to_csv('bnm_experiments/bnm_map_{}_size_{}_run_{}_visited.csv'.format(filename,size,r),index=False)


                data[n-1,r]=max(count)
                cols=['Run {}'.format(i) for i in range(runs)]
                df = pd.DataFrame(data, columns = cols)
                df.to_csv('bnm_experiments/results_bnm_map_{}_size_{}.csv'.format(filename,size),index=False)
                c=False
            except:
                print('Run {} with {} agents failed retrying'.format(r,n))
                continue