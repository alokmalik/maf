import numpy as np
from maf import MAFMap, Agent,Grid
from brick2 import BNMAgent, BrickMap
import matplotlib.pyplot as plt
import json
from matplotlib import colors
import time
from tqdm import tqdm
import pandas as pd

filename='room_map_200.json'
num_agents=20
runs=10
data=np.zeros((num_agents,runs))

#plt.imshow(grid)

for r in tqdm(range(runs)):
    for n in tqdm(range(1,num_agents+1)):
        g=Grid(filename)
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
        count=[0 for _ in range(n)]
        while m.marked_visited()<1:
            if agents[i].state():
                crds[i]=int(agents[i].next(crds))
                count[i]+=1
            i+=1
            i%=n
        #print('Run {}, Agents {}, Count {}'.format(r,n,count))
        data[n-1,r]=max(count)
        cols=['Run {}'.format(i) for i in range(runs)]
        df = pd.DataFrame(data, columns = cols)
        df.to_csv('bnm_results_{}'.format(filename),index=False)

