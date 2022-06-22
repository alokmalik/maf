import numpy as np
from macpp import MACPPAgent, MACPP
from maf import Grid
import pandas as pd
from tqdm import tqdm

filename='room_map_100.json'
num_agents=20
runs=10
data=np.zeros((num_agents,runs))

for r in tqdm(range(runs)):
    for n in tqdm(range(1,num_agents+1)):
        g=Grid(filename)
        grid=g.return_grid()
        m=MACPP(grid,4)
        agents=[]
        x,y=m.convert(0,0)
        #print(m.n*x+y)
        crds=np.ones(n)*(m.n*x+y)
        for i in range(n):
            agents.append(MACPPAgent(0,0,i,m))
        #print('agents made')
        i=0
        count=[0 for _ in range(n)]
        while m.marked_visited()<1:
            if agents[i].state():
                crds[i]=agents[i].next(crds)
                count[i]+=1
            i+=1
            i%=n
            #if i==0:
            #print(m.marked_visited(),end=' | ')
        data[n-1,r]=max(count)

cols=['Run {}'.format(i) for i in range(runs)]

df = pd.DataFrame(data, columns = cols)
df.to_csv('results_{}'.format(filename),index=False)