import numpy as np
from macpp import MACPPAgent
from macpp_online import MACPPOnline
from maf import Grid
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

filename='room_map_200.json'
num_agents=20
runs=10
data=np.zeros((num_agents,runs))
g=Grid(filename)
grid=g.return_grid()
plt.imshow(grid)

for r in tqdm(range(runs)):
    for n in tqdm(range(1,num_agents+1)):
        g=Grid(filename)
        grid=g.return_grid()
        m=MACPPOnline(grid,4)
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
        count=[0 for _ in range(n)]
        while m.marked_visited()<1:
            if agents[i].state():
                crds[i]=int(agents[i].next(crds))
                count[i]+=1
            i+=1
            i%=n
            #if i==0:
            #print(m.marked_visited(),end=' | ')
        data[n-1,r]=max(count)
        cols=['Run {}'.format(i) for i in range(runs)]

        df = pd.DataFrame(data, columns = cols)
        df.to_csv('results_onlinemacpp_{}.csv'.format(200),index=False)