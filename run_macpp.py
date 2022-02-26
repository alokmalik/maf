import numpy as np
from macpp import MACPPAgent, MACPP
from maf import Grid
import matplotlib.pyplot as plt
import json
from matplotlib import colors

l,b=50,50
grid=np.zeros((l,b))
grid[int(l/2),:]=-1
grid[int(l/2),int(b/2)]=0
grid[int(l/2),int(b/2)+1]=0
plt.imshow(grid)

#g=Grid('new_map.json')
#grid=g.return_grid()
m=MACPP(grid,4)
agents=[]
n=20
x,y=m.convert(0,0)
#print(m.n*x+y)
crds=np.ones(n)*(m.n*x+y)
for i in range(n):
    agents.append(MACPPAgent(0,0,i,m))

i=0
while m.marked_visited()<1:
    if agents[i].state():
        crds[i]=agents[i].next(crds)
    i+=1
    i%=n
    if i==0:
        print(i,m.covered(),end=' | ')

print('\n')
print('map finished')