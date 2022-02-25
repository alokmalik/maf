import numpy as np
from maf import MAFMap, Agent,Grid
import matplotlib.pyplot as plt
import json
from matplotlib import colors
import time

g=Grid('new_map.json')
grid=g.return_grid()
m=MAFMap(grid,8,10)
agents=[]
n=20
for i in range(n):
    agents.append(Agent(0,0,2,i%8,m))


t1=time.time()
agent_mode=[0]*n
moves=0
while m.poi_covered()<.9 and (m.covered()<.95 or moves<250000):
    agents[i].next()
    if agents[i].mode!=agent_mode[i]:
        print(i,agents[i].mode,agents[i].sc,moves,m.covered())
        agent_mode[i]=agents[i].mode
    i+=1
    i=i%n
    if i==0:
        moves+=1
        if moves%50000==0:
            print(moves,m.covered(),m.poi_covered(), end= ' ')
            print('elapsed time: {}'.format(time.time()-t1))
t2=time.time()
print('finished in {}'.format(t2-t1))