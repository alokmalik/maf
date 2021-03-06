import numpy as np
import matplotlib.pyplot as plt
from hmap import HAgent, HGrid, HMap
import cv2
from tqdm import tqdm
from task import Baseline, GeneticAlgorithm
from collections import defaultdict



filename='room_map_100.json'
g=HGrid(filename)


grid,doors=g.return_grid()
plt.imshow(grid[0,:,:])

rooms=np.zeros(len(doors))

new_doors=defaultdict(list)
for room in doors.keys():
    rooms[int(room)-1]=np.sum(grid[1,:,:]==int(room))
    new_doors[int(room)]=doors[room].copy()
doors=new_doors

n=5
speeds=np.array([9,4,10,6,6])
speeds=np.flip(np.sort(speeds))

mutation_probability=.1
population=10

g=GeneticAlgorithm(rooms,speeds,population,mutation_probability)

t,cost,costs=g.optimize(100)

print('tasks',t)
for i,assignment in enumerate(t):
    for j,r in enumerate(assignment):
        t[i][j]+=1

print('tasks',t)


mapob=HMap(grid,4,doors)

agents=[]

x,y=mapob.convert(0,0)
#print(m.n*x+y)
crds=np.ones(n)*(mapob.n*x+y)

telemetry=[]
for i in range(n):
    agents.append(HAgent(0,99,speeds[i],i,mapob,t[i]))
    telemetry.append([])

print('sucess!')

#crds=[mapob.n*x+y]*n
i=0
modes=[1,1,1,1,1]
tasks=[1,1,1,1,1]
steps=0

while mapob.covered()!=1:
    v=0
    while v<agents[i].speed and agents[i].mode!=2:
        v+=1
        if agents[i].mode!=2:
            telemetry[i].append(agents[i].next())
            #print(x)
        if modes[i]!=agents[i].mode:
            print('agent {} changed mode from {} to {}'.format(i,modes[i],agents[i].mode))
            print(mapob.covered())
            modes[i]=agents[i].mode
        if tasks[i]!=len(agents[i].tasks):
            print('agent {} started a task'.format(i))
            tasks[i]=len(agents[i].tasks)

    steps+=1
    if steps%1000==0:
        print('steps {} covered {}'.format(steps,mapob.covered()))
        for j in range(len(agents)):
            if agents[j].mode==0:
                print('Agent {} covering room with remaining cells {}'.format(j,len(agents[j].lmap.graph)))

    i+=1
    i=i%len(agents)

print('Completion took {} steps'.format(steps))
#print(telemetry)
for tel in telemetry:
    print(len(tel))

g=HGrid(filename)
grid,doors=g.return_grid()
grid=grid[0,:,:]
plt.imshow(grid)
l,b=grid.shape


def frame(i):
    crds=[]
    for tel in telemetry:
        if i<len(tel):
            x,y=tel[i]//b,tel[i]%b
            crds.append([x,y])
            grid[x,y]=2
        else:
            crds.append([0,0])
    return grid,crds





size = l,b
frames = max([len(tel) for tel in telemetry])
fps = 25
out = cv2.VideoWriter('video_{}.mp4'.format(filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
for i in tqdm(range(frames)):
    g,crds=frame(i)
    prev=[]
    i=0
    for x,y in crds:
        prev.append(g[x,y])
        g[x,y]=float(i*3)
        i+=1
    gray = cv2.normalize(g, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gray_3c = cv2.merge([gray, gray, gray])
    out.write(gray_3c)
    i=0
    for x,y in crds:
        g[x,y]=prev[i]
        i+=1
out.release()

print('Done!')