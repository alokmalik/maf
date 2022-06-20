import numpy as np
from macpp import MACPPAgent, MACPP
from maf import Grid
import matplotlib.pyplot as plt
import json
from matplotlib import colors
from hmap import HAgent, HGrid, HMap
import matplotlib.animation as animation
import cv2
from tqdm import tqdm


g=HGrid('room_map_100.json')


grid,doors=g.return_grid()
plt.imshow(grid[0,:,:])


mapob=HMap(grid,4,doors)

agents=[]
n=3
x,y=mapob.convert(0,0)
#print(m.n*x+y)
crds=np.ones(n)*(mapob.n*x+y)
t=dict()
t[0]=['1','2']
t[1]=['3','4']
t[2]=['5']
telemetry=[]
for i in range(n):
    agents.append(HAgent(0,99,i,mapob,t[i]))
    telemetry.append([])

print('sucess!')

#crds=[mapob.n*x+y]*n
i=0
modes=[1,1,1]
tasks=[2,2,2]
steps=0

while mapob.covered()!=1:
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

g=HGrid('room_map_100.json')
grid,doors=g.return_grid()
grid=grid[0,:,:]
plt.imshow(grid)
l,b=grid.shape


def frame(i):
    for tel in telemetry:
        if i<len(tel):
            x,y=tel[i]//b,tel[i]%b
            grid[x,y]=2
    return grid

#plt.rcParams['animation.ffmpeg_path']='/home/alokmalik/anaconda3/envs/marl/bin/ffmpeg*'
fps = 30
nSeconds = 5
fig = plt.figure(figsize=(16,16))
a = frame(55)
im=plt.imshow(a, interpolation='none', aspect='auto', vmin=-1, vmax=3)





size = l,b
frames = max([len(tel) for tel in telemetry])
fps = 25
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
for i in tqdm(range(frames)):
    gray = cv2.normalize(frame(i), None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gray_3c = cv2.merge([gray, gray, gray])
    out.write(gray_3c)
out.release()

print('Done!')