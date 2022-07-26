#%matplotlib widget
import numpy as np
from scipy.optimize import linear_sum_assignment
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from math import ceil
import random
from tqdm import tqdm

class Baseline:
    def __init__(self,rooms,agents):
        self.rooms=rooms
        self.agents=agents
        self.num_rooms=len(rooms)
        self.num_agents=len(agents)
        self.costmat= self.calculateCostMat()
        self.mincost=self.optimizeGreedy()
        self.assignment=[]
        self.visited=0
        '''print('There are {} agents and {} rooms, total number of assignments possible are {}'.format(self.num_agents,\
                                                                                                    self.num_rooms,self.num_agents**self.num_rooms))'''
    def calculateCostMat(self):
        dummy_speeds=np.reciprocal(self.agents.astype('float64'))
        return np.matmul(dummy_speeds.reshape(self.num_agents,1),self.rooms.reshape(1,self.num_rooms))
    
    def baseline(self,cost,room,assignment):
        if room ==self.num_rooms:
            self.visited+=1
            if max(cost)<self.mincost:
                #rint(assignment)
                self.mincost=max(cost)
                self.assignment=[]
                for a in range(self.num_agents):
                    self.assignment.append([])
                    for room in assignment[a]:
                        self.assignment[-1].append(room)
                        
        elif max(cost)<self.mincost:
            self.visited+=1
            for a in range(self.num_agents):
                cost[a]+=self.costmat[a][room]
                assignment[a].append(room)
                self.baseline(cost,room+1,assignment)
                assignment[a].pop()
                cost[a]-=self.costmat[a][room]
                
                
    def get_baseline(self):
        self.baseline([0]*self.num_agents,0,defaultdict(list))
        #print('Combinations Tried: ',self.visited)
        return self.assignment,self.mincost
    
    def optimizeGreedy(self):
        costs=[0]*self.num_agents
        assignments=[[] for _ in range(self.num_agents)]
        roomidx=list(np.argsort(self.rooms))
        
        while roomidx:
            idx=roomidx.pop()
            area=self.rooms[idx]
            cost=float('inf')
            minidx=0
            #print('hi')
            for a in range(self.num_agents):
                #print(a)
                c=float(area)/self.agents[a]
                #print(costs[a],c,costs[a]+c)
                if costs[a]+c<cost:
                    #print('hi',a)
                    minidx=a
                    cost=costs[a]+c
                    
            costs[minidx]+=float(area)/self.agents[minidx]
            assignments[minidx].append(idx)
        return max(costs)
        
class GeneticAlgorithm(Baseline):
    def __init__(self,rooms,agents,pop,mutprob):
        super().__init__(rooms,agents)
        self.mutprob=mutprob
        self.population=pop
        self.assignment=self.initialAssignment()
        
        #print("Initial Assignment Cost: ",self.calculateCost(self.assignment))
        
    def initialAssignment(self):
        assignment=[]
        assignment=np.random.randint(low=0,high=self.num_agents,size=(self.population,self.num_rooms))
        '''for room in range(self.num_rooms):
            a=np.random.randint(low=0,high=self.num_agents)
            assignment.append(a)
        assignment=np.array(assignment)'''
        return assignment
            
    def calculateCost(self,assignments):
        cost=[0]*self.num_agents
        #print('Assignments',assignments)
        for room,agent in enumerate(assignments):
            #print(room,agent)
            cost[int(agent)]+=self.costmat[int(agent),int(room)]
        return max(cost)
    
    def nextGeneration(self,assignment):
        childs=np.zeros((self.population+1,self.num_rooms))
        #print(assignment.shape)
        for i in range(self.population+1):
            childs[i,:]=assignment
        costs=np.zeros(self.population+1)
        costs[0]=self.calculateCost(childs[0])
        
        mut=np.random.random((self.population+1,self.num_rooms))<self.mutprob
        assigns=np.random.randint(low=0,high=self.num_agents,size=(self.population+1,self.num_rooms))
        mut[0]=False
        np.putmask(childs, mut, assigns)
        for i in range(1,self.population+1):
            costs[i]=self.calculateCost(childs[i,:])
        
        return childs,costs
    
    def optimize(self,numgenerations):
        '''parent=np.zeros((self.population,self.num_rooms))
        for i in range(numchild):
            parent[i,:]=self.assignment'''
        parent=self.assignment
        gen=0
        costs=[]
        while gen<numgenerations:
            for i in range(self.population):
                \
                if i==0:
                    child,cost=self.nextGeneration(parent[i,:])
                else:
                    tchild,tcost=self.nextGeneration(parent[i,:])
                    child=np.append(child,tchild,axis=0)
                    cost=np.append(cost,tcost,axis=0)
                    
            indices = np.argsort(cost)
            parent=child[indices[:self.population]]
            costs.append(cost[indices[0]])
            gen+=1
        assignments=[[] for _ in range(self.num_agents)]
        
        for room,agent in enumerate(list(parent[0])):
            assignments[int(agent)].append(int(room))
        return assignments,costs[-1],costs
    
    def optimizeGreedy(self):
        costs=[0]*self.num_agents
        assignments=[[] for _ in range(self.num_agents)]
        roomidx=list(np.argsort(self.rooms))
        
        while roomidx:
            idx=roomidx.pop()
            area=self.rooms[idx]
            cost=float('inf')
            minidx=0
            #print('hi')
            for a in range(self.num_agents):
                #print(a)
                c=float(area)/self.agents[a]
                #print(costs[a],c,costs[a]+c)
                if costs[a]+c<cost:
                    #print('hi',a)
                    minidx=a
                    cost=costs[a]+c
                    
            costs[minidx]+=float(area)/self.agents[minidx]
            assignments[minidx].append(idx)
        return assignments,max(costs)
    
    def optimizegreedygenetic(self,numgenerations):
        #print(self.optimizeGreedy())
        assignment,gc=self.optimizeGreedy()
        self.assignment=np.zeros(shape=(self.population,self.num_rooms))
        for i in range(self.population):
            for robot,rooms in enumerate(assignment):
                for room in rooms:
                    self.assignment[i,room]=robot
        assignments,cost,costs=self.optimize(numgenerations)
        if round(gc,2)<round(cost,2):
            print('error')
        
        self.assignment=self.initialAssignment()
        return assignments,cost,costs
            

def run(num_agents,num_rooms,population,mut_prob):
    min_speed=2
    max_speed=10
    min_area=10
    max_area=100
    agents=np.flip(np.sort(np.random.randint(low=min_speed,high=max_speed+1,size=num_agents)))
    rooms=np.random.randint(low=min_area,high=max_area,size=num_rooms)
    bl=Baseline(rooms,agents)
    bl_assignments=bl.get_baseline()
    ga=GeneticAlgorithm(rooms,agents,population,mut_prob)
    ga_assignments=ga.optimize(100)
    gr_assignments=ga.optimizeGreedy()
    gga_assignments=ga.optimizegreedygenetic(100)
    return bl_assignments[1],ga_assignments[1],gr_assignments[1],gga_assignments[1]
            
            
if __name__=='__main__':
    bl=np.zeros((10,10))
    ga=np.zeros((10,10))
    gr=np.zeros((10,10))
    gga=np.zeros((10,10))

    population=10
    mut_prob=10
    for _ in tqdm(range(10)):
        for room in range(1,11):
            for agent in range(1,11):
                a,b,c,d=run(agent,room,population,mut_prob)
                bl[agent-1,room-1]+=a
                ga[agent-1,room-1]+=b
                gr[agent-1,room-1]+=c
                gga[agent-1,room-1]+=d


    bl=bl/10
    ga=ga/10
    gr=gr/10
    gga=gga/10

    for a in range(1,11):
        plt.figure(figsize=(15,7))
        plt.plot(bl[a-1,:],label='baseline',color='blue')
        plt.plot(ga[a-1,:],label='genetic algorithm',color='green')
        plt.plot(gr[a-1,:],label='greedy',color='red')
        plt.plot(gga[a-1,:],label='genetic algo with greedy init',color='black')
        plt.title('Optimization for {} agent'.format(a),fontsize = 25)
        plt.xlabel('Num rooms',fontsize = 25,)
        plt.ylabel('Time to cover',fontsize = 25)
        plt.xlim(1,11)
        plt.xticks(np.arange(1,11,1),np.arange(1,11,1),fontsize=25)
        plt.grid(which='Major')
        plt.legend()
        plt.savefig("agents_{}.pdf".format(a), format="pdf", bbox_inches="tight")
        plt.show()

    nx, ny = 10, 10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    plt.title('Baseline',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, bl,cmap=cm.jet)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Time Taken',fontsize = 25)

    ha.view_init(30, 150)
    plt.savefig("baseline.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    nx, ny = 10, 10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    plt.title('Genetic Algorithm',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, ga,cmap=cm.jet)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Time Taken',fontsize = 25)

    ha.view_init(30, 150)
    plt.savefig("genetic.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    nx, ny = 10, 10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    plt.title('Greedy Assignment',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, gr,cmap=cm.jet)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Time Taken',fontsize = 25)

    ha.view_init(30, 150)
    plt.savefig("greedy.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    nx, ny = 10, 10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    plt.title('Genetic Algorithm with greedy assignment',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, gga,cmap=cm.jet)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Time Taken',fontsize = 25)

    ha.view_init(30, 150)
    plt.savefig("greedygenetic.pdf", format="pdf", bbox_inches="tight")

    plt.show()


                