import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from math import ceil
import random
from tqdm import tqdm
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
from functools import wraps
import time

def timer(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        if len(args[1])>5:
            print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

class Baseline:
    '''
    Class to calculate the baseline cost of the problem
    '''
    def __init__(self, rooms , speeds, distance):
        '''
        :param rooms: list of room areas
        :param speeds: list of agent speeds
        :param distance: square matrix of distance between rooms
        '''
        self.rooms = rooms
        self.speeds = speeds
        self.distance = distance
        self.num_rooms = len(rooms)
        self.num_agents = len(speeds)
        self.assignment = []
        self.visited = 0
        self.min_dist_order_cache = defaultdict(tuple)
        self.costmat = self.calculate_cost_matrix()
        self.greedy_allocation = self.optimize_greedy()
        self.min_cost = self.greedy_allocation[1]

    def calculate_cost_matrix(self):
        """
        Calculate the cost matrix representing the time each agent takes to clean each room.
        """
        dummy_speeds = np.reciprocal(self.speeds.astype('float64'))
        return np.matmul(dummy_speeds.reshape(self.num_agents, 1), self.rooms.reshape(1, self.num_rooms))
    
    def permutations(self, x: list):
        """
        Generate all permutations of the elements inside a list.

        :param x: a list
        :return: all permutations of the elements inside the list
        """
        if len(x) == 1:
            return [x]
        else:
            ans = []
            for i in range(len(x)):
                for p in self.permutations(x[:i] + x[i + 1:]):
                    ans.append([x[i]] + p)
            return ans
  
    def min_dist_order_cached(self, order):
        '''
        Stores last 10k results of min_dist_order to avoid recalculating
        '''
        
        if len(order)<5:
            return self.min_dist_order(order)
        orderhash = ''.join([str(x) for x in order])
        if orderhash in self.min_dist_order_cache:
            return self.min_dist_order_cache[orderhash]
        else:
            order, dist = self.min_dist_order(order)
            self.min_dist_order_cache[orderhash] = (order, dist)
            return order, dist

    def min_dist_order(self, order):
        """
        For one single agent find the order with minimum distance cost.
        :param order: list of room indices
        :return: optimal order of rooms for the agent to cover and the total distance covered
        """
        if len(order) < 2:
            return order, 0
        max_dist = float('inf')
        ans = order[:]
        orders = self.permutations(ans)
        for permutation in orders:
            currrent_distance = 0
            for i in range(len(permutation) - 1):
                if currrent_distance > max_dist:
                    break
                currrent_distance += self.distance[permutation[i], permutation[i + 1]]
            if currrent_distance < max_dist:
                max_dist = currrent_distance
                ans = permutation[:]
        return ans, max_dist
    
    def greedy_dist_order(self, order):
        '''
        For one sigle agent fine the order to travel rooms with closest room always.
        '''
        if len(order) < 2:
            return order, 0
        distance = 0
        temp_order = order[:]
        ans = []
        current_room = temp_order.pop(0)
        while temp_order:
            min_dist = float('inf')
            for room in temp_order:
                if self.distance[current_room, room] < min_dist:
                    min_dist = self.distance[current_room, room]
                    next_room = room
            distance += min_dist
            ans.append(current_room)
            current_room = next_room
            temp_order.remove(current_room)
        return ans, distance

    
    def min_dist_order_cost(self, agent, order):
        """
        For one single agent find the order with minimum distance cost.
        :param order: list of room indices
        :return: optimal order of rooms for the agent to cover and the total distance covered
        """
        area_cost = sum([self.rooms[i] for i in order]) / self.speeds[agent]
        if area_cost > self.min_cost:
            return order, area_cost
        ans, max_dist = self.min_dist_order_cached(order)
        distance_cost = max_dist / self.speeds[agent]
        total_cost = distance_cost + area_cost
        return ans, total_cost
    
    def baseline(self, cost, room, assignment):
        """
        Recursive method for finding the optimal task allocation for the agents.
        Inputs:
            cost: list of current costs for each agent
            room: current room to be assigned
            assignment: dictionary of lists of rooms assigned to each agent
        """
        # if all rooms are assigned, check if the cost is less than the current minimum cost
        if room == self.num_rooms:
            self.visited += 1
            # if the cost is less than the current minimum cost, 
            # update the minimum cost and assignment
            if max(cost) < self.min_cost:
                # update the minimum cost
                self.min_cost = max(cost)
                # empty the assignment list which stores the optimal assignment till now
                self.assignment = []
                # update the assignment list
                for a in range(self.num_agents):
                    self.assignment.append(assignment[a][:])
        
        # if the cost is already greater than the current minimum cost, return
        elif max(cost) < self.min_cost:
            self.visited += 1
            for a in range(self.num_agents):
                # add the cost of covering the room to the agent's cost
                prev_cost= cost[a]
                # add the room to the agent's assignment
                assignment[a].append(room)
                # find the optimal order of rooms for the agent to cover 
                # and the total distance covered
                order, cost[a] = self.min_dist_order_cost(a,assignment[a])
                # update the agent's assignment
                assignment[a] = order[:]
                # recursively call the method for the next room
                self.baseline(cost, room + 1, assignment)
                # remove the room from the agent's assignment
                assignment[a].remove(room)
                # update the agent's cost
                cost[a] = prev_cost

    def get_baseline(self):
        """
        Retrieve the optimal task allocation for the agents.
        """
        self.baseline([0] * self.num_agents, 0, defaultdict(list))
        return self.assignment, self.min_cost

    def optimize_greedy(self):
        """
        Perform greedy allocation of rooms to agents.
        Without considering the order of rooms to be covered by each agent.
        And add cost of covering the distance between rooms to the cost at the end.
        """
        # initialize the cost and assignment for each agent
        costs = [0] * self.num_agents
        assignments = [[] for _ in range(self.num_agents)]
        # get the indices of the rooms sorted by their area
        roomidx=list(np.argsort(self.rooms))

        while roomidx:
            # get the room with the largest area
            idx=roomidx.pop()
            area=self.rooms[idx]
            # find the agent with the smallest cost
            cost=float('inf')
            minidx=None
            for a in range(self.num_agents):
                # calculate the cost if the room is assigned to the agent
                c=float(area)/self.speeds[a]
                # if the cost is smaller than the current minimum, update the minimum
                if costs[a]+c<cost:
                    minidx=a
                    cost=costs[a]+c
            # assign the room to the agent with the smallest cost
            costs[minidx]+=float(area)/self.speeds[minidx]
            assignments[minidx].append(idx)
        # find the optimal order for each agent
        for robot in range(len(assignments)):
            order,dist=self.min_dist_order_cached(assignments[robot])
            assignments[robot]=order[:]
            costs[robot]+=dist/self.speeds[robot]
        return assignments,max(costs)
    

class GeneticAlgorithm(Baseline):
    '''
    Genetic Algorithm for task allocation.
    '''
    def __init__(self, rooms, agents, distance, population, mutation_prob):
        super().__init__(rooms, agents, distance)
        # mutation probability
        self.mutation_prob = mutation_prob
        # population size
        self.population = population
        # initialize the assignment of each room to agents randomly for each individual in the population
        self.assignments = self.initial_assignment()
        self.greedy_optimized = None
        self.baseline_optimized = None

    def initial_assignment(self):
        '''
        Initialize the assignment of each room to agents randomly.
        Returns: A numpy array of size (population, rooms) containing the assignment of rooms to agents.
        '''
        return np.random.randint(low=0, high=self.num_agents, size=(self.population, self.num_rooms))
    
    def get_optimize_greedy(self):
        if not self.greedy_optimized:
            self.greedy_optimized = super().optimize_greedy()
        return self.greedy_optimized
    
    def get_calculated_baseline(self):
        if not self.baseline_optimized:
            self.baseline_optimized = super().get_baseline()
        return self.baseline_optimized

    def distance_cost(self, assignment):
        '''
        input: A list containing the assignment of rooms to agents of size rooms.
        Calculate the total cost of distance covered by each agent with current assignment.
        '''
        # create a dictionary of rooms assigned to each agent
        assignments = [[] for _ in range(self.num_agents)]
        for room, agent in enumerate(assignment):
            assignments[int(agent)].append(int(room))
        cost = np.zeros(self.num_agents)
        for i, robot in enumerate(assignments):
            _, dist = self.min_dist_order_cached(robot)
            # add the distance cost to the agent's cost
            cost[i] = dist / self.speeds[i]
        return cost
    
    def calculate_cost(self, assignments, max_cost=None):
        '''
        input: A list containing the assignment of rooms to agents of size rooms.
        '''
        cost = [0] * self.num_agents
        for room, agent in enumerate(assignments):
            cost[int(agent)] += self.costmat[int(agent), int(room)]
        
        # since agents are covering the rooms in parallel, 
        # the cost is the maximum of the cost of each agent
        if max_cost and max(cost)>max_cost:
            return max(cost)
        dist_cost = self.distance_cost(assignments)
        return max(cost + dist_cost)

    def next_generation(self, assignment):
        '''
        Generate the next generation of assignments.
        Input: assignment of the current generation
        Returns: children of the next generation
        '''
        # children will store the assignments of the next generation
        # Row 0 will store the assignment of the current generation
        # Row 1 to Row population will store the assignments of the next generation
        # Each row will store the the agent assigned to the room on that index
        children = np.zeros((self.population + 1, self.num_rooms))
        for i in range(self.population + 1):
            children[i, :] = assignment[:]
        # costs will store cost of each child in the next generation
        costs = np.zeros(self.population + 1)
        # calculate the cost of the current generation
        costs[0] = self.calculate_cost(children[0])
        max_cost = costs[0]
        # Select the rooms in next generation to be mutated from the current generation
        mutations = np.random.random((self.population + 1, self.num_rooms)) < self.mutation_prob
        # Generate the new random assignments for entire generation
        random_assignments = np.random.randint(low=0, high=self.num_agents, size=(self.population + 1, self.num_rooms))
        # Do not put mutated values in the current assignment
        mutations[0] = False
        # Put the mutated values in the children
        np.putmask(children, mutations, random_assignments)
        # Calculate the cost of the children except the current generation
        for i in range(1, self.population + 1):
            costs[i] = self.calculate_cost(children[i, :], max_cost)
        return children, costs

    def optimize(self, num_generations):
        '''
        Optimize the assignment of rooms to agents using genetic algorithm.
        Input: number of generations to run the algorithm
        Returns: assignment of rooms to agents, cost of the assignment, cost of each generation
        '''
        parent = self.assignments
        generation = 0
        # costs will store the cost of each generation
        costs = []

        # Run the algorithm for num_generations
        while generation < num_generations:
            # Generate the next generation of assignments
            for i in range(self.population):
                # Generate the children of the first parent
                if i == 0:
                    child, cost = self.next_generation(parent[i, :])
                # Generate the children of the rest of the parents
                else:
                    temp_child, temp_cost = self.next_generation(parent[i, :])
                    child = np.append(child, temp_child, axis=0)
                    cost = np.append(cost, temp_cost, axis=0)
            # Sort the children based on the cost and get the indices of the sorted children
            indices = np.argsort(cost)
            # Select the best children for the next generation with lowest cost
            parent = child[indices[:self.population]]
            # Store the cost of the best child in the current generation
            costs.append(cost[indices[0]])
            # Increment the generation
            generation += 1
        
        assignment = [[] for _ in range(self.num_agents)]
        # Assign the rooms to agents
        for room, agent in enumerate(list(parent[0])):
            assignment[int(agent)].append(int(room))
        return assignment, costs[-1], costs

    def optimize_greedy_genetic(self, num_generations):
        '''
        Optimize the assignment of rooms to agents by using genetic algorithm with greedy initialization.
        Input: number of generations to run the algorithm
        Returns: assignment of rooms to agents, cost of the assignment, cost of each generation
        '''
        # Get the assignment from greedy algorithm
        greedy_assignment, greedy_cost = self.optimize_greedy()
        # reset the assignment of rooms to agents
        self.assignments = np.zeros(shape=(self.population, self.num_rooms))
        baseline_cost = self.get_baseline()[1]
        # Initialize the assignment of rooms to agents with greedy algorithm
        for i in range(self.population):
            for robot, rooms in enumerate(greedy_assignment):
                for room in rooms:
                    self.assignments[i, room] = robot
        
        assignment, cost, costs = self.optimize(num_generations)

        if (round(greedy_cost, 2) < round(cost, 2)):
            print('Greedy Cost: ', greedy_cost, ' Genetic Cost: ', cost)
            raise Exception('Cost Greater than Greedy')
        elif  (round(baseline_cost,2) > round(cost,2)):
            raise Exception('Baseline Violated')
        self.assignments = self.initial_assignment()
        return assignment, cost, costs
           



def run_simulation(num_agents,num_rooms,population,mutation_prob):
    '''
    Run the simulation for the given parameters.
    Input: number of agents, number of rooms, population, mutation probability
    '''
    min_speed = 1
    max_speed = 10
    min_area = 200
    max_area = 1000
    min_dist = 10
    max_dist = 100
    agents=np.flip(np.sort(np.random.randint(low=min_speed,high=max_speed+1,size=num_agents)))
    rooms=np.random.randint(low=min_area,high=max_area+1,size=num_rooms)
    distance=np.random.randint(low=min_dist,high=max_dist+1,size=(num_rooms,num_rooms))
    distance=np.triu(distance)+np.triu(distance).T
    np.fill_diagonal(distance,0)
    algorithms = GeneticAlgorithm(rooms, agents, distance, population, mutation_prob)
    greedy = algorithms.get_optimize_greedy()
    baseline = algorithms.get_calculated_baseline()
    genetic = algorithms.optimize(100)
    genetic_greedy = algorithms.optimize_greedy_genetic(100)
    if (round(genetic_greedy[1],2)>round(greedy[1],2)):
        print('Greedy Cost: ', greedy[1], ' Genetic Cost: ', genetic_greedy[1])
        raise Exception('Cost Greater than Greedy')
    return baseline[1], greedy[1], genetic[1], genetic_greedy[1]


# Run the simulation for all the combinations of agents and rooms for all algorithms
if __name__=='__main__':
    # initialize the arrays to store the results for agents 1 to 10 and rooms 1 to 10
    # bl: baseline, ga: genetic algorithm, gr: greedy algorithm, 
    # gga: genetic algorithm with greedy initialization
    baseline=np.zeros((10,10))
    genetic=np.zeros((10,10))
    greedy=np.zeros((10,10))
    genetic_greedy=np.zeros((10,10))

    population=10
    mut_prob=.1
    #run the simulation for 10 times
    for r in tqdm(range(10)):
        # initialize the array to store results for current iteration
        temp_baseline=np.zeros((10,10))
        temp_gentic=np.zeros((10,10))
        temp_greedy=np.zeros((10,10))
        temp_genetic_greedy=np.zeros((10,10))
        for room in range(1,11):
            for agent in range(1,11):
                bl, gr, ge, gegr =run_simulation(agent,room,population,mut_prob)
                baseline[agent-1,room-1]+=bl
                greedy[agent-1,room-1]+=gr
                genetic[agent-1,room-1]+=ge
                genetic_greedy[agent-1,room-1]+=gegr
                temp_baseline[agent-1,room-1]=bl
                temp_greedy[agent-1,room-1]=gr
                temp_gentic[agent-1,room-1]=ge
                temp_genetic_greedy[agent-1,room-1]=gegr
        # save the results for current iteration in csv file
        cols=['{} Rooms'.format(i) for i in range(1,11)]
        df=pd.DataFrame(temp_baseline,columns=cols)
        df['Agents']=[i for i in range(1,11)]
        df.to_csv("task_allocation/baseline_run_{}.csv".format(r))
        df=pd.DataFrame(temp_greedy,columns=cols)
        df['Agents']=[i for i in range(1,11)]
        df.to_csv("task_allocation/genetic_run_{}.csv".format(r))
        df=pd.DataFrame(temp_gentic,columns=cols)
        df['Agents']=[i for i in range(1,11)]
        df.to_csv("task_allocation/greedy_run_{}.csv".format(r))
        df=pd.DataFrame(temp_genetic_greedy,columns=cols)
        df['Agents']=[i for i in range(1,11)]
        df.to_csv("task_allocation/greedy_genetic_run_{}.csv".format(r))

    # calculate the average results
    baseline=baseline/10
    greedy=greedy/10
    genetic=genetic/10
    genetic_greedy=genetic_greedy/10


    cols=['{} Rooms'.format(i) for i in range(1,11)]
    df=pd.DataFrame(baseline,columns=cols)
    df['Agents']=[i for i in range(1,11)]
    df.to_csv("task_allocation/baseline_average.csv".format(r))
    df=pd.DataFrame(genetic,columns=cols)
    df['Agents']=[i for i in range(1,11)]
    df.to_csv("task_allocation/genetic_average.csv".format(r))
    df=pd.DataFrame(greedy,columns=cols)
    df['Agents']=[i for i in range(1,11)]
    df.to_csv("task_allocation/greedy_average.csv".format(r))
    df=pd.DataFrame(genetic_greedy,columns=cols)
    df['Agents']=[i for i in range(1,11)]
    df.to_csv("task_allocation/greedy_genetic_average.csv".format(r))


    msega=np.zeros((10,10))
    msegr=np.zeros((10,10))
    msegga=np.zeros((10,10))


    
    # plot the results of task allocation Time vs Number of Rooms for all agents.
    for a in range(1,11):
        plt.figure(figsize=(15,7))
        plt.plot(baseline[a-1,:],label='baseline',color='blue')
        plt.plot(genetic[a-1,:],label='genetic algorithm',color='green')
        plt.plot(greedy[a-1,:],label='greedy algorithm',color='red')
        plt.plot(genetic_greedy[a-1,:],label='genetic algo with greedy init',color='black')
        plt.title('Optimization for {} agent'.format(a),fontsize = 25)
        plt.xlabel('Num rooms',fontsize = 25,)
        plt.ylabel('Time to cover',fontsize = 25)
        plt.xlim(1,11)
        plt.xticks(np.arange(0,10,1),np.arange(1,11,1),fontsize=25)
        plt.grid(which='Major')
        plt.legend(fontsize=25)
        plt.savefig("task_allocation/agents_{}.pdf".format(a), format="pdf", bbox_inches="tight")
        plt.show()
        # calculate the mean error for each algorithm from baseline
        for r in range(1,11):
            msega[a-1,r-1]=-(baseline[a-1,r-1]-genetic[a-1,r-1])/baseline[a-1,r-1]*100
            msegr[a-1,r-1]=-(baseline[a-1,r-1]-greedy[a-1,r-1])/baseline[a-1,r-1]*100
            msegga[a-1,r-1]=-(baseline[a-1,r-1]-genetic_greedy[a-1,r-1])/baseline[a-1,r-1]*100

    # 3d plot of all algorithms results with num robots as x axis
    nx, ny = 10,10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    #plt.title('Performance of ',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, msega, color='blue',alpha=.2,label='Genetic Algorithm')
    ha.plot_surface(X, Y, msegr, color='green',alpha=0.2)
    ha.plot_surface(X, Y, msegga, color='yellow',alpha=0.2)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Percent gap from baseline',fontsize = 25)
    #ha.xticks(np.arange(0,10,1),np.arange(1,11,1),fontsize=25)

    f1 = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
    f2 = mpl.lines.Line2D([0],[0], linestyle="none", c='green', marker = 'o')
    f3 = mpl.lines.Line2D([0],[0], linestyle="none", c='yellow', marker = 'o')
    ha.legend([f1,f2,f3], ['Genetic Algorithm','Greedy Algorithm','Genetic Algorithm with greedy init'], numpoints = 1,fontsize=25)
    ha.view_init(0, 0)
    #ha.legend()
    plt.savefig("task_allocation/nrobots3d.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    # 3d plot of all algorithms results with num rooms as x axis
    nx, ny = 10, 10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    #plt.title('Baseline',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, msega, color='blue',alpha=.2,label='Genetic Algorithm')
    ha.plot_surface(X, Y, msegr, color='green',alpha=0.2)
    ha.plot_surface(X, Y, msegga, color='yellow',alpha=0.2)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Percent gap from baseline',fontsize = 25)
    #ha.xticks(np.arange(0,10,1),np.arange(1,11,1),fontsize=25)

    f1 = mpl.lines.Line2D([0],[0], linestyle="none", c='blue', marker = 'o')
    f2 = mpl.lines.Line2D([0],[0], linestyle="none", c='green', marker = 'o')
    f3 = mpl.lines.Line2D([0],[0], linestyle="none", c='yellow', marker = 'o')
    ha.legend([f1,f2,f3], ['Genetic Algorithm','Greedy Algorithm','Genetic Algorithm with greedy init'], numpoints = 1,fontsize=25)
    ha.view_init(0, 90)
    #ha.legend()
    plt.savefig("task_allocation/nrooms3d.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    # 3d plot of all rooms with eagle eye view
    nx, ny = 10, 10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    #plt.title('Baseline',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, msega,color='blue',alpha=.2,label='Genetic Algorithm')
    ha.plot_surface(X, Y, msegr,color='green',alpha=0.2)
    ha.plot_surface(X, Y, msegga,color='yellow',alpha=0.2)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Percentage gap from baseline',fontsize = 25)
    #ha.xticks(np.arange(0,10,1),np.arange(1,11,1),fontsize=25)

    f1 = mpl.lines.Line2D([0],[0], linestyle="none", c='blue', marker = 'o')
    f2 = mpl.lines.Line2D([0],[0], linestyle="none", c='green', marker = 'o')
    f3 = mpl.lines.Line2D([0],[0], linestyle="none", c='yellow', marker = 'o')
    ha.legend([f1,f2,f3], ['Genetic Algorithm','Greedy Algorithm','Genetic Algorithm with greedy init'], numpoints = 1,fontsize=15)
    ha.view_init(30, 150)
    #ha.legend()
    plt.savefig("task_allocation/gap3d.pdf", format="pdf", bbox_inches="tight")

    plt.show()


    # 3d plot of Percentage Gap from baseline for Genetic Algorithm
    nx, ny = 10, 10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    plt.title('Genetic Algorithm',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, msega,cmap=cm.jet,alpha=1,label='Genetic Algorithm')
    #ha.plot_surface(X, Y, msegr,cmap=cm.cubehelix,alpha=0.2)
    #ha.plot_surface(X, Y, msegga,cmap=cm.gnuplot,alpha=0.2)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Percentage gap from baseline',fontsize = 25)
    #ha.xticks(np.arange(0,10,1),np.arange(1,11,1),fontsize=25)

    ha.view_init(30, 150)
    plt.savefig("task_allocation/ega.pdf", format="pdf", bbox_inches="tight")

    plt.show()


    # 3d plot of Percentage Gap from baseline for Greedy Algorithm
    nx, ny = 10, 10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    plt.title('Greedy Algorithm',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, msegr,cmap=cm.jet,alpha=1,label='Greedy Algorithm')
    #ha.plot_surface(X, Y, msegr,cmap=cm.cubehelix,alpha=0.2)
    #ha.plot_surface(X, Y, msegga,cmap=cm.gnuplot,alpha=0.2)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Percentage gap from baseline',fontsize = 25)

    ha.view_init(30, 150)
    plt.savefig("task_allocation/egr.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    # 3d plot of Percentage Gap from baseline for Genetic Algorithm with greedy init
    nx, ny = 10, 10
    x = range(1,nx+1)
    y = range(1,ny+1)
    plt.figure(figsize=(15,15))
    hf = plt.figure(figsize=(15,15))
    ha = hf.add_subplot(111, projection='3d')
    plt.title('Genetic Algorithm with greedy initialization',fontsize = 25)
    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, msegga,cmap=cm.jet,alpha=1,label='Genetic Algorithm with greedy init')
    #ha.plot_surface(X, Y, msegr,cmap=cm.cubehelix,alpha=0.2)
    #ha.plot_surface(X, Y, msegga,cmap=cm.gnuplot,alpha=0.2)

    ha.set_xlabel('Num Rooms',fontsize = 25)
    ha.set_ylabel('Num Robots',fontsize = 25)
    ha.set_zlabel('Percentage gap from baseline',fontsize = 25)

    ha.view_init(30, 150)
    plt.savefig("task_allocation/egga.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    # 3d plot of time take for baseline
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
    plt.savefig("task_allocation/baseline.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    # 3d plot of time take for Genetic algorithm
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
    plt.savefig("task_allocation/genetic.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    # 3d plot of time take for Greedy algorithm
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
    plt.savefig("task_allocation/greedy.pdf", format="pdf", bbox_inches="tight")

    plt.show()
    # 3d plot of time take for Genetic algorithm with greedy assignment
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
    plt.savefig("task_allocation/greedygenetic.pdf", format="pdf", bbox_inches="tight")
