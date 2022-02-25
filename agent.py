     
        
class Agent:
    def __init__(self,x,y,sc,direction,map_object):
        '''
        input:
        x and y: start position of agent
        direction: initial direction agent is facing
        map_object: The map with agent will explore created from Map class
        '''
        #start position of agent
        self.startx=x
        self.starty=y
        #current x and y coordinates
        self.x=x
        self.y=y
        #map object is constructed from map class
        self.map=map_object
        
