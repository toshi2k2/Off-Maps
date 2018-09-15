import networkx as nx
import numpy as np
import scipy
from scipy.spatial import distance
from scipy.signal import correlate
import sys
from itertools import islice
#from googlemaps import convert
import random
import mpu
import math

def elevation(client, locations):
    """
    Provides elevation data for locations provided on the surface of the
    earth, including depth locations on the ocean floor (which return negative
    values)
    :param locations: List of latitude/longitude values from which you wish
        to calculate elevation data.
    :type locations: a single location, or a list of locations, where a
        location is a string, dict, list, or tuple
    :rtype: list of elevation data responses
    """
    params = {"locations": convert.shortest_path(locations)}
    return client._request("/maps/api/elevation/json", params).get("results", [])

def elevation_along_path(client, path, samples):
    """
    Provides elevation data sampled along a path on the surface of the earth.
    :param path: An encoded polyline string, or a list of latitude/longitude
        values from which you wish to calculate elevation data.
    :type path: string, dict, list, or tuple
    :param samples: The number of sample points along a path for which to
        return elevation data.
    :type samples: int
    :rtype: list of elevation data responses
    """

    if type(path) is str:
        path = "enc:%s" % path
    else:
        path = convert.shortest_path(path)

    params = {
        "path": path,
        "samples": samples
    }

    return client._request("/maps/api/elevation/json", params).get("results", [])

def create_graph(locations):
    """locations is a list of tuples containing the location data tuple <lat, long, elevation>"""
    coords = [x[0:2] for x in locations]
    adjacency_matrix = distance.cdist(coords, coords, 'euclidean')
    neighbors = []
    for idx in range(adjacency_matrix.shape[0]):
        neighbors.append(adjacency_matrix[idx].argsort()[:8]) #8-adjacency being used here!

    G = nx.Graph()                   #creating graph
    G.add_nodes_from(range(len(locations)))                #defining number of nodes i.e. no. of tuples i.e. locations
    for idx in range(len(locations)):
        for i in range(len(neighbors[idx])):
            if neighbors[idx][i] >= idx:
                G.add_weighted_edges_from([(idx,neighbors[idx][i],abs(locations[idx][2]-locations[neighbors[idx][i]][2]))])  #adding edge weights for only adjacent edges
    #print(G.edges(data=True))
    return(G)
    pass

def search_shortest_path(g, location, start_loc, end_loc):
    """start and end nodes are the index of the location tuples"""
    start_node = location.index(start_loc)
    end_node = location.index(end_loc)
    path = nx.astar_path(g, start_node, end_node) #returns shortest path
    return(path)
    pass

def k_shortest_paths(G, source, target, k=9, weight=None):                                #returns shortest paths
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def pattern_match(current_path, shortest_paths, locations):
    Cor = []                   #correlation of paths list
    #Cor2 = []
    det = []
    print("current path:", current_path)
    #print(shortest_paths[0])#[:len(current_path)])
    for idx in range(len(shortest_paths)):
        Cor.append(np.corrcoef(current_path, shortest_paths[idx][:len(current_path)]))
        #Cor.append(np.dot(current_path, shortest_paths[idx][:len(current_path)]))
        #Cor2.append(scipy.signal.correlate(current_path, shortest_paths[idx][:len(current_path)]))
    for i in range(len(Cor)):
        cor = np.array(Cor[i])
        det.append(np.linalg.det(cor))
    #print("cor", cor)
    #print("Cor2", Cor2)
    #return(locations[shortest_paths[np.argmin(cor)][-1]])        #returns the PRESENT location tuple
    return (locations[shortest_paths[np.argmin(det)][-1]])
    pass

def data_create(lat_range, long_range, elev_range, datapoints):
    random.seed()
    mylist = [(random.uniform(lat_range[0], lat_range[1]), random.uniform(long_range[0], long_range[1]), random.uniform(0, elev_range)) for k in range(datapoints)]
    return(mylist)
    pass

def distance_from_coords(start, end):
    """inputs are tuples of lat-lon"""
    return(mpu.haversine_distance(start, end))

def total_distance(path, locations):
    """total distance between locations including altitude"""
    dist = 0
    for i in range(len(path)):
        x1 = locations[i][2] * math.cos(locations[i][0]) * math.sin(locations[i][1])
        y1 = locations[i][2] * math.sin(locations[i][0])
        z1 = locations[i][2] * math.cos(locations[i][0]) * math.cos(locations[i][1])
        x2 = locations[i+1][2] * math.cos(locations[i+1][0]) * math.sin(locations[i+1][1])
        y2 = locations[i+1][2] * math.sin(locations[i+1][0])
        z2 = locations[i+1][2] * math.cos(locations[i+1][0]) * math.cos(locations[i+1][1])
        a = np.array([x1, y1, z1])
        b = np.array([x2, y2, z2])
        dist += np.linalg.norm(a-b)
    return(dist)
    pass

def curr_path_cal(elevec, veclen):
    a = np.array(elevec)
    print("current vec length: ", len(a))
    a = a.reshape(-1, veclen).mean(axis=1)
    return(a)
    pass


locations = data_create(lat_range = [32.018539, 31.018539], long_range =[77.510593, 77.61], elev_range = 600, datapoints = 3000)

if len(locations) == 0:
    #client =
    lat_lon_data = np.loadtxt('loc_data.txt', dtype = float)      #data .txt file in the same folder
    all_elevation = elevation(client, lat_lon_data)                #check the format of lat/lon data
else:
    pass

start_loc, exit_loc = locations[random.randint(0,len(locations))], locations[random.randint(0,len(locations))]  #take this data from app

All_G = create_graph(locations)
Shortest_path = search_shortest_path(All_G, locations, start_loc, exit_loc)
K_Short_paths = k_shortest_paths(All_G, locations.index(start_loc), locations.index(exit_loc))

step_length = .3 #in metres - get from app
window_len = 100 #window of no. of steps - get from app

total_steps = 1200              #get from app
check_num = (total_steps*window_len)/step_length

pedo_distance = 30                 #from app -distance from start by app

tot_dist = 0
mod_path = []
vector_length = 0
for idx in range(len(Shortest_path)):
    mod_path.append(Shortest_path[idx])
    tot_dist += total_distance(mod_path, locations)
    if tot_dist >= pedo_distance:
        vector_length = len(mod_path)

pedo_elev_vector = []            #from app
for i in range(vector_length*2):
    pedo_elev_vector.append(random.uniform(0, 600))
#print("pedo_vec: ", pedo_elev_vector)
current_path = curr_path_cal(pedo_elev_vector, vector_length)                      #sensor elevation vector

Out = pattern_match(current_path, K_Short_paths, locations)

print(Out) #take this output

