import os
import numpy as np
import pickle
import json
import string
import networkx as nx
import time
from numpy.core.multiarray import dtype

def getUserList():
    return np.sort(pickle.load(open("users.p", "rb")).astype(int))

def createGraph():
    usersOrig = np.sort(np.loadtxt('usersinnetwork.txt',dtype=np.int32))
    print "User loaded"
    users = dict()
    usersRev = dict()
    nodes = len(usersOrig)
    for i in range(nodes):
        users[i] = usersOrig[i]
        usersRev[usersOrig[i]] = i
    print "Hashmap created"
    G1 = nx.Graph()
    for i in range(nodes):
        G1.add_node(i)
    print "Node added to graph"
    f = open('networkx.txt')
    adjacencyMatrix = dict()
    for line in iter(f):
        if line.strip():
            line_splitted = line.split('\t')
            user1 = usersRev[int(line_splitted[0])]
            user2 = usersRev[int(line_splitted[1])]
            print users[user1]
            if user1 in adjacencyMatrix:
                list1 = adjacencyMatrix[user1]
                if user2 not in list1: 
                    adjacencyMatrix[user1].append(user2)
                    G1.add_edge(user1, user2, weight=1)
            else:
                adjacencyMatrix[user1] = [user2]
                G1.add_edge(user1, user2, weight=1)
               
            if user2 in adjacencyMatrix:
                list1 = adjacencyMatrix[user2]
                if user1 not in list1: 
                    adjacencyMatrix[user2].append(user1)
            else:
                adjacencyMatrix[user2] = [user1]
    return G1
def checkGraph():
#     usersOrig = np.sort(np.loadtxt('usersinnetwork.txt',dtype=np.int32))
#     print "User loaded"
#     users = dict()
#     usersRev = dict()
#     nodes = len(usersOrig)
#     for i in range(nodes):
#         users[i] = usersOrig[i]
#         usersRev[usersOrig[i]] = i
#     print "Hashmap created"
#     
#     otputFile = open('map.txt', 'w')
#     for key,val in users.items():
#         otputFile.write(str(key)+"\t"+str(val)+"\n")
#     otputFile.close()
    
    start_time = time.time()
    G1 = pickle.load(open("graph.p", "rb"))
    print "Time to load", (time.time()-start_time),"seconds"
     
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    degrees = np.zeros((nodes))
    for NI in range(nodes):
        deg = G1.degree(NI)
        degrees[NI] = deg
    degreessorted = np.sort(degrees)
#     np.savetxt('myfile.txt', degrees.transpose(),fmt='%i')
    np.savetxt('myfile1.txt', degreessorted.transpose(),fmt='%i')
    
def graphDiameter():
    start_time = time.time()
    G1 = pickle.load(open("graph.p", "rb"))
    print "Time to load", (time.time()-start_time),"seconds"
    if (nx.is_connected(G1)):
        diameter = nx.diameter(G1)
        print "\tDiameter is ", diameter
    else:
        print nx.number_connected_components(G1)
        graphs = list(nx.connected_component_subgraphs(G1))
        start_time = time.time()
        print "\tDiameter is ", nx.diameter(graphs[0])
        print "Time to count diameter", (time.time()-start_time),"seconds"
        print "\tDiameter is ", nx.diameter(graphs[1])
        print "\tDiameter is ", nx.diameter(graphs[2])
        print "\tDiameter is ", nx.diameter(graphs[3])
        print "\tDiameter is ", nx.diameter(graphs[4])
    
def main():
    graphDiameter()

if __name__ == '__main__':
    main()
