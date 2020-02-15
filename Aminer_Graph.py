import os
import operator
import numpy as np
from numpy.core.multiarray import dtype
import networkx as nx
import time
import pickle
import sys
    
def readGraph():
    start_time = time.time()
    G1 = pickle.load(open("Aminer_Graph.p", "rb"))
    print "Time to load", (time.time() - start_time), "seconds"
    
    nodes = G1.number_of_nodes()
    edges = G1.number_of_edges()
    
    print "nodes =", nodes, ", Edges =", edges
    print "Is connected = ", nx.is_connected(G1)
    print "Number of connected components = ", nx.number_connected_components(G1)
    Gc = max(nx.connected_component_subgraphs(G1), key=len)
    print "Largest one is "
    print "nodes =", Gc.number_of_nodes(), ", Edges =", Gc.number_of_edges()
    
def checkGraph():  
    start_time = time.time()
    G1 = pickle.load(open("Aminer_Graph.p", "rb"))
    print "Time to load", (time.time() - start_time), "seconds"
    graphs = sorted(nx.connected_component_subgraphs(G1), key=len, reverse=True)
    otputFile = open('test.txt', 'w')
    for i in range(len(graphs)):
        G1 = graphs[i]
        otputFile.write(str(i)+"\t"+str(G1.number_of_nodes())+"\t"+str(G1.number_of_edges())+'\n')    
    otputFile.close()
    
def graphDiameter():
    start_time = time.time()
    G1 = pickle.load(open("Aminer_Graph.p", "rb"))
    print "Time to load", (time.time() - start_time), "seconds"
    if (nx.is_connected(G1)):
        diameter = nx.diameter(G1)
        print "\tDiameter is ", diameter
    else:
        print nx.number_connected_components(G1)
        graphs = list(nx.connected_component_subgraphs(G1))
        for i in range(len(graphs)):
            start_time = time.time() 
            print i, "\tDiameter is ", nx.diameter(graphs[i])
            print "Time to count diameter", (time.time() - start_time), "seconds"
def main():
    checkGraph()

if __name__ == '__main__':
    main()
