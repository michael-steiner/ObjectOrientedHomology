import networkx as nx
import itertools as it
import numpy as np
import pandas as pd
import os



class GraphWorker:

    def __init__(self, name='blank'):
        self.name = name

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def create_directed_graph(self, number_of_nodes, standard_deviation=1,center_of_distribution = 0,list_of_edges = []):

        num_nodes = number_of_nodes
        edge_list = list_of_edges
        cent_dist = center_of_distribution
        std_dist = standard_deviation
        node_values_G = []
        Graph = nx.DiGraph()
        list_of_nodes = [n for n in range(num_nodes)]

        if(edge_list == []):
            edge_list_iterator = it.combinations(list_of_nodes, 2)
            for i in edge_list_iterator:
                edge_list.append(i)

        for ii in range(num_nodes):
            node_values_G.append(np.random.normal(cent_dist, std_dist))


        for i in range(len(list_of_nodes)):
            Graph.add_node(list_of_nodes[i], node_value=node_values_G[i])

        Graph.add_edges_from(edge_list)

        return Graph

    def find_paths_no_loops(self, Graph, start_node, length_of_path):
        if length_of_path==0:
            return [[start_node]]
        paths = []

        for neighbor in Graph.neighbors(start_node):
            for path in self.find_paths_no_loops(Graph, neighbor, (length_of_path-1)):
                if start_node not in path:
                    paths.append([start_node]+path)
        return paths

    def create_cloud_of_points(self,graph, dimension_of_cloud):
        cloud_dim = dimension_of_cloud
        graph_g = graph
        list_of_paths = []
        list_of_points_in_n_space = []
        for g_node in graph_g.nodes():
            list_of_paths.extend(self.find_paths_no_loops(graph_g, g_node, cloud_dim))

        for n_walk in list_of_paths:
            temp_point_list = []
            for n_node in n_walk:
                self.temp_point_list.append(graph_g.node[n_node]['node_value'])
            self.list_of_points_in_n_space.append(self.temp_point_list)

        df_points = pd.DataFrame( list_of_points_in_n_space)

        return df_points

    def generate_save_point_clouds(self,cloud_name,number_of_clouds, number_of_nodes, dim_of_cloud, save_path = "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\",stDev=1, center_of_dist = 0 ):

        cloudName = cloud_name
        numClouds = number_of_clouds
        numNodes = number_of_nodes
        dim_of_cloud = (dim_of_cloud -1)
        savePath = save_path
        standardDev = stDev
        centerOfDist = center_of_dist

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for cloudNumber in range(numClouds):
            G_temp = self.create_directed_graph(numNodes,standardDev,centerOfDist)
            tempCloud = self.create_cloud_of_points(G_temp,dim_of_cloud)
            tempCloud.to_csv(savePath + cloudName + str(cloudNumber) + ".csv", index=False)








