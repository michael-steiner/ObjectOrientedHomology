
import networkx as nx
import itertools as it
import numpy as np
import pandas as pd


import graph_work1_0_0 as gw

cloud_worker = gw.GraphWorker()

#Graph_G1 = worker.create_directed_graph(10)

#print(worker.find_paths_no_loops(Graph_G1,1,2))

#df_cloud = worker.create_cloud_of_points(Graph_G1,3)
#df_cloud.to_csv("C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\test2.csv", index=False)

###Create and save point clouds###



#Create class_0 clouds
number_of_class_0_clouds = 10
standard_dev_0 = 1
mu_0 = 0
dimension_of_cloud = 4
path_to_save = "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\clouds_class_0\\"
cloud_name = "cloud_class_0"

for cloud_counter_0 in range(number_of_class_0_clouds):
    save_path = path_to_save + cloud_name +  "dim" + str(dimension_of_cloud) + "mu" + str(mu_0) + "std" + str(standard_dev_0) + "cloud" +  str(cloud_counter_0) + ".csv"
    graph_temp_0 = cloud_worker.create_directed_graph(10,standard_dev_0,mu_0)
    df_temp_0 = cloud_worker.create_cloud_of_points(graph_temp_0,dimension_of_cloud)
    df_temp_0.to_csv(save_path, index=False)


#Create class_1 clouds
number_of_class_1_clouds = 10
standard_dev_1 = 1.5
mu_1 = 0
dimension_of_cloud = 4
path_to_save = "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\clouds_class_1\\"
cloud_name = "cloud_class_1"

for cloud_counter_1 in range(number_of_class_1_clouds):
    save_path = path_to_save + cloud_name +  "dim" + str(dimension_of_cloud) + "mu" + str(mu_1) + "std" + str(standard_dev_1) + "cloud" +  str(cloud_counter_1) + ".csv"
    graph_temp_1 = cloud_worker.create_directed_graph(10,standard_dev_1,mu_1)
    df_temp_1 = cloud_worker.create_cloud_of_points(graph_temp_1,dimension_of_cloud)
    df_temp_1.to_csv(save_path, index=False)






