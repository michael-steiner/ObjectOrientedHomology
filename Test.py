
import networkx as nx
import itertools as it
import numpy as np
import pandas as pd
import subprocess
import os
import glob
from sklearn.cross_validation import train_test_split
import networkx as nx
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
import pandas as pd
import math
import os
import glob

from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from bokeh.plotting import figure,show,output_notebook
from bokeh.models import Range1d


import graph_work1_0_0 as gw
import graph_analysis as ga

cloud_worker = gw.GraphWorker()

#Graph_G1 = worker.create_directed_graph(10)

#print(worker.find_paths_no_loops(Graph_G1,1,2))

#df_cloud = worker.create_cloud_of_points(Graph_G1,3)
#df_cloud.to_csv("C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\test2.csv", index=False)

###Create and save point clouds###

number_of_nodes_in_graph = 10

#Create class_0 clouds
number_of_class_0_clouds = 100
standard_dev_0 = 1
mu_0 = 0
dimension_of_cloud = 3
path_to_save = "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\clouds_class_0\\"
cloud_name = "cloud_class_0"

for cloud_counter_0 in range(number_of_class_0_clouds):
    save_path = path_to_save + cloud_name +  "dim" + str(dimension_of_cloud) + "mu" + str(mu_0) + "std" + str(standard_dev_0) + "cloud" +  str(cloud_counter_0) + ".csv"
    graph_temp_0 = cloud_worker.create_directed_graph(number_of_nodes_in_graph,standard_dev_0,mu_0)
    df_temp_0 = cloud_worker.create_cloud_of_points(graph_temp_0,dimension_of_cloud)
    df_temp_0.to_csv(save_path, index=False)


#Create class_1 clouds
number_of_class_1_clouds = 100
standard_dev_1 = 2.0
mu_1 = 0
dimension_of_cloud = 3
path_to_save = "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\clouds_class_1\\"
cloud_name = "cloud_class_1"

for cloud_counter_1 in range(number_of_class_1_clouds):
    save_path = path_to_save + cloud_name +  "dim" + str(dimension_of_cloud) + "mu" + str(mu_1) + "std" + str(standard_dev_1) + "cloud" +  str(cloud_counter_1) + ".csv"
    graph_temp_1 = cloud_worker.create_directed_graph(number_of_nodes_in_graph,standard_dev_1,mu_1)
    df_temp_1 = cloud_worker.create_cloud_of_points(graph_temp_1,dimension_of_cloud)
    df_temp_1.to_csv(save_path, index=False)





## call R code to perform filtration
command = 'Rscript'
path2script = 'Rcode\\TDA_betti1_1_0.R'
path_to_class_0_data = "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\clouds_class_0\\"
path_to_save_class_0_data =  "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\betti0output\\betti_data_0.csv"

path_to_class_1_data = "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\clouds_class_1\\"
path_to_save_class_1_data =  "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\betti0output\\betti_data_1.csv"

args_list =[[path_to_class_0_data, path_to_save_class_0_data], [path_to_class_1_data, path_to_save_class_1_data]]
for args in args_list:
    cmd =  [command, path2script] + args
    x = subprocess.check_output(cmd, universal_newlines=True)


# prepare the data for machine learning
features, target = ga.prepare_data_for_machine_learning(["C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\betti0output\\betti_data_0.csv", "C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\betti0output\\betti_data_1.csv"])
features_train, features_test, target_train, target_test = train_test_split(features,target,test_size=.33, random_state=0)

from sklearn.linear_model import LogisticRegression
#instantiate the classifier
clf_lr = LogisticRegression(C=1)


details_of_experiment_class_0 = [number_of_class_0_clouds,number_of_nodes_in_graph,dimension_of_cloud,standard_dev_0,mu_0] #number of clouds, number of nodes, dimension of clouds, standard deviation, mean
details_of_experiment_class_1 = [number_of_class_1_clouds,number_of_nodes_in_graph,dimension_of_cloud,standard_dev_1,mu_1]
results_save="C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\results\\results_09-05-2016__seed_0__nornal_dist.csv"

details_of_exp = details_of_experiment_class_0
details_of_exp.append(details_of_experiment_class_1[3])
details_of_exp.append(details_of_experiment_class_1[4])

#fit the classifier
ga.train_and_evaluate(clf_lr,features_train, features_test,target_train,target_test,details_of_exp,results_save)


def file_clean_up():
    folder_list = ["clouds_class_0", "clouds_class_1", "betti_data"]
    for folder in folder_list:
        filelist = glob.glob("C:\\Users\\micha\\PycharmProjects\\PersistHomologyOOP\\cloud_data\\"+folder+"\\*.csv")
        for f in filelist:
            os.remove(f)

file_clean_up()



