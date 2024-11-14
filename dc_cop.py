import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
#-------------------------------FUCTIONS-----------------------------------------------------------#

# master function for smaller functions
# because there are too many fucking lines
def kmeans_master_function(dataframe, cols_list):
    x = dataframe.iloc[:,cols_list].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    lis = kmeanz(x)
    y_predict,kmeans,wcss_list = lis[0], lis[1], lis[2]

    k=elbowpoint(wcss_list)
    print(f'The elbow point of this cluster: {k}')

    #plot_kmeans(k,x,y_predict,kmeans)

    labels = kmeans.labels_
    score = silhouette_score(x, labels, metric='cosine')
    print(f'Silhouette Score of this cluster: {score}')
    return score

# K means shortcut
def kmeanz(x):

    wcss_list= []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
        kmeans.fit(x)
        wcss_list.append(kmeans.inertia_)
    '''
    plt.plot(range(1, 11), wcss_list)
    plt.title('Elbow Graph')
    plt.xlabel('Number of clusters(k)')
    plt.ylabel('wcss_list')
    plt.show()'''
    kmeans = KMeans(n_clusters= elbowpoint(wcss_list), init='k-means++', random_state= 42)
    y_predict= kmeans.fit_predict(x)
    return [y_predict, kmeans,wcss_list]

# Obtaining elbow point (k)
def elbowpoint(wss):
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    knee_locator = KneeLocator(k_values, wss, curve='convex', direction='decreasing')
    elbow_point = knee_locator.elbow
    return elbow_point

# Plotting the clusters
def plot_kmeans(k,x,y_predict,kmeans):
    for i in range(k):
        colormap = plt.cm.autumn
        colors = colormap(np.linspace(0, 1, k))
        plt.scatter(x[y_predict == i,0], x[y_predict == i,1], s = 100, c = colors[i], label = f'Cluster {i+1}')
    plt.title("Clusters")
    plt.show()

# Prints Dtaframe information, Null Values, Columns, Duplicates (T or F)
def dataframe_info(dataframe):
    print(f'{dataframe.info()}\n')
    print(f'Columns = \n{dataframe.columns}\n')
    print(f'Null Values: \n{dataframe.isnull().sum()}\n')
    print(f'Duplicated rows?: {dataframe.duplicated().any()}\n')



#Replaces 0's with NaN
def replace_with_nan(dataframe):
    dataframe.replace({'CPU_MThread': 0, 'CPU_Utilization': 0, 'RAM_Utilization': 0,
                       'CPU_Speed':0, 'CPU_Core': 0, 'Storage_Size_GB': 0,
                       'Storage_Used_Size_GB': 0, 'RAM_Size_GB':0
                    }, np.nan,
                    inplace = True)
    print("0 replaced with null !!\n")
    return dataframe

# Fills NaN Values With median
def fill_nan(dataframe, columns):
    dataframe[columns] = dataframe[columns].fillna(dataframe[columns].median())
    print("Null Values Fixed! \n")
    return dataframe

# irq method for removing outliers just in case
def irqmethod(dataframe, column):
    q1 = dataframe[column].quantile(.25)
    q3 = dataframe[column].quantile(.75)
    irq = q3 - q1
    dataframe = dataframe[(dataframe[column] > q1 - 1.5*irq) & (dataframe[column] < q3 + 1.5*irq)]
    return dataframe

#-------------------------------FUCTIONS------------------------------------------------------------#

# Read The Dataframe
dataframe = pd.read_csv("DataCenter.csv")

# Print Basic Info of the dataframe
print(dataframe.columns,'\n')
# Ram size seems to be categorical. Can be changed to int beause all the other size values are in int/float
dataframe["RAM_Size_GB"]=dataframe["RAM_Size_GB"].str.replace("GB",'')
dataframe["RAM_Size_GB"]=dataframe["RAM_Size_GB"].astype(int)

dataframe["CPU_Speed"]=dataframe["CPU_Speed"].str.replace("GHz",'')
dataframe["CPU_Speed"]=dataframe["CPU_Speed"].astype(float)

# This column for the most parts has obj vals that can be changed to int, but has one value "Eight Core" that needs to be looked into.
# (Why the fuck would he add a value called fucking "Eight"?????)
# We need to change this before we can change the whole column

dataframe.loc[dataframe['CPU_Core'] == 'Eight Core', 'CPU_Core'] = '8'
dataframe["CPU_Core"]=dataframe["CPU_Core"].astype(int)

#Replace Zeros with NaN to identify ALL Null values
replace_with_nan(dataframe)


print(dataframe_info(dataframe))

# CPU_Utilization has 13 NaN Values, Storage_Used_Size_GB has 6 NaN Values, RAM_Utilization has 8 NaN Values
print("\nNull Values:")
print(f'{dataframe.CPU_Utilization.isnull().sum()*100/dataframe.CPU_Utilization.shape[0]}% of CPU_Utilization values are Null.')
print(f'{dataframe.Storage_Used_Size_GB.isnull().sum()*100/dataframe.Storage_Used_Size_GB.shape[0]}% of Storage_Used_Size_GB values are Null.')
print(f'{dataframe.RAM_Utilization.isnull().sum()*100/dataframe.RAM_Utilization.shape[0]}% of RAM_Utilization values are Null.\n')

'''
# Plotting histogram before NaN replacement
fig, axes = plt.subplots(2,4, figsize=(18, 5))

groups = ["CPU_Speed","CPU_Core","CPU_MThread","CPU_Utilization","RAM_Size_GB","RAM_Utilization","Storage_Size_GB","Storage_Used_Size_GB"]

for i, group in enumerate(groups):
    sns.histplot(x= group, data= dataframe,kde= True,ax=axes.flatten()[i])

plt.tight_layout()
plt.savefig("Histogram_Before")
plt.show()
'''
# Using median to fill in the NaN values in column where NaN is present
dataframe = fill_nan(dataframe, ["CPU_Utilization","RAM_Utilization","Storage_Used_Size_GB"])


print(f'Null Values: \n{dataframe.isnull().sum()}')

'''
#Plotting histogram after NaN replacement
fig, axes = plt.subplots(2,4, figsize=(18, 5))

for i, group in enumerate(groups):
    sns.histplot(x= group, data= dataframe,kde = True,ax=axes.flatten()[i])
plt.tight_layout()
plt.savefig("Histogram_After")
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(15, 5))

for i, group in enumerate(groups):
    sns.boxplot(x= group, data= dataframe,ax=axes.flatten()[i])
plt.tight_layout(pad=1.0)
plt.savefig("Boxplot")
plt.show()

countsdf = pd.DataFrame(dataframe.dtypes.value_counts())

# Datatype Frequencies
sns.barplot(x = countsdf.index, y = countsdf["count"], palette= 'pastel')
plt.xlabel("Data Types")
plt.savefig("Barplot")
plt.show()

# Server Type Frequencies
sns.countplot(x = "Server Type", data= dataframe)
plt.savefig("Countplot")
plt.show()

#Enoding Server Type for Pairplot
label = LabelEncoder()
dataframe["Server Type"] = label.fit_transform(dataframe["Server Type"])

plot = sns.pairplot(dataframe, hue="Server Type")
legend = plot.legend
legend.set_title('Server')
legend.texts[0].set_text('Physical')
legend.texts[1].set_text('Virtual')
plt.savefig("Pairplot")
plt.show()

# Columns For Heatmap
heatmapcolumns = ["CPU_Speed","CPU_Core","CPU_MThread","CPU_Utilization","RAM_Size_GB","RAM_Utilization","Storage_Size_GB","Storage_Used_Size_GB"]
heatmap_dataframe = dataframe[heatmapcolumns]

# Generating Heatmap
sns.heatmap(data=heatmap_dataframe.corr(),annot=True, cmap='coolwarm')
plt.savefig("Heatmap")
plt.show()
'''

dataframe.to_csv("cleaned_datacenter.csv")

dataframe = irqmethod(dataframe,'CPU_Utilization')
dataframe = irqmethod(dataframe,'RAM_Utilization')

dataframe = irqmethod(dataframe, 'RAM_Size_GB')
dataframe = irqmethod(dataframe, 'Storage_Used_Size_GB')


dataframe = irqmethod(dataframe, "CPU_Core")
dataframe = irqmethod(dataframe, "CPU_MThread")


list_of_columns = dataframe.columns

cluster_features, scores_list, features_list = [[2,3], [2,4], [2,5], [2,6], [2,7], [2,8], [2,9],
                                                [3,5], [3,7], [3,8], [3,9],
                                                [4,5], [4,7], [4,8], [4,9],
                                                [5,6], [5,7], [5,8], [5,9],
                                                [6,7], [6,8], [6,9],
                                                [7,8], [7,9],
                                                [8,9]],[],[]
'''print("Overall CLuster")
sils2 = kmeans_master_function(dataframe, [2, 6, 9])
scores_list.append(sils2)
features_list.append("CPU_Speed, RAM_Size_GB, Storage_Used_Size_GB")
'''

for i, feature in enumerate(cluster_features):
    versus = f'{list_of_columns[feature[0]]} vs {list_of_columns[feature[1]]}'
    print(f"Cluster {i+1}: {list_of_columns[feature[0]]} vs {list_of_columns[feature[1]]}")
    silscore = kmeans_master_function(dataframe, feature)
    scores_list.append(silscore)
    features_list.append(versus)

scores_dataframe = pd.DataFrame({"Features": features_list, "Score": scores_list})
print(scores_dataframe)


print(sum(scores_list)/len(scores_list))

plt.figure(figsize= (18,7))
sns.barplot(x = "Score", y = "Features", data= scores_dataframe)
plt.savefig("ScoresVSFeatures.png")
plt.show()

# 2 3 4 5 6 7 8 9
# 2,3 2,4 2,5 2,6 2,7 2,8 2,9
# 3,5 3,7 3,8 3,9
# 4,5 4,7 4,8 4,9
# 5,6 5,7 5,8 5,9
# 6,7 6,8 6,9
# 7,8 7,9
# 8,9