
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import geopy
import matplotlib.pyplot as plt
import plotly_express as px
import folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm.notebook import tqdm
from shapely.geometry import LineString
from time import time
from google.colab import drive
import haversine as hs
from haversine import Unit
from collections import defaultdict
from math import sqrt
import os

drive.mount('/content/drive')
tqdm.pandas()

"""**Merge Datasets**"""

#To edit a file path
path  = '..\\..\\DigiLabs - Data Science Accelerator Program\\Week-2\\Digilabs Group 9\\Datasets - BFA'

files = os.listdir(path)

merged_df = pd.DataFrame()

for file in files:
    temp_df = pd.read_csv(path + '\\' + file)   # read csv file
    merged_df = merged_df.append(temp_df)     # append to merged_df

print(merged_df.info())

merged_df.fillna(merged_df.mean(), inplace=True)
print(merged_df.info())
merged_df.to_csv('merged_df.csv', index=False)

df = merged_df
df.drop(['Timestamp', 'Horizontal Accuracy',
         'Elevation Change'], axis=1, inplace=True)
coordinates = df.to_numpy()

print(coordinates)

"""# Cleaning with Ramer-Douglas-Peucker (RDP)

"""

line = LineString(coordinates)
tolerance = 0.00001
simplified_line = line.simplify(tolerance, preserve_topology=False)
print(len(line.coords), 'coordinate pairs in full data set')
print(len(simplified_line.coords), 'coordinate pairs in simplified data set')
print(round(((1 - float(len(simplified_line.coords)) /
              float(len(line.coords))) * 100), 1), 'percent compressed')

lon = pd.Series(pd.Series(simplified_line.coords.xy)[1])
lat = pd.Series(pd.Series(simplified_line.coords.xy)[0])
si = pd.DataFrame({'Longitude': lon, 'Latitude': lat})

print(si.tail())

#new empty dataframe
rs = pd.DataFrame()

start_time = time()
for si_i, si_row in si.iterrows():
    #To keep track of the count, might take a long process as there are 5106 rows
    print(si_i)
    si_coords = (si_row['Latitude'], si_row['Longitude'])
    for df_i, df_row in df.iterrows():
        if si_coords == (df_row['Latitude'], df_row['Longitude']):
            #add row to rs
            rs = pd.concat([rs, pd.DataFrame.from_records([df_row])], ignore_index=True)
            break
print ('process took %s seconds' % round(time() - start_time, 2))

print(rs.tail())

#rs to csv
rs.to_csv('rs.csv', index=False)

# Plot as scatter plot 
plt.figure(figsize=(80, 56), dpi=200)
rs_scatter = plt.scatter(rs['Longitude'], rs['Latitude'], c='m', alpha=.8, s=250)
df_scatter = plt.scatter(df['Longitude'], df['Latitude'], c='k', alpha=.3, s=10)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('simplified set of coordinate points vs original full set')
plt.legend((rs_scatter, df_scatter),
           ('simplified', 'original'), loc='upper left')
plt.show()

# Folium Map 
map = folium.Map(location=[rs.Latitude.mean(), rs.Longitude.mean()], zoom_start=14, control_scale=True)
                 
for index, location_info in rs.iterrows():
    folium.Marker([rs["Latitude"], rs["Longitude"]]).add_to(map)
  
map

df = rs
df2 = df[["Latitude", "Longitude"]]
df2["geom"] = df2["Latitude"].map(str) + "," + df2["Longitude"].map(str)
locator = Nominatim(user_agent="myGeocoder", timeout=10)
rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.0001)
df2["address"] = df2["geom"].progress_apply(rgeocode)
df2.to_csv("address.csv")

region_dict = {
    'North' : ['Admirality','Kranji','Woodlands','Sembawang','Yishun','Yio Chu Kang', 'Seletar', 'Sengkang'],
    'South' : ['Holland','Queenstown','Bukit Merah','Telok Blangah','Pasir Panjang','Sentosa','Bukit Timah','Newton','City','Marina South'],
    'East' : ['Serangoon','Punggol','Hougang','Tampines','Pasir Ris','Loyang','Simei','Kallang','Katong','East Coast','Macpherson','Bedok','Pulau Ubin','Pulau Tekong'],
    'West' : ['Lim Chu Kang','Choa Chu Kang','Bukit Panjang','Tuas','Jurong East','Jurong West','Jurong Industrial Estate','Bukit Batok','Hillview','West Coast','Clementi'],
    'Central' : ['Thomson','Marymount','Sin Ming','Ang Mo Kio','Bishan','Serangoon Gardens','MacRitchie','Toa Payoh']
}

temp_df = pd.read_csv('address.csv')

#copy test_df
new_df = temp_df.copy()
#drop address
new_df.drop(['address'], axis=1, inplace=True)

#for loop in df.adress 
for i in range(len(temp_df)):
        address_arr = temp_df.iloc[i]['address'].split(',')
        location = address_arr[-4].strip()
        region = address_arr[-3].strip()
        postalCode = address_arr[-2].strip()
        for key,value in region_dict.items():
            if region in value:
                #create new col in new_df
                new_df.loc[i,'level_1'] = key
                new_df.loc[i,'level_2'] = region 
                new_df.loc[i, 'level_3'] = postalCode
                new_df.loc[i, 'level_4'] = location
                break
            else:
                new_df.loc[i,'level_1'] = 'Others'
                new_df.loc[i,'level_2'] = region 
                new_df.loc[i, 'level_3'] = postalCode 
                new_df.loc[i, 'level_4'] = location

new_df.to_csv('cleaned_data.csv', index=False)

print(new_df)

#Importing Clean Data
df = pd.read_csv("cleaned_data.csv")

#convert df['Latitude'] and df['Longitude'] to list of lists
df_coords = [[float(df['Latitude'][i]), float(df['Longitude'][i])] for i in range(len(df))]

def distance_between_coords(x1, y1, x2, y2):
    loc1 = (x1, y1)
    loc2 = (x2, y2)
    return hs.haversine(loc1, loc2, unit=Unit.METERS)

# Adds "names" to coordinates to use as keys for edge detection
#Use number instead of names as nodes are very close to each other and are repetitve (e.g. Ang Mo Kio Hospital)
def name_coords(coords):
    coord_count = 0
    for coord in coords:
        coord_count += 1
        coord.append(coord_count)
    return coords

#Generate a weighted graph by comparing all nodes to all other nodes by lat and long (undirected)
# Returns named coordinates and their connected edges as a dictionary
def graph(coords):
    coords = name_coords(coords)
    graph = defaultdict(list)
    edges = {}
    for current in coords:
        for comparer in coords:
            if comparer == current:
                continue
            else:
                weight = distance_between_coords(current[0], current[1],
                                                 comparer[0], comparer[1])
                graph[current[2]].append(comparer[2])
                edges[current[2], comparer[2]] = weight
    return coords, edges

#Dijkstra's Algorithm
# Returns a path to all nodes with least weight as a list of names
# from a specific node
# Try and tested for all nodes, showing only the best path and shortest route at the end
def shortest_path(node_list, edges, start):
    neighbor = 0
    unvisited = []
    visited = []
    total_weight = 0
    current_node = start
    for node in node_list:
        if node[2] == start:
            visited.append(start)
        else:
            unvisited.append(node[2])
    while unvisited:
        for index, neighbor in enumerate(unvisited):
            if index == 0:
                current_weight = edges[start, neighbor]
                current_node = neighbor
            elif edges[start, neighbor] < current_weight:
                current_weight = edges[start, neighbor]
                current_node = neighbor
        total_weight += current_weight
        unvisited.remove(current_node)
        visited.append(current_node)
    return visited, total_weight

#Main Function to run and test all the algorithms
def driver():
    coords = df_coords
    coords, edges = graph(coords)
    shortest_path(coords, edges, 3)
    shortest_path_taken = []
    shortest_path_weight = 0

    for index, node in enumerate(coords):
        path, weight = shortest_path(coords, edges, index + 1)
        print('--------------------------------------')
        print("Path", index + 1, "=", path)
        print("Weight =", weight)
        if index == 0:
            shortest_path_weight = weight
            shortest_path_taken = path
        elif weight < shortest_path_weight:
            shortest_path_weight = weight
            shortest_path_taken = path
    print('--------------------------------------')
    print('Final Output, most efficient path:')
    print("The shortest path to all nodes is:", shortest_path_taken)
    print("The weight of the path is:", shortest_path_weight)

driver()