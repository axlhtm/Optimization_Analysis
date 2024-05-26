# =============================================================================
# GENETIC ALGORITHM OPTIMIZATION - TRAVELING SALESMAN PROBLEM (TSP)
# =============================================================================

# Import Python Library 
import folium
import os 
import pandas as pd 
import plotly.express as px
import pygad


# Change Working Directories 
os.chdir("C:/Users/axelh/OneDrive/Documents/Hydroinformatics/Optimization/")

# Import Datasets
data = pd.read_csv('Starbucks_Route.csv')
df = data[data['countryCode']=='GB']
df.reset_index(inplace=True)


vis = df.groupby('city').storeNumber.count().reset_index()
px.bar(vis, x='city', y='storeNumber', template='seaborn')

map = folium.Map(location=[51.509685, -0.118092], zoom_start=6, tiles="stamentoner")
for _, r in df.iterrows():
  folium.Marker(
      [r['latitude'], r['longitude']], popup=f'<i>{r["storeNumber"]}</i>'
  ).add_to(map)