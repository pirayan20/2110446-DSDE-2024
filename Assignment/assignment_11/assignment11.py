import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import pydeck as pdk

# Load dataset
df = pd.read_csv('RainDaily_Tabular.csv')

# Convert 'date' column to datetime for time series processing
df['date'] = pd.to_datetime(df['date'])

# K-means Clustering for Locations
coordinates = df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=5, random_state=0).fit(coordinates)
df['cluster'] = kmeans.labels_  # Ensure the cluster labels are added here

# Calculate centroids and map them to cluster labels for more readable naming
centroids = kmeans.cluster_centers_
cluster_names = {i: f'Cluster {i} (Lat: {centroids[i][0]:.4f}, Lon: {centroids[i][1]:.4f})' for i in range(5)}
df['cluster_name'] = df['cluster'].map(cluster_names)

# Prepare data for rainfall by date and cluster
average_rain_by_date_cluster = df.groupby(['date', 'cluster_name'])['rain'].mean().reset_index()

# Prepare data for rainfall by province
average_rain_by_province = df.groupby('province')['rain'].mean().reset_index()
average_rain_by_province = average_rain_by_province.sort_values(by='rain', ascending=True)

# Title of the app
st.title('Rainfall Analysis App')

# Section for Rainfall by Province
st.header('Average Rainfall by Province')

# Checkbox for provinces
selected_provinces = []
for province in average_rain_by_province['province']:
    checkbox_state = st.sidebar.checkbox(province, value=True, key=province + 'prov')
    if checkbox_state:
        selected_provinces.append(province)

# Filter data based on selected provinces
selected_data_province = average_rain_by_province[average_rain_by_province['province'].isin(selected_provinces)]

# Plot for rainfall by province
fig_province = px.bar(selected_data_province,
                      x='rain',
                      y='province',
                      orientation='h',
                      title='Average Rainfall by Province',
                      labels={'rain': 'Average Rainfall (mm)', 'province': 'Province'})
fig_province.update_layout(height=800, width=800)
st.plotly_chart(fig_province)

# Section for Clustering and Rainfall over Time
st.header('Average Rainfall by Date and Cluster')

# Slider for selecting date range
min_date = df['date'].min().date()
max_date = df['date'].max().date()
start_date, end_date = st.sidebar.slider('Select Date Range for Clusters:'
                                         , min_value=min_date
                                         , max_value=max_date
                                         , value=(min_date, max_date)
                                         , format='YYYY-MM-DD')
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Checkboxes for selecting clusters
cluster_selection = []
for cluster_label, cluster_desc in cluster_names.items():
    if st.sidebar.checkbox(f'Show {cluster_desc}', True, key=cluster_desc):
        cluster_selection.append(cluster_desc)

# Filtering data for selected date range and clusters
selected_data_cluster = average_rain_by_date_cluster[(average_rain_by_date_cluster['date'] >= start_date) & (average_rain_by_date_cluster['date'] <= end_date)]
selected_data_cluster = selected_data_cluster[selected_data_cluster['cluster_name'].isin(cluster_selection)]

# Plot for rainfall by cluster
fig_cluster = px.line(selected_data_cluster,
                      x='date',
                      y='rain',
                      color='cluster_name',
                      title='Average Rainfall by Date and Cluster',
                      labels={'rain': 'Average Rainfall (mm)', 'date': 'Date', 'cluster_name': 'Cluster'})
fig_cluster.update_layout(height=600, width=800)
st.plotly_chart(fig_cluster)

st.header("Centroid Map")

# Start with an empty figure
fig_centroids = go.Figure()

# Add the centroids to the figure as scatter geo points
for idx, centroid in enumerate(centroids):
    fig_centroids.add_trace(go.Scattergeo(
        lon=[centroid[1]],
        lat=[centroid[0]],
        mode='markers+text',
        text=[f'Cluster {idx}'],
        textposition='bottom center',
        marker=dict(
            size=10,
            color='red',  # Red color to highlight the centroids
            line=dict(
                width=2,
                color='black'
            ),
            symbol='x'
        ),
        name=f'Centroid {idx}'
    ))

# Set the layout for the map
fig_centroids.update_layout(
    title='Centroids of Rainfall Clusters',
    geo=dict(
        scope='asia',
        center=dict(lat=13.736717, lon=100.523186),
        showland=True,
        landcolor='lightgrey',
        countrycolor='darkgrey',
        showcountries=True,
        showocean=True,
        oceancolor='lightblue',
        lataxis=dict(range=[5.5, 20.5]),  # Range of latitudes to "zoom" in on Thailand
        lonaxis=dict(range=[97.5, 105.5])
    ),
    height=800,
    width=800,
    margin=dict(t=0, l=0, r=0, b=0)
)

# Show the centroids map in Streamlit
st.plotly_chart(fig_centroids, use_container_width=True)

df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

st.header('Map of Rainfall Locations')

layer = pdk.Layer(
    "HeatmapLayer",
    df,
    get_position=["logitude", "latitude"],
    get_weight="rain",
    radius=100,  # Adjust the radius to increase the area of influence
    intensity=1,  # Adjust intensity to make the heatmap more sensitive to lower values
    threshold=0.1  
)

view_state = pdk.ViewState(latitude=13.736717, longitude=100.523186, zoom=10)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style='mapbox://styles/mapbox/light-v9'
)

st.pydeck_chart(r)

# Ensure latitude and longitude are of type float, and 'rain' is available
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)
df['rain'] = df['rain'].astype(float)  # Ensure rain is a float if you want to use it in any calculations

df['size'] = df['rain'].apply(lambda x: x / max(df['rain']) * 1)  # Normalize for visual

# Display the map with the data
st.map(df[['latitude', 'longitude']])

text_content = """ For the analysis of tha rain data with the correlation between the location and the date.
At first glance, I though using the K-mean clustering would clear things up, and would tell us what region of thailand will have the most rain in the period of time, unfortunately, we can see that the centroids for each region is divided evenly across Thailand. Then we look at the bar graph and notice that, in the south-region of Thailand tends to have more
avg rainfall than the other region. 
Which is not surprising, because they are surrounded with ocean and seas
"""

# Display the text content in a text area
st.header("Analysis")
st.text_area("Detailed Explanation", text_content, height=200, disabled=True)


dynamic_code = """
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import plotly.graph_objs as go

# Load dataset
df = pd.read_csv('RainDaily_Tabular.csv')

# Convert 'date' column to datetime for time series processing
df['date'] = pd.to_datetime(df['date'])

# K-means Clustering for Locations
coordinates = df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=5, random_state=0).fit(coordinates)
df['cluster'] = kmeans.labels_  # Ensure the cluster labels are added here

# Calculate centroids and map them to cluster labels for more readable naming
centroids = kmeans.cluster_centers_
cluster_names = {i: f'Cluster {i} (Lat: {centroids[i][0]:.4f}, Lon: {centroids[i][1]:.4f})' for i in range(5)}
df['cluster_name'] = df['cluster'].map(cluster_names)

# Prepare data for rainfall by date and cluster
average_rain_by_date_cluster = df.groupby(['date', 'cluster_name'])['rain'].mean().reset_index()

# Prepare data for rainfall by province
average_rain_by_province = df.groupby('province')['rain'].mean().reset_index()
average_rain_by_province = average_rain_by_province.sort_values(by='rain', ascending=True)

# Title of the app
st.title('Rainfall Analysis App')

# Section for Rainfall by Province
st.header('Average Rainfall by Province')

# Checkbox for provinces
selected_provinces = []
for province in average_rain_by_province['province']:
    checkbox_state = st.sidebar.checkbox(province, value=True, key=province + 'prov')
    if checkbox_state:
        selected_provinces.append(province)

# Filter data based on selected provinces
selected_data_province = average_rain_by_province[average_rain_by_province['province'].isin(selected_provinces)]

# Plot for rainfall by province
fig_province = px.bar(selected_data_province,
                      x='rain',
                      y='province',
                      orientation='h',
                      title='Average Rainfall by Province',
                      labels={'rain': 'Average Rainfall (mm)', 'province': 'Province'})
fig_province.update_layout(height=800, width=800)
st.plotly_chart(fig_province)

# Section for Clustering and Rainfall over Time
st.header('Average Rainfall by Date and Cluster')

# Slider for selecting date range
min_date = df['date'].min().date()
max_date = df['date'].max().date()
start_date, end_date = st.sidebar.slider('Select Date Range for Clusters:'
                                         , min_value=min_date
                                         , max_value=max_date
                                         , value=(min_date, max_date)
                                         , format='YYYY-MM-DD')
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Checkboxes for selecting clusters
cluster_selection = []
for cluster_label, cluster_desc in cluster_names.items():
    if st.sidebar.checkbox(f'Show {cluster_desc}', True, key=cluster_desc):
        cluster_selection.append(cluster_desc)

# Filtering data for selected date range and clusters
selected_data_cluster = average_rain_by_date_cluster[(average_rain_by_date_cluster['date'] >= start_date) & (average_rain_by_date_cluster['date'] <= end_date)]
selected_data_cluster = selected_data_cluster[selected_data_cluster['cluster_name'].isin(cluster_selection)]

# Plot for rainfall by cluster
fig_cluster = px.line(selected_data_cluster,
                      x='date',
                      y='rain',
                      color='cluster_name',
                      title='Average Rainfall by Date and Cluster',
                      labels={'rain': 'Average Rainfall (mm)', 'date': 'Date', 'cluster_name': 'Cluster'})
fig_cluster.update_layout(height=600, width=800)
st.plotly_chart(fig_cluster)

st.header("Centroid Map")

# Start with an empty figure
fig_centroids = go.Figure()

# Add the centroids to the figure as scatter geo points
for idx, centroid in enumerate(centroids):
    fig_centroids.add_trace(go.Scattergeo(
        lon=[centroid[1]],
        lat=[centroid[0]],
        mode='markers+text',
        text=[f'Cluster {idx}'],
        textposition='bottom center',
        marker=dict(
            size=10,
            color='red',  # Red color to highlight the centroids
            line=dict(
                width=2,
                color='black'
            ),
            symbol='x'
        ),
        name=f'Centroid {idx}'
    ))

# Set the layout for the map
fig_centroids.update_layout(
    title='Centroids of Rainfall Clusters',
    geo=dict(
        scope='asia',
        center=dict(lat=13.736717, lon=100.523186),
        showland=True,
        landcolor='lightgrey',
        countrycolor='darkgrey',
        showcountries=True,
        showocean=True,
        oceancolor='lightblue',
        lataxis=dict(range=[5.5, 20.5]),  # Range of latitudes to "zoom" in on Thailand
        lonaxis=dict(range=[97.5, 105.5])
    ),
    height=800,
    width=800,
    margin=dict(t=0, l=0, r=0, b=0)
)

# Show the centroids map in Streamlit
st.plotly_chart(fig_centroids, use_container_width=True)


st.header('Map of Rainfall Locations')

# Ensure latitude and longitude are of type float, and 'rain' is available
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)
df['rain'] = df['rain'].astype(float)  # Ensure rain is a float if you want to use it in any calculations

# Optionally, you could create a size column based on rain to hint at rainfall amount
# This won't affect the st.map marker size, but you can use it in tooltips in other types of maps
df['size'] = df['rain'].apply(lambda x: x / max(df['rain']) * 1)  # Normalize for visual

# Display the map with the data
st.map(df[['latitude', 'longitude']])

"""

st.header('Code')
st.code(dynamic_code, language='python')