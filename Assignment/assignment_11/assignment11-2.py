import streamlit as st
import pandas as pd
import pydeck as pdk

df = pd.read_csv('RainDaily_Tabular.csv')

st.sidebar.header('Sidebar')
province = st.sidebar.selectbox(
    'Province',
    df['province'].unique()
)
date = st.sidebar.date_input(
    'Date',
    min_value=pd.to_datetime(df['date']).dt.date.min(),
    max_value=pd.to_datetime(df['date']).dt.date.max(),
    value=pd.to_datetime(df['date']).dt.date.min()
)
area = st.sidebar.selectbox(
    'Area',
    df[df['province'] == province]['name'].unique()
)

st.title('Streamlit Rain Daily Tabular')

st.subheader('Province Data', province)

col1, col2 = st.columns(2)

province_data = df[df['province'] == province]
with col1:
    st.write('Rainfall Amount on', date.strftime('%d/%m/%Y'))
    st.bar_chart(province_data[province_data['date'] == str(date)].set_index('name')['rain'])


with col2:
    st.write('Average Rainfall Amount')
    st.bar_chart(province_data.groupby('name')['rain'].mean())


st.write('Rainfall Amount for Area', area, 'in Province', province)
column1, column2 = st.columns([3, 1])

with column1:
    st.line_chart(province_data[province_data['name'] == area].set_index('date')['rain'])

with column2:
    st.write(province_data[province_data['name'] == area]['rain'].describe())


st.write('Average Rainfall Amount for Province', province)
column1, column2 = st.columns([3, 1])
with column1:
    st.line_chart(province_data.groupby('date')['rain'].mean())

with column2:
    st.write(province_data.groupby('date')['rain'].mean().describe())


st.write('Rainfall Amount for All Areas in Province', province)
st.line_chart(province_data.groupby(['date', 'name'])['rain'].mean().unstack())
  
st.subheader('Country-wide Data')

st.write('Average Rainfall Amount for the Entire Country on', date.strftime('%d/%m/%Y'))
date_data_df = df[df['date'] == str(date)]
column1, column2 = st.columns([9, 1.5])
with column1:
    st.bar_chart(date_data_df.groupby('province')['rain'].mean())

with column2:
    st.write(date_data_df.groupby('province')['rain'].mean().describe())

st.write('Average Rainfall Amount for the Entire Country')
column1, column2 = st.columns([3, 1])
with column1:
    st.bar_chart(df.groupby('province')['rain'].mean())

with column2:
    st.write(df.groupby('province')['rain'].mean().describe())


def create_map(dataframe):
    layer = pdk.Layer(
        "HeatmapLayer",
        dataframe,
        get_position=["longitude", "latitude"],
        opacity=0.5,
        pickable=True
    )

    view_state = pdk.ViewState(
        longitude=dataframe['longitude'].mean(),
        latitude=dataframe['latitude'].mean(),
        zoom=1
    )

    return pdk.Deck(layers=[layer], initial_view_state=view_state)

st.subheader('Map of Rainfall Amount on' + date.strftime('%d/%m/%Y'))
map = create_map(date_data_df[['latitude', 'longitude', 'rain']])
st.pydeck_chart(map)

st.subheader('Summary of Rainfall Analysis')
st.write("The analysis above provides insights into the rainfall patterns in different areas over time. It includes visualizations such as bar charts, line charts, and a heatmap to illustrate the distribution of rainfall amounts across provinces, areas, and dates. This information can be valuable for understanding and predicting precipitation trends for various regions.")
