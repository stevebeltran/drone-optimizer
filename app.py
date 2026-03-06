import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import os
import itertools

# --- PAGE CONFIG ---
st.set_page_config(page_title="Drone Logistics Portal", layout="wide", page_icon="🛰️")

# --- 1. INITIALIZE SESSION STATE ---
if 'box_open' not in st.session_state:
    st.session_state.box_open = True

# --- 2. HEADER ---
if st.sidebar.button("🔄 Reset & Upload New Data"):
    st.session_state.box_open = True
    st.rerun()

st.title("🛰️ Strategic Drone Deployment Optimizer")
st.markdown("#### **Geospatial Operations Analysis Tool**")

# --- SPEED OPTIMIZATION: CACHING ---
@st.cache_data
def process_geo_data(shp_path, selection):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None: gdf.set_crs(epsg=4269, inplace=True)
    gdf['geometry'] = gdf['geometry'].simplify(0.0001, preserve_topology=True)
    name_col = 'DISTRICT' if 'DISTRICT' in gdf.columns else 'NAME'
    
    if selection == "SHOW ALL DISTRICTS":
        active_gdf = gdf.to_crs(epsg=4326)
        boundary = unary_union(active_gdf.geometry)
    else:
        active_gdf = gdf[gdf[name_col] == selection].to_crs(epsg=4326)
        boundary = active_gdf.iloc[0].geometry
    return gdf, active_gdf, boundary, name_col

# --- 3. DATA IMPORT ---
call_data, station_data, shot_data, shape_components = None, None, None, []

with st.expander("📁 Secure Data Import", expanded=st.session_state.box_open):
    uploaded_files = st.file_uploader("Upload Incident CSVs, 'shots.csv', and Shapefiles", accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        fname = f.name.lower()
        if fname == "calls.csv": call_data = f
        elif fname == "stations.csv": station_data = f
        elif fname == "shots.csv": shot_data = f
        elif any(fname.endswith(ext) for ext in ['.shp', '.shx', '.dbf', '.prj']):
            shape_components.append(f)

    if call_data and station_data and len(shape_components) >= 3:
        if st.session_state.box_open:
            st.session_state.box_open = False
            st.rerun()

STATION_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#800000", "#333333", "#000075"]

# --- 4. MAIN ANALYSIS ENGINE ---
if call_data and station_data and len(shape_components) >= 3:
    if not os.path.exists("temp"): os.mkdir("temp")
    for f in shape_components:
        with open(os.path.join("temp", f.name), "wb") as buffer:
            buffer.write(f.getbuffer())
    
    try:
        shp_path = [os.path.join("temp", f.name) for f in shape_components if f.name.endswith('.shp')][0]
        temp_gdf = gpd.read_file(shp_path)
        name_col_init = 'DISTRICT' if 'DISTRICT' in temp_gdf.columns else 'NAME'
        options = ["SHOW ALL DISTRICTS"] + sorted(temp_gdf[name_col_init].unique().tolist())
        
        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        selection = ctrl_col1.selectbox("📍 Active Jurisdiction Focus", options)

        gdf_all, active_gdf, city_boundary, name_col = process_geo_data(shp_path, selection)
        
        # Calculate UTM Zone for accurate buffering
        utm_zone = int((city_boundary.centroid.x + 180) / 6) + 1
        epsg_code = f"326{utm_zone}" if city_boundary.centroid.y > 0 else f"327{utm_zone}"
        city_m = active_gdf.to_crs(epsg=epsg_code).unary_union
        
        df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
        df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])
        
        gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
        calls_in_city = gdf_calls[gdf_calls.within(city_boundary)].to_crs(epsg=epsg_code)
        calls_in_city['point_idx'] = range(len(calls_in_city))
        
        # PRE-CALC (Dual Profiles)
        radius_resp_m = 3218.69   # 2 Miles
        radius_guard_m = 12874.75 # 8 Miles
        
        station_metadata = []
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            
            # Responder Profile (2-Mile)
            mask_2m = calls_in_city.geometry.distance(s_pt_m) <= radius_resp_m
            indices_2m = set(calls_in_city[mask_2m]['point_idx'])
            clipped_2m = s_pt_m.buffer(radius_resp_m).intersection(city_m)
            
            # Guardian Profile (8-Mile)
            mask_8m = calls_in_city.geometry.distance(s_pt_m) <= radius_guard_m
            indices_8m = set(calls_in_city[mask_8m]['point_idx'])
            clipped_8
