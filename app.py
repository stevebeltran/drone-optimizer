import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import os
import itertools
import glob
import math
import simplekml
from concurrent.futures import ThreadPoolExecutor

# --- PAGE CONFIG ---
st.set_page_config(page_title="brinc COS Drone Optimizer", layout="wide")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    html, body, [class*="css"]  { font-size: 18px !important; }
    div[role="radiogroup"] label div { font-size: 20px !important; }
    .stRadio label p, .stMultiSelect label p { font-size: 22px !important; font-weight: bold !important; }
    div[data-baseweb="select"] span { font-size: 18px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- INITIALIZE GLOBAL VARIABLES ---
# This prevents NameErrors during reruns
city_boundary_geom = None
calls_in_city = pd.DataFrame()
center_lat, center_lon = 0.0, 0.0
dynamic_zoom = 12

# --- LOGO ---
try:
    st.sidebar.image("logo.png", use_container_width=True)
except FileNotFoundError:
    pass

st.title("🛰️ BRINC COS Drone Optimizer")

# --- CONFIGURATION ---
SHAPEFILE_DIR = "jurisdiction_data" 
if not os.path.exists(SHAPEFILE_DIR):
    os.makedirs(SHAPEFILE_DIR)

# --- SIDEBAR: MAP LIBRARY MANAGER ---
with st.sidebar.expander("🗺️ Map Library Manager"):
    st.write("Upload shapefiles here to populate the 'jurisdiction_data' folder.")
    map_files = st.file_uploader("Drop .shp, .shx, .dbf, .prj files", accept_multiple_files=True)
    if map_files:
        count = 0
        for f in map_files:
            with open(os.path.join(SHAPEFILE_DIR, f.name), "wb") as buffer:
                buffer.write(f.getbuffer())
            count += 1
        st.success(f"Saved {count} map files to library!")

# --- MAIN UPLOAD SECTION ---
if 'csvs_ready' not in st.session_state:
    st.session_state['csvs_ready'] = False

with st.expander("📁 Upload Mission Data (CSVs)", expanded=not st.session_state['csvs_ready']):
    uploaded_files = st.file_uploader("Upload 'calls.csv' and 'stations.csv'", accept_multiple_files=True)

# --- HELPER FUNCTIONS ---
def get_circle_coords(lat, lon, r_mi=2.0):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

@st.cache_data
def find_relevant_jurisdictions(calls_df, stations_df, shapefile_dir):
    points_list = []
    if calls_df is not None: points_list.append(calls_df[['lat', 'lon']])
    if stations_df is not None: points_list.append(stations_df[['lat', 'lon']])
    if not points_list: return None
    
    full_points = pd.concat(points_list)
    full_points = full_points[(full_points.lat.abs() > 1) & (full_points.lon.abs() > 1)]
    scan_points = full_points.sample(min(len(full_points), 50000), random_state=42)

    points_gdf = gpd.GeoDataFrame(scan_points, geometry=gpd.points_from_xy(scan_points.lon, scan_points.lat), crs="EPSG:4326")
    total_bounds = points_gdf.total_bounds
    shp_files = glob.glob(os.path.join(shapefile_dir, "*.shp"))
    relevant_polys = []
    
    for shp_path in shp_files:
        try:
            gdf_chunk = gpd.read_file(shp_path, bbox=tuple(total_bounds))
            if not gdf_chunk.empty:
                if gdf_chunk.crs is None: gdf_chunk.set_crs(epsg=4269, inplace=True)
                gdf_chunk = gdf_chunk.to_crs(epsg=4326)
                hits = gpd.sjoin(gdf_chunk, points_gdf, how="inner", predicate="intersects")
                if not hits.empty:
                    valid_indices = hits.index.unique()
                    subset = gdf_chunk.loc[valid_indices].copy()
                    subset['data_count'] = hits.index.value_counts()
                    name_col = next((c for c in ['NAME', 'DISTRICT', 'NAMELSAD'] if c in subset.columns), subset.columns[0])
                    subset['DISPLAY_NAME'] = subset[name_col].astype(str)
                    relevant_polys.append(subset)
        except: continue
            
    if not relevant_polys: return None
    master_gdf = pd.concat(relevant_polys, ignore_index=True).sort_values(by='data_count', ascending=False)
    master_gdf['pct_share'] = master_gdf['data_count'] / master_gdf['data_count'].sum()
    master_gdf['cum_share'] = master_gdf['pct_share'].cumsum()
    mask = (master_gdf['cum_share'] <= 0.98) | (master_gdf['pct_share'] > 0.01)
    mask.iloc[0] = True
    return master_gdf[mask]

def calculate_zoom(min_lon, max_lon):
    width = max_lon - min_lon
    if width <= 0: return 12
    zoom = 10.5 - np.log(width)
    return min(max(zoom, 10), 15)

# --- FILE ROUTING ---
call_data, station_data = None, None
if uploaded_files:
    for f in uploaded_files:
        fname = f.name.lower()
        if fname == "calls.csv": call_data = f
        elif fname == "stations.csv": station_data = f

# --- MAIN LOGIC BLOCK ---
if call_data and station_data:
    if not st.session_state['csvs_ready']:
        st.session_state['csvs_ready'] = True
        st.rerun()

    df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
    df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])

    with st.spinner("🌍 Identifying dominant jurisdictions..."):
        master_gdf = find_relevant_jurisdictions(df_calls, df_stations_all, SHAPEFILE_DIR)

    if master_gdf is None or master_gdf.empty:
        st.error("❌ No matching jurisdictions found.")
        st.stop()

    st.sidebar.success(f"**Found {len(master_gdf)} Significant Zones**")
    st.markdown("---")
    
    # Selection Controls
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    total_pts = master_gdf['data_count'].sum()
    master_gdf['LABEL'] = master_gdf['DISPLAY_NAME'] + " (" + (master_gdf['data_count']/total_pts*100).round(1).astype(str) + "%)"
    options_map = dict(zip(master_gdf['LABEL'], master_gdf['DISPLAY_NAME']))
    selected_labels = ctrl_col1.multiselect("📍 Active Jurisdictions", options=master_gdf['LABEL'].tolist(), default=master_gdf['LABEL'].tolist())
    
    if not selected_labels:
        st.warning("Please select at least one jurisdiction.")
        st.stop()
        
    selected_names = [options_map[l] for l in selected_labels]
    active_gdf = master_gdf[master_gdf['DISPLAY_NAME'].isin(selected_names)]

    # Coordinate Reference System Handling
    center_lon, center_lat = df_calls['lon'].mean(), df_calls['lat'].mean()
    utm_zone = int((center_lon + 180) / 6) + 1
    epsg_code = f"326{utm_zone}" if center_lat > 0 else f"327{utm_zone}"
    
    try:
        active_utm = active_gdf.to_crs(epsg=epsg_code)
        if hasattr(active_utm.geometry, 'union_all'):
            full_boundary_utm = active_utm.geometry.buffer(0.1).union_all().buffer(-0.1)
        else:
            full_boundary_utm = active_utm.geometry.buffer(0.1).unary_union.buffer(-0.1)
            
        city_boundary_geom = gpd.GeoSeries([full_boundary_utm], crs=epsg_code).to_crs(epsg=4326).iloc[0]
        
        gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
        gdf_calls_utm = gdf_calls.to_crs(epsg=epsg_code)
        calls_in_city = gdf_calls_utm[gdf_calls_utm.within(full_boundary_utm)]
    except Exception as e:
        st.error(f"Geometry Error: {e}")

    # --- MAP RENDERING (NOW INSIDE THE DATA GATE) ---
    fig = go.Figure()

    if city_boundary_geom is not None and not city_boundary_geom.is_empty:
        min_lon, min_lat, max_lon, max_lat = city_boundary_geom.bounds
        center_lon, center_lat = (min_lon + max_lon)/2, (min_lat + max_lat)/2
        dynamic_zoom = calculate_zoom(min_lon, max_lon)
    
    mapbox_config = dict(
        center=dict(lat=center_lat, lon=center_lon), 
        zoom=dynamic_zoom, 
        style="open-street-map"
    )

    fig.update_layout(
        uirevision="MAP_LOCK", 
        mapbox=mapbox_config, 
        margin=dict(l=0,r=0,t=0,b=0), 
        height=800, 
        font=dict(size=18)
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

else:
    st.info("👋 Welcome! Please upload your 'calls.csv' and 'stations.csv' to begin.")
