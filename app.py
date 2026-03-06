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

# --- MAP STATE MEMORY (ADDED) ---
if "map_state" not in st.session_state:
    st.session_state.map_state = {
        "center_lat": None,
        "center_lon": None,
        "zoom": None
    }


# --- CUSTOM CSS FOR FONT SIZES ---
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-size: 18px !important; 
    }

    div[role="radiogroup"] label div {
        font-size: 20px !important;
    }

    .stRadio label p, .stMultiSelect label p {
        font-size: 22px !important;
        font-weight: bold !important;
    }

    div[data-baseweb="select"] span {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

STATION_COLORS = [
    "#E6194B","#3CB44B","#4363D8","#F58231","#911EB4",
    "#800000","#333333","#000075","#808000","#9A6324"
]


def get_circle_coords(lat, lon, r_mi=2.0):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons


# ---------- (ALL YOUR EXISTING FUNCTIONS REMAIN IDENTICAL) ----------
# generate_kml(...)
# find_relevant_jurisdictions(...)
# optimizer logic
# metrics
# sidebar controls
# etc.

# (Nothing changed in these large sections)

# -------------------------------------------------------------------


# --- MAP RENDERING ---
fig = go.Figure()

def calculate_zoom(min_lon, max_lon):
    width = max_lon - min_lon
    if width <= 0:
        return 12
    zoom = 10.5 - np.log(width)
    return min(max(zoom, 10), 15)


# (all your existing traces remain identical)


# ----------- REPLACED MAP VIEW BLOCK -----------

if city_boundary_geom is not None and not city_boundary_geom.is_empty:

    min_lon, min_lat, max_lon, max_lat = city_boundary_geom.bounds
    default_center_lon = (min_lon + max_lon) / 2
    default_center_lat = (min_lat + max_lat) / 2
    default_zoom = calculate_zoom(min_lon, max_lon)

elif not calls_in_city.empty:

    q_low = calls_in_city.geometry.x.quantile(0.01)
    q_high = calls_in_city.geometry.x.quantile(0.99)

    clean_pts = calls_in_city[
        (calls_in_city.geometry.x >= q_low) &
        (calls_in_city.geometry.x <= q_high)
    ]

    if clean_pts.empty:
        clean_pts = calls_in_city

    min_lon, min_lat = clean_pts.geometry.x.min(), clean_pts.geometry.y.min()
    max_lon, max_lat = clean_pts.geometry.x.max(), clean_pts.geometry.y.max()

    default_center_lon = (min_lon + max_lon) / 2
    default_center_lat = (min_lat + max_lat) / 2
    default_zoom = calculate_zoom(min_lon, max_lon)

else:

    default_center_lat = 39
    default_center_lon = -96
    default_zoom = 4


if st.session_state.map_state["center_lat"] is None:

    st.session_state.map_state["center_lat"] = default_center_lat
    st.session_state.map_state["center_lon"] = default_center_lon
    st.session_state.map_state["zoom"] = default_zoom


center_lat = st.session_state.map_state["center_lat"]
center_lon = st.session_state.map_state["center_lon"]
dynamic_zoom = st.session_state.map_state["zoom"]


# ----------- MAPBOX CONFIG -----------

mapbox_config = dict(
    center=dict(lat=center_lat, lon=center_lon),
    zoom=dynamic_zoom,
    style="open-street-map"
)


if show_satellite:

    mapbox_config["style"] = "carto-positron"

    mapbox_config["layers"] = [
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "Esri, Maxar, Earthstar Geographics",
            "source": [
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ]
        }
    ]


# ----------- MAP LOCK (CHANGED) -----------

data_signature = "MAP_LOCK"


fig.update_layout(
    uirevision=data_signature,
    mapbox=mapbox_config,
    margin=dict(l=0, r=0, t=0, b=0),
    height=800,
    font=dict(size=18)
)


st.plotly_chart(
    fig,
    use_container_width=True,
    config={"scrollZoom": True}
)
