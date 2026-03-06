import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import os
import itertools
import glob
import math
import simplekml
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="BRINC COS Drone Optimizer", layout="wide")

# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------

RESP_RADIUS_MI = 2
GUARD_RADIUS_MI = 8

RESP_RADIUS_M = 3218.69
GUARD_RADIUS_M = 12874.75

STATION_COLORS = [
"#E6194B","#3CB44B","#4363D8","#F58231","#911EB4",
"#800000","#333333","#000075","#808000","#9A6324"
]

# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

def get_circle_coords(lat, lon, r_mi=2.0):

    angles = np.linspace(0, 2*np.pi, 100)

    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)

    return c_lats, c_lons


def calculate_zoom(min_lon, max_lon):

    width = max_lon - min_lon

    if width <= 0:
        return 12

    zoom = 10.5 - np.log(width)

    return min(max(zoom, 10), 15)


# -------------------------------------------------------
# LOAD CSV DATA
# -------------------------------------------------------

uploaded_files = st.file_uploader(
"Upload calls.csv and stations.csv",
accept_multiple_files=True
)

call_data = None
station_data = None

if uploaded_files:

    for f in uploaded_files:

        if f.name.lower() == "calls.csv":
            call_data = f

        if f.name.lower() == "stations.csv":
            station_data = f


if call_data is None or station_data is None:

    st.info("Upload calls.csv and stations.csv to begin.")
    st.stop()


df_calls = pd.read_csv(call_data).dropna(subset=["lat","lon"])
df_stations = pd.read_csv(station_data).dropna(subset=["lat","lon"])

# -------------------------------------------------------
# CONVERT TO GEODATA
# -------------------------------------------------------

gdf_calls = gpd.GeoDataFrame(
df_calls,
geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat),
crs="EPSG:4326"
)

center_lat = df_calls.lat.mean()
center_lon = df_calls.lon.mean()

utm_zone = int((center_lon + 180) / 6) + 1
epsg_code = f"326{utm_zone}"

gdf_calls_utm = gdf_calls.to_crs(epsg=epsg_code)

# -------------------------------------------------------
# STATION METADATA BUILD
# -------------------------------------------------------

station_metadata = []

calls_array = np.array(list(zip(gdf_calls_utm.geometry.x, gdf_calls_utm.geometry.y)))

for _, row in df_stations.iterrows():

    pt = gpd.GeoSeries(
        [Point(row.lon, row.lat)],
        crs="EPSG:4326"
    ).to_crs(epsg=epsg_code).iloc[0]

    dists = np.sqrt(
        (calls_array[:,0] - pt.x)**2 +
        (calls_array[:,1] - pt.y)**2
    )

    resp_mask = dists <= RESP_RADIUS_M
    guard_mask = dists <= GUARD_RADIUS_M

    station_metadata.append({
        "name":row["name"],
        "lat":row["lat"],
        "lon":row["lon"],
        "resp_idx":set(np.where(resp_mask)[0]),
        "guard_idx":set(np.where(guard_mask)[0])
    })


n = len(station_metadata)
total_calls = len(gdf_calls)

# -------------------------------------------------------
# BUILD COVERAGE MATRICES
# -------------------------------------------------------

resp_matrix = np.zeros((n,total_calls),dtype=bool)
guard_matrix = np.zeros((n,total_calls),dtype=bool)

for i,s in enumerate(station_metadata):

    resp_matrix[i,list(s["resp_idx"])] = True
    guard_matrix[i,list(s["guard_idx"])] = True


# -------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------

st.sidebar.header("Deployment Controls")

k_responder = st.sidebar.slider(
"Responder Drones",
0,
n,
min(1,n)
)

k_guardian = st.sidebar.slider(
"Guardian Drones",
0,
n,
0
)

show_satellite = st.sidebar.toggle("Satellite Map",False)

# -------------------------------------------------------
# OPTIMIZER
# -------------------------------------------------------

def evaluate_combo(resp_combo, guard_combo):

    resp_cov = resp_matrix[list(resp_combo)].any(axis=0)

    if len(guard_combo)>0:

        guard_cov = guard_matrix[list(guard_combo)].any(axis=0)
        combined = np.logical_or(resp_cov, guard_cov)

    else:

        combined = resp_cov

    return combined.sum()


station_indices = list(range(n))

resp_combos = list(itertools.combinations(station_indices,k_responder))

best_score = -1
best_combo = None


def search_resp_combo(resp_combo):

    best_local = (-1,None)

    remaining = [i for i in station_indices if i not in resp_combo]

    guard_combos = list(itertools.combinations(remaining,k_guardian))

    for g in guard_combos:

        score = evaluate_combo(resp_combo,g)

        if score > best_local[0]:

            best_local = (score,(resp_combo,g))

    return best_local


with st.spinner("Optimizing deployment..."):

    with ThreadPoolExecutor() as ex:

        results = list(ex.map(search_resp_combo,resp_combos))

    for score,combo in results:

        if score > best_score:

            best_score = score
            best_combo = combo


best_resp_names=[]
best_guard_names=[]

if best_combo:

    r,g = best_combo

    best_resp_names=[station_metadata[i]["name"] for i in r]
    best_guard_names=[station_metadata[i]["name"] for i in g]


# -------------------------------------------------------
# USER SELECTION
# -------------------------------------------------------

active_resp = st.multiselect(
"Active Responders",
df_stations.name.tolist(),
default=best_resp_names
)

active_guard = st.multiselect(
"Active Guardians",
df_stations.name.tolist(),
default=best_guard_names
)

# -------------------------------------------------------
# METRICS
# -------------------------------------------------------

active_resp_idx=[i for i,s in enumerate(station_metadata) if s["name"] in active_resp]
active_guard_idx=[i for i,s in enumerate(station_metadata) if s["name"] in active_guard]

resp_cov = resp_matrix[active_resp_idx].any(axis=0) if active_resp_idx else np.zeros(total_calls,bool)
guard_cov = guard_matrix[active_guard_idx].any(axis=0) if active_guard_idx else np.zeros(total_calls,bool)

combined=np.logical_or(resp_cov,guard_cov)

coverage_pct=combined.sum()/total_calls*100

st.metric("Incident Coverage",f"{coverage_pct:.1f}%")

# -------------------------------------------------------
# MAP
# -------------------------------------------------------

fig = go.Figure()

sample_calls = gdf_calls.sample(min(5000,len(gdf_calls)),random_state=42)

fig.add_trace(go.Scattermapbox(
lat=sample_calls.geometry.y,
lon=sample_calls.geometry.x,
mode="markers",
marker=dict(size=4,color="#000080",opacity=0.35),
name="Incidents"
))


for i,row in df_stations.iterrows():

    name=row["name"]
    color=STATION_COLORS[i%len(STATION_COLORS)]

    if name in active_resp:

        clats,clons=get_circle_coords(row.lat,row.lon,2)

    elif name in active_guard:

        clats,clons=get_circle_coords(row.lat,row.lon,8)

    else:
        continue

    fig.add_trace(go.Scattermapbox(
        lat=clats,
        lon=clons,
        mode="lines",
        line=dict(color=color,width=4),
        name=name
    ))

fig.update_layout(
uirevision="LOCKED_MAP",
mapbox=dict(
center=dict(lat=center_lat,lon=center_lon),
zoom=12,
style="satellite" if show_satellite else "open-street-map"
),
margin=dict(l=0,r=0,t=0,b=0),
height=800
)

st.plotly_chart(fig,use_container_width=True)
