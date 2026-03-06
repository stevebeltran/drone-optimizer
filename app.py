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

# --- PAGE CONFIG ---
st.set_page_config(page_title="brinc COS Drone Optimizer", layout="wide")

# --- CUSTOM CSS FOR FONT SIZES ---
st.markdown(
    """
    <style>
    /* 1. Global font size for standard text */
    html, body, [class*="css"]  {
        font-size: 18px !important; 
    }

    /* 2. Change the font size of the Radio Button Options (Maximize Land Coverage, etc.) */
    div[role="radiogroup"] label div {
        font-size: 20px !important;
    }

    /* 3. Change the font size of the main Widget Titles (Optimization Goal:, Active Responders, etc.) */
    .stRadio label p, .stMultiSelect label p {
        font-size: 22px !important;
        font-weight: bold !important;
    }

    /* 4. Change the font size of the Multi-Select box items */
    div[data-baseweb="select"] span {
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- LOGO ---
# Save your logo file in the same folder as this script. 
# Change "logo.png" to your actual file name (e.g., "brinc_logo.jpg")
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

# --- MAIN UPLOAD SECTION (CSVs ONLY) ---
if 'csvs_ready' not in st.session_state:
    st.session_state['csvs_ready'] = False

with st.expander("📁 Upload Mission Data (CSVs)", expanded=not st.session_state['csvs_ready']):
    uploaded_files = st.file_uploader("Upload 'calls.csv' and 'stations.csv'", accept_multiple_files=True)

# High-Contrast Palette
STATION_COLORS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", 
    "#800000", "#333333", "#000075", "#808000", "#9A6324"
]

def get_circle_coords(lat, lon, r_mi=2.0):
    """Generates Lat/Lon coordinates for a circle."""
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

# --- KML EXPORT FUNCTION (UPDATED FOR DUAL FLEET) ---
def generate_kml(active_gdf, df_stations_all, active_resp_names, active_guard_names, calls_gdf):
    kml = simplekml.Kml()
    
    # 1. Add Boundaries
    fol_bounds = kml.newfolder(name="Jurisdictions")
    for _, row in active_gdf.iterrows():
        geoms = [row.geometry] if isinstance(row.geometry, Polygon) else row.geometry.geoms
        for geom in geoms:
            pol = fol_bounds.newpolygon(name=row.get('DISPLAY_NAME', 'Boundary'))
            pol.outerboundaryis = list(geom.exterior.coords)
            pol.style.linestyle.color = simplekml.Color.red
            pol.style.linestyle.width = 3
            pol.style.polystyle.color = simplekml.Color.changealphaint(30, simplekml.Color.red)

    # 2. Add Active Stations & RINGS
    fol_stations = kml.newfolder(name="Stations Points")
    fol_rings = kml.newfolder(name="Coverage Rings")

    def add_kml_station(row, radius, color, name_prefix):
        pnt = fol_stations.newpoint(name=f"{name_prefix} {row['name']}")
        pnt.coords = [(row['lon'], row['lat'])]
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/blu-blank.png'
        lats, lons = get_circle_coords(row['lat'], row['lon'], r_mi=radius)
        ring_coords = list(zip(lons, lats))
        ring_coords.append(ring_coords[0])
        pol = fol_rings.newpolygon(name=f"Range: {row['name']}")
        pol.outerboundaryis = ring_coords
        pol.style.linestyle.color = color
        pol.style.linestyle.width = 2
        pol.style.polystyle.color = simplekml.Color.changealphaint(60, color)

    for _, row in df_stations_all[df_stations_all['name'].isin(active_resp_names)].iterrows():
        add_kml_station(row, 2.0, simplekml.Color.blue, "[Responder]")
        
    for _, row in df_stations_all[df_stations_all['name'].isin(active_guard_names)].iterrows():
        add_kml_station(row, 8.0, simplekml.Color.orange, "[Guardian]")

    # 3. Add Sample of Calls (Also given fixed random state here for consistency!)
    fol_calls = kml.newfolder(name="Incident Data (Sample)")
    calls_export = calls_gdf.to_crs(epsg=4326)
    if len(calls_export) > 2000:
        calls_export = calls_export.sample(2000, random_state=42)
        
    for _, row in calls_export.iterrows():
        pnt = fol_calls.newpoint()
        pnt.coords = [(row.geometry.x, row.geometry.y)]
        pnt.style.iconstyle.scale = 0.5
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'

    return kml.kml()

# --- INTELLIGENT SCANNER ---
@st.cache_data
def find_relevant_jurisdictions(calls_df, stations_df, shapefile_dir):
    points_list = []
    if calls_df is not None:
        points_list.append(calls_df[['lat', 'lon']])
    if stations_df is not None:
        points_list.append(stations_df[['lat', 'lon']])
    
    if not points_list: return None
    
    full_points = pd.concat(points_list)
    full_points = full_points[(full_points.lat.abs() > 1) & (full_points.lon.abs() > 1)]
    
    if len(full_points) > 50000:
        scan_points = full_points.sample(50000, random_state=42)
    else:
        scan_points = full_points

    points_gdf = gpd.GeoDataFrame(
        scan_points, 
        geometry=gpd.points_from_xy(scan_points.lon, scan_points.lat), 
        crs="EPSG:4326"
    )
    
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
        except Exception:
            continue
            
    if not relevant_polys: return None
        
    master_gdf = pd.concat(relevant_polys, ignore_index=True)
    master_gdf = master_gdf.sort_values(by='data_count', ascending=False)
    
    total_scanned_points = master_gdf['data_count'].sum()
    
    if total_scanned_points > 0:
        master_gdf['pct_share'] = master_gdf['data_count'] / total_scanned_points
        master_gdf['cum_share'] = master_gdf['pct_share'].cumsum()
        mask = (master_gdf['cum_share'] <= 0.98) | (master_gdf['pct_share'] > 0.01)
        mask.iloc[0] = True
        return master_gdf[mask]
    
    return master_gdf

# --- FILE ROUTING ---
call_data, station_data = None, None
if uploaded_files:
    for f in uploaded_files:
        fname = f.name.lower()
        if fname == "calls.csv": call_data = f
        elif fname == "stations.csv": station_data = f

# --- MAIN LOGIC ---
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
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    
    total_pts = master_gdf['data_count'].sum()
    master_gdf['LABEL'] = master_gdf['DISPLAY_NAME'] + " (" + (master_gdf['data_count']/total_pts*100).round(1).astype(str) + "%)"
    
    options_map = dict(zip(master_gdf['LABEL'], master_gdf['DISPLAY_NAME']))
    all_options = master_gdf['LABEL'].tolist()
    
    selected_labels = ctrl_col1.multiselect(
        "📍 Active Jurisdictions", 
        options=all_options, 
        default=all_options
    )
    
    if not selected_labels:
        st.warning("Please select at least one jurisdiction.")
        st.stop()
        
    selected_names = [options_map[l] for l in selected_labels]
    active_gdf = master_gdf[master_gdf['DISPLAY_NAME'].isin(selected_names)]

    center_lon = df_calls['lon'].mean()
    center_lat = df_calls['lat'].mean()
    utm_zone = int((center_lon + 180) / 6) + 1
    epsg_code = f"326{utm_zone}" if center_lat > 0 else f"327{utm_zone}"
    
    city_m = None
    city_boundary_geom = None
    
    try:
        active_utm = active_gdf.to_crs(epsg=epsg_code)
        
        if hasattr(active_utm.geometry, 'union_all'):
            full_boundary_utm = active_utm.geometry.buffer(0.1).union_all().buffer(-0.1)
        else:
            full_boundary_utm = active_utm.geometry.buffer(0.1).unary_union.buffer(-0.1)
            
        city_m = full_boundary_utm
        city_boundary_geom = gpd.GeoSeries([full_boundary_utm], crs=epsg_code).to_crs(epsg=4326).iloc[0]
        
    except Exception as e:
        st.error(f"Geometry Error: {e}")
        st.stop()

    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    gdf_calls_utm = gdf_calls.to_crs(epsg=epsg_code)
    
    try:
        calls_in_city = gdf_calls_utm[gdf_calls_utm.within(city_m)]
    except:
        calls_in_city = gdf_calls_utm
        
    calls_in_city['point_idx'] = range(len(calls_in_city))
    
    # --- PRE-CALC DUAL PROFILES ---
    radius_resp_m = 3218.69   # 2 Miles
    radius_guard_m = 12874.75 # 8 Miles
    
    station_metadata = []
    
    if not calls_in_city.empty:
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            
            mask_2m = calls_in_city.geometry.distance(s_pt_m) <= radius_resp_m
            indices_2m = set(calls_in_city[mask_2m]['point_idx'])
            full_buf_2m = s_pt_m.buffer(radius_resp_m)
            try: clipped_2m = full_buf_2m.intersection(city_m)
            except: clipped_2m = full_buf_2m

            mask_8m = calls_in_city.geometry.distance(s_pt_m) <= radius_guard_m
            indices_8m = set(calls_in_city[mask_8m]['point_idx'])
            full_buf_8m = s_pt_m.buffer(radius_guard_m)
            try: clipped_8m = full_buf_8m.intersection(city_m)
            except: clipped_8m = full_buf_8m
                
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_2m': clipped_2m, 'indices_2m': indices_2m,
                'clipped_8m': clipped_8m, 'indices_8m': indices_8m
            })

    # --- OPTIMIZER CONTROLS ---
    st.sidebar.header("🎯 Optimizer Controls")
    opt_strategy = st.sidebar.radio("Optimization Goal:", ("Maximize Call Coverage", "Maximize Land Coverage"), index=0)
    
    n = len(station_metadata)
    k_responder = st.sidebar.slider("🚁 Responder Drones (2-Mile)", 0, n, min(1, n))
    k_guardian = st.sidebar.slider("🦅 Guardian Drones (8-Mile)", 0, n, 0)
    
    show_boundaries = st.sidebar.checkbox("Show Jurisdiction Boundaries", value=True)
    show_health = st.sidebar.toggle("Show Health Score Banner", value=True)
    show_satellite = st.sidebar.toggle("🌍 Satellite View", value=False)
    
    best_resp_names, best_guard_names = [], []
    
    if k_responder + k_guardian > n:
        st.error("⚠️ Over-Deployment: Total requested drones exceed available stations.")
    elif k_responder > 0 or k_guardian > 0:
        
        total_resp_combos = math.comb(n, k_responder)
        total_guard_combos = math.comb(n - k_responder, k_guardian) if n >= k_responder else 0
        total_possible = total_resp_combos * total_guard_combos
        
        combos = []
        if total_possible > 3000:
            st.toast(f"Optimization Mode: Sampling ({total_possible:,} options)")
            for _ in range(3000):
                chosen = np.random.choice(range(n), k_responder + k_guardian, replace=False)
                r_c = tuple(chosen[:k_responder])
                g_c = tuple(chosen[k_responder:])
                combos.append((r_c, g_c))
        else:
            stations_idx = list(range(n))
            for r_c in itertools.combinations(stations_idx, k_responder):
                rem = [x for x in stations_idx if x not in r_c]
                for g_c in itertools.combinations(rem, k_guardian):
                    combos.append((r_c, g_c))
        
        max_val = -1
        best_combo = None
        
        with st.spinner(f"Optimizing {len(combos)} configurations..."):
            for r_combo, g_combo in combos:
                if opt_strategy == "Maximize Call Coverage":
                    union_set = set().union(*(station_metadata[i]['indices_2m'] for i in r_combo), 
                                            *(station_metadata[i]['indices_8m'] for i in g_combo))
                    val = len(union_set)
                else:
                    union_geo = unary_union([station_metadata[i]['clipped_2m'] for i in r_combo] + 
                                            [station_metadata[i]['clipped_8m'] for i in g_combo])
                    val = union_geo.area
                
                if val > max_val:
                    max_val = val
                    best_combo = (r_combo, g_combo)
            
            if best_combo is not None:
                r_best, g_best = best_combo
                best_resp_names = [station_metadata[i]['name'] for i in r_best]
                best
