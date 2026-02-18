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

# --- PAGE CONFIG ---
st.set_page_config(page_title="brinc COS Drone Optimizer", layout="wide")
st.title("🛰️ brinc COS Drone Optimizer")

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

def get_circle_coords(lat, lon, r_mi=2):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

# --- STRICT SCANNER LOGIC ---
@st.cache_data
def find_relevant_jurisdictions(calls_df, stations_df, shapefile_dir):
    """
    Scans shapefiles and returns ONLY the specific polygons that contain data.
    Discards all empty polygons to prevent 'too much land' issues.
    """
    # 1. Prepare Points DataFrame (Calls + Stations)
    points_list = []
    if calls_df is not None:
        points_list.append(calls_df[['lat', 'lon']])
    if stations_df is not None:
        points_list.append(stations_df[['lat', 'lon']])
    
    if not points_list: return None
    
    # 2. Create Geometry for fast lookup
    # We use a sample of calls if dataset is huge (>50k) to speed up the scan
    # But we include ALL stations because they are critical
    full_points = pd.concat(points_list)
    
    # Filter outliers (0,0) coordinates
    full_points = full_points[(full_points.lat.abs() > 1) & (full_points.lon.abs() > 1)]
    
    if len(full_points) > 20000:
        scan_points = full_points.sample(20000, random_state=42)
    else:
        scan_points = full_points

    points_gdf = gpd.GeoDataFrame(
        scan_points, 
        geometry=gpd.points_from_xy(scan_points.lon, scan_points.lat), 
        crs="EPSG:4326"
    )
    
    # Calculate bounds to skip irrelevant files
    total_bounds = points_gdf.total_bounds
    
    shp_files = glob.glob(os.path.join(shapefile_dir, "*.shp"))
    relevant_polys = []
    
    for shp_path in shp_files:
        try:
            # A. Fast Bbox Filter (File Level)
            gdf_chunk = gpd.read_file(shp_path, bbox=tuple(total_bounds))
            
            if not gdf_chunk.empty:
                if gdf_chunk.crs is None: 
                    # Heuristic for common US projections
                    gdf_chunk.set_crs(epsg=4269, inplace=True)
                
                gdf_chunk = gdf_chunk.to_crs(epsg=4326)
                
                # B. Strict Polygon Filter (Row Level)
                # sjoin determines which specific polygon contains points
                hits = gpd.sjoin(gdf_chunk, points_gdf, how="inner", predicate="intersects")
                
                if not hits.empty:
                    # Get unique polygons that actually have data
                    valid_indices = hits.index.unique()
                    subset = gdf_chunk.loc[valid_indices].copy()
                    
                    # Calculate "Call Density" for ranking
                    # (Count how many times each polygon index appears in 'hits')
                    subset['data_count'] = hits.index.value_counts()
                    
                    # Normalize Name for Display
                    name_col = next((c for c in ['NAME', 'DISTRICT', 'NAMELSAD'] if c in subset.columns), subset.columns[0])
                    subset['DISPLAY_NAME'] = subset[name_col].astype(str)
                    subset['SOURCE_FILE'] = os.path.basename(shp_path)
                    
                    relevant_polys.append(subset)
        except Exception:
            continue
            
    if not relevant_polys:
        return None
        
    # Combine and Sort by Data Density (Highest count first)
    master_gdf = pd.concat(relevant_polys, ignore_index=True)
    master_gdf = master_gdf.sort_values(by='data_count', ascending=False)
    
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

    # --- SCANNING ---
    with st.spinner("🌍 Identifying exact jurisdictions..."):
        master_gdf = find_relevant_jurisdictions(df_calls, df_stations_all, SHAPEFILE_DIR)

    if master_gdf is None or master_gdf.empty:
        st.error("❌ No matching jurisdictions found.")
        st.info("Ensure your shapefiles overlap with your CSV coordinates.")
        st.stop()

    st.sidebar.success(f"**Found {len(master_gdf)} Active Zones**")
    
    st.markdown("---")
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    
    # --- MULTI-SELECT WIDGET (Ranked) ---
    # Create labels that show the count, e.g. "Schaumburg (15,402 calls)"
    master_gdf['LABEL'] = master_gdf['DISPLAY_NAME'] + " (" + master_gdf['data_count'].astype(str) + " pts)"
    options_map = dict(zip(master_gdf['LABEL'], master_gdf['DISPLAY_NAME']))
    
    # Default: Select top 5 most active areas automatically
    default_selections = master_gdf['LABEL'].tolist()[:13] 
    
    selected_labels = ctrl_col1.multiselect(
        "📍 Active Jurisdictions (Sorted by Activity)", 
        options=master_gdf['LABEL'].tolist(), 
        default=default_selections
    )
    
    if not selected_labels:
        st.warning("Please select at least one jurisdiction.")
        st.stop()
        
    # Filter Active GDF based on selection
    selected_names = [options_map[l] for l in selected_labels]
    active_gdf = master_gdf[master_gdf['DISPLAY_NAME'].isin(selected_names)]

    # --- GEOMETRY PROCESSING (ROBUST) ---
    # 1. Determine UTM Zone from Data Center
    center_lon = df_calls['lon'].mean()
    center_lat = df_calls['lat'].mean()
    utm_zone = int((center_lon + 180) / 6) + 1
    epsg_code = f"326{utm_zone}" if center_lat > 0 else f"327{utm_zone}"
    
    city_m = None
    city_boundary_geom = None
    
    try:
        # Reproject to UTM for safe merging
        active_utm = active_gdf.to_crs(epsg=epsg_code)
        
        # Buffer(0) trick fixes topology errors (gaps/overlaps) that cause crashes
        # Use union_all if available, else unary_union
        if hasattr(active_utm.geometry, 'union_all'):
            # Buffer by 1 meter to close gaps, merge, then unbuffer
            merged_poly = active_utm.geometry.buffer(1.0).union_all().buffer(-1.0)
        else:
            merged_poly = active_utm.geometry.buffer(1.0).unary_union.buffer(-1.0)
            
        city_m = merged_poly
        
        # Convert back to Lat/Lon for display
        city_boundary_geom = gpd.GeoSeries([merged_poly], crs=epsg_code).to_crs(epsg=4326).iloc[0]
        
    except Exception as e:
        st.error(f"Geometry Merge Error: {e}")
        st.stop()

    # --- FILTER CALLS ---
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    gdf_calls_utm = gdf_calls.to_crs(epsg=epsg_code)
    
    # Filter calls to strictly inside the selected boundary
    try:
        # Using within logic on UTM for accuracy
        # (This handles the "multiple shape files" requirement)
        mask = gdf_calls_utm.within(city_m)
        calls_in_city = gdf_calls_utm[mask]
    except:
        calls_in_city = gdf_calls_utm
        
    calls_in_city['point_idx'] = range(len(calls_in_city))
    
    # 3. PRE-CALC STATIONS
    radius_m = 3218.69 
    station_metadata = []
    
    if not calls_in_city.empty:
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            
            # Fast distance filter first
            dist_mask = calls_in_city.geometry.distance(s_pt_m) <= radius_m
            covered_indices = set(calls_in_city[dist_mask]['point_idx'])
            
            full_buf = s_pt_m.buffer(radius_m)
            try:
                clipped_buf = full_buf.intersection(city_m)
            except:
                clipped_buf = full_buf
                
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_m': clipped_buf, 'indices': covered_indices, 'count': len(covered_indices)
            })

    # --- 4. OPTIMIZER (CRASH PROOF) ---
    st.sidebar.header("🎯 Optimizer Controls")
    opt_strategy = st.sidebar.radio("Optimization Goal:", ("Maximize Call Coverage", "Maximize Land Coverage"), index=0)
    
    max_k = len(station_metadata) if station_metadata else 1
    k = st.sidebar.slider("Number of Stations to Deploy", 1, max_k, min(2, max_k) if max_k > 1 else 1)
    
    show_boundaries = st.sidebar.checkbox("Show Jurisdiction Boundaries", value=True)
    show_health = st.sidebar.toggle("Show Health Score Banner", value=True)
    
    # --- SMART OPTIMIZER (SAMPLING) ---
    n = len(station_metadata)
    total_possible = math.comb(n, k)
    
    combos = []
    is_random_sample = False
    
    if total_possible > 5000:
        is_random_sample = True
        st.toast(f"Running Statistical Optimization (Total combinations: {total_possible:,})")
        # Sample 5000 random scenarios
        for _ in range(5000):
            combos.append(np.random.choice(range(n), k, replace=False))
    else:
        combos = list(itertools.combinations(range(n), k))
    
    best_names = []
    max_val = -1
    
    with st.spinner(f"Optimizing..."):
        for combo in combos:
            if opt_strategy == "Maximize Call Coverage":
                union_set = set().union(*(station_metadata[i]['indices'] for i in combo))
                val = len(union_set)
            else:
                union_geo = unary_union([station_metadata[i]['clipped_m'] for i in combo])
                val = union_geo.area
            
            if val > max_val:
                max_val = val
                best_combo = combo
        
        if best_combo is not None:
            best_names = [station_metadata[i]['name'] for i in best_combo]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 Deployment Plan")
    for name in best_names: st.sidebar.write(f"✅ {name}")
    
    active_names = ctrl_col2.multiselect("📡 Active Deployment", options=df_stations_all['name'].tolist(), default=best_names)
    
    # Stats Calculation
    area_covered_perc, overlap_perc, calls_covered_perc = 0.0, 0.0, 0.0
    if active_names and station_metadata:
        active_data = [s for s in station_metadata if s['name'] in active_names]
        active_buffers = [s['clipped_m'] for s in active_data]
        active_indices = [s['indices'] for s in active_data]
        
        if not city_m.is_empty:
            area_covered_perc = (unary_union(active_buffers).area / city_m.area) * 100
        
        if len(calls_in_city) > 0:
            calls_covered_perc = (len(set().union(*active_indices)) / len(calls_in_city)) * 100
        
        inters = []
        for i in range(len(active_buffers)):
            for j in range(i+1, len(active_buffers)):
                over = active_buffers[i].intersection(active_buffers[j])
                if not over.is_empty: inters.append(over)
        if not city_m.is_empty:
            overlap_perc = (unary_union(inters).area / city_m.area * 100) if inters else 0.0

    st.markdown("---")
    if show_health:
        norm_redundancy = min(overlap_perc / 35.0, 1.0) * 100
        health_score = (calls_covered_perc * 0.50) + (area_covered_perc * 0.25) + (norm_redundancy * 0.25)
        if health_score >= 85: h_color, h_label = "#28a745", "OPTIMAL"
        elif health_score >= 75: h_color, h_label = "#94c11f", "SUFFICIENT"
        elif health_score >= 55: h_color, h_label = "#ffc107", "MARGINAL"
        else: h_color, h_label = "#dc3545", "CRITICAL"
        
        st.markdown(f"""
            <div style="background-color: {h_color}; padding: 10px; border-radius: 5px; color: white; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 1.2em; font-weight: bold;">Department Health Score: {health_score:.1f}%</span>
                <span style="font-size: 1.1em; background: rgba(0,0,0,0.2); padding: 2px 10px; border-radius: 4px;">{h_label}</span>
            </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Incident Points", f"{len(calls_in_city):,}")
    m2.metric("Response Capacity %", f"{calls_covered_perc:.1f}%")
    m3.metric("Land Covered", f"{area_covered_perc:.1f}%")
    m4.metric("Redundancy (Overlap)", f"{overlap_perc:.1f}%")

    fig = go.Figure()
    
    # Auto-Zoom (Robust)
    def calculate_zoom(min_lon, max_lon):
        width = max_lon - min_lon
        if width <= 0: return 12
        zoom = 10.5 - np.log(width)
        return min(max(zoom, 10), 15)

    def add_boundary_to_map(geom):
        if geom is None or geom.is_empty: return
        if isinstance(geom, Polygon):
            bx, by = geom.exterior.coords.xy
            fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#222", width=3), name="Jurisdiction Boundary", hoverinfo='skip'))
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                bx, by = poly.exterior.coords.xy
                fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#222", width=3), name="Jurisdiction Boundary", hoverinfo='skip', showlegend=False))

    if show_boundaries:
        add_boundary_to_map(city_boundary_geom)
        
    if len(calls_in_city) > 0:
        # Sample calls for performance
        display_calls = calls_in_city.sample(min(5000, len(calls_in_city))).to_crs(epsg=4326)
        fig.add_trace(go.Scattermap(lat=display_calls.geometry.y, lon=display_calls.geometry.x, mode='markers', marker=dict(size=4, color='#000080', opacity=0.35), name="Incident Data", hoverinfo='skip'))

    all_names = df_stations_all['name'].tolist()
    for i, s in enumerate(station_metadata):
        if s['name'] in active_names:
            color = STATION_COLORS[all_names.index(s['name']) % len(STATION_COLORS)]
            clats, clons = get_circle_coords(s['lat'], s['lon'])
            fig.add_trace(go.Scattermap(
                lat=list(clats) + [None, s['lat']], lon=list(clons) + [None, s['lon']], 
                mode='lines+markers', marker=dict(size=[0]*len(clats) + [0, 20], color=color), 
                line=dict(color=color, width=4.5), fill='toself', fillcolor='rgba(0,0,0,0)', 
                name=f"{s['name']}", hoverinfo='name'))

    # Center on filtered data
    data_points = calls_in_city.to_crs(epsg=4326)
    if not data_points.empty:
        # Outlier rejection for zoom
        clean_points = data_points # You can add quantile filtering here if needed
        min_lon, min_lat = clean_points.geometry.x.min(), clean_points.geometry.y.min()
        max_lon, max_lat = clean_points.geometry.x.max(), clean_points.geometry.y.max()
        dynamic_zoom = calculate_zoom(min_lon, max_lon)
        center_lat, center_lon = (min_lat + max_lat)/2, (min_lon + max_lon)/2
    else:
        # Fallback
        dynamic_zoom, center_lat, center_lon = 12, 42.0, -88.0

    fig.update_layout(map_style="open-street-map", map_zoom=dynamic_zoom, map_center={"lat": center_lat, "lon": center_lon}, margin={"r":0,"t":0,"l":0,"b":0}, height=800)
    st.plotly_chart(fig, width='stretch')

else:
    st.info("👋 Upload CSV data to begin. The map will auto-detect matching jurisdictions from the library.")
