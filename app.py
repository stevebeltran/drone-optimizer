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

# --- PAGE CONFIG ---
st.set_page_config(page_title="brinc COS Drone Optimizer", layout="wide")
st.title("🛰️ brinc COS Drone Optimizer")

# --- CONFIGURATION ---
SHAPEFILE_DIR = "jurisdiction_data" 

# --- UPLOAD SECTION ---
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

@st.cache_data
def load_and_match_shapefile(center_point, shapefile_dir):
    """
    Optimized Scanner: Checks bounding boxes first to avoid loading 
    massive files for states we aren't currently in.
    """
    shp_files = glob.glob(os.path.join(shapefile_dir, "*.shp"))
    
    if not shp_files:
        return None, None, "No shapefiles found in 'jurisdiction_data' folder."

    # 1. Fast Scan (Bounding Box Check)
    candidate_files = []
    for shp_path in shp_files:
        try:
            # Read only the bounds/metadata first (Very Fast)
            info = gpd.read_file(shp_path, rows=1) 
            if info.crs is None: info.set_crs(epsg=4269, inplace=True)
            info = info.to_crs(epsg=4326)
            
            # Rough check: Is our point roughly in the total bounds of this file?
            # We read the whole file bounds using a lighter method if possible, 
            # but for simplicity, we load the file with a spatial filter in step 2.
            candidate_files.append(shp_path)
        except:
            continue
            
    # 2. Deep Scan (Detailed Polygon Check on Candidates)
    for shp_path in candidate_files:
        try:
            # Load file using a "bbox" filter to only load relevant geometry
            # This prevents loading "Texas" if we are in "Maine"
            gdf = gpd.read_file(shp_path, bbox=center_point)
            
            if not gdf.empty:
                # If we got data back, it means the point is inside this file's bounding box
                # Now verify exact polygon containment
                if gdf.crs is None: gdf.set_crs(epsg=4269, inplace=True)
                gdf = gdf.to_crs(epsg=4326)
                
                matching_row = gdf[gdf.contains(center_point)]
                
                if not matching_row.empty:
                    # Found it! Now we must load the REST of the file (for context/map display)
                    # We reload the full file because the previous read was filtered to just 1 point
                    full_gdf = gpd.read_file(shp_path)
                    if full_gdf.crs is None: full_gdf.set_crs(epsg=4269, inplace=True)
                    full_gdf = full_gdf.to_crs(epsg=4326)
                    
                    # Find the specific row in the full GDF
                    # We match based on the first unique ID column we find
                    id_col = next((c for c in ['GEOID', 'COUSUBFP', 'NAME'] if c in matching_row.columns), matching_row.columns[0])
                    match_val = matching_row.iloc[0][id_col]
                    full_row = full_gdf[full_gdf[id_col] == match_val].iloc[0]
                    
                    return full_gdf, full_row, os.path.basename(shp_path)
        except Exception as e:
            continue

    return None, None, "No matching jurisdiction found in library."

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

    avg_lat = df_calls['lat'].mean()
    avg_lon = df_calls['lon'].mean()
    center_point = Point(avg_lon, avg_lat)

    with st.spinner("🌍 Scanning map library..."):
        city_gdf_all, city_boundary_row, match_source = load_and_match_shapefile(center_point, SHAPEFILE_DIR)

    if city_gdf_all is None:
        st.error(f"❌ Auto-Detection Failed: {match_source}")
        st.info("Ensure your 'jurisdiction_data' folder contains the correct .shp files.")
        st.stop()

    name_col = next((c for c in ['NAME', 'DISTRICT', 'NAMELSAD'] if c in city_boundary_row.index), city_boundary_row.index[0])
    detected_name = city_boundary_row[name_col]

    st.markdown("---")
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    
    city_list = sorted(city_gdf_all[name_col].astype(str).unique())
    default_ix = city_list.index(detected_name) if detected_name in city_list else 0
    
    target_city = ctrl_col1.selectbox("📍 Jurisdiction", city_list, index=default_ix)
    ctrl_col1.success(f"Auto-loaded from **{match_source}**")

    city_gdf = city_gdf_all[city_gdf_all[name_col] == target_city].to_crs(epsg=4326)
    city_boundary = city_gdf.iloc[0].geometry
    
    utm_zone = int((city_boundary.centroid.x + 180) / 6) + 1
    epsg_code = f"326{utm_zone}" if city_boundary.centroid.y > 0 else f"327{utm_zone}"
    city_m = city_gdf.to_crs(epsg=epsg_code).geometry.union_all()
    
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    calls_in_city = gdf_calls[gdf_calls.within(city_boundary)].to_crs(epsg=epsg_code)
    calls_in_city['point_idx'] = range(len(calls_in_city))
    
    radius_m = 3218.69 
    station_metadata = []
    for i, row in df_stations_all.iterrows():
        s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
        mask = calls_in_city.geometry.distance(s_pt_m) <= radius_m
        covered_indices = set(calls_in_city[mask]['point_idx'])
        full_buf = s_pt_m.buffer(radius_m)
        clipped_buf = full_buf.intersection(city_m)
        station_metadata.append({
            'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
            'clipped_m': clipped_buf, 'indices': covered_indices, 'count': len(covered_indices)
        })

    st.sidebar.header("🎯 Optimizer Controls")
    
    opt_strategy = st.sidebar.radio(
        "Optimization Goal:",
        ("Maximize Call Coverage", "Maximize Land Coverage"),
        index=0
    )
    
    k = st.sidebar.slider("Number of Stations to Deploy", 1, len(station_metadata), min(2, len(station_metadata)))
    show_health = st.sidebar.toggle("Show Health Score Banner", value=True)
    
    combos = list(itertools.combinations(range(len(station_metadata)), k))
    if len(combos) > 2000: combos = combos[:2000]
    
    best_combo = None
    max_val = -1
    
    with st.spinner(f"Optimizing for {opt_strategy}..."):
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
        
        best_names = [station_metadata[i]['name'] for i in best_combo]

    st.sidebar.markdown("---")
    if opt_strategy == "Maximize Call Coverage":
        st.sidebar.subheader("🏆 Optimal Stations (Calls)")
        caption_text = f"Covers {max_val:,} total incident points"
    else:
        st.sidebar.subheader("🌍 Optimal Stations (Land)")
        caption_text = f"Covers {(max_val/city_m.area)*100:.1f}% of total area"
        
    for name in best_names: st.sidebar.write(f"✅ {name}")
    st.sidebar.caption(caption_text)

    active_names = ctrl_col2.multiselect("📡 Active Deployment", options=df_stations_all['name'].tolist(), default=best_names)
    
    area_covered_perc, overlap_perc, calls_covered_perc = 0.0, 0.0, 0.0
    if active_names:
        active_data = [s for s in station_metadata if s['name'] in active_names]
        active_buffers = [s['clipped_m'] for s in active_data]
        active_indices = [s['indices'] for s in active_data]
        
        area_covered_perc = (unary_union(active_buffers).area / city_m.area) * 100
        if len(calls_in_city) > 0:
            calls_covered_perc = (len(set().union(*active_indices)) / len(calls_in_city)) * 100
        
        inters = []
        for i in range(len(active_buffers)):
            for j in range(i+1, len(active_buffers)):
                over = active_buffers[i].intersection(active_buffers[j])
                if not over.is_empty: inters.append(over)
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
            </div>
            """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Incident Points", f"{len(calls_in_city):,}")
    m2.metric("Response Capacity %", f"{calls_covered_perc:.1f}%")
    m3.metric("Land Covered", f"{area_covered_perc:.1f}%")
    m4.metric("Redundancy (Overlap)", f"{overlap_perc:.1f}%")

    fig = go.Figure()
    
    def add_boundary_to_map(geom):
        if isinstance(geom, Polygon):
            bx, by = geom.exterior.coords.xy
            fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#222", width=3), name="Jurisdiction Boundary", hoverinfo='skip'))
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                bx, by = poly.exterior.coords.xy
                fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#222", width=3), name="Jurisdiction Boundary", hoverinfo='skip', showlegend=False))

    add_boundary_to_map(city_boundary)
    
    if len(calls_in_city) > 0:
        sample_size = min(5000, len(calls_in_city))
        calls_map = calls_in_city.to_crs(epsg=4326).sample(sample_size)
        fig.add_trace(go.Scattermap(lat=calls_map.geometry.y, lon=calls_map.geometry.x, mode='markers', marker=dict(size=4, color='#000080', opacity=0.35), name="Incident Data", hoverinfo='skip'))
    
    all_names = df_stations_all['name'].tolist()
    for i, s in enumerate(station_metadata):
        if s['name'] in active_names:
            color = STATION_COLORS[all_names.index(s['name']) % len(STATION_COLORS)]
            clats, clons = get_circle_coords(s['lat'], s['lon'])
            fig.add_trace(go.Scattermap(
                lat=list(clats) + [None, s['lat']], 
                lon=list(clons) + [None, s['lon']], 
                mode='lines+markers', 
                marker=dict(size=[0]*len(clats) + [0, 20], color=color), 
                line=dict(color=color, width=4.5), 
                fill='toself', fillcolor='rgba(0,0,0,0)', 
                name=f"{s['name']}",
                hoverinfo='name'
            ))

    fig.update_layout(map_style="open-street-map", map_zoom=12, map_center={"lat": city_boundary.centroid.y, "lon": city_boundary.centroid.x}, margin={"r":0,"t":0,"l":0,"b":0}, height=800)
    st.plotly_chart(fig, width='stretch')

else:
    st.info("👋 Upload CSV data to begin. The map will auto-select the jurisdiction from the library.")
