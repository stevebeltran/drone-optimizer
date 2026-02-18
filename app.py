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

# --- PAGE CONFIG ---
st.set_page_config(page_title="brinc COS Drone Optimizer", layout="wide")
st.title("🛰️ brinc COS Drone Optimizer")

# --- CONFIGURATION ---
SHAPEFILE_DIR = "jurisdiction_data" 

# Ensure the directory exists
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

@st.cache_data
def identify_best_jurisdiction(calls_df, shapefile_dir):
    """
    Scans all shapefiles and finds the one that contains the MOST calls.
    Returns the file and the specific jurisdiction (polygon) with the highest count.
    """
    # 1. Create a lightweight sample of points for fast scanning
    # 2000 points is enough to statistically determine the correct jurisdiction
    sample_size = min(len(calls_df), 2000)
    sample_df = calls_df.sample(sample_size, random_state=42)
    
    calls_gdf = gpd.GeoDataFrame(
        sample_df, 
        geometry=gpd.points_from_xy(sample_df.lon, sample_df.lat), 
        crs="EPSG:4326"
    )
    
    # Get bounds of calls to filter files quickly
    calls_bounds = calls_gdf.total_bounds
    
    shp_files = glob.glob(os.path.join(shapefile_dir, "*.shp"))
    if not shp_files:
        return None, None, "No shapefiles found."

    best_file_gdf = None
    best_poly_row = None
    best_filename = None
    max_matches = -1

    for shp_path in shp_files:
        try:
            # Check if file CRS is likely compatible or readable
            # We read just the bounds/metadata first if possible, but 
            # reading with bbox filter is the standard efficient way.
            
            # Heuristic: Read file filtered by Call Box
            gdf_chunk = gpd.read_file(shp_path, bbox=tuple(calls_bounds))
            
            if gdf_chunk.empty:
                continue
                
            if gdf_chunk.crs is None: gdf_chunk.set_crs(epsg=4269, inplace=True)
            gdf_chunk = gdf_chunk.to_crs(epsg=4326)
            
            # Perform Spatial Join (Count points in polygons)
            joined = gpd.sjoin(gdf_chunk, calls_gdf, how="inner", predicate="contains")
            
            # Total matches in this file
            total_matches_in_file = len(joined)
            
            if total_matches_in_file > max_matches:
                max_matches = total_matches_in_file
                best_filename = os.path.basename(shp_path)
                
                # Find the specific Polygon (Jurisdiction) with the most calls
                # 'index' in joined refers to the index of the polygon in gdf_chunk
                counts_by_poly = joined.index.value_counts()
                best_poly_idx = counts_by_poly.idxmax()
                
                # We need to reload the FULL file so the user can see other jurisdictions in the dropdown
                # (gdf_chunk was only a partial load)
                full_gdf = gpd.read_file(shp_path)
                if full_gdf.crs is None: full_gdf.set_crs(epsg=4269, inplace=True)
                full_gdf = full_gdf.to_crs(epsg=4326)
                
                # Find the matching row in the full file (using ID or Name if available, else index)
                # To be safe, we'll assume the index might differ if bbox was used. 
                # Let's match by the unique Name/ID found in the chunk.
                chunk_row = gdf_chunk.loc[best_poly_idx]
                
                # Try to find a unique ID column
                id_col = next((c for c in ['GEOID', 'COUSUBFP', 'NAME', 'NAMELSAD'] if c in chunk_row.index), chunk_row.index[0])
                match_val = chunk_row[id_col]
                
                best_file_gdf = full_gdf
                best_poly_row = full_gdf[full_gdf[id_col] == match_val].iloc[0]

        except Exception as e:
            continue

    if best_file_gdf is None:
        return None, None, "No matching jurisdictions found for call data."
        
    return best_file_gdf, best_poly_row, best_filename

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

    # --- INTELLIGENT SCANNING ---
    with st.spinner("🌍 Analyzing call density to find jurisdiction..."):
        # We pass the DataFrame now, not a point
        city_gdf_all, city_boundary_row, match_source = identify_best_jurisdiction(df_calls, SHAPEFILE_DIR)

    if city_gdf_all is None:
        st.error(f"❌ Auto-Detection Failed: {match_source}")
        st.warning("Go to the **Sidebar > Map Library Manager** and upload your shapefiles.")
        st.stop()

    name_col = next((c for c in ['NAME', 'DISTRICT', 'NAMELSAD'] if c in city_boundary_row.index), city_boundary_row.index[0])
    detected_name = city_boundary_row[name_col]

    st.sidebar.success(f"Best Match: **{detected_name}**\n(Source: {match_source})")

    st.markdown("---")
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    
    # Dropdown (Pre-selected with the Best Match)
    city_list = sorted(city_gdf_all[name_col].astype(str).unique())
    default_ix = city_list.index(detected_name) if detected_name in city_list else 0
    
    target_city = ctrl_col1.selectbox("📍 Jurisdiction", city_list, index=default_ix)

    city_gdf = city_gdf_all[city_gdf_all[name_col] == target_city].to_crs(epsg=4326)
    city_boundary = city_gdf.iloc[0].geometry
    
    # UTM Projection
    utm_zone = int((city_boundary.centroid.x + 180) / 6) + 1
    epsg_code = f"326{utm_zone}" if city_boundary.centroid.y > 0 else f"327{utm_zone}"
    city_m = city_gdf.to_crs(epsg=epsg_code).geometry.union_all()
    
    # Clip Calls
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    calls_in_city = gdf_calls[gdf_calls.within(city_boundary)].to_crs(epsg=epsg_code)
    calls_in_city['point_idx'] = range(len(calls_in_city))
    
    # 3. PRE-CALC
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

    # --- 4. OPTIMIZER ---
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

    # --- MAP CENTER FIX ---
    # Center based on calls, not the shapefile centroid
    center_lat = df_calls['lat'].mean()
    center_lon = df_calls['lon'].mean()
    
    fig.update_layout(map_style="open-street-map", map_zoom=11, map_center={"lat": center_lat, "lon": center_lon}, margin={"r":0,"t":0,"l":0,"b":0}, height=800)
    st.plotly_chart(fig, width='stretch')

else:
    st.info("👋 Upload CSV data to begin. The map will auto-select the jurisdiction from the library.")
