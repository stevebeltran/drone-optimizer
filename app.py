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

# --- CACHED FUNCTION ---
@st.cache_data
def consolidate_jurisdictions(calls_df, stations_df, shapefile_dir):
    """
    Scans library and returns polygons that intersect with data points.
    Returns: GeoDataFrame (EPSG:4326), List of Source Files
    """
    # 1. Prepare Points (Calls + Stations)
    points_list = []
    if calls_df is not None:
        points_list.append(calls_df[['lat', 'lon']])
    if stations_df is not None:
        points_list.append(stations_df[['lat', 'lon']])
        
    if not points_list: return None, "No data points found."
    
    all_points = pd.concat(points_list)
    
    # Create GeoDataFrame locally
    points_gdf = gpd.GeoDataFrame(
        all_points, 
        geometry=gpd.points_from_xy(all_points.lon, all_points.lat), 
        crs="EPSG:4326"
    )
    
    # 2. Scan Library
    shp_files = glob.glob(os.path.join(shapefile_dir, "*.shp"))
    if not shp_files: return None, "No shapefiles found in library."

    matched_gdfs = []
    detected_sources = []
    
    total_bounds = points_gdf.total_bounds 
    
    for shp_path in shp_files:
        try:
            # Fast Filter
            gdf_chunk = gpd.read_file(shp_path, bbox=tuple(total_bounds))
            
            if not gdf_chunk.empty:
                # Check / Set CRS
                if gdf_chunk.crs is None: 
                    # Heuristic: If coords are small (<180), assume 4269. If huge, assume 3857.
                    if gdf_chunk.total_bounds[0] > -180 and gdf_chunk.total_bounds[2] < 180:
                        gdf_chunk.set_crs(epsg=4269, inplace=True)
                    else:
                        gdf_chunk.set_crs(epsg=3857, inplace=True) # Common fallback
                
                gdf_chunk = gdf_chunk.to_crs(epsg=4326)
                
                # Strict Intersection Filter
                matches = gpd.sjoin(gdf_chunk, points_gdf, how="inner", predicate="intersects")
                
                if not matches.empty:
                    valid_indices = matches.index.unique()
                    final_polys = gdf_chunk.loc[valid_indices].copy()
                    
                    name_col = next((c for c in ['NAME', 'DISTRICT', 'NAMELSAD'] if c in final_polys.columns), final_polys.columns[0])
                    final_polys['DISPLAY_NAME'] = final_polys[name_col].astype(str)
                    
                    matched_gdfs.append(final_polys)
                    detected_sources.append(os.path.basename(shp_path))
        except Exception:
            continue
            
    if not matched_gdfs:
        return None, "No matching geometry found."
        
    master_gdf = pd.concat(matched_gdfs, ignore_index=True)
    return master_gdf, detected_sources

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
    with st.spinner("🌍 Identifying active jurisdictions..."):
        master_gdf, match_sources = consolidate_jurisdictions(df_calls, df_stations_all, SHAPEFILE_DIR)

    if match_sources:
        st.sidebar.success(f"**Loaded {len(match_sources)} Map Files**")
    else:
        st.sidebar.warning("No boundaries found. Using data-only view.")

    st.markdown("---")
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    
    # --- COMBINED SELECTION ---
    if master_gdf is not None:
        available_areas = sorted(master_gdf['DISPLAY_NAME'].unique())
        dropdown_options = ["Combined (All Detected)"] + available_areas if len(available_areas) > 1 else available_areas
        target_selection = ctrl_col1.selectbox("📍 Jurisdiction Filter", dropdown_options, index=0)
        
        if target_selection == "Combined (All Detected)":
            active_gdf = master_gdf
        else:
            active_gdf = master_gdf[master_gdf['DISPLAY_NAME'] == target_selection]
    else:
        active_gdf = None
        target_selection = "None"

    # --- GEOMETRY PROCESSING ---
    city_m = None
    city_boundary_geom = None
    
    # Calculate Data Bounds (Crucial for Center/Zoom)
    data_points = pd.concat([df_calls[['lat', 'lon']], df_stations_all[['lat', 'lon']]])
    min_lon, min_lat = data_points['lon'].min(), data_points['lat'].min()
    max_lon, max_lat = data_points['lon'].max(), data_points['lat'].max()
    
    # Auto-Calc UTM Zone based on DATA centroid, not map centroid
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    utm_zone = int((center_lon + 180) / 6) + 1
    epsg_code = f"326{utm_zone}" if center_lat > 0 else f"327{utm_zone}"

    if active_gdf is not None:
        try:
            # Combine polygons
            if hasattr(active_gdf.geometry, 'union_all'):
                full_boundary = active_gdf.geometry.union_all()
            else:
                full_boundary = unary_union(active_gdf.geometry)
            
            # --- SMART CROP ---
            # Create a bounding box around the DATA (Calls + Stations) + 10% Buffer
            lat_buff = max((max_lat - min_lat) * 0.1, 0.02)
            lon_buff = max((max_lon - min_lon) * 0.1, 0.02)
            focus_box = box(min_lon - lon_buff, min_lat - lat_buff, max_lon + lon_buff, max_lat + lat_buff)

            # Visually clip boundary to this box
            city_boundary_geom = full_boundary.intersection(focus_box)
            
            # Project for Analysis (Use full boundary for accurate area)
            if hasattr(active_gdf.geometry, 'union_all'):
                city_m = active_gdf.to_crs(epsg=epsg_code).geometry.union_all()
            else:
                city_m = unary_union(active_gdf.to_crs(epsg=epsg_code).geometry)
                
        except Exception as e:
            st.error(f"Geometry Error: {e}")
            city_boundary_geom = None

    # Clip Calls (If boundary exists, use it. Else use bbox)
    gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
    
    if city_boundary_geom and not city_boundary_geom.is_empty:
        calls_in_city = gdf_calls[gdf_calls.within(full_boundary)].to_crs(epsg=epsg_code)
    else:
        # Fallback: Just use all calls if no boundary
        calls_in_city = gdf_calls.to_crs(epsg=epsg_code)
        
    calls_in_city['point_idx'] = range(len(calls_in_city))
    
    # 3. PRE-CALC STATIONS
    radius_m = 3218.69 
    station_metadata = []
    
    if not calls_in_city.empty:
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            
            mask = calls_in_city.geometry.distance(s_pt_m) <= radius_m
            covered_indices = set(calls_in_city[mask]['point_idx'])
            
            # Calculate intersection area if map exists
            full_buf = s_pt_m.buffer(radius_m)
            if city_m:
                clipped_buf = full_buf.intersection(city_m)
            else:
                clipped_buf = full_buf # Fallback to full circle
                
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'],
                'clipped_m': clipped_buf, 'indices': covered_indices, 'count': len(covered_indices)
            })

    # --- 4. OPTIMIZER ---
    st.sidebar.header("🎯 Optimizer Controls")
    
    opt_strategy = st.sidebar.radio("Optimization Goal:", ("Maximize Call Coverage", "Maximize Land Coverage"), index=0)
    
    max_k = len(station_metadata) if station_metadata else 1
    k = st.sidebar.slider("Number of Stations to Deploy", 1, max_k, min(2, max_k) if max_k > 1 else 1)
    
    # --- NEW: BOUNDARY TOGGLE ---
    show_boundaries = st.sidebar.checkbox("Show Jurisdiction Boundaries", value=True)
    show_health = st.sidebar.toggle("Show Health Score Banner", value=True)
    
    best_names = []
    max_val = 0
    if len(station_metadata) > 0:
        combos = list(itertools.combinations(range(len(station_metadata)), k))
        if len(combos) > 2000: combos = combos[:2000]
        
        best_combo = None
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
            
            if best_combo:
                best_names = [station_metadata[i]['name'] for i in best_combo]

    st.sidebar.markdown("---")
    if opt_strategy == "Maximize Call Coverage":
        st.sidebar.subheader("🏆 Optimal Stations (Calls)")
        caption_text = f"Covers {max_val:,} total incident points"
    else:
        st.sidebar.subheader("🌍 Optimal Stations (Land)")
        total_area = city_m.area if city_m else 1
        caption_text = f"Covers {(max_val/total_area)*100:.1f}% of total area"
        
    for name in best_names: st.sidebar.write(f"✅ {name}")
    st.sidebar.caption(caption_text)

    active_names = ctrl_col2.multiselect("📡 Active Deployment", options=df_stations_all['name'].tolist(), default=best_names)
    
    area_covered_perc, overlap_perc, calls_covered_perc = 0.0, 0.0, 0.0
    if active_names and station_metadata:
        active_data = [s for s in station_metadata if s['name'] in active_names]
        active_buffers = [s['clipped_m'] for s in active_data]
        active_indices = [s['indices'] for s in active_data]
        
        if city_m and not city_m.is_empty:
            area_covered_perc = (unary_union(active_buffers).area / city_m.area) * 100
        
        if len(calls_in_city) > 0:
            calls_covered_perc = (len(set().union(*active_indices)) / len(calls_in_city)) * 100
        
        inters = []
        for i in range(len(active_buffers)):
            for j in range(i+1, len(active_buffers)):
                over = active_buffers[i].intersection(active_buffers[j])
                if not over.is_empty: inters.append(over)
        
        if city_m and not city_m.is_empty:
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
    
    # Auto-Zoom Calculation (Based on DATA, not shapefile)
    def calculate_zoom(min_lon, max_lon):
        width = max_lon - min_lon
        if width <= 0: return 12
        zoom = 10.5 - np.log(width)
        return min(max(zoom, 10), 15) 

    # Draw Display Boundary (Only if checkbox is checked)
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

    # Center map on the DATA CENTROID
    dynamic_zoom = calculate_zoom(min_lon, max_lon)
    fig.update_layout(map_style="open-street-map", map_zoom=dynamic_zoom, map_center={"lat": center_lat, "lon": center_lon}, margin={"r":0,"t":0,"l":0,"b":0}, height=800)
    st.plotly_chart(fig, width='stretch')

else:
    st.info("👋 Upload CSV data to begin. The map will auto-detect matching jurisdictions from the library.")
