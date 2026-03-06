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
    # If the logo isn't found, we just show a text placeholder or nothing
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
        # Point
        pnt = fol_stations.newpoint(name=f"{name_prefix} {row['name']}")
        pnt.coords = [(row['lon'], row['lat'])]
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/blu-blank.png'
        # Ring
        lats, lons = get_circle_coords(row['lat'], row['lon'], r_mi=radius)
        ring_coords = list(zip(lons, lats))
        ring_coords.append(ring_coords[0])
        pol = fol_rings.newpolygon(name=f"Range: {row['name']}")
        pol.outerboundaryis = ring_coords
        pol.style.linestyle.color = color
        pol.style.linestyle.width = 2
        pol.style.polystyle.color = simplekml.Color.changealphaint(60, color)

    # Add Responders (Blue, 2-Mile)
    for _, row in df_stations_all[df_stations_all['name'].isin(active_resp_names)].iterrows():
        add_kml_station(row, 2.0, simplekml.Color.blue, "[Responder]")
        
    # Add Guardians (Orange, 8-Mile)
    for _, row in df_stations_all[df_stations_all['name'].isin(active_guard_names)].iterrows():
        add_kml_station(row, 8.0, simplekml.Color.orange, "[Guardian]")

    # 3. Add Sample of Calls
    fol_calls = kml.newfolder(name="Incident Data (Sample)")
    calls_export = calls_gdf.to_crs(epsg=4326)
    if len(calls_export) > 2000:
        calls_export = calls_export.sample(2000)
        
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
            
            # Responder Profile (2-Mile)
            mask_2m = calls_in_city.geometry.distance(s_pt_m) <= radius_resp_m
            indices_2m = set(calls_in_city[mask_2m]['point_idx'])
            full_buf_2m = s_pt_m.buffer(radius_resp_m)
            try: clipped_2m = full_buf_2m.intersection(city_m)
            except: clipped_2m = full_buf_2m

            # Guardian Profile (8-Mile)
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
    
    best_resp_names, best_guard_names = [], []
    
    if k_responder + k_guardian > n:
        st.error("⚠️ Over-Deployment: Total requested drones exceed available stations.")
    elif k_responder > 0 or k_guardian > 0:
        
        # Calculate possible combinations
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
                best_guard_names = [station_metadata[i]['name'] for i in g_best]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 Recommended Deployment")
    for name in best_resp_names: st.sidebar.write(f"🚁 {name} (Responder)")
    for name in best_guard_names: st.sidebar.write(f"🦅 {name} (Guardian)")
    
    # --- UI SELECTION ---
    active_resp_names = ctrl_col2.multiselect("🚁 Active Responders (2-Mile)", options=df_stations_all['name'].tolist(), default=best_resp_names)
    active_guard_names = ctrl_col2.multiselect("🦅 Active Guardians (8-Mile)", options=df_stations_all['name'].tolist(), default=best_guard_names)
    
    # --- METRICS CALCULATION ---
    area_covered_perc, overlap_perc, calls_covered_perc = 0.0, 0.0, 0.0
    
    active_resp_data = [s for s in station_metadata if s['name'] in active_resp_names]
    active_guard_data = [s for s in station_metadata if s['name'] in active_guard_names]
    
    active_geos = [s['clipped_2m'] for s in active_resp_data] + [s['clipped_8m'] for s in active_guard_data]
    active_indices = [s['indices_2m'] for s in active_resp_data] + [s['indices_8m'] for s in active_guard_data]

    if active_geos:
        if not city_m.is_empty:
            area_covered_perc = (unary_union(active_geos).area / city_m.area) * 100
        
        if len(calls_in_city) > 0:
            calls_covered_perc = (len(set().union(*active_indices)) / len(calls_in_city)) * 100
        
        inters = []
        for i in range(len(active_geos)):
            for j in range(i+1, len(active_geos)):
                over = active_geos[i].intersection(active_geos[j])
                if not over.is_empty: inters.append(over)
        if not city_m.is_empty:
            overlap_perc = (unary_union(inters).area / city_m.area * 100) if inters else 0.0

    st.markdown("---")
    if show_health:
        norm_redundancy = min(overlap_perc / 35.0, 1.0) * 100
        health_score = (calls_covered_perc * 0.50) + (area_covered_perc * 0.35) + (norm_redundancy * 0.15)
        if health_score >= 80: h_color, h_label = "#28a745", "OPTIMAL"
        elif health_score >= 70: h_color, h_label = "#94c11f", "SUFFICIENT"
        elif health_score >= 55: h_color, h_label = "#ffc107", "MARGINAL"
        else: h_color, h_label = "#dc3545", "ESSENTIAL"
        
        st.markdown(f"""
            <div style="background-color: {h_color}; padding: 10px; border-radius: 5px; color: white; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 1.5em; font-weight: bold;">Department Health Score: {health_score:.1f}%</span>
                <span style="font-size: 1.3em; background: rgba(0,0,0,0.2); padding: 2px 10px; border-radius: 4px;">{h_label}</span>
            </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Incident Points", f"{len(calls_in_city):,}")
    m2.metric("Response Capacity %", f"{calls_covered_perc:.1f}%")
    m3.metric("Land Covered", f"{area_covered_perc:.1f}%")
    m4.metric("Redundancy (Overlap)", f"{overlap_perc:.1f}%")

    # --- KML EXPORT ---
    kml_data = generate_kml(
        active_gdf, 
        df_stations_all, 
        active_resp_names,
        active_guard_names,
        calls_in_city
    )
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="🌏 Download for Google Earth",
        data=kml_data,
        file_name="drone_deployment.kml",
        mime="application/vnd.google-earth.kml+xml"
    )

    # --- MAP RENDERING ---
    fig = go.Figure()
    
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
        display_calls = calls_in_city.sample(min(5000, len(calls_in_city))).to_crs(epsg=4326)
        fig.add_trace(go.Scattermap(lat=display_calls.geometry.y, lon=display_calls.geometry.x, mode='markers', marker=dict(size=4, color='#000080', opacity=0.35), name="Incident Data", hoverinfo='skip'))

    all_names = df_stations_all['name'].tolist()

    def plot_ring(s, radius_mi, drone_type):
        color = STATION_COLORS[all_names.index(s['name']) % len(STATION_COLORS)]
        clats, clons = get_circle_coords(s['lat'], s['lon'], r_mi=radius_mi)
        fig.add_trace(go.Scattermap(
            lat=list(clats) + [None, s['lat']], lon=list(clons) + [None, s['lon']], 
            mode='lines+markers', marker=dict(size=[0]*len(clats) + [0, 20], color=color), 
            line=dict(color=color, width=4.5), fill='toself', fillcolor='rgba(0,0,0,0)', 
            name=f"{s['name']} ({drone_type})", hoverinfo='name'))

    # Plot Responders
    for s in active_resp_data:
        plot_ring(s, 2.0, "Responder")
        
    # Plot Guardians
    for s in active_guard_data:
        plot_ring(s, 8.0, "Guardian")

    data_points = calls_in_city.to_crs(epsg=4326)
    if not data_points.empty:
        q_low = data_points.geometry.x.quantile(0.01)
        q_high = data_points.geometry.x.quantile(0.99)
        clean_pts = data_points[(data_points.geometry.x >= q_low) & (data_points.geometry.x <= q_high)]
        if clean_pts.empty: clean_pts = data_points
        min_lon, min_lat = clean_pts.geometry.x.min(), clean_pts.geometry.y.min()
        max_lon, max_lat = clean_pts.geometry.x.max(), clean_pts.geometry.y.max()
        dynamic_zoom = calculate_zoom(min_lon, max_lon)
        center_lat, center_lon = (min_lat + max_lat)/2, (min_lon + max_lon)/2
    else:
        dynamic_zoom, center_lat, center_lon = 12, 42.0, -88.0

    fig.update_layout(
        map_style="open-street-map", 
        map_zoom=dynamic_zoom, 
        map_center={"lat": center_lat, "lon": center_lon}, 
        margin={"r":0,"t":0,"l":0,"b":0}, 
        height=800,
        font=dict(size=18)
    )
    st.plotly_chart(fig, width='stretch')

else:
    st.info("👋 Upload CSV data to begin. The map will auto-detect matching jurisdictions from the library.")
