import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import os
import shutil
import itertools

# --- PAGE CONFIG ---
st.set_page_config(page_title="High-Performance Drone Optimizer", layout="wide")
st.title("🛰️ High-Performance Drone Optimizer")

# --- UPLOAD SECTION ---
if 'files_ready' not in st.session_state:
    st.session_state['files_ready'] = False

with st.expander("📁 Upload Data Files", expanded=not st.session_state['files_ready']):
    uploaded_files = st.file_uploader("Drop all 6 files here", accept_multiple_files=True)

# High-Contrast "Heavy" Palette
STATION_COLORS = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", 
    "#800000", "#333333", "#000075", "#808000", "#9A6324", 
    "#5E35B1", "#1B5E20", "#B71C1C", "#0D47A1", "#212121"
]

def get_circle_coords(lat, lon, r_mi=2):
    angles = np.linspace(0, 2*np.pi, 100)
    c_lats = lat + (r_mi/69.172) * np.sin(angles)
    c_lons = lon + (r_mi/(69.172 * np.cos(np.radians(lat)))) * np.cos(angles)
    return c_lats, c_lons

# --- FILE ROUTING ---
call_data, station_data, shape_components = None, None, []
for f in uploaded_files:
    fname = f.name.lower()
    if fname == "calls.csv": call_data = f
    elif fname == "stations.csv": station_data = f
    elif any(fname.endswith(ext) for ext in ['.shp', '.shx', '.dbf', '.prj']):
        shape_components.append(f)

if call_data and station_data and len(shape_components) >= 3:
    st.session_state['files_ready'] = True
    if os.path.exists("temp"): shutil.rmtree("temp")
    os.mkdir("temp")
    for f in shape_components:
        with open(os.path.join("temp", f.name), "wb") as buffer:
            buffer.write(f.getbuffer())
    
    try:
        # 1. LOAD GEOGRAPHY
        shp_path = [os.path.join("temp", f.name) for f in shape_components if f.name.endswith('.shp')][0]
        city_gdf_all = gpd.read_file(shp_path)
        if city_gdf_all.crs is None: city_gdf_all.set_crs(epsg=4269, inplace=True)
        city_list = sorted(city_gdf_all['NAME'].unique())
        
        default_ix = city_list.index("Boston") if "Boston" in city_list else (city_list.index("Benton") if "Benton" in city_list else 0)
        
        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        target_city = ctrl_col1.selectbox("📍 Jurisdiction", city_list, index=default_ix)
        
        city_gdf = city_gdf_all[city_gdf_all['NAME'] == target_city].to_crs(epsg=4326)
        city_boundary = city_gdf.iloc[0].geometry
        utm_zone = int((city_boundary.centroid.x + 180) / 6) + 1
        epsg_code = f"326{utm_zone}" if city_boundary.centroid.y > 0 else f"327{utm_zone}"
        city_m = city_gdf.to_crs(epsg=epsg_code).iloc[0].geometry
        
        # 2. LOAD DATA
        df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
        df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])
        
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
        k = st.sidebar.slider("Number of Stations to Deploy", 1, len(station_metadata), min(2, len(station_metadata)))
        
        combos = list(itertools.combinations(range(len(station_metadata)), k))
        if len(combos) > 2000: combos = combos[:2000]
        
        best_call_combo, max_calls = None, -1
        best_geo_combo, max_area = None, -1
        
        with st.spinner("Analyzing optimal placements..."):
            for combo in combos:
                union_set = set().union(*(station_metadata[i]['indices'] for i in combo))
                if len(union_set) > max_calls:
                    max_calls = len(union_set); best_call_combo = combo
                
                union_geo = unary_union([station_metadata[i]['clipped_m'] for i in combo])
                if union_geo.area > max_area:
                    max_area = union_geo.area; best_geo_combo = combo
            
            best_call_names = [station_metadata[i]['name'] for i in best_call_combo]
            best_geo_names = [station_metadata[i]['name'] for i in best_geo_combo]

        # --- SIDEBAR RANKINGS ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🏆 Best for Call Response")
        for name in best_call_names: st.sidebar.write(f"✅ {name}")
        st.sidebar.caption(f"Covers {max_calls:,} total incident points")

        st.sidebar.markdown("---")
        st.sidebar.subheader("🌍 Best for Land Coverage")
        for name in best_geo_names: st.sidebar.write(f"📍 {name}")
        st.sidebar.caption(f"Covers {(max_area/city_m.area)*100:.1f}% of total area")

        # --- MAIN INTERFACE ---
        active_names = ctrl_col2.multiselect("📡 Active Deployment", options=df_stations_all['name'].tolist(), default=best_call_names)
        
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
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Incident Points", f"{len(calls_in_city):,}")
        m2.metric("Response Capacity %", f"{calls_covered_perc:.1f}%")
        m3.metric("Land Covered", f"{area_covered_perc:.1f}%")
        m4.metric("Redundancy (Overlap)", f"{overlap_perc:.1f}%")

        # --- THE MAP ---
        fig = go.Figure()
        
        # 1. FIX FOR MULTIPOLYGON: Handle city boundary correctly
        def add_boundary_to_map(geom):
            if isinstance(geom, Polygon):
                bx, by = geom.exterior.coords.xy
                fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#222", width=3), name="Jurisdiction Boundary", hoverinfo='skip'))
            elif isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    bx, by = poly.exterior.coords.xy
                    fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#222", width=3), name="Jurisdiction Boundary", hoverinfo='skip', showlegend=False))

        add_boundary_to_map(city_boundary)
        
        # 2. Service Calls (Midnight Blue)
        if len(calls_in_city) > 0:
            sample_size = min(5000, len(calls_in_city))
            calls_map = calls_in_city.to_crs(epsg=4326).sample(sample_size)
            fig.add_trace(go.Scattermap(lat=calls_map.geometry.y, lon=calls_map.geometry.x, mode='markers', marker=dict(size=4, color='#000080', opacity=0.35), name="Incident Data", hoverinfo='skip'))
        
        # 3. Stations & Heavy Rings
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

    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    st.info("👋 Upload data files to begin tactical analysis.")
