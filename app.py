import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import os
import itertools

# --- PAGE CONFIG ---
st.set_page_config(page_title="Drone Logistics Portal", layout="wide", page_icon="🛰️")

# --- 1. INITIALIZE SESSION STATE ---
if 'box_open' not in st.session_state:
    st.session_state.box_open = True

# --- 2. HEADER ---
if st.sidebar.button("🔄 Reset & Upload New Data"):
    st.session_state.box_open = True
    st.rerun()

st.title("🛰️ Strategic Drone Deployment Optimizer")
st.markdown("#### **Geospatial Operations Analysis Tool**")

# --- SPEED OPTIMIZATION: CACHING ---
@st.cache_data
def process_geo_data(shp_path, selection):
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None: gdf.set_crs(epsg=4269, inplace=True)
    gdf['geometry'] = gdf['geometry'].simplify(0.0001, preserve_topology=True)
    name_col = 'DISTRICT' if 'DISTRICT' in gdf.columns else 'NAME'
    
    if selection == "SHOW ALL DISTRICTS":
        active_gdf = gdf.to_crs(epsg=4326)
        boundary = unary_union(active_gdf.geometry)
    else:
        active_gdf = gdf[gdf[name_col] == selection].to_crs(epsg=4326)
        boundary = active_gdf.iloc[0].geometry
    return gdf, active_gdf, boundary, name_col

# --- 3. DATA IMPORT ---
call_data, station_data, shot_data, shape_components = None, None, None, []

with st.expander("📁 Secure Data Import", expanded=st.session_state.box_open):
    uploaded_files = st.file_uploader("Upload Incident CSVs, 'shots.csv', and Shapefiles", accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        fname = f.name.lower()
        if fname == "calls.csv": call_data = f
        elif fname == "stations.csv": station_data = f
        elif fname == "shots.csv": shot_data = f
        elif any(fname.endswith(ext) for ext in ['.shp', '.shx', '.dbf', '.prj']):
            shape_components.append(f)

    if call_data and station_data and len(shape_components) >= 3:
        if st.session_state.box_open:
            st.session_state.box_open = False
            st.rerun()

STATION_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#800000", "#333333", "#000075"]

# --- 4. MAIN ANALYSIS ENGINE ---
if call_data and station_data and len(shape_components) >= 3:
    if not os.path.exists("temp"): os.mkdir("temp")
    for f in shape_components:
        with open(os.path.join("temp", f.name), "wb") as buffer:
            buffer.write(f.getbuffer())
    
    try:
        shp_path = [os.path.join("temp", f.name) for f in shape_components if f.name.endswith('.shp')][0]
        temp_gdf = gpd.read_file(shp_path)
        name_col_init = 'DISTRICT' if 'DISTRICT' in temp_gdf.columns else 'NAME'
        options = ["SHOW ALL DISTRICTS"] + sorted(temp_gdf[name_col_init].unique().tolist())
        
        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        selection = ctrl_col1.selectbox("📍 Active Jurisdiction Focus", options)

        gdf_all, active_gdf, city_boundary, name_col = process_geo_data(shp_path, selection)
        
        # Calculate UTM Zone for accurate buffering
        utm_zone = int((city_boundary.centroid.x + 180) / 6) + 1
        epsg_code = f"326{utm_zone}" if city_boundary.centroid.y > 0 else f"327{utm_zone}"
        city_m = active_gdf.to_crs(epsg=epsg_code).unary_union
        
        df_calls = pd.read_csv(call_data).dropna(subset=['lat', 'lon'])
        df_stations_all = pd.read_csv(station_data).dropna(subset=['lat', 'lon'])
        
        gdf_calls = gpd.GeoDataFrame(df_calls, geometry=gpd.points_from_xy(df_calls.lon, df_calls.lat), crs="EPSG:4326")
        calls_in_city = gdf_calls[gdf_calls.within(city_boundary)].to_crs(epsg=epsg_code)
        calls_in_city['point_idx'] = range(len(calls_in_city))
        
        # PRE-CALC (Dual Profiles)
        radius_resp_m = 3218.69   # 2 Miles
        radius_guard_m = 12874.75 # 8 Miles
        
        station_metadata = []
        for i, row in df_stations_all.iterrows():
            s_pt_m = gpd.GeoSeries([Point(row['lon'], row['lat'])], crs="EPSG:4326").to_crs(epsg=epsg_code).iloc[0]
            
            # Responder Profile (2-Mile)
            mask_2m = calls_in_city.geometry.distance(s_pt_m) <= radius_resp_m
            indices_2m = set(calls_in_city[mask_2m]['point_idx'])
            clipped_2m = s_pt_m.buffer(radius_resp_m).intersection(city_m)
            
            # Guardian Profile (8-Mile)
            mask_8m = calls_in_city.geometry.distance(s_pt_m) <= radius_guard_m
            indices_8m = set(calls_in_city[mask_8m]['point_idx'])
            clipped_8m = s_pt_m.buffer(radius_guard_m).intersection(city_m)
            
            station_metadata.append({
                'name': row['name'], 'lat': row['lat'], 'lon': row['lon'], 
                'clipped_2m': clipped_2m, 'indices_2m': indices_2m,
                'clipped_8m': clipped_8m, 'indices_8m': indices_8m
            })

        # OPTIMIZER
        st.sidebar.header("🎯 Optimizer Controls")
        
        # Split Sliders for Responder and Guardian
        k_responder = st.sidebar.slider("🚁 Responder Drones (2-Mile)", 0, len(station_metadata), 0)
        k_guardian = st.sidebar.slider("🦅 Guardian Drones (8-Mile)", 0, len(station_metadata), 0)
        strategy = st.sidebar.radio("Optimization Goal", ("Maximize Call Volume", "Maximize Land Equity"))

        if k_responder == 0 and k_guardian == 0:
            active_resp_names = []
            active_guard_names = []
            st.sidebar.info("🧪 **Generation Mode**: Enable 'Suggested Sites' below to view results.")
            
        elif k_responder + k_guardian > len(station_metadata):
            st.error("⚠️ Over-Deployment: You are trying to deploy more drones than you have available launch stations.")
            active_resp_names = []
            active_guard_names = []
        else:
            # Combinatorial Optimization for mixed fleets
            stations_idx = list(range(len(station_metadata)))
            combos = []
            
            # Generate valid disjoint sets (a station can't be both simultaneously in the solver)
            for resp_combo in itertools.combinations(stations_idx, k_responder):
                rem_stations = [s for s in stations_idx if s not in resp_combo]
                for guard_combo in itertools.combinations(rem_stations, k_guardian):
                    combos.append((resp_combo, guard_combo))
                    if len(combos) > 600: break # Cap computations
                if len(combos) > 600: break
            
            best_combo = None
            max_calls, max_area = -1, -1
            
            for r_combo, g_combo in combos:
                if strategy == "Maximize Call Volume":
                    u_set = set().union(*(station_metadata[i]['indices_2m'] for i in r_combo), 
                                        *(station_metadata[i]['indices_8m'] for i in g_combo))
                    if len(u_set) > max_calls: 
                        max_calls = len(u_set)
                        best_combo = (r_combo, g_combo)
                else:
                    u_geo = unary_union([station_metadata[i]['clipped_2m'] for i in r_combo] + 
                                        [station_metadata[i]['clipped_8m'] for i in g_combo])
                    if u_geo.area > max_area: 
                        max_area = u_geo.area
                        best_combo = (r_combo, g_combo)
            
            r_best, g_best = best_combo if best_combo else ([], [])
            default_resp = [station_metadata[i]['name'] for i in r_best]
            default_guard = [station_metadata[i]['name'] for i in g_best]
            
            # Split Selection Interface
            active_resp_names = ctrl_col2.multiselect("🚁 Responder Assignments (2-Mile)", options=df_stations_all['name'].tolist(), default=default_resp)
            active_guard_names = ctrl_col2.multiselect("🦅 Guardian Assignments (8-Mile)", options=df_stations_all['name'].tolist(), default=default_guard)
        
        # --- LAYER CONTROLS ---
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Layer Controls")
        
        show_shots = False
        df_shots = None
        if shot_data:
            show_shots = st.sidebar.toggle("Show Shot Detection Events", value=False)
            if show_shots:
                try: df_shots = pd.read_csv(shot_data)
                except: pass
        
        show_suggestions = st.sidebar.toggle("Show Suggested Coverage Sites", value=False)

        # --- METRICS ---
        active_resp_data = [s for s in station_metadata if s['name'] in active_resp_names]
        active_guard_data = [s for s in station_metadata if s['name'] in active_guard_names]
        
        # Aggregate active sets
        active_indices = [s['indices_2m'] for s in active_resp_data] + [s['indices_8m'] for s in active_guard_data]
        all_ids = set().union(*active_indices) if active_indices else set()
        cap_perc = (len(all_ids) / len(calls_in_city)) * 100 if len(calls_in_city) > 0 else 0
        
        active_geos = [s['clipped_2m'] for s in active_resp_data] + [s['clipped_8m'] for s in active_guard_data]
        total_union_geo = unary_union(active_geos) if active_geos else None
        land_perc = (total_union_geo.area / city_m.area * 100) if total_union_geo else 0
        
        overlap_perc = 0.0
        if len(active_geos) > 1:
            inters = [active_geos[i].intersection(active_geos[j]) for i in range(len(active_geos)) for j in range(i+1, len(active_geos)) if not active_geos[i].intersection(active_geos[j]).is_empty]
            overlap_perc = (unary_union(inters).area / city_m.area * 100) if inters else 0.0

        # --- ITERATIVE COVERAGE GENERATOR ---
        suggested_coords = []
        if land_perc < 99.0:
            current_covered = total_union_geo if total_union_geo else Polygon()
            uncovered_poly = city_m.difference(current_covered)
            max_iterations = 25 
            
            for _ in range(max_iterations):
                if uncovered_poly.is_empty or (uncovered_poly.area / city_m.area) < 0.01:
                    break 
                
                if isinstance(uncovered_poly, MultiPolygon):
                    valid_geoms = [g for g in uncovered_poly.geoms if g.area > 1000] 
                    if not valid_geoms: break
                    target_chunk = max(valid_geoms, key=lambda g: g.area)
                else:
                    target_chunk = uncovered_poly
                
                new_site_pt = target_chunk.representative_point()
                p_geo = gpd.GeoSeries([new_site_pt], crs=epsg_code).to_crs(epsg=4326).iloc[0]
                suggested_coords.append({'lat': p_geo.y, 'lon': p_geo.x})
                
                # Gap generation defaults to using the 2-mile Responder radius footprint 
                new_coverage = new_site_pt.buffer(radius_resp_m)
                uncovered_poly = uncovered_poly.difference(new_coverage)

        # --- HEALTH SCORE ---
        norm_redundancy = min(overlap_perc / 39.0, 1.0) * 100
        health_score = (cap_perc * 0.50) + (land_perc * 0.25) + (norm_redundancy * 0.25)

        if health_score >= 85: h_color, h_label = "#28a745", "OPTIMAL"
        elif health_score >= 75: h_color, h_label = "#94c11f", "SUFFICIENT"
        elif health_score >= 55: h_color, h_label = "#ffc107", "MARGINAL"
        else: h_color, h_label = "#dc3545", "CRITICAL"

        # BANNER
        st.markdown(f"""
            <div style="background-color: {h_color}; padding: 10px; border-radius: 5px; color: white; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 1.2em; font-weight: bold;">Department Health Score: {health_score:.1f}%</span>
                <span style="font-size: 1.1em; background: rgba(0,0,0,0.2); padding: 2px 10px; border-radius: 4px;">{h_label}</span>
            </div>
            """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Response Capacity", f"{cap_perc:.1f}%")
        m2.metric("Land Covered", f"{land_perc:.1f}%")
        m3.metric("Redundancy", f"{overlap_perc:.1f}%")
        m4.metric("Uncovered Calls", f"{len(calls_in_city) - len(all_ids):,}")

        # --- SIDEBAR SCORECARD & SUGGESTIONS ---
        st.sidebar.markdown("---")
        with st.sidebar.expander("📝 Tactical Scorecard", expanded=True):
            summary_text = f"""DRONE DEPLOYMENT ANALYSIS
---------------------------------
Jurisdiction: {selection}
Coverage: {land_perc:.1f}%
Status: {h_label}

🚁 RESPONDERS (2-Mile):
""" + "\n".join([f"✅ {name}" for name in active_resp_names]) + f"""

🦅 GUARDIANS (8-Mile):
""" + "\n".join([f"✅ {name}" for name in active_guard_names])
            
            if suggested_coords and show_suggestions:
                summary_text += "\n\n⚠️ SUGGESTED EXPANSIONS:\n"
                for i, c in enumerate(suggested_coords):
                    summary_text += f"📍 Site {i+1}: {c['lat']:.5f}, {c['lon']:.5f}\n"

            st.text_area("Copy Report:", summary_text, height=350)

        if suggested_coords and show_suggestions:
            st.sidebar.markdown("### 💡 Recommended Sites")
            st.sidebar.info(f"Generated {len(suggested_coords)} Responder-sized sites to close gaps.")
            for i, c in enumerate(suggested_coords):
                st.sidebar.code(f"{c['lat']:.5f}, {c['lon']:.5f}", language="text")

        # --- THE MAP ---
        fig = go.Figure()
        
        # 1. District Lines
        for _, row in gdf_all.to_crs(epsg=4326).iterrows():
            geom = row.geometry
            p_list = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
            for p in p_list:
                bx, by = p.exterior.coords.xy
                fig.add_trace(go.Scattermap(mode="lines", lon=list(bx), lat=list(by), line=dict(color="#444", width=1), showlegend=False, hoverinfo='skip'))
        
        # 2. Incidents
        sample = calls_in_city.to_crs(epsg=4326).sample(min(2000, len(calls_in_city)))
        fig.add_trace(go.Scattermap(
            lat=sample.geometry.y, 
            lon=sample.geometry.x, 
            mode='markers', 
            marker=dict(size=4, color='#000080', opacity=0.3), 
            name="Incidents",
            hoverinfo='skip'
        ))

        # 3. Shot Detection
        if show_shots and df_shots is not None:
             fig.add_trace(go.Scattermap(
                lat=df_shots['lat'],
                lon=df_shots['lon'],
                mode='markers',
                marker=dict(symbol='triangle', size=10, color='#FF4500', opacity=0.9),
                name="Shot Detection",
                text=df_shots['point_id'] if 'point_id' in df_shots.columns else None,
                hoverinfo='text+lat+lon'
            ))

        # 4. SUGGESTED SITES (Controlled by Toggle)
        if show_suggestions:
            for i, c in enumerate(suggested_coords):
                angles = np.linspace(0, 2*np.pi, 100)
                clats = c['lat'] + (2.0/69.172) * np.sin(angles)  # Drawn at 2-mile Responder Size
                clons = c['lon'] + (2.0/(69.172 * np.cos(np.radians(c['lat'])))) * np.cos(angles)
                
                fig.add_trace(go.Scattermap(
                    lat=list(clats),
                    lon=list(clons),
                    mode='markers',
                    marker=dict(size=4, color='#FF00FF'), 
                    name=f"Proposed Coverage {i+1}",
                    hoverinfo='skip',
                    showlegend=False
                ))
                fig.add_trace(go.Scattermap(
                    lat=[c['lat']],
                    lon=[c['lon']],
                    mode='markers+text',
                    marker=dict(size=12, color='#FF00FF', symbol='circle'), 
                    text=[f"NEW SITE {i+1}"],
                    textposition="top center",
                    name=f"Suggestion {i+1}",
                    hoverinfo='text'
                ))
        
        # 5. Active Stations Plotting Function
        all_st_names = df_stations_all['name'].tolist()

        def plot_drone_ring(fig, s, radius_miles, color, drone_type):
            angles = np.linspace(0, 2*np.pi, 100)
            clats = s['lat'] + (radius_miles/69.172) * np.sin(angles)
            clons = s['lon'] + (radius_miles/(69.172 * np.cos(np.radians(s['lat'])))) * np.cos(angles)
            
            fig.add_trace(go.Scattermap(
                lat=list(clats) + [clats[0]], 
                lon=list(clons) + [clons[0]], 
                mode='lines', 
                line=dict(color=color, width=4.5), 
                hoverinfo='skip',
                showlegend=False
            ))
            fig.add_trace(go.Scattermap(
                lat=[s['lat']], 
                lon=[s['lon']], 
                mode='markers', 
                marker=dict(size=12, color=color), 
                name=f"{s['name']} ({drone_type})",
                hoverinfo='name'
            ))

        # Render Responders (2-mile rings)
        for s in active_resp_data:
            color = STATION_COLORS[all_st_names.index(s['name']) % len(STATION_COLORS)]
            plot_drone_ring(fig, s, 2.0, color, "Responder")
            
        # Render Guardians (8-mile rings)
        for s in active_guard_data:
            color = STATION_COLORS[all_st_names.index(s['name']) % len(STATION_COLORS)]
            plot_drone_ring(fig, s, 8.0, color, "Guardian")

        fig.update_layout(map_style="open-street-map", map_zoom=11, map_center={"lat": city_boundary.centroid.y, "lon": city_boundary.centroid.x}, margin={"r":0,"t":0,"l":0,"b":0}, height=750)
        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("System Ready: Please upload deployment files above to initialize session.")
