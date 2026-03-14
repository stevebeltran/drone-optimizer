import streamlit as st
import pandas as pd
import folium
from folium import plugins
from folium.features import DivIcon
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import time
import random
import math
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Tactical Drone Command", initial_sidebar_state="collapsed")

# --- Session State Initialization ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'map_center' not in st.session_state: st.session_state.map_center = [39.8283, -98.5795]
if 'map_zoom' not in st.session_state: st.session_state.map_zoom = 4 
if 'base' not in st.session_state: st.session_state.base = None
if 'target' not in st.session_state: st.session_state.target = None
if 'inc_type' not in st.session_state: st.session_state.inc_type = None
if 'squad_cars' not in st.session_state: st.session_state.squad_cars = []
if 'sim_completed' not in st.session_state: st.session_state.sim_completed = False
if 'has_run_once' not in st.session_state: st.session_state.has_run_once = False
if 'best_officer_sq' not in st.session_state: st.session_state.best_officer_sq = None
if 't_officers' not in st.session_state: st.session_state.t_officers = None

# --- CUSTOM CSS: CLEAN COCKPIT THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Manrope:wght@400;600;700&display=swap');

    header[data-testid="stHeader"] { display: none; }
    
    /* Global Typography & Colors */
    .stApp { 
        background-color: #050505 !important; 
        color: #797979; 
        font-family: 'Manrope', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 { 
        color: #ffffff !important; 
        font-family: 'Manrope', sans-serif;
        margin-bottom: 0px !important; 
        padding-bottom: 0px !important; 
    }
    
    h3 { font-size: 1.2rem !important; }
    
    .block-container { padding-top: 3rem !important; padding-bottom: 1rem !important; }
    div.stVerticalBlock > div { gap: 0.2rem !important; } 
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        color: #00D2FF; 
        font-family: 'IBM Plex Mono', monospace;
    }
    div[data-testid="stMetricLabel"] { 
        font-size: 0.6rem !important; 
        color: #797979; 
        margin-bottom: -5px; 
    }

    .stProgress > div > div { height: 6px !important; }
    .stProgress > div > div > div > div { background-color: #00D2FF; }

    /* Button and Popover Styling */
    div.stButton > button, div[data-testid="stPopover"] > button {
        background-color: #111 !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
        font-size: 0.8rem;
        font-family: 'Manrope', sans-serif;
    }
    
    div.stButton > button:hover, div[data-testid="stPopover"] > button:hover {
        border-color: #00D2FF !important;
        color: #00D2FF !important;
    }
    
    div.stButton > button:focus, div.stButton > button:active,
    div[data-testid="stPopover"] > button:focus, div[data-testid="stPopover"] > button:active,
    div[data-testid="stPopover"] > button[aria-expanded="true"] {
        background-color: #111 !important;
        color: #00D2FF !important;
        border-color: #00D2FF !important;
    }

    div[data-testid="stPopoverBody"], div[role="dialog"] {
        background-color: #050505 !important;
        border: 1px solid #333 !important;
    }
    
    .stTextInput input { 
        background-color: #111 !important; 
        color: #ffffff !important; 
        border: 1px solid #444 !important; 
        font-family: 'IBM Plex Mono', monospace;
        height: 38px !important;
        font-size: 0.85rem !important;
    }

    hr { margin: 0.5em 0 !important; border-color: #333 !important; }

    /* --- DRONE NAME PULSE (BRINC BLUE) --- */
    @keyframes dronePulse {
        0%, 49% { color: #797979; text-shadow: none; }
        50%, 100% { color: #00D2FF; text-shadow: 0 0 8px #00D2FF; }
    }
    .drone-active { animation: dronePulse 0.8s infinite; font-weight: bold; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; display: block; margin-bottom: -10px; }
    .drone-static { color: #ffffff; font-weight: bold; text-shadow: none; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; display: block; margin-bottom: -10px; }

    /* --- INCIDENT LOG CSS --- */
    .incident-log {
        background-color: #111;
        border: 1px solid #333;
        border-radius: 5px;
        padding: 8px; 
        margin-bottom: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        min-height: 80px;  
    }
    .log-header { color: #ffffff; font-size: 0.85rem; border-bottom: 1px solid #333; margin-bottom: 6px; padding-bottom: 4px; font-weight: bold; }
    .log-entry { margin-bottom: 2px; color: #797979; }
    .log-time { color: #797979; margin-right: 12px; }
    
    .log-critical { color: #ffffff; font-weight: bold; }
    .log-action { color: #00D2FF; font-weight: bold; }
    .log-success { color: #00D2FF; font-weight: bold; }
    .log-info { color: #797979; font-weight: normal; }

    /* --- DRONE METRICS CARD --- */
    .drone-card {
        background-color: #080808;
        border: 1px solid #222;
        border-radius: 4px;
        padding: 6px 10px; 
        margin-top: 2px;
        margin-bottom: 0px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 6px; 
    }
    .m-box { display: flex; flex-direction: column; }
    .m-label {
        color: #797979;
        font-size: 0.55rem; 
        font-family: 'Manrope', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0px; 
    }
    .m-val {
        color: #00D2FF;
        font-size: 0.95rem; 
        font-family: 'IBM Plex Mono', monospace;
        font-weight: bold;
    }
    .m-val-dim {
        color: #444444;
        font-size: 0.95rem; 
        font-family: 'IBM Plex Mono', monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def load_data():
    try:
        df = pd.read_csv('drones.csv')
        df = df.dropna(subset=['model'])
        df['model'] = df['model'].astype(str)
        df.columns = df.columns.str.strip()
        return df
    except:
        data = {
            'model': ['RESPONDER', 'GUARDIAN', 'SKYDIO X-10', 'MATRICE 4TD'],
            'flight_time_min': [42, 60, 40, 54],
            'speed_mph': [22, 30, 36, 34],
            'range_miles': [5.0, 12.0, 7.5, 6.2]
        }
        return pd.DataFrame(data)

def get_full_recharge_time(model_name):
    mapping = {
        'RESPONDER': 25,
        'GUARDIAN': 1, 
        'SKYDIO': 90,
        'MATRICE': 55
    }
    for key, val in mapping.items():
        if key in model_name.upper():
            return val
    return 60

@st.cache_data(show_spinner=False)
def get_lat_lon_from_zip(zip_code):
    geolocator = Nominatim(user_agent="tactical_drone_command_ui_v3", timeout=10)
    try:
        location = geolocator.geocode({"postalcode": zip_code, "country": "US"})
        if location: 
            return [location.latitude, location.longitude]
    except Exception as e:
        print(f"Geocode Error: {e}")
        return None
    return None

def get_distance_miles(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5 * 69

def generate_incident():
    incidents = [
        ("SHOTS FIRED", "critical"),
        ("ARMED ROBBERY", "critical"),
        ("OFFICER IN DISTRESS", "critical"),
        ("BURGLARY IN PROGRESS", "action"),
        ("VEHICLE PURSUIT", "action"),
        ("MISSING PERSON", "info"),
        ("SUSPICIOUS ACTIVITY", "info")
    ]
    inc, severity = random.choice(incidents)
    st.session_state.inc_type = inc
    st.session_state.inc_severity = severity
    
    hr = random.choice(list(range(18, 24)) + list(range(0, 4)))
    mn = random.randint(0, 59)
    sc = random.randint(0, 59)
    
    base_time = datetime.now().replace(hour=hr, minute=mn, second=sc)
    st.session_state.t_call = base_time
    st.session_state.t_launch = base_time + timedelta(seconds=random.randint(45, 120))

def randomize_squads():
    if st.session_state.base:
        st.session_state.squad_cars = []
        num_cars = random.randint(3, 5) 
        for _ in range(num_cars):
            r_mi = random.uniform(0.5, 9.0) 
            angle = random.uniform(0, 2 * math.pi)
            d_lat = (r_mi * math.sin(angle)) / 69.172
            d_lon = (r_mi * math.cos(angle)) / (69.172 * math.cos(math.radians(st.session_state.base[0])))
            st.session_state.squad_cars.append([st.session_state.base[0] + d_lat, st.session_state.base[1] + d_lon])

def calculate_responding_officer():
    best_dist = float('inf')
    best_sq = None
    if st.session_state.base and st.session_state.target and st.session_state.squad_cars:
        for sq in st.session_state.squad_cars:
            d = get_distance_miles(sq, st.session_state.target)
            if d < best_dist:
                best_dist = d
                best_sq = sq
        
        if best_dist == float('inf'): 
            best_dist = get_distance_miles(st.session_state.base, st.session_state.target)
            best_sq = st.session_state.base

        st.session_state.best_officer_sq = best_sq
        
        if 't_call' in st.session_state:
            t_officer_dispatch = st.session_state.t_call + timedelta(seconds=60)
            officer_travel_sec = (best_dist * 1.4) / (35.0 / 3600.0)
            st.session_state.t_officers = t_officer_dispatch + timedelta(seconds=officer_travel_sec)

# --- Dynamic Map Marker Functions ---
def get_drone_pos(stage, base, target, index):
    if stage == 0 or stage == 4:
        base_lat, base_lon = base[0], base[1]
    elif stage == 1 or stage == 3:
        base_lat, base_lon = (base[0] + target[0]) / 2.0, (base[1] + target[1]) / 2.0
    else: # stage 2 (On Scene)
        base_lat, base_lon = target[0], target[1]
        
    lat_offset = 0.0015 if index in [0, 1] else -0.0015
    lon_offset = 0.0015 if index in [0, 2] else -0.0015
    
    return [base_lat + lat_offset, base_lon + lon_offset]

def generate_base_map(drones_to_draw=None):
    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles="CartoDB dark_matter")
    
    if st.session_state.base:
        base_html = """<div style="color: #00D2FF; font-size: 24px; text-shadow: 0 0 5px #000;"><i class="fa fa-home"></i></div>"""
        folium.Marker(st.session_state.base, icon=DivIcon(html=base_html, icon_anchor=(10,10))).add_to(m)
        
        rings = [2, 4, 6, 8]
        for r in rings:
            folium.Circle(location=st.session_state.base, radius=r * 1609.34, color='#00D2FF', weight=1, fill=False, opacity=0.5, dash_array='4, 8').add_to(m)
            lat_offset = (r / 69.0)
            folium.map.Marker([st.session_state.base[0] + lat_offset, st.session_state.base[1]], icon=DivIcon(icon_size=(100,20), icon_anchor=(50,10), html=f'<div style="font-family: \'Manrope\', sans-serif; font-size:10px; font-weight:600; color:#00D2FF; text-shadow: 0 0 5px #000;">{r} MI</div>')).add_to(m)

        is_responding = st.session_state.step == 3 and not st.session_state.sim_completed

        for sq in st.session_state.squad_cars:
            if is_responding:
                car_color = "#FF0000" if sq == st.session_state.best_officer_sq else "#00D2FF"
            else:
                car_color = "#00D2FF"
            car_html = f"""<div style="color: {car_color}; font-size: 18px; text-shadow: 0 0 5px #000;"><i class="fa fa-car"></i></div>"""
            folium.Marker(sq, icon=DivIcon(html=car_html)).add_to(m)

    if st.session_state.target:
        target_html = """<div style="color: #FF0000; font-size: 24px; text-shadow: 0 0 5px #000;"><i class="fa fa-crosshairs"></i></div>"""
        folium.Marker(st.session_state.target, icon=DivIcon(html=target_html, icon_anchor=(10,10))).add_to(m)
        
        plugins.AntPath(locations=[st.session_state.base, st.session_state.target], color="#00D2FF", pulse_color="#ffffff", weight=3, delay=800, dash_array=[10, 20]).add_to(m)
        
        if st.session_state.step == 3 and not st.session_state.sim_completed and st.session_state.best_officer_sq:
            plugins.AntPath(locations=[st.session_state.best_officer_sq, st.session_state.target], color="#FF0000", pulse_color="#ffffff", weight=3, delay=400, dash_array=[15, 30]).add_to(m)

    if drones_to_draw and st.session_state.base and st.session_state.target:
        for idx, d_info in enumerate(drones_to_draw):
            pos = get_drone_pos(d_info['stage'], st.session_state.base, st.session_state.target, idx)
            color = d_info['color']
            letter = d_info['letter']
            
            html = f"""
                <div style="
                    background-color: {color}; 
                    color: #000; 
                    border-radius: 50%; 
                    width: 22px; 
                    height: 22px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    font-weight: 900; 
                    font-family: 'IBM Plex Mono', monospace;
                    font-size: 14px;
                    box-shadow: 0 0 12px {color};
                    border: 2px solid #fff;
                ">
                    {letter}
                </div>
            """
            folium.Marker(pos, icon=DivIcon(html=html, icon_anchor=(11,11))).add_to(m)

    return m

# --- Layout: Dynamic Columns ---
left_col, mid_col = st.columns([7, 3])

# ==========================================
# COLUMN 2: OPS CENTER & ASSET COST
# ==========================================
with mid_col:
    if st.session_state.step == 1:
        st.markdown("### OPS CENTER")
        zip_col, space_col, logo_col = st.columns([1, 1, 2])
        with zip_col:
            zip_in = st.text_input("ZIP", placeholder="ZIP + ENTER", label_visibility="collapsed", max_chars=5, key="zip_input")
        with logo_col:
            st.image("logo.png", use_container_width=True)
            
        if zip_in and len(zip_in) == 5:
            coords = get_lat_lon_from_zip(zip_in)
            if coords:
                st.session_state.map_center = coords
                st.session_state.map_zoom = 13 
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("Invalid ZIP code. (Try again)")

    elif st.session_state.step >= 2:
        asset_col, space_col, logo_col = st.columns([1.5, 0.5, 2])
        with logo_col:
            st.image("logo.png", use_container_width=True)
            
        with asset_col:
            if st.session_state.target and st.session_state.base:
                heli_cost = 1.0 * 1300 
                
                with st.popover("AIR ASSET", use_container_width=True):
                    st.markdown(f"""
                    <div style="background-color: #050505; padding: 10px; border-radius: 5px;">
                        <div style="text-align: center; margin-bottom: 20px; margin-top: 10px;">
                            <div style="display: inline-block; border: 1px solid rgba(0, 210, 255, 0.3); border-radius: 50%; padding: 4px;">
                                <div style="border: 2px solid rgba(0, 210, 255, 0.6); border-radius: 50%; padding: 6px;">
                                    <div style="border: 2px solid #00D2FF; border-radius: 50%; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 10px rgba(0, 210, 255, 0.5);">
                                        <span style="color: #00D2FF; font-weight: bold; font-family: sans-serif; font-size: 16px;">$</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <hr style="border-color: #333; margin-bottom: 20px;">
                        <div style="background-color: #000000; border: 1px solid #222; padding: 20px; border-radius: 4px; text-align: center; margin-bottom: 15px;">
                            <h6 style="color: #ffffff; margin: 0; font-size: 0.85rem; letter-spacing: 1px; font-family: 'Manrope', sans-serif;">EST. HELICOPTER COST FOR THIS CALL</h6>
                            <h2 style="color: #00D2FF; margin: 15px 0; font-family: 'IBM Plex Mono', monospace; font-size: 2.5rem;">${heli_cost:,.2f}</h2>
                            <div style="color: #797979; font-size: 0.7rem;">BASED ON $1,300/HR OP COST</div>
                        </div>
                        <div style="border: 1px solid #222; padding: 15px; border-radius: 4px; background-color: #000000; font-family: 'Manrope', sans-serif;">
                            <div style="color: #797979; font-size: 0.9rem;">TOTAL FLIGHT TIME (W/ HOVER): <span style="color:#ffffff;">60 MIN</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if not st.session_state.base:
            st.warning("SET BASE")
        elif not st.session_state.target:
            st.info("SET TARGET")
        else:
            incident_placeholder = st.empty()

        if st.session_state.step == 3:
            df = load_data()
            drone_ui_elements = [] 
            for index, row in df.iterrows():
                with st.container():
                    head_c1, head_c2 = st.columns([1.8, 1])
                    name_placeholder = head_c1.empty()
                    name_placeholder.markdown(f"<span class='drone-static'>{row['model']}</span>", unsafe_allow_html=True)
                    status_placeholder = head_c2.empty()
                    
                    flight_bar = st.progress(0)
                    metrics_placeholder = st.empty() 
                    
                    ui_obj = {
                        'specs': row,
                        'name_text': name_placeholder,
                        'status_text': status_placeholder,
                        'flight_bar': flight_bar,
                        'metrics_html': metrics_placeholder 
                    }
                    drone_ui_elements.append(ui_obj)
                    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

# ==========================================
# COLUMN 1: MAP
# ==========================================
with left_col:
    map_placeholder = st.empty()
    is_simulating = st.session_state.step == 3 and not st.session_state.sim_completed
    
    if not is_simulating:
        with map_placeholder:
            m_static = generate_base_map()
            map_data = st_folium(m_static, height=850, use_container_width=True, key="static_map")
            
            if map_data['last_clicked']:
                coords = [map_data['last_clicked']['lat'], map_data['last_clicked']['lng']]
                if not st.session_state.base:
                    st.session_state.base = coords
                    st.session_state.map_zoom = 12 
                    randomize_squads() 
                    st.session_state.sim_completed = False
                    st.rerun()
                elif st.session_state.target != coords:
                    st.session_state.target = coords
                    randomize_squads() 
                    generate_incident() 
                    calculate_responding_officer() 
                    st.session_state.step = 3
                    st.session_state.sim_completed = False
                    st.rerun()

# ==========================================
# SIMULATION LOOP
# ==========================================
if st.session_state.step == 3 and st.session_state.base and st.session_state.target:
    dist_one_way = get_distance_miles(st.session_state.base, st.session_state.target)
    
    fleet_sim_data = []
    for drone in drone_ui_elements:
        specs = drone['specs']
        max_v = float(specs['speed_mph'])
        
        t_out = dist_one_way / (max_v / 3600)
        batt_sec = float(specs['flight_time_min']) * 60
        
        reserve_sec = 5 * 60
        hover_sec = (batt_sec - reserve_sec) - (t_out * 2)
        
        possible = hover_sec >= 0 and dist_one_way <= float(specs['range_miles'])
        
        used_batt_sec = (t_out * 2) + (hover_sec if possible else 0)
        batt_used_pct = used_batt_sec / batt_sec
        full_recharge_min = get_full_recharge_time(specs['model'])
        
        if specs['model'].upper() == 'GUARDIAN':
            turnaround_min = 1.0 
        else:
            turnaround_min = full_recharge_min * batt_used_pct

        fleet_sim_data.append({
            'ui': drone, 't_out': t_out, 't_hov': hover_sec if possible else 0,
            't_total': (t_out * 2) + (hover_sec if possible else 0),
            'batt_cap': batt_sec, 'possible': possible,
            'turnaround_min': turnaround_min,
            'fail_msg': "FUEL" if hover_sec < 0 else "RANGE"
        })

    valid = [d for d in fleet_sim_data if d['possible']]
    valid.sort(key=lambda x: x['t_total'], reverse=True) 
    
    fastest_t_out = min([d['t_out'] for d in valid]) if valid else 0
    t_drone_arrival = st.session_state.t_launch + timedelta(seconds=fastest_t_out)
    sim_dur = max([d['t_total'] for d in valid]) if valid else 5
    
    DRONE_COLORS = ["#FF00FF", "#00D2FF", "#39FF14", "#FF6F00"] 

    def render_ui_state(curr_time, log_html_override=None):
        log_events = [
            (st.session_state.t_call, f'<span class="log-{st.session_state.inc_severity}">{st.session_state.inc_type} - TARGET: {dist_one_way:.2f} MI</span>'),
            (st.session_state.t_launch, '<span class="log-action">DRONE LAUNCHED</span>')
        ]
        
        if curr_time >= fastest_t_out and valid:
            log_events.append((t_drone_arrival, '<span class="log-success">DRONE ON SCENE</span>'))

        officer_sec_since_launch = (st.session_state.t_officers - st.session_state.t_launch).total_seconds()
        if curr_time >= officer_sec_since_launch:
            log_events.append((st.session_state.t_officers, '<span class="log-info">OFFICERS ARRIVE</span>'))

        log_events.sort(key=lambda x: x[0])

        if log_html_override is None:
            log_html = f"""<div class="incident-log"><div class="log-header">INCIDENT LOG</div>"""
            for dt, html_str in log_events:
                log_html += f'<div class="log-entry"><span class="log-time">{dt.strftime("%H:%M:%S")}</span>{html_str}</div>'
            log_html += "</div>"
        else:
            log_html = log_html_override
            
        incident_placeholder.markdown(log_html, unsafe_allow_html=True)

        for d in fleet_sim_data:
            ui = d['ui']
            if not d['possible']:
                ui['status_text'].markdown(f"<div style='text-align:right; margin-bottom:-10px;'><span style='color:#797979; font-size:0.8rem; font-weight:bold; font-family: \"IBM Plex Mono\", monospace;'>{d['fail_msg']}</span></div>", unsafe_allow_html=True)
                ui['name_text'].markdown(f"<span class='drone-static'>{ui['specs']['model']}</span>", unsafe_allow_html=True)
                ui['flight_bar'].progress(0.0)
                
                card_html = f"""
                <div class="drone-card">
                    <div class="metric-grid">
                        <div class="m-box"><div class="m-label">TIME TO TGT</div><div class="m-val-dim">N/A</div></div>
                        <div class="m-box"><div class="m-label">ON SCENE</div><div class="m-val-dim">N/A</div></div>
                        <div class="m-box"><div class="m-label">BATTERY</div><div class="m-val-dim">N/A</div></div>
                    </div>
                </div>
                """
                ui['metrics_html'].markdown(card_html, unsafe_allow_html=True)
                continue
            
            phase_txt, phase_col, site_time = "", "#00D2FF", 0
            is_active = False
            is_rtb_complete = False
            
            if curr_time < d['t_out']:
                phase_txt = "OUTBOUND"
                flight_prog = curr_time / d['t_out']
                is_active = True
            elif curr_time < (d['t_out'] + d['t_hov']):
                phase_txt, site_time = "ON SCENE", curr_time - d['t_out']
                flight_prog = 1.0
                is_active = True
            elif curr_time < d['t_total']:
                phase_txt = "RTB"
                site_time = d['t_hov']
                flight_prog = 1.0 - ((curr_time - d['t_out'] - d['t_hov']) / d['t_out'])
                is_active = True
            else:
                if ui['specs']['model'].upper() == 'GUARDIAN':
                    phase_txt = "SWAPPING BATT"
                else:
                    phase_txt = "RECHARGING"
                
                phase_col = "#FFC300" 
                site_time = d['t_hov']
                flight_prog = 0.0
                is_active = False
                is_rtb_complete = True

            name_class = "drone-active" if is_active else "drone-static"
            ui['name_text'].markdown(f"<span class='{name_class}'>{ui['specs']['model']}</span>", unsafe_allow_html=True)
            ui['status_text'].markdown(f"<div style='text-align:right; margin-bottom:-10px;'><span style='color:{phase_col}; font-size:0.8rem; font-weight:bold; font-family: \"IBM Plex Mono\", monospace;'>{phase_txt}</span></div>", unsafe_allow_html=True)
            ui['flight_bar'].progress(max(0.0, min(flight_prog, 1.0)))
            
            used = min(curr_time, d['t_out']) + max(0, min(curr_time - d['t_out'], d['t_hov'])) + max(0, min(curr_time - (d['t_out'] + d['t_hov']), d['t_out']))
            
            eta_label = "TIME TO TGT"
            display_time = min(curr_time, d['t_out'])
            eta_val = f"{int(display_time/60):02d}:{int(display_time%60):02d}"
            hov_val = f"{int(site_time/60):02d}:{int(site_time%60):02d}"
            pct = max(0, 100 - (used / d['batt_cap'] * 100))
            
            if is_rtb_complete:
                mission_progress = used / d['t_total'] if d['t_total'] > 0 else 0
                current_recharge_min = d['turnaround_min'] * mission_progress
                t_min = int(current_recharge_min)
                t_sec = int((current_recharge_min * 60) % 60)
                bat_label = "<span style='color: #ffffff;'>RECHARGE</span>"
                bat_val = f"{t_min:02d}m {t_sec:02d}s"
            else:
                bat_label = "BATTERY"
                bat_val = f"{int(pct)}%"

            card_html = f"""
            <div class="drone-card">
                <div class="metric-grid">
                    <div class="m-box"><div class="m-label">{eta_label}</div><div class="m-val">{eta_val}</div></div>
                    <div class="m-box"><div class="m-label">ON SCENE</div><div class="m-val">{hov_val}</div></div>
                    <div class="m-box"><div class="m-label">{bat_label}</div><div class="m-val">{bat_val}</div></div>
                </div>
            </div>
            """
            ui['metrics_html'].markdown(card_html, unsafe_allow_html=True)

    if not st.session_state.sim_completed:
        last_stages_hash = None
        
        for tick in range(101):
            curr_time = (tick / 100) * sim_dur
            render_ui_state(curr_time)
            
            current_drones = []
            for idx, d in enumerate(valid):
                t_out = d['t_out']
                t_hov = d['t_hov']
                t_total = d['t_total']
                
                if curr_time < t_out * 0.15:
                    stage = 0 
                elif curr_time < t_out * 0.85:
                    stage = 1 
                elif curr_time < t_out + t_hov * 0.85:
                    stage = 2 
                elif curr_time < t_total - (t_out * 0.15):
                    stage = 3 
                else:
                    stage = 4 
                    
                current_drones.append({
                    'stage': stage,
                    'color': DRONE_COLORS[idx % 4],
                    'letter': d['ui']['specs']['model'][0].upper()
                })
                
            stages_hash = "-".join([str(x['stage']) for x in current_drones])
            
            if stages_hash != last_stages_hash:
                with map_placeholder:
                    m_sim = generate_base_map(drones_to_draw=current_drones)
                    st_folium(m_sim, height=850, use_container_width=True, key=f"sim_map_{stages_hash}")
                last_stages_hash = stages_hash
                
            time.sleep(0.16)

        time.sleep(3.0) 
        st.session_state.sim_completed = True
        st.session_state.has_run_once = True 
        st.rerun()
        
    else:
        render_ui_state(sim_dur)
