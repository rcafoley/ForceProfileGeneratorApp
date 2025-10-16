import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import base64
from io import BytesIO
import openpyxl
from datetime import datetime
import os

"""
Force profile generator for complex fatigue modeling
Copyright Ryan C. A. Foley 2023-10-29
Updated with Streamlit implementation 2025-01-24
v4.1.1
"""

# Page configuration
st.set_page_config(
    page_title="Force Profile Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ontario Tech University branding colors
COLORS = {
    'white': '#FFFFFF',
    'future_blue': '#003C71',
    'simcoe_blue': '#C00B2E',
    'tech_tangerine': '#E75D2A',
    'warm_grey': '#ACA39A',
    'cool_grey': '#A7A8AA',
    'dark_grey': '#5B6770',
    'spirit_navy': '#00283C'
}

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main content area */
    .main {
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #003C71;
        background-color: white;
        color: #003C71;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #003C71;
        color: white;
    }
    
    /* Remove the grey boxes between elements */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Card-like containers for subtasks */
    .subtask-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Title styling */
    h1 {
        color: #003C71;
        border-bottom: 3px solid #E75D2A;
        padding-bottom: 10px;
    }
    
    h2, h3 {
        color: #003C71;
    }
    
    /* Hide streamlit's default container padding */
    .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'subtasks' not in st.session_state:
    st.session_state.subtasks = []
if 'saved_profiles' not in st.session_state:
    st.session_state.saved_profiles = {}
if 'combined_profiles' not in st.session_state:
    st.session_state.combined_profiles = []
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 10.0
if 'export_df' not in st.session_state:
    st.session_state.export_df = pd.DataFrame()
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Profile Builder"

# Sidebar navigation
with st.sidebar:
    # Profile section
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://github.com/rcafoley.png", width=80)
    with col2:
        st.markdown("**Ryan C. A. Foley**")
        st.markdown("PhD Candidate, CSEP-CEP")
    
    st.markdown("[Website](https://rcafoley.github.io) | [GitHub](https://github.com/rcafoley) | [Email](mailto:ryan.foley@ontariotechu.ca)")
    st.markdown("[ryan@afferentresearch.com](mailto:ryan@afferentresearch.com)")
    st.markdown("---")
    
    # Navigation menu
    st.markdown("### Navigation")
    pages = {
        "Profile Builder": "Profile Builder",
        "Visualize & Store": "Visualize & Store",
        "Import Data": "Import Data",
        "Combine Profiles": "Combine Profiles",
        "Export Data": "Export Data",
        "About": "About"
    }
    
    for title, page_name in pages.items():
        if st.button(title, key=f"nav_{page_name}"):
            st.session_state.current_page = page_name
    
    # Visual indicator of current page
    st.markdown("---")
    st.markdown(f"**Current Page:** {st.session_state.current_page}")

# Helper functions
def create_subtask(index=None):
    """Create a new subtask dictionary"""
    if index is None:
        index = len(st.session_state.subtasks)
    
    return {
        'index': index,
        'type': 'Force',  # Default to Force type
        'description': '',
        'timing': 0.0,
        'force': 0.0,
        'start_force': 0.0,
        'end_force': 0.0,
        'midpoint_force': 0.0,
        'flexion': 5.0
    }

def validate_force_input(value, min_val=0.0, max_val=100.0):
    """Validate force input to be within 0-100 range"""
    try:
        val = float(value) if value else 0.0
        return max(min_val, min(val, max_val))
    except:
        return 0.0

def validate_time_input(value):
    """Validate time input to be non-negative"""
    try:
        val = float(value) if value else 0.0
        return max(0.0, val)
    except:
        return 0.0

def move_subtask(index, direction):
    """Move subtask up or down in the list"""
    subtasks = st.session_state.subtasks
    if direction == 'up' and index > 0:
        subtasks[index], subtasks[index-1] = subtasks[index-1], subtasks[index]
    elif direction == 'down' and index < len(subtasks) - 1:
        subtasks[index], subtasks[index+1] = subtasks[index+1], subtasks[index]
    st.session_state.subtasks = subtasks

def delete_subtask(index):
    """Delete a subtask"""
    st.session_state.subtasks.pop(index)

def render_subtask(subtask, index):
    """Render a single subtask UI component"""
    # Use a container for each subtask
    with st.container():
        # Create columns for the subtask layout
        col_num, col_type, col_desc, col_time, col_force, col_controls = st.columns([0.5, 1.5, 2, 1.5, 3, 1.5])
        
        # Subtask number
        col_num.markdown(f"**#{index + 1}**")
        
        # Subtask type dropdown
        subtask_types = ['Force', 'Slope', 'Ramp', 'Precision', 'Rest']
        new_type = col_type.selectbox(
            "Type",
            options=subtask_types,
            index=subtask_types.index(subtask['type']),
            key=f"type_{index}",
            label_visibility="collapsed"
        )
        
        # Update type if changed
        if new_type != subtask['type']:
            subtask['type'] = new_type
            st.rerun()
        
        # Description
        subtask['description'] = col_desc.text_input(
            "Description",
            value=subtask['description'],
            key=f"desc_{index}",
            label_visibility="collapsed",
            placeholder="Describe the subtask"
        )
        
        # Timing (with validation)
        time_input = col_time.text_input(
            "Time (s)",
            value=str(subtask['timing']),
            key=f"time_{index}",
            label_visibility="collapsed",
            placeholder="Time (s)"
        )
        subtask['timing'] = validate_time_input(time_input)
        
        # Force parameters based on type
        if subtask['type'] == 'Force':
            force_input = col_force.text_input(
                "Force (%MVC)",
                value=str(subtask['force']),
                key=f"force_{index}",
                label_visibility="collapsed",
                placeholder="Force (0-100%)"
            )
            subtask['force'] = validate_force_input(force_input)
            
        elif subtask['type'] == 'Slope':
            subcols = col_force.columns(2)
            subcols[0].markdown("**Start →**", help="Initial force level at the beginning of the slope (0-100% MVC)")
            start_input = subcols[0].text_input(
                "Start",
                value=str(subtask['start_force']),
                key=f"start_{index}",
                label_visibility="collapsed",
                placeholder="Start Force (%MVC)",
                help="Starting force level (0-100%)"
            )
            subcols[1].markdown("**→ End**", help="Final force level at the end of the slope (0-100% MVC)")
            end_input = subcols[1].text_input(
                "End",
                value=str(subtask['end_force']),
                key=f"end_{index}",
                label_visibility="collapsed",
                placeholder="End Force (%MVC)",
                help="Ending force level (0-100%)"
            )
            subtask['start_force'] = validate_force_input(start_input)
            subtask['end_force'] = validate_force_input(end_input)
            
        elif subtask['type'] == 'Ramp':
            subcols = col_force.columns(3)
            subcols[0].markdown("**Start**", help="Initial force level")
            start_input = subcols[0].text_input(
                "Start",
                value=str(subtask['start_force']),
                key=f"ramp_start_{index}",
                label_visibility="collapsed",
                placeholder="Start (%MVC)",
                help="Starting force (0-100%)"
            )
            subcols[1].markdown("**Peak**", help="Maximum force level reached")
            mid_input = subcols[1].text_input(
                "Peak",
                value=str(subtask['midpoint_force']),
                key=f"ramp_mid_{index}",
                label_visibility="collapsed",
                placeholder="Peak (%MVC)",
                help="Peak force at midpoint (0-100%)"
            )
            subcols[2].markdown("**End**", help="Final force level")
            end_input = subcols[2].text_input(
                "End",
                value=str(subtask['end_force']),
                key=f"ramp_end_{index}",
                label_visibility="collapsed",
                placeholder="End (%MVC)",
                help="Ending force (0-100%)"
            )
            subtask['start_force'] = validate_force_input(start_input)
            subtask['midpoint_force'] = validate_force_input(mid_input)
            subtask['end_force'] = validate_force_input(end_input)
            
        elif subtask['type'] == 'Precision':
            subtask['flexion'] = col_force.selectbox(
                "Shoulder Flexion",
                options=[5.0, 12.0, 13.5],
                format_func=lambda x: {5.0: "0-45°", 12.0: "45-90°", 13.5: ">90°"}[x],
                index=0 if subtask['flexion'] is None else [5.0, 12.0, 13.5].index(float(subtask['flexion'])) if subtask['flexion'] in [5, 5.0, 12, 12.0, 13.5] else 0,
                key=f"flexion_{index}",
                label_visibility="collapsed"
            )
            
        elif subtask['type'] == 'Rest':
            col_force.markdown("*Rest Period*")
        
        # Control buttons
        button_cols = col_controls.columns(3)
        if button_cols[0].button("↑", key=f"up_{index}", help="Move up"):
            move_subtask(index, 'up')
            st.rerun()
            
        if button_cols[1].button("↓", key=f"down_{index}", help="Move down"):
            move_subtask(index, 'down')
            st.rerun()
            
        if button_cols[2].button("×", key=f"del_{index}", help="Delete"):
            delete_subtask(index)
            st.rerun()

def generate_force_profile(subtasks, sample_rate):
    """Generate force profile data from subtasks"""
    force_profile = []
    time_array = []
    task_descriptions = []
    subtask_segments = []  # Store info about where each subtask starts/ends
    
    colors = {
        'Rest': COLORS['warm_grey'],
        'Force': COLORS['future_blue'],
        'Precision': COLORS['spirit_navy'],
        'Slope': COLORS['simcoe_blue'],
        'Ramp': COLORS['tech_tangerine']
    }
    
    fig = go.Figure()
    current_time = 0
    
    for subtask in subtasks:
        timing = float(subtask['timing']) if subtask['timing'] else 0.0
        if timing <= 0:
            continue
            
        n_points = max(1, int(timing * sample_rate))
        time_segment = [current_time + i / sample_rate for i in range(n_points)]
        
        # Store segment info
        segment_start = len(force_profile)
        
        # Calculate force values based on type
        if subtask['type'] == 'Rest':
            force_segment = [0] * n_points
        elif subtask['type'] == 'Force':
            force_val = float(subtask['force']) if subtask['force'] else 0.0
            force_segment = [force_val] * n_points
        elif subtask['type'] == 'Precision':
            force_val = float(subtask['flexion']) if subtask['flexion'] else 0.0
            force_segment = [force_val] * n_points
        elif subtask['type'] == 'Slope':
            start = float(subtask['start_force']) if subtask['start_force'] else 0.0
            end = float(subtask['end_force']) if subtask['end_force'] else 0.0
            force_segment = np.linspace(start, end, n_points).tolist()
        elif subtask['type'] == 'Ramp':
            start = float(subtask['start_force']) if subtask['start_force'] else 0.0
            mid = float(subtask['midpoint_force']) if subtask['midpoint_force'] else 0.0
            end = float(subtask['end_force']) if subtask['end_force'] else 0.0
            half_point = n_points // 2
            first_half = np.linspace(start, mid, half_point)
            second_half = np.linspace(mid, end, n_points - half_point)
            force_segment = np.concatenate([first_half, second_half]).tolist()
        
        # Add segments to our arrays
        force_profile.extend(force_segment)
        time_array.extend(time_segment)
        task_descriptions.extend([subtask['description']] * n_points)
        
        # Store segment info for hover data
        subtask_segments.append({
            'start': segment_start,
            'end': len(force_profile),
            'type': subtask['type'],
            'description': subtask['description'] or subtask['type']
        })
        
        current_time = time_segment[-1] + (1.0 / sample_rate)
    
    # Create one continuous trace
    if force_profile:
        # Create custom hover text
        hover_text = []
        for i in range(len(force_profile)):
            # Find which segment this point belongs to
            segment_info = None
            for seg in subtask_segments:
                if seg['start'] <= i < seg['end']:
                    segment_info = seg
                    break
            
            if segment_info:
                hover_text.append(
                    f"Time: {time_array[i]:.2f}s<br>" +
                    f"Force: {force_profile[i]:.1f}%MVC<br>" +
                    f"Type: {segment_info['type']}<br>" +
                    f"Description: {segment_info['description']}"
                )
            else:
                hover_text.append(f"Time: {time_array[i]:.2f}s<br>Force: {force_profile[i]:.1f}%MVC")
        
        # Add the single continuous trace
        fig.add_trace(go.Scatter(
            x=time_array,
            y=force_profile,
            mode='lines',
            name='Force Profile',
            line=dict(color=COLORS['future_blue'], width=3),
            hovertext=hover_text,
            hoverinfo='text'
        ))
        
        # Add vertical lines to show subtask boundaries
        for i, seg in enumerate(subtask_segments[:-1]):  # All but last segment
            boundary_time = time_array[seg['end']-1] if seg['end'] > 0 else 0
            fig.add_vline(
                x=boundary_time,
                line_width=1,
                line_dash="dash",
                line_color="gray",
                opacity=0.5
            )
        
        # Add annotations for subtask types at the top
        for seg in subtask_segments:
            if seg['end'] > seg['start']:
                mid_point = (seg['start'] + seg['end']) // 2
                if mid_point < len(time_array):
                    fig.add_annotation(
                        x=time_array[mid_point],
                        y=max(force_profile) * 1.1 if max(force_profile) > 0 else 10,
                        text=seg['type'],
                        showarrow=False,
                        font=dict(size=10, color=colors.get(seg['type'], 'black')),
                        bgcolor="white",
                        bordercolor=colors.get(seg['type'], 'black'),
                        borderwidth=1,
                        borderpad=2
                    )
    
    fig.update_layout(
        title="Force Profile Visualization",
        xaxis_title="Time (s)",
        yaxis_title="Force (%MVC)",
        hovermode="closest",
        height=500,
        showlegend=False,
        yaxis=dict(range=[0, max(force_profile) * 1.2] if force_profile else [0, 100])
    )
    
    return fig, force_profile, time_array, task_descriptions

# Main content area
if st.session_state.current_page == "Profile Builder":
    st.title("Force Profile Builder")
    
    # Sample rate setting
    col1, col2 = st.columns([1, 3])
    with col1:
        sample_rate_input = st.text_input(
            "Sample Rate (Hz)",
            value=str(st.session_state.sample_rate),
            help="Enter sample rate in Hz (must be positive)"
        )
        st.session_state.sample_rate = max(1.0, float(sample_rate_input) if sample_rate_input else 10.0)
    
    st.markdown("---")
    
    # Add subtask button
    st.subheader("Subtask Manager")
    if st.button("Add Subtask", type="primary"):
        st.session_state.subtasks.append(create_subtask())
        st.rerun()
    
    # Display subtasks
    if st.session_state.subtasks:
        st.subheader("Current Subtasks")
        
        # Header row
        header_cols = st.columns([0.5, 1.5, 2, 1.5, 3, 1.5])
        header_cols[0].markdown("**#**")
        header_cols[1].markdown("**Type**")
        header_cols[2].markdown("**Description**")
        header_cols[3].markdown("**Time (s)**")
        header_cols[4].markdown("**Force Parameters**")
        header_cols[5].markdown("**Actions**")
        
        # Render each subtask
        for i, subtask in enumerate(st.session_state.subtasks):
            render_subtask(subtask, i)
        
        # Summary
        total_time = sum(float(s['timing']) if s['timing'] else 0.0 for s in st.session_state.subtasks)
        st.info(f"**Total Cycle Time:** {total_time:.1f} seconds")
        
        # Clear all button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col3:
            if st.button("Clear All Subtasks", type="secondary"):
                st.session_state.subtasks = []
                st.rerun()
    else:
        st.info("No subtasks added yet. Click 'Add Subtask' to begin.")

elif st.session_state.current_page == "Visualize & Store":
    st.title("Visualize & Store Force Profiles")
    
    if st.session_state.subtasks:
        # Generate and display the force profile
        fig, force_profile, time_array, task_descriptions = generate_force_profile(
            st.session_state.subtasks, 
            st.session_state.sample_rate
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save profile section
        st.markdown("---")
        st.subheader("Save Current Profile")
        
        col1, col2 = st.columns([3, 1])
        profile_name = col1.text_input("Profile Name", placeholder="Enter a name for this profile")
        
        if col2.button("Save Profile", type="primary"):
            if profile_name:
                profile_id = f"profile_{len(st.session_state.saved_profiles) + 1}"
                st.session_state.saved_profiles[profile_id] = {
                    'name': profile_name,
                    'subtasks': st.session_state.subtasks.copy(),
                    'force_profile': force_profile,
                    'time_array': time_array,
                    'task_descriptions': task_descriptions,
                    'sample_rate': st.session_state.sample_rate
                }
                st.success(f"Profile '{profile_name}' saved successfully!")
            else:
                st.error("Please enter a profile name.")
        
        # Export section with explanation
        st.markdown("---")
        st.subheader("Export Options")
        
        st.info("""
        **Export Full Profile:** Includes metadata, subtask details, and all force-time data
        
        **Export Subtask Template:** Saves only the subtask definitions for reuse (no time series data)
        """)
        
        # Export current profile
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Full Profile to Excel"):
                df = pd.DataFrame({
                    'Time (s)': time_array,
                    'Force (%MVC)': force_profile,
                    'Task Description': task_descriptions
                })
                
                # Create metadata
                metadata = {
                    'Parameter': ['Profile Name', 'Sample Rate (Hz)', 'Total Duration (s)', 'Number of Subtasks', 'Export Date'],
                    'Value': [
                        profile_name or 'Unnamed Profile',
                        st.session_state.sample_rate,
                        f"{time_array[-1] if time_array else 0:.2f}",
                        len(st.session_state.subtasks),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                metadata_df = pd.DataFrame(metadata)
                
                # Create subtask details
                subtask_details = []
                for i, subtask in enumerate(st.session_state.subtasks):
                    details = {
                        'Subtask #': i + 1,
                        'Type': subtask['type'],
                        'Description': subtask['description'] or 'No description',
                        'Duration (s)': subtask['timing'],
                    }
                    
                    if subtask['type'] == 'Force':
                        details['Force (%MVC)'] = subtask['force']
                    elif subtask['type'] == 'Slope':
                        details['Start Force (%MVC)'] = subtask['start_force']
                        details['End Force (%MVC)'] = subtask['end_force']
                    elif subtask['type'] == 'Ramp':
                        details['Start Force (%MVC)'] = subtask['start_force']
                        details['Peak Force (%MVC)'] = subtask['midpoint_force']
                        details['End Force (%MVC)'] = subtask['end_force']
                    elif subtask['type'] == 'Precision':
                        details['Shoulder Flexion'] = {5.0: "0-45°", 12.0: "45-90°", 13.5: ">90°"}.get(subtask['flexion'], "Unknown")
                        details['Force (%MVC)'] = subtask['flexion']
                    elif subtask['type'] == 'Rest':
                        details['Force (%MVC)'] = 0
                    
                    subtask_details.append(details)
                
                subtask_df = pd.DataFrame(subtask_details)
                
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    # Write metadata
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                    
                    # Write subtask details
                    subtask_df.to_excel(writer, sheet_name='Subtask Details', index=False)
                    
                    # Write force profile data
                    df.to_excel(writer, sheet_name='Force Profile Data', index=False)
                    
                    # Format the sheets
                    for sheet_name in writer.sheets:
                        worksheet = writer.sheets[sheet_name]
                        for column in worksheet.columns:
                            max_length = 0
                            column = [cell for cell in column]
                            for cell in column:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(str(cell.value))
                                except:
                                    pass
                            adjusted_width = min(max_length + 2, 50)
                            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
                
                st.download_button(
                    label="Download Full Profile",
                    data=buffer.getvalue(),
                    file_name=f"{profile_name or 'force_profile'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            if st.button("Export Subtask Template"):
                # Export just the subtask definitions for reuse
                subtask_template = []
                for i, subtask in enumerate(st.session_state.subtasks):
                    template = {
                        'Order': i + 1,
                        'Type': subtask['type'],
                        'Description': subtask['description'] or '',
                        'Duration (s)': subtask['timing'],
                        'Force': subtask.get('force', ''),
                        'Start Force': subtask.get('start_force', ''),
                        'Peak/Mid Force': subtask.get('midpoint_force', ''),
                        'End Force': subtask.get('end_force', ''),
                        'Shoulder Flexion': subtask.get('flexion', '')
                    }
                    subtask_template.append(template)
                
                template_df = pd.DataFrame(subtask_template)
                
                buffer = BytesIO()
                template_df.to_excel(buffer, index=False)
                
                st.download_button(
                    label="Download Template",
                    data=buffer.getvalue(),
                    file_name=f"subtask_template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.warning("No subtasks to visualize. Please add subtasks in the Profile Builder.")

elif st.session_state.current_page == "Import Data":
    st.title("Import Force Profile Data")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload a force profile Excel file"
    )
    
    if uploaded_file is not None:
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            # Check if this is a multi-sheet export or single sheet
            if 'Force Profile Data' in sheet_names:
                # New format with metadata
                metadata_df = pd.read_excel(uploaded_file, sheet_name='Metadata')
                subtask_df = pd.read_excel(uploaded_file, sheet_name='Subtask Details')
                df = pd.read_excel(uploaded_file, sheet_name='Force Profile Data')
                
                st.success("File uploaded successfully!")
                
                # Display metadata
                st.subheader("Profile Metadata")
                for _, row in metadata_df.iterrows():
                    st.write(f"**{row['Parameter']}:** {row['Value']}")
                
                # Display subtask details
                st.subheader("Subtask Details")
                st.dataframe(subtask_df)
                
                # Extract sample rate from metadata
                sample_rate_row = metadata_df[metadata_df['Parameter'] == 'Sample Rate (Hz)']
                imported_sample_rate = float(sample_rate_row['Value'].iloc[0]) if not sample_rate_row.empty else 10.0
                
            else:
                # Old format or generic Excel
                df = pd.read_excel(uploaded_file)
                st.success("File uploaded successfully!")
                imported_sample_rate = st.session_state.sample_rate
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Detect columns
            time_col = None
            force_col = None
            desc_col = None
            
            # Look for columns by name (case-insensitive)
            for col in df.columns:
                col_lower = col.lower()
                if 'time' in col_lower:
                    time_col = col
                elif 'force' in col_lower and '%mvc' in col_lower:
                    force_col = col
                elif 'description' in col_lower or 'task' in col_lower:
                    desc_col = col
            
            # If force column not found, try first numeric column that's not time
            if not force_col:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if time_col and time_col in numeric_cols:
                    numeric_cols.remove(time_col)
                if numeric_cols:
                    force_col = numeric_cols[0]
            
            if force_col:
                st.info(f"Detected force column: '{force_col}'")
            else:
                st.warning("Could not automatically detect force column. Please select from available columns:")
                force_col = st.selectbox("Select Force Column", df.columns.tolist())
            
            # Import options
            col1, col2 = st.columns(2)
            
            with col1:
                profile_name = st.text_input("Profile Name", value=uploaded_file.name.split('.')[0])
            
            with col2:
                if st.button("Import as New Profile", type="primary"):
                    # Extract force values
                    force_values = df[force_col].tolist() if force_col in df.columns else df.iloc[:, 0].tolist()
                    
                    # Extract time values if available
                    if time_col and time_col in df.columns:
                        time_array = df[time_col].tolist()
                    else:
                        # Generate time array based on imported sample rate
                        time_array = [i / imported_sample_rate for i in range(len(force_values))]
                    
                    # Extract descriptions if available
                    if desc_col and desc_col in df.columns:
                        task_descriptions = df[desc_col].tolist()
                    else:
                        task_descriptions = ['Imported'] * len(force_values)
                    
                    profile_id = f"imported_{len(st.session_state.saved_profiles) + 1}"
                    st.session_state.saved_profiles[profile_id] = {
                        'name': profile_name,
                        'subtasks': [],  # No subtask breakdown for imported data
                        'force_profile': force_values,
                        'time_array': time_array,
                        'task_descriptions': task_descriptions,
                        'sample_rate': imported_sample_rate
                    }
                    st.success(f"Profile '{profile_name}' imported successfully with {len(force_values)} data points!")
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif st.session_state.current_page == "Combine Profiles":
    st.title("Combine Force Profiles")
    
    if st.session_state.saved_profiles:
        # Display saved profiles
        st.subheader("Available Profiles")
        
        profile_list = []
        for pid, profile in st.session_state.saved_profiles.items():
            profile_list.append(f"{profile['name']} (ID: {pid})")
        
        selected_profile = st.selectbox("Select a profile to add", profile_list)
        
        if st.button("Add to Combined Profile"):
            # Extract profile ID from selection
            pid = selected_profile.split("(ID: ")[1].rstrip(")")
            st.session_state.combined_profiles.append(st.session_state.saved_profiles[pid])
            st.success(f"Added '{st.session_state.saved_profiles[pid]['name']}' to combined profile")
        
        # Display combined profiles
        if st.session_state.combined_profiles:
            st.markdown("---")
            st.subheader("Combined Profile Components")
            
            for i, profile in enumerate(st.session_state.combined_profiles):
                col1, col2 = st.columns([4, 1])
                col1.write(f"{i+1}. {profile['name']}")
                if col2.button("Remove", key=f"remove_{i}"):
                    st.session_state.combined_profiles.pop(i)
                    st.rerun()
            
            # Generate combined profile visualization
            combined_force = []
            combined_time = []
            combined_descriptions = []
            current_time = 0
            
            fig = go.Figure()
            
            # Create one continuous line for the combined profile
            for profile in st.session_state.combined_profiles:
                # Add this profile's data
                profile_time = [t + current_time for t in profile['time_array']]
                combined_force.extend(profile['force_profile'])
                combined_time.extend(profile_time)
                combined_descriptions.extend(profile['task_descriptions'])
                
                # Add a vertical line at profile boundaries
                if current_time > 0:
                    fig.add_vline(
                        x=current_time,
                        line_width=1,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.5
                    )
                
                # Add annotation for profile name
                if profile_time:
                    mid_time = (profile_time[0] + profile_time[-1]) / 2
                    max_force_in_segment = max(profile['force_profile']) if profile['force_profile'] else 0
                    fig.add_annotation(
                        x=mid_time,
                        y=max_force_in_segment * 1.1 if max_force_in_segment > 0 else 10,
                        text=profile['name'],
                        showarrow=False,
                        font=dict(size=12, color='black'),
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4
                    )
                
                current_time = profile_time[-1] + (1.0 / st.session_state.sample_rate)
            
            # Add the single continuous trace
            if combined_force:
                fig.add_trace(go.Scatter(
                    x=combined_time,
                    y=combined_force,
                    mode='lines',
                    name='Combined Profile',
                    line=dict(color=COLORS['future_blue'], width=3)
                ))
            
            fig.update_layout(
                title="Combined Force Profile",
                xaxis_title="Time (s)",
                yaxis_title="Force (%MVC)",
                height=500,
                showlegend=False,
                yaxis=dict(range=[0, max(combined_force) * 1.2] if combined_force else [0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Push to dataframe button
            if st.button("Push to Export DataFrame", type="primary"):
                # Create dataframe from combined profile
                profile_id = len(st.session_state.export_df.columns) // 4 + 1
                
                new_data = pd.DataFrame({
                    f'Timing_{profile_id}': combined_time,
                    f'Force Values_{profile_id}': combined_force,
                    f'Task Descriptions_{profile_id}': combined_descriptions,
                    f'Profile ID_{profile_id}': [profile_id] * len(combined_time)
                })
                
                # Merge with existing export dataframe
                if st.session_state.export_df.empty:
                    st.session_state.export_df = new_data
                else:
                    # Align indices
                    max_len = max(len(st.session_state.export_df), len(new_data))
                    st.session_state.export_df = st.session_state.export_df.reindex(range(max_len), fill_value=np.nan)
                    new_data = new_data.reindex(range(max_len), fill_value=np.nan)
                    st.session_state.export_df = pd.concat([st.session_state.export_df, new_data], axis=1)
                
                st.success("Combined profile added to export dataframe!")
                #st.session_state.combined_profiles = []  # Clear after pushing
            
            # Clear button
            if st.button("Clear Combined Profiles", type="secondary"):
                st.session_state.combined_profiles = []
                st.rerun()
    else:
        st.warning("No saved profiles available. Please create and save profiles first.")

elif st.session_state.current_page == "Export Data":
    st.title("Export Force Profile Data")
    
    if not st.session_state.export_df.empty:
        st.subheader("Export DataFrame Preview")
        st.dataframe(st.session_state.export_df.head(20))
        
        st.info(f"DataFrame contains {len(st.session_state.export_df)} rows and {len(st.session_state.export_df.columns)} columns")
        
        # Export button
        if st.button("Export to Excel", type="primary"):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Create metadata for the combined export
                metadata = {
                    'Parameter': ['Export Type', 'Number of Profiles', 'Total Data Points', 'Export Date'],
                    'Value': [
                        'Combined Force Profiles',
                        len(st.session_state.export_df.columns) // 4,
                        len(st.session_state.export_df),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                metadata_df = pd.DataFrame(metadata)
                
                # Write metadata
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Write the main data
                st.session_state.export_df.to_excel(writer, sheet_name='Combined Profile Data', index=False)
                
                # Format the sheets
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column = [cell for cell in column]
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
            
            st.download_button(
                label="Download Excel File",
                data=buffer.getvalue(),
                file_name=f"force_profiles_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Clear button
        if st.button("Clear Export Data", type="secondary"):
            st.session_state.export_df = pd.DataFrame()
            st.success("Export data cleared!")
            st.rerun()
    else:
        st.warning("No data to export. Please combine profiles and push them to the export dataframe.")

elif st.session_state.current_page == "About":
    st.title("About Force Profile Generator")
    
    st.markdown("---")
    
    st.markdown("""
    ## Force Profile Generator for Neuromuscular Fatigue Modelling
    
    **Version:** 4.1.1  
    **Original Author:** Ryan C. A. Foley, PhD Candidate, CSEP Clinical Exercise Physiologist  
    **Current Laboratory:** Occupational Neuromechanics and Ergonomics (ONE) Laboratory – Dr. Nicholas La Delfa  
    **Institutional Affiliation:** Ontario Tech University, Oshawa, Ontario, Canada  
    
    ### Contact
    - **Academic Email:** ryan.foley@ontariotechu.ca
    - **Research Email:** ryan@afferentresearch.com
    - **GitHub:** [github.com/rcafoley](https://github.com/rcafoley)
    
    ### Purpose
    This application generates force-time profiles for use in neuromuscular fatigue modelling. It allows researchers and clinicians to:
    - Create complex force profiles with multiple subtask types
    - Label the subtasks for easy adjustment and recall at a later time
    - Visualize force patterns over time to verify the structure
    - Repeat a profile or combine multiple profiles for complex work simulations
    - Export data for further analysis via a muscle fatigue model
    
    ### Features
    - **Multiple Subtask Types:** Constant force, linear slopes, ramp contractions, shoulder activity during precision hand work, and rest periods
    - **Visual Profile Builder:** Intuitive interface with drag-and-drop reordering
    - **Profile Management:** Save, load, and combine multiple profiles
    - **Data Export:** Export to Excel for further analysis
    - **Real-time Visualization:** Interactive Plotly charts
    
    ### Technical Details
    - **Sample Rate:** Configurable from 1-1000 Hz
    - **Force Units:** Percentage of Maximum Voluntary Contraction (%MVC)
    - **Time Units:** Seconds
    - **Export Format:** Excel (.xlsx) with metadata, subtask details, and force profile data
    
    ### Precision Force Assumptions
    The **Precision** subtask type is designed to estimate shoulder muscle activation during precision manual work tasks. This feature accounts for the postural demands of maintaining shoulder flexion during fine motor activities that have no external loads applied.
    
    The force values for different shoulder flexion angles are derived from:
    1. **Published Research:** Brookham, R. L., Wong, J. M., & Dickerson, C. R. (2010). Upper limb posture and submaximal hand tasks influence shoulder muscle activity. *International Journal of Industrial Ergonomics*, *40*(3), 337-344.
    2. **Unpublished Data:** Overhead work EMG data from the ONE Laboratory (Ontario Tech University)
    
    There is the assumption of a relatively upright posture, such that gravity is accelerating the upper limb segments inferiorly and the angles are expressed as the humero-thoracic angle. The implemented values represent anterior deltoid activation as a percentage of maximum voluntary contraction (%MVC):
    - **0-45° shoulder flexion:** 5% MVC
    - **45-90° shoulder flexion:** 12% MVC
    - **>90° shoulder flexion:** 13.5% MVC
    
    These values provide reasonable estimates for fatigue modelling during precision work tasks requiring sustained shoulder postures.
    
    ---
    *© 2023-2025 Ryan C. A. Foley. All rights reserved.*
    """)
# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: {COLORS['dark_grey']};'>"
    f"Muscle Fatigue Analysis System v4.1.1 | © 2025 ONE Laboratory, Ontario Tech University"
    f"</div>",
    unsafe_allow_html=True
)