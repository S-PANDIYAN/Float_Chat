import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import folium
from folium import plugins
import logging

logger = logging.getLogger(__name__)

class ArgoVisualizer:
    """Visualization components for ARGO oceanographic data"""
    
    def __init__(self):
        self.color_schemes = {
            'temperature': 'RdYlBu_r',
            'salinity': 'viridis',
            'pressure': 'plasma'
        }
    
    def plot_temperature_salinity_profile(self, profile_data: Dict) -> go.Figure:
        """Create temperature and salinity profile plot"""
        try:
            # Extract data
            temp_data = profile_data.get('temperature_data', [])
            sal_data = profile_data.get('salinity_data', [])
            pressure_data = profile_data.get('pressure_data', [])
            
            if not (temp_data and pressure_data):
                raise ValueError("Insufficient data for profile plot")
            
            fig = go.Figure()
            
            # Temperature profile
            if temp_data:
                fig.add_trace(go.Scatter(
                    x=temp_data,
                    y=pressure_data,
                    mode='lines+markers',
                    name='Temperature (°C)',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                    xaxis='x1'
                ))
            
            # Salinity profile
            if sal_data:
                fig.add_trace(go.Scatter(
                    x=sal_data,
                    y=pressure_data,
                    mode='lines+markers',
                    name='Salinity (PSU)',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4),
                    xaxis='x2'
                ))
            
            # Layout configuration
            fig.update_layout(
                title=f"ARGO Profile - Float {profile_data.get('float_id', 'Unknown')} "
                      f"Cycle {profile_data.get('cycle_number', 'N/A')}",
                xaxis=dict(
                    title="Temperature (°C)",
                    side='top',
                    color='red'
                ),
                xaxis2=dict(
                    title="Salinity (PSU)",
                    side='bottom',
                    overlaying='x',
                    color='blue'
                ),
                yaxis=dict(
                    title="Pressure (dbar)",
                    autorange='reversed'  # Depth increases downward
                ),
                height=600,
                hovermode='y unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating profile plot: {e}")
            raise
    
    def plot_trajectory_map(self, profiles: List[Dict]) -> folium.Map:
        """Create trajectory map showing float paths"""
        try:
            if not profiles:
                raise ValueError("No profile data provided")
            
            # Extract coordinates
            coordinates = []
            dates = []
            float_ids = []
            
            for profile in profiles:
                lat = profile.get('latitude')
                lon = profile.get('longitude')
                date = profile.get('profile_date')
                float_id = profile.get('float_id')
                
                if lat is not None and lon is not None:
                    coordinates.append([lat, lon])
                    dates.append(date)
                    float_ids.append(float_id)
            
            if not coordinates:
                raise ValueError("No valid coordinates found")
            
            # Calculate map center
            center_lat = np.mean([coord[0] for coord in coordinates])
            center_lon = np.mean([coord[1] for coord in coordinates])
            
            # Create map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=4,
                tiles='OpenStreetMap'
            )
            
            # Group by float ID for different colors
            float_groups = {}
            for i, float_id in enumerate(float_ids):
                if float_id not in float_groups:
                    float_groups[float_id] = []
                float_groups[float_id].append({
                    'coords': coordinates[i],
                    'date': dates[i],
                    'index': i
                })
            
            # Color palette for different floats
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'darkred', 
                     'lightred', 'beige', 'darkblue', 'darkgreen']
            
            # Add trajectories for each float
            for j, (float_id, points) in enumerate(float_groups.items()):
                color = colors[j % len(colors)]
                
                # Sort points by date
                points.sort(key=lambda x: x['date'] if x['date'] else '1900-01-01')
                
                # Create trajectory line
                trajectory_coords = [point['coords'] for point in points]
                
                folium.PolyLine(
                    locations=trajectory_coords,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Float {float_id} trajectory"
                ).add_to(m)
                
                # Add markers for profile locations
                for i, point in enumerate(points):
                    folium.CircleMarker(
                        location=point['coords'],
                        radius=5,
                        popup=f"Float {float_id}<br>"
                              f"Date: {point['date']}<br>"
                              f"Position: {point['coords'][0]:.3f}, {point['coords'][1]:.3f}",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(m)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; left: 50px; width: 150px; height: auto; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:14px; padding: 10px">
            <p><b>ARGO Float Trajectories</b></p>
            '''
            
            for j, float_id in enumerate(list(float_groups.keys())[:5]):  # Show first 5
                color = colors[j % len(colors)]
                legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> Float {float_id}</p>'
            
            legend_html += '</div>'
            m.get_root().html.add_child(folium.Element(legend_html))
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating trajectory map: {e}")
            raise
    
    def plot_depth_time_series(self, profiles: List[Dict], 
                              variable: str = 'temperature') -> go.Figure:
        """Create depth-time contour plot"""
        try:
            if not profiles:
                raise ValueError("No profile data provided")
            
            # Prepare data matrices
            times = []
            depths = []
            values = []
            
            for profile in profiles:
                date = profile.get('profile_date')
                pressure_data = profile.get('pressure_data', [])
                var_data = profile.get(f'{variable}_data', [])
                
                if date and pressure_data and var_data:
                    # Ensure same length
                    min_len = min(len(pressure_data), len(var_data))
                    
                    for i in range(min_len):
                        times.append(date)
                        depths.append(pressure_data[i])
                        values.append(var_data[i])
            
            if not values:
                raise ValueError(f"No {variable} data found")
            
            # Create DataFrame
            df = pd.DataFrame({
                'time': times,
                'depth': depths,
                'value': values
            })
            
            # Create pivot table for contour plot
            df_pivot = df.pivot_table(
                index='depth', 
                columns='time', 
                values='value', 
                aggfunc='mean'
            )
            
            # Create contour plot
            fig = go.Figure(data=go.Contour(
                z=df_pivot.values,
                x=df_pivot.columns,
                y=df_pivot.index,
                colorscale=self.color_schemes.get(variable, 'viridis'),
                colorbar=dict(
                    title=f"{variable.title()} ({'°C' if variable == 'temperature' else 'PSU' if variable == 'salinity' else 'dbar'})"
                )
            ))
            
            fig.update_layout(
                title=f"{variable.title()} Depth-Time Series",
                xaxis_title="Time",
                yaxis_title="Depth (dbar)",
                yaxis=dict(autorange='reversed'),
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating depth-time series: {e}")
            raise
    
    def plot_profile_comparison(self, profiles: List[Dict], 
                               variable: str = 'temperature') -> go.Figure:
        """Compare multiple profiles"""
        try:
            if len(profiles) < 2:
                raise ValueError("At least 2 profiles required for comparison")
            
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            
            for i, profile in enumerate(profiles[:10]):  # Limit to 10 profiles
                pressure_data = profile.get('pressure_data', [])
                var_data = profile.get(f'{variable}_data', [])
                
                if pressure_data and var_data:
                    min_len = min(len(pressure_data), len(var_data))
                    
                    fig.add_trace(go.Scatter(
                        x=var_data[:min_len],
                        y=pressure_data[:min_len],
                        mode='lines',
                        name=f"Float {profile.get('float_id', 'Unknown')} "
                             f"Cycle {profile.get('cycle_number', 'N/A')}",
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            
            fig.update_layout(
                title=f"{variable.title()} Profile Comparison",
                xaxis_title=f"{variable.title()} ({'°C' if variable == 'temperature' else 'PSU' if variable == 'salinity' else 'dbar'})",
                yaxis_title="Pressure (dbar)",
                yaxis=dict(autorange='reversed'),
                height=600,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating profile comparison: {e}")
            raise
    
    def plot_temperature_salinity_diagram(self, profiles: List[Dict]) -> go.Figure:
        """Create T-S diagram"""
        try:
            if not profiles:
                raise ValueError("No profile data provided")
            
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            
            for i, profile in enumerate(profiles[:5]):  # Limit to 5 profiles
                temp_data = profile.get('temperature_data', [])
                sal_data = profile.get('salinity_data', [])
                pressure_data = profile.get('pressure_data', [])
                
                if temp_data and sal_data:
                    min_len = min(len(temp_data), len(sal_data), len(pressure_data))
                    
                    fig.add_trace(go.Scatter(
                        x=sal_data[:min_len],
                        y=temp_data[:min_len],
                        mode='markers+lines',
                        name=f"Float {profile.get('float_id', 'Unknown')} "
                             f"Cycle {profile.get('cycle_number', 'N/A')}",
                        marker=dict(
                            color=pressure_data[:min_len],
                            colorscale='viridis',
                            colorbar=dict(title="Pressure (dbar)"),
                            size=6
                        ),
                        line=dict(color=colors[i % len(colors)], width=1)
                    ))
            
            fig.update_layout(
                title="Temperature-Salinity Diagram",
                xaxis_title="Salinity (PSU)",
                yaxis_title="Temperature (°C)",
                height=600,
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating T-S diagram: {e}")
            raise
    
    def create_statistics_dashboard(self, profiles: List[Dict]) -> Dict[str, go.Figure]:
        """Create dashboard with multiple statistical plots"""
        try:
            if not profiles:
                raise ValueError("No profile data provided")
            
            dashboard = {}
            
            # 1. Geographic distribution
            lats = [p.get('latitude') for p in profiles if p.get('latitude')]
            lons = [p.get('longitude') for p in profiles if p.get('longitude')]
            
            if lats and lons:
                geo_fig = go.Figure(data=go.Scatter(
                    x=lons,
                    y=lats,
                    mode='markers',
                    marker=dict(size=8, opacity=0.6),
                    text=[f"Float {p.get('float_id', 'Unknown')}" for p in profiles],
                    hovertemplate="<b>%{text}</b><br>Lat: %{y:.2f}<br>Lon: %{x:.2f}<extra></extra>"
                ))
                
                geo_fig.update_layout(
                    title="Geographic Distribution of Profiles",
                    xaxis_title="Longitude",
                    yaxis_title="Latitude",
                    height=400
                )
                dashboard['geographic'] = geo_fig
            
            # 2. Temporal distribution
            dates = [p.get('profile_date') for p in profiles if p.get('profile_date')]
            if dates:
                temporal_fig = go.Figure(data=go.Histogram(
                    x=dates,
                    nbinsx=20,
                    marker_color='lightblue'
                ))
                
                temporal_fig.update_layout(
                    title="Temporal Distribution of Profiles",
                    xaxis_title="Date",
                    yaxis_title="Number of Profiles",
                    height=400
                )
                dashboard['temporal'] = temporal_fig
            
            # 3. Temperature distribution
            all_temps = []
            for profile in profiles:
                temp_data = profile.get('temperature_data', [])
                all_temps.extend(temp_data)
            
            if all_temps:
                temp_fig = go.Figure(data=go.Histogram(
                    x=all_temps,
                    nbinsx=30,
                    marker_color='red',
                    opacity=0.7
                ))
                
                temp_fig.update_layout(
                    title="Temperature Distribution",
                    xaxis_title="Temperature (°C)",
                    yaxis_title="Frequency",
                    height=400
                )
                dashboard['temperature_dist'] = temp_fig
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating statistics dashboard: {e}")
            return {}