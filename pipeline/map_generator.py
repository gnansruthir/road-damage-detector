import folium
from folium.plugins import HeatMap
import random

# Center coordinates for simulated smart city network (New Delhi area)
CITY_CENTER = [28.6139, 77.2090]

class CivicMapGenerator:
    def __init__(self, center_lat=28.6139, center_lng=77.2090):
        self.center = [center_lat, center_lng]
        # Store active damage list: each item is [lat, lng, weight, class, priority]
        self.damages = self._generate_mock_damages()

    def _generate_mock_damages(self):
        """Generates pre-existing GPS-tagged damages for the heatmap visualization."""
        mock_list = []
        # Generate 15 points clustered near center
        for _ in range(15):
            lat = self.center[0] + random.uniform(-0.015, 0.015)
            lng = self.center[1] + random.uniform(-0.015, 0.015)
            severity = random.choice(["Critical", "Medium", "Small"])
            weight = 0.9 if severity == "Critical" else 0.5 if severity == "Medium" else 0.2
            mock_list.append({
                "lat": lat,
                "lng": lng,
                "weight": weight,
                "class": random.choice(["Pothole", "Longitudinal Crack", "Transverse Crack"]),
                "severity": severity,
                "priority": random.choice([3, 4, 5])
            })
        return mock_list

    def add_damage_point(self, severity, class_name, priority):
        """Adds a newly detected damage point near the center and returns its coords."""
        lat = self.center[0] + random.uniform(-0.005, 0.005)
        lng = self.center[1] + random.uniform(-0.005, 0.005)
        weight = 1.0 if severity == "Critical" else 0.6 if severity == "Medium" else 0.3
        
        point = {
            "lat": lat,
            "lng": lng,
            "weight": weight,
            "class": class_name,
            "severity": severity,
            "priority": priority
        }
        self.damages.append(point)
        return lat, lng

    def generate_map_html(self, output_path=None):
        """
        Creates the Folium map with Dark Matter tiles, a heatmap layer,
        and custom pulsing markers.
        """
        # Create map with CartoDB Dark Matter tiles for the "Command Room" aesthetic
        m = folium.Map(
            location=self.center,
            zoom_start=14,
            tiles="CartoDB dark_matter",
            control_scale=True
        )

        # 1. Add Heatmap Layer
        heat_data = [[d["lat"], d["lng"], d["weight"]] for d in self.damages]
        HeatMap(heat_data, radius=25, blur=15, min_opacity=0.4).add_to(m)

        # 2. Add Pins for each damage segment
        for d in self.damages:
            color = "#F5A623" # Amber (Critical/Medium fallback)
            if d["severity"] == "Critical":
                color = "#FF3B30" # Red
            elif d["severity"] == "Small":
                color = "#4CAF82" # Muted Green
                
            popup_html = f"""
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 12px; color: #333;">
                <b>Finding:</b> {d['class']}<br/>
                <b>Severity:</b> <span style="color: {color}; font-weight: bold;">{d['severity']}</span><br/>
                <b>Priority:</b> level {d['priority']}/5
            </div>
            """
            
            # Draw circles with solid outline and translucent fill
            folium.CircleMarker(
                location=[d["lat"], d["lng"]],
                radius=8,
                popup=folium.Popup(popup_html, max_width=200),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                weight=2
            ).add_to(m)

        if output_path:
            m.save(output_path)
            return output_path
            
        return m._repr_html_()
