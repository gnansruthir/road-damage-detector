import folium
from folium.plugins import HeatMap

# Default map view only; findings are added only with real coordinates.
CITY_CENTER = [28.6139, 77.2090]

class CivicMapGenerator:
    def __init__(self, center_lat=28.6139, center_lng=77.2090):
        self.center = [center_lat, center_lng]
        self.damages = []

    def add_damage_point(self, severity, class_name, priority, latitude, longitude):
        """Adds a finding at coordinates extracted from the source image."""
        weight = 1.0 if severity == "Critical" else 0.6 if severity == "Medium" else 0.3
        
        point = {
            "lat": latitude,
            "lng": longitude,
            "weight": weight,
            "class": class_name,
            "severity": severity,
            "priority": priority
        }
        self.damages.append(point)
        return latitude, longitude

    def generate_map_html(self, output_path=None):
        """
        Creates the Folium map with Dark Matter tiles and real GPS findings.
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
