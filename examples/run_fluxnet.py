import pandas as pd
import os
import subprocess
from pathlib import Path
import time
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import contextily as ctx # type: ignore

# Try to import optional libraries for enhanced visualizations
try:
    import cartopy.crs as ccrs # type: ignore
    import cartopy.feature as cfeature # type: ignore
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Cartopy not available. Map visualizations will be limited.")


# Köppen-Geiger climate classification colors
KG_COLORS = {
    'Af': '#0000FF',  # Tropical rainforest - Dark Blue
    'Am': '#0077FF',  # Tropical monsoon - Medium Blue
    'Aw': '#00FFFF',  # Tropical savanna - Light Blue
    'BWh': '#FF0000', # Hot desert - Red
    'BWk': '#FF5500', # Cold desert - Orange-Red
    'BSh': '#FFAA00', # Hot steppe - Orange
    'BSk': '#FFFF00', # Cold steppe - Yellow
    'Csa': '#66FF00', # Mediterranean hot summer - Light Green
    'Csb': '#33CC00', # Mediterranean warm summer - Medium Green
    'Csc': '#009900', # Mediterranean cold summer - Dark Green
    'Cwa': '#99FF66', # Humid subtropical hot summer - Light Yellow-Green
    'Cwb': '#66CC33', # Humid subtropical warm summer - Medium Yellow-Green
    'Cwc': '#339900', # Humid subtropical cold summer - Dark Yellow-Green
    'Cfa': '#00FF99', # Humid subtropical without dry season - Light Blue-Green
    'Cfb': '#009966', # Oceanic - Medium Blue-Green
    'Cfc': '#006633', # Subpolar oceanic - Dark Blue-Green
    'Dsa': '#FF99FF', # Hot-summer humid continental - Light Pink
    'Dsb': '#FF66CC', # Warm-summer humid continental - Medium Pink
    'Dsc': '#CC3399', # Cool-summer humid continental - Dark Pink
    'Dsd': '#990066', # Very cold humid continental - Very Dark Pink
    'Dwa': '#FFCCFF', # Monsoon-influenced hot-summer humid continental - Very Light Pink
    'Dwb': '#FF99CC', # Monsoon-influenced warm-summer humid continental - Light-Medium Pink
    'Dwc': '#CC6699', # Monsoon-influenced cool-summer humid continental - Medium-Dark Pink
    'Dwd': '#993366', # Monsoon-influenced very cold humid continental - Dark Pink
    'Dfa': '#CCFFCC', # Hot-summer humid continental without dry season - Very Light Green
    'Dfb': '#99CC99', # Warm-summer humid continental without dry season - Light Green
    'Dfc': '#669966', # Subarctic without dry season - Medium Green
    'Dfd': '#336633', # Very cold subarctic without dry season - Dark Green
    'ET': '#CCCCCC',  # Tundra - Light Gray
    'EF': '#999999',  # Ice cap - Medium Gray
}

# Holdridge life zones (HLZ) simplified colors
HLZ_COLORS = {
    'L1': '#006600',  # Tropical rainforest - Dark Green
    'L2': '#00cc00',  # Tropical wet forest - Medium Green
    'L3': '#66ff66',  # Tropical moist forest - Light Green
    'L4': '#ffcc00',  # Tropical dry forest - Gold
    'L5': '#ff9900',  # Tropical very dry forest - Orange
    'L6': '#ff6600',  # Tropical thorn woodland - Dark Orange
    'L7': '#ff0000',  # Tropical desert - Red
    'L8': '#cc00ff',  # Subtropical rainforest - Purple
    'L9': '#9900cc',  # Subtropical wet forest - Dark Purple
    'L10': '#cc99ff', # Subtropical moist forest - Light Purple
    'L11': '#ffff00', # Subtropical dry forest - Yellow
    'L12': '#cc9900', # Subtropical thorn woodland - Brown
    'L13': '#996600', # Subtropical desert - Dark Brown
    'L14': '#0099ff', # Warm temperate rainforest - Medium Blue
    'L15': '#3366ff', # Warm temperate wet forest - Darker Blue
    'L16': '#99ccff', # Warm temperate moist forest - Light Blue
    'L17': '#cccc00', # Warm temperate dry forest - Olive
    'L18': '#999900', # Warm temperate thorn steppe - Dark Olive
    'L19': '#666600', # Warm temperate desert - Very Dark Olive
    'L20': '#00ffff', # Cool temperate rainforest - Cyan
    'L21': '#00cccc', # Cool temperate wet forest - Dark Cyan
    'L22': '#66ffff', # Cool temperate moist forest - Light Cyan
    'L23': '#999966', # Cool temperate steppe - Greige
    'L24': '#666633', # Cool temperate desert - Dark Greige
    'L25': '#ccffff', # Boreal rainforest - Very Light Blue
    'L26': '#99cccc', # Boreal wet forest - Grayish Blue
    'L27': '#669999', # Boreal moist forest - Dark Grayish Blue
    'L28': '#666666', # Boreal dry forest - Gray
    'L29': '#cccccc', # Polar desert - Light Gray
    'L30': '#ffffff', # Ice - White
}

# Land cover color scheme
LC_COLORS = {
    'Forest': '#006400',        # Dark green
    'Grassland': '#ADFF2F',     # Green-yellow
    'Cropland': '#FFD700',      # Gold
    'Wetland': '#00FFFF',       # Cyan
    'Urban': '#808080',         # Gray
    'Barren': '#A0522D',        # Brown
    'Snow/Ice': '#FFFFFF',      # White
    'Water': '#0000FF',         # Blue
    'Shrubland': '#9ACD32',     # Yellow-green
    'Savanna': '#8B4513',       # Saddle brown
    'Mixed': '#7CFC00',         # Lawn green
}

def determine_em_earth_region(latitude, longitude):
    """
    Determine the EM-Earth region based on latitude and longitude coordinates.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
    
    Returns:
        str: Region name matching EM-Earth directory structure
        
    Available regions: Africa, Asia, Europe, NorthAmerica, Oceania, SouthAmerica
    """
    lat = float(latitude)
    lon = float(longitude)
    
    # Define regional boundaries (approximate)
    # These boundaries are designed to match typical meteorological data coverage
    # and may overlap in some areas - priority is given to the most likely region
    
    # North America: Generally western hemisphere, north of ~10°N
    if -180 <= lon <= -30 and 10 <= lat <= 90:
        return "NorthAmerica"
    
    # Central America and Caribbean are often included with North America
    if -120 <= lon <= -60 and 5 <= lat <= 35:
        return "NorthAmerica"
    
    # South America: Western hemisphere, generally south of North America
    if -90 <= lon <= -30 and -60 <= lat <= 15:
        return "SouthAmerica"
    
    # Europe: Roughly -10°W to 40°E, 35°N to 75°N
    if -15 <= lon <= 45 and 35 <= lat <= 75:
        return "Europe"
    
    # Africa: Roughly -20°W to 55°E, -35°S to 40°N
    if -25 <= lon <= 60 and -40 <= lat <= 40:
        # Exclude Europe overlap region
        if not (-15 <= lon <= 45 and 35 <= lat <= 75):
            return "Africa"
    
    # Oceania: Pacific region including Australia, New Zealand, Pacific Islands
    # Australia and New Zealand
    if 110 <= lon <= 180 and -50 <= lat <= -10:
        return "Oceania"
    # Pacific Islands
    if 130 <= lon <= -130 and -30 <= lat <= 30:
        return "Oceania"
    # Handle longitude wraparound for Pacific
    if lon >= 130 or lon <= -130:
        if -30 <= lat <= 30:
            return "Oceania"
    
    # Asia: Large region covering the rest of Eurasia
    # Main Asian landmass
    if 40 <= lon <= 180 and 5 <= lat <= 80:
        return "Asia"
    # Eastern Russia and far east
    if 100 <= lon <= 180 and 40 <= lat <= 75:
        return "Asia"
    # Middle East (transition zone, but often grouped with Asia)
    if 35 <= lon <= 65 and 10 <= lat <= 45:
        return "Asia"
    # Indian subcontinent and Southeast Asia
    if 65 <= lon <= 140 and -10 <= lat <= 40:
        return "Asia"
    
    # Default fallback logic based on hemisphere and latitude
    if lon < -30:  # Western hemisphere
        if lat > 10:
            return "NorthAmerica"
        else:
            return "SouthAmerica"
    else:  # Eastern hemisphere
        if lat > 35:
            if lon < 45:
                return "Europe"
            else:
                return "Asia"
        elif lat > -35:
            if lon > 110:
                return "Oceania"
            else:
                return "Africa"
        else:
            return "Oceania"

def parse_pour_point_coords(pour_point_str):
    """
    Parse pour point coordinates from string format.
    
    Args:
        pour_point_str: String in format "latitude/longitude"
    
    Returns:
        tuple: (latitude, longitude) as floats
    """
    try:
        parts = pour_point_str.split('/')
        if len(parts) == 2:
            lat = float(parts[0])
            lon = float(parts[1])
            return lat, lon
        else:
            raise ValueError(f"Invalid pour point format: {pour_point_str}")
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Error parsing pour point coordinates '{pour_point_str}': {e}")

def generate_config_file(template_path, output_path, domain_name, pour_point, bounding_box):
    """
    Generate a new config file based on the template with updated parameters
    
    Args:
        template_path: Path to the template config file
        output_path: Path to save the new config file
        domain_name: Name of the domain to set
        pour_point: Pour point coordinates to set
        bounding_box: Bounding box coordinates to set
    """
    # Read the template config file
    with open(template_path, 'r') as f:
        config_content = f.read()
    
    # Parse pour point coordinates to determine region
    try:
        lat, lon = parse_pour_point_coords(pour_point)
        em_earth_region = determine_em_earth_region(lat, lon)
        print(f"Determined EM-Earth region for {domain_name}: {em_earth_region} (lat: {lat}, lon: {lon})")
    except ValueError as e:
        print(f"Warning: Could not parse pour point coordinates for {domain_name}: {e}")
        print("Defaulting to Europe region")
        em_earth_region = "Europe"
    
    # Update the domain name using regex
    config_content = re.sub(r'DOMAIN_NAME:.*', f'DOMAIN_NAME: "{domain_name}"', config_content)
    
    # Update the pour point coordinates using regex
    config_content = re.sub(r'POUR_POINT_COORDS:.*', f'POUR_POINT_COORDS: {pour_point}', config_content)
    
    # Update the bounding box coordinates using regex
    config_content = re.sub(r'BOUNDING_BOX_COORDS:.*', f'BOUNDING_BOX_COORDS: {bounding_box}', config_content)
    
    # Update CONFLUENCE_DATA_DIR for ISMN data
    config_content = re.sub(r'CONFLUENCE_DATA_DIR:.*', f'CONFLUENCE_DATA_DIR: "/anvil/projects/x-ees240082/data/CONFLUENCE_data/fluxnet"', config_content)

    # Update EM-Earth paths based on determined region
    em_earth_base_path = "/anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly"
    config_content = re.sub(r'EM_EARTH_PRCP_DIR:.*', f'EM_EARTH_PRCP_DIR: {em_earth_base_path}/prcp/{em_earth_region}', config_content)
    config_content = re.sub(r'EM_EARTH_TMEAN_DIR:.*', f'EM_EARTH_TMEAN_DIR: {em_earth_base_path}/tmean/{em_earth_region}', config_content)
    config_content = re.sub(r'EM_EARTH_REGION:.*', f'EM_EARTH_REGION: {em_earth_region}', config_content)

    # Save the new config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    # Verify the changes were made
    print(f"Config file created at {output_path}")
    print(f"Checking for proper updates...")
    
    with open(output_path, 'r') as f:
        new_content = f.read()
    
    # Verify domain name was updated
    domain_pattern = re.compile(r'DOMAIN_NAME:.*')
    domain_match = domain_pattern.search(new_content)
    if domain_match:
        print(f"Domain name setting: {domain_match.group().strip()}")
    else:
        print("Warning: Domain name not found in config!")
    
    # Verify pour point was updated
    pour_point_pattern = re.compile(r'POUR_POINT_COORDS:.*')
    pour_point_match = pour_point_pattern.search(new_content)
    if pour_point_match:
        print(f"Pour point setting: {pour_point_match.group().strip()}")
    else:
        print("Warning: Pour point not found in config!")
    
    # Verify bounding box was updated
    bbox_pattern = re.compile(r'BOUNDING_BOX_COORDS:.*')
    bbox_match = bbox_pattern.search(new_content)
    if bbox_match:
        print(f"Bounding box setting: {bbox_match.group().strip()}")
    else:
        print("Warning: Bounding box not found in config!")
    
    # Verify EM-Earth region was updated
    em_earth_pattern = re.compile(r'EM_EARTH_REGION:.*')
    em_earth_match = em_earth_pattern.search(new_content)
    if em_earth_match:
        print(f"EM-Earth region setting: {em_earth_match.group().strip()}")
    else:
        print("Warning: EM-Earth region not found in config!")
    
    return output_path

def run_confluence(config_path, watershed_name):
    """
    Run CONFLUENCE with the specified config file
    
    Args:
        config_path: Path to the config file
        watershed_id: Name of the watershed for job naming
    """
    # Create a temporary batch script for this specific run
    batch_script = f"run_{watershed_name}.sh"
    
    with open(batch_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={watershed_name}
#SBATCH --output=CONFLUENCE_{watershed_name}_%j.log
#SBATCH --error=CONFLUENCE_{watershed_name}_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

# Load necessary modules
module restore confluence_modules

# Activate Python environment

conda activate confluence

# Run CONFLUENCE with the specified config
python ../CONFLUENCE/CONFLUENCE.py --config {config_path}

echo "CONFLUENCE job for {watershed_name} complete"
""")
    
    # Make the script executable
    os.chmod(batch_script, 0o755)
    
    # Submit the job
    result = subprocess.run(['sbatch', batch_script], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job for {watershed_name}, job ID: {job_id}")
        return job_id
    else:
        print(f"Failed to submit job for {watershed_name}: {result.stderr}")
        return None

class WatershedVisualizer:
    """Class to handle watershed visualization tasks"""
    
    def __init__(self, watershed_df, output_dir="visualizations"):
        """
        Initialize the visualizer with watershed data
        
        Args:
            watershed_df: Pandas DataFrame containing watershed data
            output_dir: Directory to save visualization outputs
        """
        self.watersheds = watershed_df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract coordinates from the dataframe
        self.pour_points = []
        self.bboxes = []
        
        for _, row in self.watersheds.iterrows():
            # Parse pour point (latitude/longitude)
            try:
                lat, lon = row['POUR_POINT_COORDS'].split('/')
                self.pour_points.append((float(lat), float(lon)))
            except (ValueError, AttributeError):
                self.pour_points.append((np.nan, np.nan))
            
            # Parse bounding box (lat_max/lon_min/lat_min/lon_max)
            try:
                parts = row['BOUNDING_BOX_COORDS'].split('/')
                if len(parts) == 4:
                    self.bboxes.append({
                        'lat_max': float(parts[0]),
                        'lon_min': float(parts[1]),
                        'lat_min': float(parts[2]),
                        'lon_max': float(parts[3])
                    })
                else:
                    self.bboxes.append(None)
            except (ValueError, AttributeError):
                self.bboxes.append(None)
        
        print(f"Visualizer initialized with {len(self.watersheds)} watersheds")
        
    def plot_global_distribution(self):
        """
        Create a global map showing the distribution of watersheds
        """
        print("Plotting global watershed distribution...")
        
        if HAS_CARTOPY:
            # Create a global map with Cartopy
            plt.figure(figsize=(15, 10))
            ax = plt.axes(projection=ccrs.Robinson())
            
            # Add map features
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
            
            # Plot pour points
            lats = [pp[0] for pp in self.pour_points]
            lons = [pp[1] for pp in self.pour_points]
            
            # Plot the pour points as red dots
            ax.scatter(lons, lats, transform=ccrs.PlateCarree(), 
                      s=30, color='red', alpha=0.7, 
                      edgecolors='black', linewidth=0.5)
            
            # Plot bounding boxes if available
            for bbox in self.bboxes:
                if bbox:
                    lon_min = bbox['lon_min']
                    lon_max = bbox['lon_max']
                    lat_min = bbox['lat_min']
                    lat_max = bbox['lat_max']
                    
                    # Create a rectangle for the bounding box
                    # Note: This works for most cases but there are edge cases with the date line
                    box = mpatches.Rectangle(
                        xy=[lon_min, lat_min],
                        width=(lon_max - lon_min),
                        height=(lat_max - lat_min),
                        facecolor='none',
                        edgecolor='blue',
                        alpha=0.5,
                        linewidth=0.5,
                        transform=ccrs.PlateCarree()
                    )
                    ax.add_patch(box)
            
            # Add a title and gridlines
            plt.title('Global Distribution of Selected Watersheds', fontsize=16)
            ax.gridlines(linestyle='--', alpha=0.5)
            
            # Save the figure
            plt.savefig(os.path.join(self.output_dir, 'global_watershed_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Global distribution map saved to {self.output_dir}/global_watershed_distribution.png")
            
        else:
            # Fallback to basic matplotlib
            plt.figure(figsize=(12, 6))
            plt.scatter([pp[1] for pp in self.pour_points], 
                       [pp[0] for pp in self.pour_points], 
                       c='red', alpha=0.7, s=20)
            
            plt.title('Global Distribution of Selected Watersheds')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Add simple world map outline if available
            try:
                # This is a simplified approach without cartopy

                ctx.add_basemap(plt.gca(), zoom=1, source=ctx.providers.OpenStreetMap.Mapnik)
            except ImportError:
                print("contextily not available for adding basemap")
            
            plt.savefig(os.path.join(self.output_dir, 'global_watershed_distribution_simple.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Basic global distribution map saved to {self.output_dir}/global_watershed_distribution_simple.png")
    
    def plot_climate_distribution(self):
        """
        Plot the distribution of Köppen-Geiger climate classifications
        """
        # Count the frequency of each climate type
        if 'KG' in self.watersheds.columns:
            kg_counts = self.watersheds['KG'].value_counts()
            
            plt.figure(figsize=(12, 8))
            
            # Use climate colors where available
            colors = [KG_COLORS.get(climate, '#AAAAAA') for climate in kg_counts.index]
            
            # Create the bar chart
            bars = plt.bar(kg_counts.index, kg_counts.values, color=colors)
            
            plt.title('Distribution of Köppen-Geiger Climate Classifications', fontsize=16)
            plt.xlabel('Climate Classification', fontsize=12)
            plt.ylabel('Number of Watersheds', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'climate_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Climate distribution chart saved to {self.output_dir}/climate_distribution.png")
            
            # Now create a map with climate colors if cartopy is available
            if HAS_CARTOPY:
                plt.figure(figsize=(15, 10))
                ax = plt.axes(projection=ccrs.Robinson())
                
                # Add map features
                ax.add_feature(cfeature.LAND, facecolor='lightgray')
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                
                # Plot pour points colored by climate type
                for i, (lat, lon) in enumerate(self.pour_points):
                    climate = self.watersheds.iloc[i]['KG']
                    color = KG_COLORS.get(climate, '#AAAAAA')
                    
                    ax.scatter(lon, lat, transform=ccrs.PlateCarree(),
                              s=50, color=color, alpha=0.8,
                              edgecolors='black', linewidth=0.5)
                
                # Create a legend
                # Group the climates by main type (first letter)
                climate_groups = {}
                for climate in kg_counts.index:
                    main_type = climate[0]
                    if main_type not in climate_groups:
                        climate_groups[main_type] = []
                    climate_groups[main_type].append(climate)
                
                # Create legend elements
                legend_elements = []
                for main_type, climates in sorted(climate_groups.items()):
                    # Choose a representative color for each main type
                    rep_climate = climates[0]
                    color = KG_COLORS.get(rep_climate, '#AAAAAA')
                    
                    # Map main types to descriptions
                    type_descriptions = {
                        'A': 'Tropical',
                        'B': 'Arid',
                        'C': 'Temperate',
                        'D': 'Continental',
                        'E': 'Polar'
                    }
                    
                    description = type_descriptions.get(main_type, main_type)
                    
                    # Add the legend element
                    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                 markerfacecolor=color, markersize=10,
                                                 label=f'{description} ({main_type})'))
                
                # Add the legend to the plot
                ax.legend(handles=legend_elements, title='Climate Types', 
                         loc='lower left', fontsize=10)
                
                plt.title('Global Distribution of Watersheds by Climate Type', fontsize=16)
                ax.gridlines(linestyle='--', alpha=0.5)
                
                plt.savefig(os.path.join(self.output_dir, 'climate_map.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Climate map saved to {self.output_dir}/climate_map.png")
        else:
            print("KG (Köppen-Geiger) climate classification not found in dataset")
    
    def plot_area_distribution(self):
        """
        Plot the distribution of watershed areas
        """
        if 'Area_km2' in self.watersheds.columns:
            plt.figure(figsize=(12, 8))
            
            # Create histogram with logarithmic scale for area
            plt.hist(self.watersheds['Area_km2'], bins=20, color='skyblue', 
                    edgecolor='black', alpha=0.7)
            
            plt.title('Distribution of Watershed Areas', fontsize=16)
            plt.xlabel('Area (km²)', fontsize=12)
            plt.ylabel('Number of Watersheds', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Convert to log scale if the range is very large
            if self.watersheds['Area_km2'].max() > self.watersheds['Area_km2'].min() * 100:
                plt.xscale('log')
                plt.xlabel('Area (km²) - Log Scale', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'area_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Area distribution chart saved to {self.output_dir}/area_distribution.png")
            
            # Create a map with points sized by area if cartopy is available
            if HAS_CARTOPY:
                plt.figure(figsize=(15, 10))
                ax = plt.axes(projection=ccrs.Robinson())
                
                # Add map features
                ax.add_feature(cfeature.LAND, facecolor='lightgray')
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                
                # Normalize point sizes based on area
                areas = self.watersheds['Area_km2'].values
                # Scale between 10 and 200 square points
                min_size = 10
                max_size = 200
                if areas.max() > areas.min():
                    size_scaled = min_size + (max_size - min_size) * (
                        (areas - areas.min()) / (areas.max() - areas.min())
                    )
                else:
                    size_scaled = np.full_like(areas, min_size)
                
                # Plot pour points with size proportional to area
                for i, (lat, lon) in enumerate(self.pour_points):
                    ax.scatter(lon, lat, transform=ccrs.PlateCarree(),
                              s=size_scaled[i], color='blue', alpha=0.6,
                              edgecolors='black', linewidth=0.5)
                
                # Create size legend
                handles = []
                labels = []
                
                # Create a few representative sizes for the legend
                area_breaks = [areas.min(), 
                              areas.min() + (areas.max() - areas.min()) * 0.33,
                              areas.min() + (areas.max() - areas.min()) * 0.67, 
                              areas.max()]
                
                for area in area_breaks:
                    if areas.max() > areas.min():
                        size = min_size + (max_size - min_size) * (
                            (area - areas.min()) / (areas.max() - areas.min())
                        )
                    else:
                        size = min_size
                    
                    handles.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='blue', markersize=np.sqrt(size/np.pi)))
                    labels.append(f'{area:.0f} km²')
                
                # Add the legend
                ax.legend(handles=handles, labels=labels, 
                         title='Watershed Area', loc='lower left',
                         fontsize=10)
                
                plt.title('Global Distribution of Watersheds by Area', fontsize=16)
                ax.gridlines(linestyle='--', alpha=0.5)
                
                plt.savefig(os.path.join(self.output_dir, 'area_map.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Area distribution map saved to {self.output_dir}/area_map.png")
        else:
            print("Area_km2 not found in dataset")
    
    def plot_landcover_distribution(self):
        """
        Plot the distribution of dominant land cover types
        """
        if 'Dominant_LC' in self.watersheds.columns:
            lc_counts = self.watersheds['Dominant_LC'].value_counts()
            
            plt.figure(figsize=(12, 8))
            
            # Use land cover colors where available
            colors = [LC_COLORS.get(lc, '#AAAAAA') for lc in lc_counts.index]
            
            # Create the bar chart
            bars = plt.bar(lc_counts.index, lc_counts.values, color=colors)
            
            plt.title('Distribution of Dominant Land Cover Types', fontsize=16)
            plt.xlabel('Land Cover Type', fontsize=12)
            plt.ylabel('Number of Watersheds', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'landcover_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Land cover distribution chart saved to {self.output_dir}/landcover_distribution.png")
            
            # Create a land cover map if cartopy is available
            if HAS_CARTOPY:
                plt.figure(figsize=(15, 10))
                ax = plt.axes(projection=ccrs.Robinson())
                
                # Add map features
                ax.add_feature(cfeature.LAND, facecolor='lightgray')
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                
                # Plot pour points colored by land cover type
                for i, (lat, lon) in enumerate(self.pour_points):
                    lc = self.watersheds.iloc[i]['Dominant_LC']
                    color = LC_COLORS.get(lc, '#AAAAAA')
                    
                    ax.scatter(lon, lat, transform=ccrs.PlateCarree(),
                              s=50, color=color, alpha=0.8,
                              edgecolors='black', linewidth=0.5)
                
                # Create a legend
                legend_elements = []
                for lc_type in lc_counts.index:
                    color = LC_COLORS.get(lc_type, '#AAAAAA')
                    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                 markerfacecolor=color, markersize=10,
                                                 label=lc_type))
                
                # Add the legend to the plot
                ax.legend(handles=legend_elements, title='Land Cover Types', 
                         loc='lower left', fontsize=10)
                
                plt.title('Global Distribution of Watersheds by Land Cover Type', fontsize=16)
                ax.gridlines(linestyle='--', alpha=0.5)
                
                plt.savefig(os.path.join(self.output_dir, 'landcover_map.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Land cover map saved to {self.output_dir}/landcover_map.png")
        else:
            print("Dominant_LC not found in dataset")
    
    def plot_human_footprint(self):
        """
        Plot the distribution of human footprint values
        """
        if 'HFP' in self.watersheds.columns:
            plt.figure(figsize=(12, 8))
            
            # Create histogram of human footprint values
            plt.hist(self.watersheds['HFP'], bins=20, color='purple', 
                    edgecolor='black', alpha=0.7)
            
            plt.title('Distribution of Human Footprint Values', fontsize=16)
            plt.xlabel('Human Footprint Index', fontsize=12)
            plt.ylabel('Number of Watersheds', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'hfp_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Human footprint distribution chart saved to {self.output_dir}/hfp_distribution.png")
            
            # Create a human footprint map if cartopy is available
            if HAS_CARTOPY:
                plt.figure(figsize=(15, 10))
                ax = plt.axes(projection=ccrs.Robinson())
                
                # Add map features
                ax.add_feature(cfeature.LAND, facecolor='lightgray')
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                
                # Define a colormap for human footprint
                cmap = plt.cm.RdYlGn_r  # Red (high impact) to Green (low impact)
                
                # Plot pour points colored by human footprint value
                hfp_values = self.watersheds['HFP'].values
                hfp_min, hfp_max = hfp_values.min(), hfp_values.max()
                
                # Normalize the HFP values for the colormap
                if hfp_max > hfp_min:
                    norm = plt.Normalize(hfp_min, hfp_max)
                    colors = cmap(norm(hfp_values))
                else:
                    colors = np.array([[0, 0.7, 0, 1]] * len(hfp_values))  # Green if all values are the same
                
                # Plot points with colors based on HFP
                scatter = ax.scatter([pp[1] for pp in self.pour_points],
                                    [pp[0] for pp in self.pour_points],
                                    transform=ccrs.PlateCarree(),
                                    s=50, c=hfp_values, cmap=cmap,
                                    edgecolors='black', linewidth=0.5)
                
                # Add a colorbar
                cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                                   pad=0.05, shrink=0.6)
                cbar.set_label('Human Footprint Index', fontsize=10)
                
                plt.title('Global Distribution of Watersheds by Human Footprint', fontsize=16)
                ax.gridlines(linestyle='--', alpha=0.5)
                
                plt.savefig(os.path.join(self.output_dir, 'hfp_map.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Human footprint map saved to {self.output_dir}/hfp_map.png")
        else:
            print("HFP (Human Footprint) not found in dataset")
    
    def plot_holdridge_zones(self):
        """
        Plot the distribution of Holdridge life zones
        """
        if 'HLZ' in self.watersheds.columns:
            hlz_counts = self.watersheds['HLZ'].value_counts()
            
            plt.figure(figsize=(12, 8))
            
            # Use Holdridge zone colors where available
            colors = [HLZ_COLORS.get(hlz, '#AAAAAA') for hlz in hlz_counts.index]
            
            # Create the bar chart
            bars = plt.bar(hlz_counts.index, hlz_counts.values, color=colors)
            
            plt.title('Distribution of Holdridge Life Zones', fontsize=16)
            plt.xlabel('Holdridge Life Zone', fontsize=12)
            plt.ylabel('Number of Watersheds', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'hlz_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Holdridge life zone distribution chart saved to {self.output_dir}/hlz_distribution.png")
            
            # Create a Holdridge life zone map if cartopy is available
            if HAS_CARTOPY:
                plt.figure(figsize=(15, 10))
                ax = plt.axes(projection=ccrs.Robinson())
                
                # Add map features
                ax.add_feature(cfeature.LAND, facecolor='lightgray')
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                
                # Plot pour points colored by Holdridge life zone
                for i, (lat, lon) in enumerate(self.pour_points):
                    hlz = self.watersheds.iloc[i]['HLZ']
                    color = HLZ_COLORS.get(hlz, '#AAAAAA')
                    
                    ax.scatter(lon, lat, transform=ccrs.PlateCarree(),
                              s=50, color=color, alpha=0.8,
                              edgecolors='black', linewidth=0.5)
                
                # Create a legend - group by similar zones if there are many
                if len(hlz_counts) > 10:
                    # Group by first digit/character if many zones
                    hlz_groups = {}
                    for hlz in hlz_counts.index:
                        group = hlz[0:2]  # Group by first two characters
                        if group not in hlz_groups:
                            hlz_groups[group] = []
                        hlz_groups[group].append(hlz)
                    
                    legend_elements = []
                    for group, zones in sorted(hlz_groups.items()):
                        # Use the first zone in the group for coloring
                        rep_zone = zones[0]
                        color = HLZ_COLORS.get(rep_zone, '#AAAAAA')
                        
                        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                     markerfacecolor=color, markersize=10,
                                                     label=f'{group}xx'))
                else:
                    # If not too many zones, show them all
                    legend_elements = []
                    for hlz in hlz_counts.index:
                        color = HLZ_COLORS.get(hlz, '#AAAAAA')
                        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                                    markerfacecolor=color, markersize=10,
                                                    label=hlz))
                
                # Add the legend to the plot
                ax.legend(handles=legend_elements, title='Holdridge Life Zones', 
                         loc='lower left', fontsize=10)
                
                plt.title('Global Distribution of Watersheds by Holdridge Life Zone', fontsize=16)
                ax.gridlines(linestyle='--', alpha=0.5)
                
                plt.savefig(os.path.join(self.output_dir, 'hlz_map.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Holdridge life zone map saved to {self.output_dir}/hlz_map.png")
        else:
            print("HLZ (Holdridge Life Zone) not found in dataset")
    
    def create_summary_report(self):
        """
        Create a summary report of the watershed dataset
        """
        print("Generating summary report...")
        
        # Create a summary text file
        report_path = os.path.join(self.output_dir, 'watershed_summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Global Watershed Dataset Summary Report\n")
            f.write("=====================================\n\n")
            
            # Basic statistics
            f.write(f"Total number of watersheds: {len(self.watersheds)}\n\n")
            
            # Area statistics if available
            if 'Area_km2' in self.watersheds.columns:
                f.write("Watershed Area Statistics:\n")
                f.write(f"  Minimum area: {self.watersheds['Area_km2'].min():.2f} km²\n")
                f.write(f"  Maximum area: {self.watersheds['Area_km2'].max():.2f} km²\n")
                f.write(f"  Mean area: {self.watersheds['Area_km2'].mean():.2f} km²\n")
                f.write(f"  Median area: {self.watersheds['Area_km2'].median():.2f} km²\n")
                f.write(f"  Total area: {self.watersheds['Area_km2'].sum():.2f} km²\n\n")
            
            # Climate distribution if available
            if 'KG' in self.watersheds.columns:
                f.write("Köppen-Geiger Climate Classification Distribution:\n")
                kg_counts = self.watersheds['KG'].value_counts()
                for climate, count in kg_counts.items():
                    f.write(f"  {climate}: {count} watersheds ({count/len(self.watersheds)*100:.1f}%)\n")
                f.write("\n")
            
            # Holdridge life zone distribution if available
            if 'HLZ' in self.watersheds.columns:
                f.write("Holdridge Life Zone Distribution:\n")
                hlz_counts = self.watersheds['HLZ'].value_counts()
                for hlz, count in hlz_counts.items():
                    f.write(f"  {hlz}: {count} watersheds ({count/len(self.watersheds)*100:.1f}%)\n")
                f.write("\n")
            
            # Land cover distribution if available
            if 'Dominant_LC' in self.watersheds.columns:
                f.write("Dominant Land Cover Distribution:\n")
                lc_counts = self.watersheds['Dominant_LC'].value_counts()
                for lc, count in lc_counts.items():
                    f.write(f"  {lc}: {count} watersheds ({count/len(self.watersheds)*100:.1f}%)\n")
                f.write("\n")
            
            # Human footprint statistics if available
            if 'HFP' in self.watersheds.columns:
                f.write("Human Footprint Statistics:\n")
                f.write(f"  Minimum HFP: {self.watersheds['HFP'].min()}\n")
                f.write(f"  Maximum HFP: {self.watersheds['HFP'].max()}\n")
                f.write(f"  Mean HFP: {self.watersheds['HFP'].mean():.2f}\n")
                f.write(f"  Median HFP: {self.watersheds['HFP'].median():.2f}\n\n")
                
                # Categorize HFP into Low, Medium, High
                hfp_categories = pd.cut(self.watersheds['HFP'], 
                                      bins=[0, 25, 50, 100], 
                                      labels=['Low', 'Medium', 'High'])
                hfp_cat_counts = hfp_categories.value_counts()
                
                f.write("Human Footprint Categories:\n")
                for cat, count in hfp_cat_counts.items():
                    f.write(f"  {cat}: {count} watersheds ({count/len(self.watersheds)*100:.1f}%)\n")
                f.write("\n")
            
            # Geographic distribution summary
            f.write("Geographic Distribution:\n")
            
            # Calculate hemisphere distribution
            north_count = sum(1 for lat, _ in self.pour_points if lat > 0)
            south_count = sum(1 for lat, _ in self.pour_points if lat < 0)
            equator_count = sum(1 for lat, _ in self.pour_points if lat == 0)
            
            f.write(f"  Northern Hemisphere: {north_count} watersheds ({north_count/len(self.watersheds)*100:.1f}%)\n")
            f.write(f"  Southern Hemisphere: {south_count} watersheds ({south_count/len(self.watersheds)*100:.1f}%)\n")
            if equator_count > 0:
                f.write(f"  On Equator: {equator_count} watersheds ({equator_count/len(self.watersheds)*100:.1f}%)\n")
            
            # Calculate continental distribution (approximate based on longitude/latitude)
            continents = {
                'North America': 0,
                'South America': 0,
                'Europe': 0,
                'Africa': 0,
                'Asia': 0,
                'Oceania': 0,
                'Antarctica': 0
            }
            
            for lat, lon in self.pour_points:
                # Very simplified continental assignment based on lat/lon
                # This is a rough approximation and will have errors
                if lat > 0 and -170 <= lon <= -30:
                    continents['North America'] += 1
                elif lat < 0 and -80 <= lon <= -35:
                    continents['South America'] += 1
                elif 36 <= lat <= 70 and -10 <= lon <= 40:
                    continents['Europe'] += 1
                elif -35 <= lat <= 37 and -18 <= lon <= 52:
                    continents['Africa'] += 1
                elif lat > 0 and ((lon > 40 and lon < 180) or (lon < -170)):
                    continents['Asia'] += 1
                elif lat < 0 and ((lon >= 110 and lon <= 180) or (lon >= -180 and lon <= -140)):
                    continents['Oceania'] += 1
                elif lat < -60:
                    continents['Antarctica'] += 1
                else:
                    # If unsure, assign based on closest continent centroid
                    # This is a fallback and not very accurate
                    if lat > 0:
                        if abs(lon) > 100:
                            continents['Asia'] += 1
                        else:
                            continents['Africa'] += 1
                    else:
                        continents['Oceania'] += 1
            
            f.write("\n  Continental Distribution (approximate):\n")
            for continent, count in continents.items():
                if count > 0:
                    f.write(f"    {continent}: {count} watersheds ({count/len(self.watersheds)*100:.1f}%)\n")
            
            f.write("\nNote: This report was generated automatically. Continental distribution is approximate based on latitude/longitude.\n")
        
        print(f"Summary report saved to {report_path}")
    
    def run_all_visualizations(self):
        """
        Run all visualization methods and create a comprehensive analysis
        """
        print("Running all watershed visualizations...")
        
        # Plot global distribution
        self.plot_global_distribution()
        
        # Plot climate distribution
        self.plot_climate_distribution()
        
        # Plot area distribution
        self.plot_area_distribution()
        
        # Plot land cover distribution
        self.plot_landcover_distribution()
        
        # Plot human footprint distribution
        self.plot_human_footprint()
        
        # Plot Holdridge life zones
        self.plot_holdridge_zones()
        
        # Create summary report
        self.create_summary_report()
        
        print("All visualizations completed!")
        print(f"Results saved to {self.output_dir}/")


def main():
    # Path to the CSV file
    csv_path = "fluxnet_towers.csv"
    
    # Path to the template config file
    template_config_path = "../CONFLUENCE/0_config_files/config_Bow_lumped.yaml"
    
    # Directory to store generated config files
    config_dir = "../CONFLUENCE/0_config_files/fluxnet"
    
    # Create the config directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    # Read the watershed data
    print(f"Reading watershed data from {csv_path}...")
    watersheds = pd.read_csv(csv_path)
    
    # Create visualizations directory
    vis_dir = "watershed_visualizations_snotel"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create and run the visualizer
    #print("Creating visualizations of the watershed dataset...")
    #visualizer = WatershedVisualizer(watersheds, output_dir=vis_dir)
    #visualizer.run_all_visualizations()
    
    # Process each watershed for CONFLUENCE runs
    submitted_jobs = []
    skipped_jobs = []
    
    # Ask if user wants to submit CONFLUENCE jobs
    submit_jobs = input("\nDo you want to submit CONFLUENCE jobs for these watersheds? (y/n): ").lower().strip()
    
    if submit_jobs == 'y':
        for _, watershed in watersheds.iterrows():
            # Get watershed parameters
            watershed_id = watershed['ID']
            watershed_name = watershed['Watershed_Name']
            pour_point = watershed['POUR_POINT_COORDS']
            bounding_box = watershed['BOUNDING_BOX_COORDS']
            
            # Create a unique domain name (using ID and name)
            domain_name = f"{watershed_name}"
            
            # Check if the simulations directory already exists
            simulation_dir = Path(f"/anvil/projects/x-ees240082/data//CONFLUENCE_data/fluxnet/domain_{domain_name}")#/simulations")
            
            if simulation_dir.exists():
                print(f"Skipping {domain_name} - simulation directory already exists: {simulation_dir}")
                skipped_jobs.append(domain_name)
                continue
            
            # Generate the config file path
            config_path = os.path.join(config_dir, f"config_{domain_name}.yaml")
            
            # Generate the config file
            print(f"Generating config file for {domain_name}...")
            generate_config_file(template_config_path, config_path, domain_name, pour_point, bounding_box)
            
            # Run CONFLUENCE with the generated config
            print(f"Submitting CONFLUENCE job for {domain_name}...")
            job_id = run_confluence(config_path, domain_name)
            
            if job_id:
                submitted_jobs.append((domain_name, job_id))
            
            # Add a small delay between job submissions to avoid overwhelming the scheduler
            time.sleep(2)
        
        # Print summary of submitted jobs
        print("\nSubmitted jobs summary:")
        for domain_name, job_id in submitted_jobs:
            print(f"Domain: {domain_name}, Job ID: {job_id}")
        
        print(f"\nTotal jobs submitted: {len(submitted_jobs)}")
        print(f"Total jobs skipped: {len(skipped_jobs)}")
        
        if skipped_jobs:
            print("\nSkipped domains (simulations already exist):")
            for domain_name in skipped_jobs:
                print(f"- {domain_name}")
    else:
        print("\nNo CONFLUENCE jobs submitted. Visualizations are available in the output directory.")
        print(f"To run CONFLUENCE jobs later, use the config files generated in {config_dir}")

if __name__ == "__main__":
    main()