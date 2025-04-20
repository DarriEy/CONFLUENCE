from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from rasterstats import zonal_stats
import rasterio
from osgeo import gdal

from scipy.stats import skew, kurtosis, circmean, circstd
from shapely.geometry import box
from rasterio.mask import mask

class attributeProcessor:
    """
    Simple attribute processor that calculates elevation, slope, and aspect 
    statistics from an existing DEM using GDAL.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize the attribute processor."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.logger.info(f'data dir: {self.data_dir}')
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.logger.info(f'domain name: {self.domain_name}')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Create or access directories
        self.dem_dir = self.project_dir / 'attributes' / 'elevation' / 'dem'
        self.slope_dir = self.project_dir / 'attributes' / 'elevation' / 'slope'
        self.aspect_dir = self.project_dir / 'attributes' / 'elevation' / 'aspect'
        
        # Create directories if they don't exist
        self.slope_dir.mkdir(parents=True, exist_ok=True)
        self.aspect_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the catchment shapefile
        self.catchment_path = self._get_catchment_path()
        
        # Initialize results dictionary
        self.results = {}
    
    def _get_catchment_path(self) -> Path:
        """Get the path to the catchment shapefile."""
        catchment_path = self.config.get('CATCHMENT_PATH')
        self.logger.info(f'catchment path: {catchment_path}')
        
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        self.logger.info(f'catchment name: {catchment_name}')
        

        if catchment_path == 'default':
            catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            catchment_path = Path(catchment_path)
        
        if catchment_name == 'default':
            # Find the catchment shapefile based on domain discretization
            discretization = self.config.get('DOMAIN_DISCRETIZATION')
            catchment_file = f"{self.domain_name}_HRUs_{discretization}.shp"
        else:
            catchment_file = catchment_name
        
        return catchment_path / catchment_file
    
    def find_dem_file(self) -> Path:
        """Find the DEM file in the elevation/dem directory."""
        dem_files = list(self.dem_dir.glob("*.tif"))
        
        if not dem_files:
            self.logger.error(f"No DEM files found in {self.dem_dir}")
            raise FileNotFoundError(f"No DEM files found in {self.dem_dir}")
        
        # Use the first found DEM file
        return dem_files[0]
    
    def generate_slope_and_aspect(self, dem_file: Path) -> Dict[str, Path]:
        """Generate slope and aspect rasters from the DEM using GDAL."""
        self.logger.info(f"Generating slope and aspect from DEM: {dem_file}")
        
        # Create output file paths
        slope_file = self.slope_dir / f"{dem_file.stem}_slope.tif"
        aspect_file = self.aspect_dir / f"{dem_file.stem}_aspect.tif"
        
        try:
            # Prepare the slope options
            slope_options = gdal.DEMProcessingOptions(
                computeEdges=True,
                slopeFormat='degree',
                alg='Horn'
            )
            
            # Generate slope
            self.logger.info("Calculating slope...")
            gdal.DEMProcessing(
                str(slope_file),
                str(dem_file),
                'slope',
                options=slope_options
            )
            
            # Prepare the aspect options
            aspect_options = gdal.DEMProcessingOptions(
                computeEdges=True,
                alg='Horn',
                zeroForFlat=True
            )
            
            # Generate aspect
            self.logger.info("Calculating aspect...")
            gdal.DEMProcessing(
                str(aspect_file),
                str(dem_file),
                'aspect',
                options=aspect_options
            )
            
            self.logger.info(f"Slope saved to: {slope_file}")
            self.logger.info(f"Aspect saved to: {aspect_file}")
            
            return {
                'dem': dem_file,
                'slope': slope_file,
                'aspect': aspect_file
            }
        
        except Exception as e:
            self.logger.error(f"Error generating slope and aspect: {str(e)}")
            raise
    
    def calculate_statistics(self, raster_file: Path, attribute_name: str) -> Dict[str, float]:
        """Calculate zonal statistics for a raster."""
        self.logger.info(f"Calculating statistics for {attribute_name} from {raster_file}")
        
        # Define statistics to calculate
        stats = ['min', 'mean', 'max', 'std']
        
        # Special handling for aspect (circular statistics)
        if attribute_name == 'aspect':
            def calc_circmean(x):
                """Calculate circular mean of angles in degrees."""
                if isinstance(x, np.ma.MaskedArray):
                    # Filter out masked values
                    x = x.compressed()
                if len(x) == 0:
                    return np.nan
                # Convert to radians, calculate circular mean, convert back to degrees
                rad = np.radians(x)
                result = np.degrees(circmean(rad))
                return result
            
            def calc_circstd(x):
                """Calculate circular standard deviation of angles in degrees."""
                if isinstance(x, np.ma.MaskedArray):
                    # Filter out masked values
                    x = x.compressed()
                if len(x) == 0:
                    return np.nan
                # Convert to radians, calculate circular std, convert back to degrees
                rad = np.radians(x)
                result = np.degrees(circstd(rad))
                return result
            
            # Add custom circular statistics
            stats = ['min', 'max']
            custom_stats = {
                'circmean': calc_circmean
            }
            
            try:
                from scipy.stats import circstd
                custom_stats['circstd'] = calc_circstd
            except ImportError:
                self.logger.warning("scipy.stats.circstd not available, skipping circular std")
            
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(raster_file), 
                stats=stats,
                add_stats=custom_stats, 
                all_touched=True
            )
        else:
            # Regular statistics for elevation and slope
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(raster_file), 
                stats=stats, 
                all_touched=True
            )
        
        # Format the results
        results = {}
        if zonal_out:
            for i, zonal_result in enumerate(zonal_out):
                # Use HRU ID as key if available, otherwise use index
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                if not is_lumped:
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    hru_id = catchment.iloc[i][hru_id_field]
                    key_prefix = f"HRU_{hru_id}_"
                else:
                    key_prefix = ""
                
                # Add each statistic to results
                for stat, value in zonal_result.items():
                    if value is not None:
                        results[f"{key_prefix}{attribute_name}_{stat}"] = value
        
        return results

    def process_attributes(self) -> pd.DataFrame:
        """
        Process  catchment attributes from  available data sources,
        creating an integrated dataset of hydrologically relevant basin characteristics.
        
        Returns:
            pd.DataFrame: Complete dataframe of catchment attributes
        """
        self.logger.info("Starting attribute processing")
        
        try:
            # Initialize results dictionary
            all_results = {}
            
            # Process base elevation attributes (DEM, slope, aspect)
            self.logger.info("Processing elevation attributes")
            elevation_results = self._process_elevation_attributes()
            all_results.update(elevation_results)
            
            # Process soil properties
            self.logger.info("Processing soil attributes")
            soil_results = self._process_soil_attributes()
            all_results.update(soil_results)
            
            # Process geological attributes
            self.logger.info("Processing geological attributes")
            geo_results = self._process_geological_attributes()
            all_results.update(geo_results)
            
            # Enhance geological attributes with derived hydraulic properties
            geo_enhanced = self.enhance_hydrogeological_attributes(geo_results)
            all_results.update(geo_enhanced)
            
            # Process land cover attributes
            self.logger.info("Processing land cover attributes")
            lc_results = self._process_landcover_attributes()
            all_results.update(lc_results)
            
            # Process vegetation attributes (LAI, forest height)
            self.logger.info("Processing vegetation attributes")
            veg_results = self._process_forest_height()
            all_results.update(veg_results)
            
            # Process composite land cover metrics
            self.logger.info("Processing composite land cover metrics")
            self.results = all_results  # Temporarily set results for composite metrics calculation
            composite_lc = self.calculate_composite_landcover_metrics()
            all_results.update(composite_lc)
            
            # Process climate attributes
            self.logger.info("Processing climate attributes")
            climate_results = self._process_climate_attributes()
            all_results.update(climate_results)
            
            # Process advanced seasonality metrics
            self.logger.info("Processing seasonality metrics")
            seasonality_results = self.calculate_seasonality_metrics()
            all_results.update(seasonality_results)
            
            # Process hydrological attributes
            self.logger.info("Processing hydrological attributes")
            hydro_results = self._process_hydrological_attributes()
            all_results.update(hydro_results)
            
            # Process enhanced river network analysis
            self.logger.info("Processing enhanced river network")
            river_results = self.enhance_river_network_analysis()
            all_results.update(river_results)
            
            # Process baseflow attributes
            self.logger.info("Processing baseflow attributes")
            baseflow_results = self.calculate_baseflow_attributes()
            all_results.update(baseflow_results)
            
            # Process streamflow signatures
            self.logger.info("Processing streamflow signatures")
            signature_results = self.calculate_streamflow_signatures()
            all_results.update(signature_results)
            
            # Process water balance
            self.logger.info("Processing water balance")
            wb_results = self.calculate_water_balance()
            all_results.update(wb_results)
            
            # Create final dataframe
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if is_lumped:
                # For lumped catchment, create a single row
                df = pd.DataFrame([all_results])
                df['basin_id'] = self.domain_name
                df = df.set_index('basin_id')
            else:
                # For distributed catchment, create multi-level DataFrame with HRUs
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                hru_ids = catchment[hru_id_field].values
                
                # Reorganize results by HRU
                hru_results = []
                for i, hru_id in enumerate(hru_ids):
                    result = {'hru_id': hru_id}
                    
                    for key, value in all_results.items():
                        if key.startswith(f"HRU_{hru_id}_"):
                            # Remove HRU prefix
                            clean_key = key.replace(f"HRU_{hru_id}_", "")
                            result[clean_key] = value
                        elif not any(k.startswith("HRU_") for k in all_results.keys()):
                            # If there are no HRU-specific results, use the same values for all HRUs
                            result[key] = value
                    
                    hru_results.append(result)
                    
                df = pd.DataFrame(hru_results)
                df['basin_id'] = self.domain_name
                df = df.set_index(['basin_id', 'hru_id'])
            
            # Save the  attributes to different formats
            output_dir = self.project_dir / 'attributes'
            
            # Use prefix to distinguish from basic attribute files
            csv_file = output_dir / f"{self.domain_name}_attributes.csv"
            df.to_csv(csv_file)
            
            # Save as Parquet
            try:
                parquet_file = output_dir / f"{self.domain_name}_attributes.parquet"
                df.to_parquet(parquet_file)
                self.logger.info(f" attributes saved as Parquet: {parquet_file}")
            except ImportError:
                self.logger.warning("pyarrow not installed, skipping Parquet output")
            
            # Save as pickle
            pickle_file = output_dir / f"{self.domain_name}_attributes.pkl"
            df.to_pickle(pickle_file)
            
            self.logger.info(f" attributes saved to {csv_file}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in attribute processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Try to return whatever we processed so far
            if 'all_results' in locals() and all_results:
                try:
                    return pd.DataFrame([all_results])
                except:
                    pass
            
            # Return empty DataFrame as fallback
            return pd.DataFrame()


    def _process_geological_attributes(self) -> Dict[str, Any]:
        """
        Process geological attributes including bedrock permeability, porosity, and lithology.
        Extracts data from GLHYMPS and other geological datasets.
        
        Returns:
            Dict[str, Any]: Dictionary of geological attributes
        """
        results = {}
        
        # Process GLHYMPS data for permeability and porosity
        glhymps_results = self._process_glhymps_data()
        results.update(glhymps_results)
        
        # Process additional lithology data if available
        litho_results = self._process_lithology_data()
        results.update(litho_results)
        
        return results

    def _process_glhymps_data(self) -> Dict[str, Any]:
        """
        Process GLHYMPS (Global Hydrogeology MaPS) data for permeability and porosity.
        Optimized for performance with spatial filtering and simplified geometries.
        
        Returns:
            Dict[str, Any]: Dictionary of hydrogeological attributes
        """
        results = {}
        
        # Define path to GLHYMPS data
        glhymps_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/glhymps/raw/glhymps.shp")
        
        # Check if GLHYMPS file exists
        if not glhymps_path.exists():
            self.logger.warning(f"GLHYMPS file not found: {glhymps_path}")
            return results
        
        # Create cache directory and define cache file
        cache_dir = self.project_dir / 'cache' / 'glhymps'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_glhymps_results.pickle"
        
        # Check if cached results exist
        if cache_file.exists():
            self.logger.info(f"Loading cached GLHYMPS results from {cache_file}")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached GLHYMPS results: {str(e)}")
        
        self.logger.info("Processing GLHYMPS data for hydrogeological attributes")
        
        try:
            # Get catchment bounding box first to filter GLHYMPS data
            catchment = gpd.read_file(self.catchment_path)
            catchment_bbox = catchment.total_bounds
            
            # Add buffer to bounding box (e.g., 1% of width/height)
            bbox_width = catchment_bbox[2] - catchment_bbox[0]
            bbox_height = catchment_bbox[3] - catchment_bbox[1]
            buffer_x = bbox_width * 0.01
            buffer_y = bbox_height * 0.01
            
            # Create expanded bbox
            expanded_bbox = (
                catchment_bbox[0] - buffer_x,
                catchment_bbox[1] - buffer_y,
                catchment_bbox[2] + buffer_x,
                catchment_bbox[3] + buffer_y
            )
            
            # Load only GLHYMPS data within the bounding box and only necessary columns
            self.logger.info("Loading GLHYMPS data within catchment bounding box")
            glhymps = gpd.read_file(
                str(glhymps_path), 
                bbox=expanded_bbox,
                # Only read necessary columns if they exist
                columns=['geometry', 'porosity', 'logK_Ice', 'Porosity', 'Permeabi_1']
            )
            
            if glhymps.empty:
                self.logger.warning("No GLHYMPS data within catchment bounding box")
                return results
            
            # Rename columns if they exist with shortened names
            if ('Porosity' in glhymps.columns) and ('Permeabi_1' in glhymps.columns):
                glhymps.rename(columns={'Porosity': 'porosity', 'Permeabi_1': 'logK_Ice'}, inplace=True)
            
            # Check for missing required columns
            required_columns = ['porosity', 'logK_Ice']
            missing_columns = [col for col in required_columns if col not in glhymps.columns]
            if missing_columns:
                self.logger.warning(f"Missing required columns in GLHYMPS: {', '.join(missing_columns)}")
                return results
            
            # Simplify geometries for faster processing
            self.logger.info("Simplifying geometries for faster processing")
            simplify_tolerance = 0.001  # Adjust based on your data units
            glhymps['geometry'] = glhymps.geometry.simplify(simplify_tolerance)
            catchment['geometry'] = catchment.geometry.simplify(simplify_tolerance)
            
            # Calculate area in equal area projection for weighting
            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
            glhymps['New_area_m2'] = glhymps.to_crs(equal_area_crs).area
            
            # Check if we're dealing with lumped or distributed catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
            
            if is_lumped:
                # For lumped catchment, intersect with catchment boundary
                try:
                    # Ensure CRS match
                    if glhymps.crs != catchment.crs:
                        glhymps = glhymps.to_crs(catchment.crs)
                    
                    self.logger.info("Intersecting GLHYMPS with catchment boundary")
                    # Intersect GLHYMPS with catchment
                    intersection = gpd.overlay(glhymps, catchment, how='intersection')
                    
                    if not intersection.empty:
                        # Convert to equal area for proper area calculation
                        intersection['New_area_m2'] = intersection.to_crs(equal_area_crs).area
                        total_area = intersection['New_area_m2'].sum()
                        
                        if total_area > 0:
                            # Calculate area-weighted averages for porosity and permeability
                            # Porosity
                            porosity_mean = (intersection['porosity'] * intersection['New_area_m2']).sum() / total_area
                            # Standard deviation calculation (area-weighted)
                            porosity_variance = ((intersection['New_area_m2'] * 
                                                (intersection['porosity'] - porosity_mean)**2).sum() / 
                                                total_area)
                            porosity_std = np.sqrt(porosity_variance)
                            
                            # Get min and max values
                            porosity_min = intersection['porosity'].min()
                            porosity_max = intersection['porosity'].max()
                            
                            # Add porosity results
                            results["geology.porosity_mean"] = porosity_mean
                            results["geology.porosity_std"] = porosity_std
                            results["geology.porosity_min"] = porosity_min
                            results["geology.porosity_max"] = porosity_max
                            
                            # Permeability (log10 values)
                            logk_mean = (intersection['logK_Ice'] * intersection['New_area_m2']).sum() / total_area
                            # Standard deviation calculation (area-weighted)
                            logk_variance = ((intersection['New_area_m2'] * 
                                            (intersection['logK_Ice'] - logk_mean)**2).sum() / 
                                            total_area)
                            logk_std = np.sqrt(logk_variance)
                            
                            # Get min and max values
                            logk_min = intersection['logK_Ice'].min()
                            logk_max = intersection['logK_Ice'].max()
                            
                            # Add permeability results
                            results["geology.log_permeability_mean"] = logk_mean
                            results["geology.log_permeability_std"] = logk_std
                            results["geology.log_permeability_min"] = logk_min
                            results["geology.log_permeability_max"] = logk_max
                            
                            # Calculate derived hydraulic properties
                            
                            # Convert log permeability to hydraulic conductivity
                            # K [m/s] = 10^(log_k) * (rho*g/mu) where rho*g/mu ≈ 10^7 for water
                            hyd_cond = 10**(logk_mean) * (10**7)  # m/s
                            results["geology.hydraulic_conductivity_m_per_s"] = hyd_cond
                            
                            # Calculate transmissivity assuming 100m aquifer thickness
                            # T = K * b where b is aquifer thickness
                            transmissivity = hyd_cond * 100  # m²/s
                            results["geology.transmissivity_m2_per_s"] = transmissivity
                    else:
                        self.logger.warning("No intersection between GLHYMPS and catchment boundary")
                except Exception as e:
                    self.logger.error(f"Error processing GLHYMPS for lumped catchment: {str(e)}")
            else:
                # For distributed catchment, process each HRU
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                # Ensure CRS match
                if glhymps.crs != catchment.crs:
                    glhymps = glhymps.to_crs(catchment.crs)
                
                # Create spatial index for GLHYMPS
                self.logger.info("Creating spatial index for faster intersections")
                import rtree
                spatial_index = rtree.index.Index()
                for idx, geom in enumerate(glhymps.geometry):
                    spatial_index.insert(idx, geom.bounds)
                
                # Process in chunks for better performance
                chunk_size = 10  # Adjust based on your data
                for i in range(0, len(catchment), chunk_size):
                    self.logger.info(f"Processing HRUs {i} to {min(i+chunk_size, len(catchment))-1}")
                    hru_chunk = catchment.iloc[i:min(i+chunk_size, len(catchment))]
                    
                    for j, hru in hru_chunk.iterrows():
                        try:
                            hru_id = hru[hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            # Create a GeoDataFrame with just this HRU
                            hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                            
                            # Use spatial index to filter candidates
                            hru_bounds = hru.geometry.bounds
                            potential_matches_idx = list(spatial_index.intersection(hru_bounds))
                            
                            if not potential_matches_idx:
                                self.logger.debug(f"No potential GLHYMPS matches for HRU {hru_id}")
                                continue
                            
                            # Subset GLHYMPS using potential matches
                            glhymps_subset = glhymps.iloc[potential_matches_idx]
                            
                            # Intersect with GLHYMPS
                            intersection = gpd.overlay(glhymps_subset, hru_gdf, how='intersection')
                            
                            if not intersection.empty:
                                # Convert to equal area for proper area calculation
                                intersection['New_area_m2'] = intersection.to_crs(equal_area_crs).area
                                total_area = intersection['New_area_m2'].sum()
                                
                                if total_area > 0:
                                    # Calculate area-weighted averages for porosity and permeability
                                    # Porosity
                                    porosity_mean = (intersection['porosity'] * intersection['New_area_m2']).sum() / total_area
                                    # Standard deviation calculation (area-weighted)
                                    porosity_variance = ((intersection['New_area_m2'] * 
                                                        (intersection['porosity'] - porosity_mean)**2).sum() / 
                                                        total_area)
                                    porosity_std = np.sqrt(porosity_variance)
                                    
                                    # Get min and max values
                                    porosity_min = intersection['porosity'].min()
                                    porosity_max = intersection['porosity'].max()
                                    
                                    # Add porosity results
                                    results[f"{prefix}geology.porosity_mean"] = porosity_mean
                                    results[f"{prefix}geology.porosity_std"] = porosity_std
                                    results[f"{prefix}geology.porosity_min"] = porosity_min
                                    results[f"{prefix}geology.porosity_max"] = porosity_max
                                    
                                    # Permeability (log10 values)
                                    logk_mean = (intersection['logK_Ice'] * intersection['New_area_m2']).sum() / total_area
                                    # Standard deviation calculation (area-weighted)
                                    logk_variance = ((intersection['New_area_m2'] * 
                                                    (intersection['logK_Ice'] - logk_mean)**2).sum() / 
                                                    total_area)
                                    logk_std = np.sqrt(logk_variance)
                                    
                                    # Get min and max values
                                    logk_min = intersection['logK_Ice'].min()
                                    logk_max = intersection['logK_Ice'].max()
                                    
                                    # Add permeability results
                                    results[f"{prefix}geology.log_permeability_mean"] = logk_mean
                                    results[f"{prefix}geology.log_permeability_std"] = logk_std
                                    results[f"{prefix}geology.log_permeability_min"] = logk_min
                                    results[f"{prefix}geology.log_permeability_max"] = logk_max
                                    
                                    # Calculate derived hydraulic properties
                                    
                                    # Convert log permeability to hydraulic conductivity
                                    # K [m/s] = 10^(log_k) * (rho*g/mu) where rho*g/mu ≈ 10^7 for water
                                    hyd_cond = 10**(logk_mean) * (10**7)  # m/s
                                    results[f"{prefix}geology.hydraulic_conductivity_m_per_s"] = hyd_cond
                                    
                                    # Calculate transmissivity assuming 100m aquifer thickness
                                    # T = K * b where b is aquifer thickness
                                    transmissivity = hyd_cond * 100  # m²/s
                                    results[f"{prefix}geology.transmissivity_m2_per_s"] = transmissivity
                            else:
                                self.logger.debug(f"No intersection between GLHYMPS and HRU {hru_id}")
                        except Exception as e:
                            self.logger.error(f"Error processing GLHYMPS for HRU {hru_id}: {str(e)}")
        
            # Cache results
            try:
                self.logger.info(f"Caching GLHYMPS results to {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except Exception as e:
                self.logger.warning(f"Error caching GLHYMPS results: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing GLHYMPS data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_lithology_data(self) -> Dict[str, Any]:
        """
        Process lithology classification data from geological maps.
        
        Returns:
            Dict[str, Any]: Dictionary of lithology attributes
        """
        results = {}
        
        # Define path to geological map data
        # This could be GMNA (Geological Map of North America) or similar dataset
        geo_map_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/geology/raw")
        
        # Check if geological map directory exists
        if not geo_map_path.exists():
            self.logger.warning(f"Geological map directory not found: {geo_map_path}")
            return results
        
        self.logger.info("Processing lithology data from geological maps")
        
        try:
            # Search for potential geological shapefiles
            shapefile_patterns = ["*geologic*.shp", "*lithology*.shp", "*bedrock*.shp", "*rock*.shp"]
            geo_files = []
            
            for pattern in shapefile_patterns:
                geo_files.extend(list(geo_map_path.glob(pattern)))
            
            if not geo_files:
                self.logger.warning(f"No geological map files found in {geo_map_path}")
                return results
            
            # Use the first matching file
            geo_file = geo_files[0]
            self.logger.info(f"Using geological map file: {geo_file}")
            
            # Load geological map shapefile
            geo_map = gpd.read_file(str(geo_file))
            
            # Identify potential lithology classification columns
            litho_columns = []
            for col in geo_map.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ['lith', 'rock', 'geol', 'form', 'unit']):
                    litho_columns.append(col)
            
            if not litho_columns:
                self.logger.warning(f"No lithology classification columns found in {geo_file}")
                return results
            
            # Use the first matching column
            litho_column = litho_columns[0]
            self.logger.info(f"Using lithology classification column: {litho_column}")
            
            # Define rock type categories for classification
            rock_categories = {
                'igneous': ['igneous', 'volcanic', 'plutonic', 'basalt', 'granite', 'diorite', 'gabbro', 'rhyolite', 'andesite'],
                'metamorphic': ['metamorphic', 'gneiss', 'schist', 'quartzite', 'marble', 'amphibolite', 'slate', 'phyllite'],
                'sedimentary': ['sedimentary', 'limestone', 'sandstone', 'shale', 'conglomerate', 'dolomite', 'chalk', 'siltstone'],
                'unconsolidated': ['alluvium', 'colluvium', 'sand', 'gravel', 'clay', 'silt', 'soil', 'till', 'glacial']
            }
            
            # Check if we're dealing with lumped or distributed catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
            
            if is_lumped:
                # For lumped catchment, intersect with catchment boundary
                catchment = gpd.read_file(self.catchment_path)
                
                # Ensure CRS match
                if geo_map.crs != catchment.crs:
                    geo_map = geo_map.to_crs(catchment.crs)
                
                # Intersect geological map with catchment
                intersection = gpd.overlay(geo_map, catchment, how='intersection')
                
                if not intersection.empty:
                    # Convert to equal area for proper area calculation
                    equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                    intersection_equal_area = intersection.to_crs(equal_area_crs)
                    
                    # Calculate area for each lithology class
                    intersection_equal_area['area_km2'] = intersection_equal_area.geometry.area / 10**6  # m² to km²
                    
                    # Calculate lithology class distribution
                    litho_areas = intersection_equal_area.groupby(litho_column)['area_km2'].sum()
                    total_area = litho_areas.sum()
                    
                    if total_area > 0:
                        # Calculate fraction for each lithology class
                        for litho_type, area in litho_areas.items():
                            if litho_type and str(litho_type).strip():  # Skip empty values
                                fraction = area / total_area
                                # Clean name for attribute key
                                clean_name = self._clean_attribute_name(str(litho_type))
                                results[f"geology.{clean_name}_fraction"] = fraction
                        
                        # Identify dominant lithology class
                        dominant_litho = litho_areas.idxmax()
                        if dominant_litho and str(dominant_litho).strip():
                            results["geology.dominant_lithology"] = str(dominant_litho)
                            results["geology.dominant_lithology_fraction"] = litho_areas[dominant_litho] / total_area
                        
                        # Calculate lithology diversity (Shannon entropy)
                        shannon_entropy = 0
                        for _, area in litho_areas.items():
                            if area > 0:
                                p = area / total_area
                                shannon_entropy -= p * np.log(p)
                        results["geology.lithology_diversity"] = shannon_entropy
                    
                    # Classify by broad rock types
                    litho_by_category = {category: 0 for category in rock_categories}
                    
                    # Iterate through each geological unit
                    for _, unit in intersection_equal_area.iterrows():
                        area = unit['area_km2']
                        unit_desc = str(unit.get(litho_column, '')).lower()
                        
                        # Classify the unit into a category based on keywords
                        classified = False
                        for category, keywords in rock_categories.items():
                            if any(keyword in unit_desc for keyword in keywords):
                                litho_by_category[category] += area
                                classified = True
                                break
                    
                    # Calculate percentages for rock categories
                    category_total = sum(litho_by_category.values())
                    if category_total > 0:
                        for category, area in litho_by_category.items():
                            results[f"geology.{category}_fraction"] = area / category_total
                        
                        # Identify dominant category
                        dominant_category = max(litho_by_category.items(), key=lambda x: x[1])[0]
                        results["geology.dominant_category"] = dominant_category
                        results["geology.dominant_category_fraction"] = litho_by_category[dominant_category] / category_total
                else:
                    self.logger.warning("No intersection between geological map and catchment boundary")
            else:
                # For distributed catchment, process each HRU
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                # Ensure CRS match
                if geo_map.crs != catchment.crs:
                    geo_map = geo_map.to_crs(catchment.crs)
                
                for i, hru in catchment.iterrows():
                    try:
                        hru_id = hru[hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Create a GeoDataFrame with just this HRU
                        hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                        
                        # Intersect with geological map
                        intersection = gpd.overlay(geo_map, hru_gdf, how='intersection')
                        
                        if not intersection.empty:
                            # Convert to equal area for proper area calculation
                            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                            intersection_equal_area = intersection.to_crs(equal_area_crs)
                            
                            # Calculate area for each lithology class
                            intersection_equal_area['area_km2'] = intersection_equal_area.geometry.area / 10**6  # m² to km²
                            
                            # Calculate lithology class distribution
                            litho_areas = intersection_equal_area.groupby(litho_column)['area_km2'].sum()
                            total_area = litho_areas.sum()
                            
                            if total_area > 0:
                                # Calculate fraction for each lithology class
                                for litho_type, area in litho_areas.items():
                                    if litho_type and str(litho_type).strip():  # Skip empty values
                                        fraction = area / total_area
                                        # Clean name for attribute key
                                        clean_name = self._clean_attribute_name(str(litho_type))
                                        results[f"{prefix}geology.{clean_name}_fraction"] = fraction
                                
                                # Identify dominant lithology class
                                dominant_litho = litho_areas.idxmax()
                                if dominant_litho and str(dominant_litho).strip():
                                    results[f"{prefix}geology.dominant_lithology"] = str(dominant_litho)
                                    results[f"{prefix}geology.dominant_lithology_fraction"] = litho_areas[dominant_litho] / total_area
                                
                                # Calculate lithology diversity (Shannon entropy)
                                shannon_entropy = 0
                                for _, area in litho_areas.items():
                                    if area > 0:
                                        p = area / total_area
                                        shannon_entropy -= p * np.log(p)
                                results[f"{prefix}geology.lithology_diversity"] = shannon_entropy
                            
                            # Classify by broad rock types
                            litho_by_category = {category: 0 for category in rock_categories}
                            
                            # Iterate through each geological unit
                            for _, unit in intersection_equal_area.iterrows():
                                area = unit['area_km2']
                                unit_desc = str(unit.get(litho_column, '')).lower()
                                
                                # Classify the unit into a category based on keywords
                                classified = False
                                for category, keywords in rock_categories.items():
                                    if any(keyword in unit_desc for keyword in keywords):
                                        litho_by_category[category] += area
                                        classified = True
                                        break
                            
                            # Calculate percentages for rock categories
                            category_total = sum(litho_by_category.values())
                            if category_total > 0:
                                for category, area in litho_by_category.items():
                                    results[f"{prefix}geology.{category}_fraction"] = area / category_total
                                
                                # Identify dominant category
                                dominant_category = max(litho_by_category.items(), key=lambda x: x[1])[0]
                                results[f"{prefix}geology.dominant_category"] = dominant_category
                        else:
                            self.logger.debug(f"No intersection between geological map and HRU {hru_id}")
                    except Exception as e:
                        self.logger.error(f"Error processing lithology for HRU {hru_id}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing lithology data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _clean_attribute_name(self, name: str) -> str:
        """
        Clean a string to be usable as an attribute name.
        
        Args:
            name: Original string
            
        Returns:
            String usable as an attribute name
        """
        if not name:
            return "unknown"
        
        # Convert to string and strip whitespace
        name_str = str(name).strip()
        if not name_str:
            return "unknown"
        
        # Replace spaces and special characters
        cleaned = name_str.lower()
        cleaned = cleaned.replace(' ', '_')
        cleaned = cleaned.replace('-', '_')
        cleaned = cleaned.replace('.', '_')
        cleaned = cleaned.replace('(', '')
        cleaned = cleaned.replace(')', '')
        cleaned = cleaned.replace(',', '')
        cleaned = cleaned.replace('/', '_')
        cleaned = cleaned.replace('\\', '_')
        cleaned = cleaned.replace('&', 'and')
        cleaned = cleaned.replace('%', 'percent')
        
        # Remove any remaining non-alphanumeric characters
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c == '_')
        
        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = 'x' + cleaned
        
        # Limit length
        if len(cleaned) > 50:
            cleaned = cleaned[:50]
        
        # Handle empty result
        if not cleaned:
            return "unknown"
        
        return cleaned

    def _process_structural_features(self, geologic_map_path: Path) -> Dict[str, Any]:
        """
        Process structural geological features like faults and folds.
        
        Args:
            geologic_map_path: Path to geological map data
            
        Returns:
            Dictionary of structural attributes
        """
        results = {}
        
        # Check for Faults.shp in the GMNA_SHAPES directory
        gmna_dir = geologic_map_path / "GMNA_SHAPES"
        if not gmna_dir.exists():
            self.logger.warning(f"GMNA_SHAPES directory not found at {gmna_dir}")
            return results
        
        # Files to process and their descriptions
        structural_files = [
            ("Faults.shp", "faults"),
            ("Geologic_contacts.shp", "contacts"),
            ("Volcanoes.shp", "volcanoes")
        ]
        
        try:
            # Read catchment shapefile
            catchment = gpd.read_file(self.catchment_path)
            catchment_area_km2 = catchment.to_crs('ESRI:102008').area.sum() / 10**6
            
            # Create output directory for clipped data
            output_dir = self.project_dir / 'attributes' / 'geology' / 'structural'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Parse bounding box coordinates for clipping
            bbox_coords = self.config.get('BOUNDING_BOX_COORDS').split('/')
            bbox = None
            if len(bbox_coords) == 4:
                bbox = (
                    float(bbox_coords[1]),  # lon_min
                    float(bbox_coords[2]),  # lat_min
                    float(bbox_coords[3]),  # lon_max
                    float(bbox_coords[0])   # lat_max
                )
            
            # Process each structural feature
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            for file_name, feature_type in structural_files:
                feature_file = gmna_dir / file_name
                if not feature_file.exists():
                    self.logger.info(f"{feature_type.capitalize()} file not found: {feature_file}")
                    continue
                
                self.logger.info(f"Processing {feature_type} data from: {feature_file}")
                
                # Create a clipped version if it doesn't exist
                clipped_file = output_dir / f"{self.domain_name}_{feature_type}.gpkg"
                
                if not clipped_file.exists() and bbox:
                    self.logger.info(f"Clipping {feature_type} to catchment bounding box")
                    try:
                        # Read features within the bounding box
                        features = gpd.read_file(feature_file, bbox=bbox)
                        
                        # Save the clipped features
                        if not features.empty:
                            self.logger.info(f"Saving {len(features)} {feature_type} to {clipped_file}")
                            features.to_file(clipped_file, driver="GPKG")
                        else:
                            self.logger.info(f"No {feature_type} found within bounding box")
                    except Exception as e:
                        self.logger.error(f"Error reading {feature_type}: {str(e)}")
                        continue
                
                # Process the features
                if clipped_file.exists():
                    try:
                        # Read the clipped features
                        features = gpd.read_file(clipped_file)
                        
                        if features.empty:
                            self.logger.info(f"No {feature_type} found in the study area")
                            continue
                        
                        # Ensure CRS match
                        if features.crs != catchment.crs:
                            features = features.to_crs(catchment.crs)
                        
                        if is_lumped:
                            # For lumped catchment - intersect with entire catchment
                            if feature_type in ['faults', 'contacts']:  # Line features
                                # Intersect with catchment
                                features_in_catchment = gpd.clip(features, catchment)
                                
                                if not features_in_catchment.empty:
                                    # Convert to equal area projection for length calculations
                                    features_equal_area = features_in_catchment.to_crs('ESRI:102008')
                                    
                                    # Calculate total length and density
                                    total_length_km = features_equal_area.length.sum() / 1000  # m to km
                                    density = total_length_km / catchment_area_km2 if catchment_area_km2 > 0 else 0
                                    
                                    # Add to results
                                    results[f"geology.{feature_type}_count"] = len(features_in_catchment)
                                    results[f"geology.{feature_type}_length_km"] = total_length_km
                                    results[f"geology.{feature_type}_density_km_per_km2"] = density
                                else:
                                    results[f"geology.{feature_type}_count"] = 0
                                    results[f"geology.{feature_type}_length_km"] = 0
                                    results[f"geology.{feature_type}_density_km_per_km2"] = 0
                            
                            elif feature_type == 'volcanoes':  # Point features
                                # Count volcanoes in catchment
                                volcanoes_in_catchment = gpd.clip(features, catchment)
                                volcano_count = len(volcanoes_in_catchment)
                                volcano_density = volcano_count / catchment_area_km2 if catchment_area_km2 > 0 else 0
                                
                                results["geology.volcano_count"] = volcano_count
                                results["geology.volcano_density_per_km2"] = volcano_density
                        else:
                            # For distributed catchment - process each HRU
                            for idx, hru in catchment.iterrows():
                                hru_id = hru[self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')]
                                prefix = f"HRU_{hru_id}_"
                                
                                # Create a GeoDataFrame with just this HRU
                                hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                                hru_area_km2 = hru_gdf.to_crs('ESRI:102008').area.sum() / 10**6
                                
                                if feature_type in ['faults', 'contacts']:  # Line features
                                    # Clip features to HRU
                                    try:
                                        features_in_hru = gpd.clip(features, hru_gdf)
                                        
                                        if not features_in_hru.empty:
                                            # Calculate length and density
                                            features_equal_area = features_in_hru.to_crs('ESRI:102008')
                                            total_length_km = features_equal_area.length.sum() / 1000
                                            density = total_length_km / hru_area_km2 if hru_area_km2 > 0 else 0
                                            
                                            results[f"{prefix}geology.{feature_type}_count"] = len(features_in_hru)
                                            results[f"{prefix}geology.{feature_type}_length_km"] = total_length_km
                                            results[f"{prefix}geology.{feature_type}_density_km_per_km2"] = density
                                        else:
                                            results[f"{prefix}geology.{feature_type}_count"] = 0
                                            results[f"{prefix}geology.{feature_type}_length_km"] = 0
                                            results[f"{prefix}geology.{feature_type}_density_km_per_km2"] = 0
                                    except Exception as e:
                                        self.logger.error(f"Error processing {feature_type} for HRU {hru_id}: {str(e)}")
                                
                                elif feature_type == 'volcanoes':  # Point features
                                    try:
                                        volcanoes_in_hru = gpd.clip(features, hru_gdf)
                                        volcano_count = len(volcanoes_in_hru)
                                        volcano_density = volcano_count / hru_area_km2 if hru_area_km2 > 0 else 0
                                        
                                        results[f"{prefix}geology.volcano_count"] = volcano_count
                                        results[f"{prefix}geology.volcano_density_per_km2"] = volcano_density
                                    except Exception as e:
                                        self.logger.error(f"Error processing volcanoes for HRU {hru_id}: {str(e)}")
                    
                    except Exception as e:
                        self.logger.error(f"Error processing {feature_type}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing structural features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results


    def _process_hydrological_attributes(self) -> Dict[str, Any]:
        """
        Process hydrological data including lakes and river networks.
        
        Sources:
        - HydroLAKES: Global database of lakes
        - MERIT Basins: River network and basin characteristics
        """
        results = {}
        
        # Find and check paths for hydrological data
        hydrolakes_path = self._get_data_path('ATTRIBUTES_HYDROLAKES_PATH', 'hydrolakes')
        merit_basins_path = self._get_data_path('ATTRIBUTES_MERIT_BASINS_PATH', 'merit_basins')
        
        self.logger.info(f"HydroLAKES path: {hydrolakes_path}")
        self.logger.info(f"MERIT Basins path: {merit_basins_path}")
        
        # Process lakes data
        lakes_results = self._process_lakes_data(hydrolakes_path)
        results.update(lakes_results)
        
        # Process river network data
        river_results = self._process_river_network_data(merit_basins_path)
        results.update(river_results)
        
        return results

    def calculate_seasonality_metrics(self) -> Dict[str, Any]:
        """
        Calculate advanced seasonality metrics including sine curve fitting.
        
        Returns:
            Dict[str, Any]: Dictionary of seasonality metrics
        """
        results = {}
        
        # Load temperature and precipitation data
        temp_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_temperature.csv"
        precip_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_precipitation.csv"
        
        if not temp_path.exists() or not precip_path.exists():
            self.logger.warning("Temperature or precipitation data not found for seasonality calculation")
            return results
        
        try:
            import pandas as pd
            import numpy as np
            from scipy.optimize import curve_fit
            
            # Define sine fitting functions
            def sine_function_temp(x, mean, delta, phase, period=1):
                return mean + delta * np.sin(2*np.pi*(x-phase)/period)
            
            def sine_function_prec(x, mean, delta, phase, period=1):
                result = mean * (1 + delta * np.sin(2*np.pi*(x-phase)/period))
                return np.where(result < 0, 0, result)
            
            # Read data
            temp_df = pd.read_csv(temp_path, parse_dates=['date'])
            precip_df = pd.read_csv(precip_path, parse_dates=['date'])
            
            # Resample to daily if needed
            temp_daily = temp_df.set_index('date').resample('D').mean()
            precip_daily = precip_df.set_index('date').resample('D').mean()
            
            # Ensure same time period
            common_idx = temp_daily.index.intersection(precip_daily.index)
            temp_daily = temp_daily.loc[common_idx]
            precip_daily = precip_daily.loc[common_idx]
            
            # Calculate day of year as fraction of year for x-axis
            day_of_year = np.array([(d.dayofyear - 1) / 365 for d in temp_daily.index])
            
            # Fit temperature sine curve
            temp_values = temp_daily['temperature'].values
            try:
                t_mean = float(np.mean(temp_values))
                t_delta = float(np.max(temp_values) - np.min(temp_values))
                t_phase = 0.5  # Initial guess for phase
                t_initial_guess = [t_mean, t_delta, t_phase]
                t_pars, _ = curve_fit(sine_function_temp, day_of_year, temp_values, p0=t_initial_guess)
                
                results["climate.temperature_mean"] = t_pars[0]
                results["climate.temperature_amplitude"] = t_pars[1]
                results["climate.temperature_phase"] = t_pars[2]
            except Exception as e:
                self.logger.warning(f"Error fitting temperature sine curve: {str(e)}")
            
            # Fit precipitation sine curve
            precip_values = precip_daily['precipitation'].values
            try:
                p_mean = float(np.mean(precip_values))
                p_delta = float(np.max(precip_values) - np.min(precip_values)) / p_mean if p_mean > 0 else 0
                p_phase = 0.5  # Initial guess for phase
                p_initial_guess = [p_mean, p_delta, p_phase]
                p_pars, _ = curve_fit(sine_function_prec, day_of_year, precip_values, p0=p_initial_guess)
                
                results["climate.precipitation_mean"] = p_pars[0]
                results["climate.precipitation_amplitude"] = p_pars[1]
                results["climate.precipitation_phase"] = p_pars[2]
                
                # Calculate Woods (2009) seasonality index
                if hasattr(t_pars, '__len__') and hasattr(p_pars, '__len__') and len(t_pars) >= 3 and len(p_pars) >= 3:
                    seasonality = p_pars[1] * np.sign(t_pars[1]) * np.cos(2*np.pi*(p_pars[2]-t_pars[2])/1)
                    results["climate.seasonality_index"] = seasonality
            except Exception as e:
                self.logger.warning(f"Error fitting precipitation sine curve: {str(e)}")
            
            # Calculate additional seasonality metrics
            
            # Walsh & Lawler (1981) seasonality index for precipitation
            try:
                # Resample to monthly
                monthly_precip = precip_df.set_index('date').resample('M').sum()
                
                # Group by month and calculate multi-year average for each month
                monthly_avg = monthly_precip.groupby(monthly_precip.index.month).mean()
                
                if len(monthly_avg) == 12:
                    annual_sum = monthly_avg.sum().iloc[0]
                    monthly_diff = np.abs(monthly_avg.values - annual_sum/12)
                    walsh_index = np.sum(monthly_diff) / annual_sum
                    
                    results["climate.precipitation_walsh_seasonality"] = walsh_index
                    
                    # Interpret the Walsh & Lawler index
                    if walsh_index < 0.19:
                        results["climate.seasonality_regime"] = "very_equable"
                    elif walsh_index < 0.40:
                        results["climate.seasonality_regime"] = "equable_but_with_a_definite_wetter_season"
                    elif walsh_index < 0.60:
                        results["climate.seasonality_regime"] = "rather_seasonal"
                    elif walsh_index < 0.80:
                        results["climate.seasonality_regime"] = "seasonal"
                    elif walsh_index < 0.99:
                        results["climate.seasonality_regime"] = "markedly_seasonal_with_a_long_drier_season"
                    elif walsh_index < 1.19:
                        results["climate.seasonality_regime"] = "most_rain_in_3_months_or_less"
                    else:
                        results["climate.seasonality_regime"] = "extreme_seasonality"
            except Exception as e:
                self.logger.warning(f"Error calculating Walsh & Lawler seasonality index: {str(e)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating seasonality metrics: {str(e)}")
            return results

    def calculate_water_balance(self) -> Dict[str, Any]:
        """
        Calculate water balance components and hydrological indices.
        
        Returns:
            Dict[str, Any]: Dictionary of water balance metrics
        """
        results = {}
        
        # Look for required data
        precip_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_precipitation.csv"
        pet_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_pet.csv"
        streamflow_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not precip_path.exists() or not streamflow_path.exists():
            self.logger.warning("Cannot calculate water balance: missing precipitation or streamflow data")
            return results
        
        try:
            import pandas as pd
            import numpy as np
            
            # Read data
            precip_df = pd.read_csv(precip_path, parse_dates=['date'])
            streamflow_df = pd.read_csv(streamflow_path, parse_dates=['date'])
            
            # Read PET if available, otherwise use None
            if pet_path.exists():
                pet_df = pd.read_csv(pet_path, parse_dates=['date'])
            else:
                pet_df = None
            
            # Set index and align data
            precip_df.set_index('date', inplace=True)
            streamflow_df.set_index('date', inplace=True)
            if pet_df is not None:
                pet_df.set_index('date', inplace=True)
            
            # Define common time period
            common_period = precip_df.index.intersection(streamflow_df.index)
            if pet_df is not None:
                common_period = common_period.intersection(pet_df.index)
            
            if len(common_period) == 0:
                self.logger.warning("No common time period between precipitation and streamflow data")
                return results
            
            # Subset data to common period
            precip = precip_df.loc[common_period, 'precipitation'].copy()
            streamflow = streamflow_df.loc[common_period, 'flow'].copy()
            if pet_df is not None:
                pet = pet_df.loc[common_period, 'pet'].copy()
            else:
                pet = None
            
            # Calculate water balance components
            
            # 1. Runoff ratio (Q/P)
            annual_precip = precip.resample('YS').sum()
            annual_streamflow = streamflow.resample('YS').sum()
            
            # Align years
            common_years = annual_precip.index.intersection(annual_streamflow.index)
            
            if len(common_years) > 0:
                annual_runoff_ratio = annual_streamflow.loc[common_years] / annual_precip.loc[common_years]
                results["hydrology.annual_runoff_ratio_mean"] = annual_runoff_ratio.mean()
                results["hydrology.annual_runoff_ratio_std"] = annual_runoff_ratio.std()
                results["hydrology.annual_runoff_ratio_cv"] = annual_runoff_ratio.std() / annual_runoff_ratio.mean() if annual_runoff_ratio.mean() > 0 else np.nan
            
            # 2. Estimate actual evapotranspiration (AET) from water balance
            # AET = P - Q - dS/dt (ignoring storage changes)
            annual_aet_estimate = annual_precip.loc[common_years] - annual_streamflow.loc[common_years]
            results["hydrology.annual_aet_estimate_mean"] = annual_aet_estimate.mean()
            results["hydrology.annual_aet_estimate_std"] = annual_aet_estimate.std()
            
            # 3. Aridity index and Budyko analysis
            if pet_df is not None:
                annual_pet = pet.resample('YS').sum()
                common_years_pet = common_years.intersection(annual_pet.index)
                
                if len(common_years_pet) > 0:
                    # Aridity index (PET/P)
                    aridity_index = annual_pet.loc[common_years_pet] / annual_precip.loc[common_years_pet]
                    results["hydrology.aridity_index_mean"] = aridity_index.mean()
                    results["hydrology.aridity_index_std"] = aridity_index.std()
                    
                    # Evaporative index (AET/P)
                    evaporative_index = annual_aet_estimate.loc[common_years_pet] / annual_precip.loc[common_years_pet]
                    results["hydrology.evaporative_index_mean"] = evaporative_index.mean()
                    
                    # Calculate Budyko curve parameters (Fu equation: ET/P = 1 + PET/P - [1 + (PET/P)^w]^(1/w))
                    # We'll use optimization to find the w parameter
                    
                    from scipy.optimize import minimize
                    
                    def fu_equation(w, pet_p):
                        return 1 + pet_p - (1 + pet_p**w)**(1/w)
                    
                    def objective(w, pet_p, et_p):
                        predicted = fu_equation(w, pet_p)
                        return np.mean((predicted - et_p)**2)
                    
                    # Filter out invalid values
                    valid = (aridity_index > 0) & (evaporative_index > 0) & (~np.isnan(aridity_index)) & (~np.isnan(evaporative_index))
                    
                    if valid.any():
                        pet_p_values = aridity_index[valid].values
                        et_p_values = evaporative_index[valid].values
                        
                        # Find optimal w parameter
                        result = minimize(objective, 2.0, args=(pet_p_values, et_p_values))
                        w_parameter = result.x[0]
                        
                        results["hydrology.budyko_parameter_w"] = w_parameter
                        
                        # Calculate Budyko-predicted ET/P
                        budyko_predicted = fu_equation(w_parameter, aridity_index.mean())
                        results["hydrology.budyko_predicted_et_ratio"] = budyko_predicted
                        
                        # Calculate residual (actual - predicted)
                        results["hydrology.budyko_residual"] = evaporative_index.mean() - budyko_predicted
            
            # 4. Flow duration curve metrics
            flow_sorted = np.sort(streamflow.dropna().values)[::-1]  # sort in descending order
            if len(flow_sorted) > 0:
                total_flow = np.sum(flow_sorted)
                
                # Calculate normalized flow duration curve
                n = len(flow_sorted)
                exceedance_prob = np.arange(1, n+1) / (n+1)  # exceedance probability
                
                # Calculate common FDC metrics
                q1 = np.interp(0.01, exceedance_prob, flow_sorted)
                q5 = np.interp(0.05, exceedance_prob, flow_sorted)
                q10 = np.interp(0.1, exceedance_prob, flow_sorted)
                q50 = np.interp(0.5, exceedance_prob, flow_sorted)
                q85 = np.interp(0.85, exceedance_prob, flow_sorted)
                q95 = np.interp(0.95, exceedance_prob, flow_sorted)
                q99 = np.interp(0.99, exceedance_prob, flow_sorted)
                
                results["hydrology.fdc_q1"] = q1
                results["hydrology.fdc_q5"] = q5
                results["hydrology.fdc_q10"] = q10
                results["hydrology.fdc_q50"] = q50
                results["hydrology.fdc_q85"] = q85
                results["hydrology.fdc_q95"] = q95
                results["hydrology.fdc_q99"] = q99
                
                # Calculate high flow (Q10/Q50) and low flow (Q50/Q90) indices
                results["hydrology.high_flow_index"] = q10 / q50 if q50 > 0 else np.nan
                results["hydrology.low_flow_index"] = q50 / q95 if q95 > 0 else np.nan
                
                # Calculate flow variability indices
                results["hydrology.flow_duration_index"] = (q10 - q95) / q50 if q50 > 0 else np.nan
                
                # Baseflow index (Q90/Q50)
                results["hydrology.baseflow_index_fdc"] = q95 / q50 if q50 > 0 else np.nan
            
            # 5. Flow seasonality and timing
            monthly_flow = streamflow.resample('M').sum()
            monthly_precip = precip.resample('M').sum()
            
            # Calculate monthly flow contribution
            annual_flow_by_month = monthly_flow.groupby(monthly_flow.index.month).mean()
            annual_flow_total = annual_flow_by_month.sum()
            
            if annual_flow_total > 0:
                monthly_flow_fraction = annual_flow_by_month / annual_flow_total
                
                # Find month with maximum flow
                max_flow_month = monthly_flow_fraction.idxmax()
                results["hydrology.peak_flow_month"] = max_flow_month
                
                # Calculate seasonality indices
                
                # Colwell's predictability index components
                # M = seasonality, C = contingency, P = predictability = M + C
                # Ranges from 0 (unpredictable) to 1 (perfectly predictable)
                
                # Use normalized monthly values for calculations
                monthly_flow_norm = monthly_flow / monthly_flow.mean() if monthly_flow.mean() > 0 else monthly_flow
                
                # Seasonality index (Walsh & Lawler)
                if len(monthly_flow_fraction) == 12:
                    monthly_diff = np.abs(monthly_flow_fraction - 1/12)
                    seasonality_index = np.sum(monthly_diff) * 0.5  # Scale to 0-1
                    results["hydrology.flow_seasonality_index"] = seasonality_index
                
                # Calculate monthly runoff ratios
                monthly_precip_avg = monthly_precip.groupby(monthly_precip.index.month).mean()
                monthly_flow_avg = monthly_flow.groupby(monthly_flow.index.month).mean()
                
                # Calculate monthly runoff ratios
                common_months = monthly_flow_avg.index.intersection(monthly_precip_avg.index)
                
                if len(common_months) > 0:
                    monthly_runoff_ratio = monthly_flow_avg.loc[common_months] / monthly_precip_avg.loc[common_months]
                    
                    # Store min, max, and range of monthly runoff ratios
                    results["hydrology.monthly_runoff_ratio_min"] = monthly_runoff_ratio.min()
                    results["hydrology.monthly_runoff_ratio_max"] = monthly_runoff_ratio.max()
                    results["hydrology.monthly_runoff_ratio_range"] = monthly_runoff_ratio.max() - monthly_runoff_ratio.min()
                    
                    # Find month with maximum and minimum runoff ratio
                    results["hydrology.max_runoff_ratio_month"] = monthly_runoff_ratio.idxmax()
                    results["hydrology.min_runoff_ratio_month"] = monthly_runoff_ratio.idxmin()
                    
                    # Calculate correlation between monthly precipitation and runoff
                    corr = np.corrcoef(monthly_precip_avg.loc[common_months], monthly_flow_avg.loc[common_months])[0, 1]
                    results["hydrology.precip_flow_correlation"] = corr
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating water balance: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return results


    def calculate_composite_landcover_metrics(self) -> Dict[str, Any]:
        """
        Calculate composite land cover metrics combining multiple land cover datasets
        and extracting ecologically meaningful indicators.
        
        Returns:
            Dict[str, Any]: Dictionary of composite land cover metrics
        """
        results = {}
        
        # Collect existing land cover results from individual datasets
        lc_datasets = {
            "glclu2019": "landcover.forest_fraction",
            "modis": "landcover.evergreen_needleleaf_fraction",
            "esa": "landcover.forest_broadleaf_evergreen_fraction"
        }
        
        # Define ecological groupings that span datasets
        ecological_groups = {
            "forest": ["forest", "tree", "woodland", "evergreen", "deciduous", "needleleaf", "broadleaf"],
            "grassland": ["grass", "savanna", "rangeland", "pasture"],
            "wetland": ["wetland", "marsh", "bog", "fen", "swamp"],
            "cropland": ["crop", "agriculture", "cultivated"],
            "urban": ["urban", "built", "city", "development"],
            "water": ["water", "lake", "river", "reservoir"],
            "barren": ["barren", "desert", "rock", "sand"]
        }
        
        try:
            # Collect all landcover attributes from results
            all_attributes = [key for key in self.results.keys() if key.startswith("landcover.")]
            
            # Group by datasets
            datasets_found = {}
            for attr in all_attributes:
                # Extract dataset name from the attribute
                parts = attr.split(".")
                if len(parts) > 1:
                    attribute = parts[1]
                    # Try to identify which dataset this comes from
                    for dataset_name, marker in lc_datasets.items():
                        if marker in attr:
                            if dataset_name not in datasets_found:
                                datasets_found[dataset_name] = []
                            datasets_found[dataset_name].append(attr)
            
            # If we found at least one dataset, calculate composite metrics
            if datasets_found:
                # First, create ecological groupings for each dataset
                dataset_eco_groups = {}
                
                for dataset, attributes in datasets_found.items():
                    dataset_eco_groups[dataset] = {group: [] for group in ecological_groups}
                    
                    # Assign attributes to ecological groups
                    for attr in attributes:
                        for group, keywords in ecological_groups.items():
                            if any(keyword in attr.lower() for keyword in keywords):
                                dataset_eco_groups[dataset][group].append(attr)
                
                # Calculate composite metrics for each ecological group
                for group, keywords in ecological_groups.items():
                    # Collect values across datasets
                    group_values = []
                    
                    for dataset, eco_groups in dataset_eco_groups.items():
                        if eco_groups[group]:
                            # Get average value for this group in this dataset
                            group_attrs = eco_groups[group]
                            attr_values = [self.results.get(attr, 0) for attr in group_attrs]
                            if attr_values:
                                group_values.append(np.mean(attr_values))
                    
                    # Calculate composite metric if we have values
                    if group_values:
                        results[f"landcover.composite_{group}_fraction"] = np.mean(group_values)
                
                # Calculate landscape diversity from composite fractions
                # Shannon diversity index
                fractions = [v for k, v in results.items() if k.startswith("landcover.composite_") and k.endswith("_fraction")]
                if fractions:
                    # Normalize fractions to sum to 1
                    total = sum(fractions)
                    if total > 0:
                        normalized = [f/total for f in fractions]
                        shannon_index = -sum(p * np.log(p) for p in normalized if p > 0)
                        results["landcover.composite_shannon_diversity"] = shannon_index
                        
                        # Simpson diversity index (1-D)
                        simpson_index = 1 - sum(p**2 for p in normalized)
                        results["landcover.composite_simpson_diversity"] = simpson_index
                
                # Calculate anthropogenic influence
                if "landcover.composite_urban_fraction" in results and "landcover.composite_cropland_fraction" in results:
                    anthropogenic = results["landcover.composite_urban_fraction"] + results["landcover.composite_cropland_fraction"]
                    results["landcover.anthropogenic_influence"] = anthropogenic
                
                # Calculate natural vegetation cover
                if "landcover.composite_forest_fraction" in results and "landcover.composite_grassland_fraction" in results and "landcover.composite_wetland_fraction" in results:
                    natural = results["landcover.composite_forest_fraction"] + results["landcover.composite_grassland_fraction"] + results["landcover.composite_wetland_fraction"]
                    results["landcover.natural_vegetation_cover"] = natural
            
            # Analyze vegetation dynamics if we have LAI data
            lai_months = [key for key in self.results.keys() if key.startswith("vegetation.lai_month")]
            if lai_months:
                # Extract monthly values
                monthly_lai = {}
                for key in lai_months:
                    if "_mean" in key:
                        # Extract month number
                        try:
                            month = int(key.split("month")[1][:2])
                            monthly_lai[month] = self.results[key]
                        except (ValueError, IndexError):
                            pass
                
                if len(monthly_lai) >= 6:  # Need at least half the year
                    # Sort by month
                    months = sorted(monthly_lai.keys())
                    lai_values = [monthly_lai[m] for m in months]
                    
                    # Calculate seasonal amplitude
                    lai_min = min(lai_values)
                    lai_max = max(lai_values)
                    results["vegetation.seasonal_lai_amplitude"] = lai_max - lai_min
                    
                    # Calculate growing season characteristics
                    if lai_min < lai_max:
                        # Define growing season threshold (e.g., 50% of range above minimum)
                        threshold = lai_min + 0.5 * (lai_max - lai_min)
                        
                        # Find months above threshold
                        growing_months = [m for m in months if monthly_lai[m] >= threshold]
                        
                        if growing_months:
                            results["vegetation.growing_season_length_months"] = len(growing_months)
                            results["vegetation.growing_season_start_month"] = min(growing_months)
                            results["vegetation.growing_season_end_month"] = max(growing_months)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error calculating composite land cover metrics: {str(e)}")
            return results

    def enhance_river_network_analysis(self) -> Dict[str, Any]:
        """
        Enhanced analysis of river network including stream order, drainage density,
        and river morphology metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of enhanced river network attributes
        """
        results = {}
        
        # Get path to river network shapefile
        river_dir = self.project_dir / 'shapefiles' / 'river_network'
        river_files = list(river_dir.glob("*.shp"))
        
        if not river_files:
            self.logger.warning("No river network shapefile found")
            return results
        
        river_file = river_files[0]
        
        try:
            import geopandas as gpd
            import numpy as np
            from scipy.stats import skew, kurtosis
            
            # Read river network and catchment shapefiles
            river_network = gpd.read_file(river_file)
            catchment = gpd.read_file(self.catchment_path)
            
            # Ensure CRS match
            if river_network.crs != catchment.crs:
                river_network = river_network.to_crs(catchment.crs)
            
            # Calculate catchment area in km²
            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
            catchment_area_km2 = catchment.to_crs(equal_area_crs).area.sum() / 10**6
            
            # Basic river network statistics
            results["hydrology.stream_segment_count"] = len(river_network)
            
            # Calculate total stream length
            if 'Length' in river_network.columns:
                length_column = 'Length'
            elif 'length' in river_network.columns:
                length_column = 'length'
            elif any(col.lower() == 'length' for col in river_network.columns):
                length_column = next(col for col in river_network.columns if col.lower() == 'length')
            else:
                # Calculate length using geometry
                river_network_equal_area = river_network.to_crs(equal_area_crs)
                river_network_equal_area['length_m'] = river_network_equal_area.geometry.length
                length_column = 'length_m'
            
            # Total stream length in km
            total_length_km = river_network[length_column].sum() / 1000 if length_column != 'length_m' else river_network[length_column].sum() / 1000
            results["hydrology.stream_total_length_km"] = total_length_km
            
            # Drainage density (km/km²)
            results["hydrology.drainage_density"] = total_length_km / catchment_area_km2
            
            # Stream order statistics if available
            stream_order_column = None
            for possible_column in ['str_order', 'order', 'strahler', 'STRAHLER']:
                if possible_column in river_network.columns:
                    stream_order_column = possible_column
                    break
            
            if stream_order_column:
                stream_orders = river_network[stream_order_column].astype(float)
                results["hydrology.stream_order_min"] = stream_orders.min()
                results["hydrology.stream_order_max"] = stream_orders.max()
                results["hydrology.stream_order_mean"] = stream_orders.mean()
                
                # Stream order frequency and length by order
                for order in range(1, int(stream_orders.max()) + 1):
                    order_segments = river_network[river_network[stream_order_column] == order]
                    if not order_segments.empty:
                        # Number of segments for this order
                        results[f"hydrology.stream_order_{order}_count"] = len(order_segments)
                        
                        # Length for this order in km
                        order_length_km = order_segments[length_column].sum() / 1000 if length_column != 'length_m' else order_segments[length_column].sum() / 1000
                        results[f"hydrology.stream_order_{order}_length_km"] = order_length_km
                        
                        # Percentage of total length
                        if total_length_km > 0:
                            results[f"hydrology.stream_order_{order}_percent"] = (order_length_km / total_length_km) * 100
                
                # Calculate bifurcation ratio (ratio of number of streams of a given order to number of streams of next higher order)
                for order in range(1, int(stream_orders.max())):
                    count_current = len(river_network[river_network[stream_order_column] == order])
                    count_next = len(river_network[river_network[stream_order_column] == order + 1])
                    
                    if count_next > 0:
                        bifurcation_ratio = count_current / count_next
                        results[f"hydrology.bifurcation_ratio_order_{order}_{order+1}"] = bifurcation_ratio
                
                # Average bifurcation ratio across all orders
                bifurcation_ratios = []
                for order in range(1, int(stream_orders.max())):
                    count_current = len(river_network[river_network[stream_order_column] == order])
                    count_next = len(river_network[river_network[stream_order_column] == order + 1])
                    
                    if count_next > 0:
                        bifurcation_ratios.append(count_current / count_next)
                
                if bifurcation_ratios:
                    results["hydrology.mean_bifurcation_ratio"] = np.mean(bifurcation_ratios)
            
            # Slope statistics if available
            slope_column = None
            for possible_column in ['slope', 'SLOPE', 'Slope']:
                if possible_column in river_network.columns:
                    slope_column = possible_column
                    break
            
            if slope_column:
                slopes = river_network[slope_column].astype(float)
                results["hydrology.stream_slope_min"] = slopes.min()
                results["hydrology.stream_slope_max"] = slopes.max()
                results["hydrology.stream_slope_mean"] = slopes.mean()
                results["hydrology.stream_slope_median"] = slopes.median()
                results["hydrology.stream_slope_std"] = slopes.std()
            
            # Calculate sinuosity if we have both actual length and straight-line distance
            sinuosity_column = None
            for possible_column in ['sinuosity', 'SINUOSITY', 'Sinuosity']:
                if possible_column in river_network.columns:
                    sinuosity_column = possible_column
                    break
            
            if sinuosity_column:
                sinuosity = river_network[sinuosity_column].astype(float)
                results["hydrology.stream_sinuosity_min"] = sinuosity.min()
                results["hydrology.stream_sinuosity_max"] = sinuosity.max()
                results["hydrology.stream_sinuosity_mean"] = sinuosity.mean()
                results["hydrology.stream_sinuosity_median"] = sinuosity.median()
            else:
                # Try to calculate sinuosity using geometry
                try:
                    # Create a new column with the straight-line distance between endpoints
                    def calculate_sinuosity(geometry):
                        try:
                            coords = list(geometry.coords)
                            if len(coords) >= 2:
                                from shapely.geometry import LineString
                                straight_line = LineString([coords[0], coords[-1]])
                                return geometry.length / straight_line.length
                            return 1.0
                        except:
                            return 1.0
                    
                    river_network['calc_sinuosity'] = river_network.geometry.apply(calculate_sinuosity)
                    
                    results["hydrology.stream_sinuosity_min"] = river_network['calc_sinuosity'].min()
                    results["hydrology.stream_sinuosity_max"] = river_network['calc_sinuosity'].max()
                    results["hydrology.stream_sinuosity_mean"] = river_network['calc_sinuosity'].mean()
                    results["hydrology.stream_sinuosity_median"] = river_network['calc_sinuosity'].median()
                except Exception as e:
                    self.logger.warning(f"Could not calculate sinuosity: {str(e)}")
            
            # Calculate headwater density
            headwater_column = None
            if stream_order_column:
                # Headwaters are typically first-order streams
                headwaters = river_network[river_network[stream_order_column] == 1]
                results["hydrology.headwater_count"] = len(headwaters)
                results["hydrology.headwater_density_per_km2"] = len(headwaters) / catchment_area_km2
            
            # If we have a pour point, calculate:
            # 1. Distance to farthest headwater (longest flow path)
            # 2. Main stem length
            # 3. Basin elongation ratio
            pour_point_file = self.project_dir / 'shapefiles' / 'pour_point' / f"{self.domain_name}_pourPoint.shp"
            
            if pour_point_file.exists():
                pour_point = gpd.read_file(pour_point_file)
                
                # Ensure CRS match
                if pour_point.crs != river_network.crs:
                    pour_point = pour_point.to_crs(river_network.crs)
                
                # Try to find the main stem by tracing upstream from the pour point
                # This would require river network connectivity information
                # Placeholder for the logic - depends on how your river network topology is structured
                
                # Estimate basin shape metrics
                # Basin length is approximately the longest flow path
                if "hydrology.longest_flowpath_km" in results:
                    basin_length = results["hydrology.longest_flowpath_km"]
                else:
                    # Estimate basin length using catchment geometry
                    # Calculate longest intra-polygon distance as rough approximation
                    from shapely.geometry import MultiPoint
                    hull = catchment.unary_union.convex_hull
                    if hasattr(hull, 'exterior'):
                        points = list(hull.exterior.coords)
                        if len(points) > 1:
                            # Find approximate longest axis through convex hull
                            point_combinations = [(points[i], points[j]) 
                                                for i in range(len(points)) 
                                                for j in range(i+1, len(points))]
                            distances = [np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
                                        for p1, p2 in point_combinations]
                            basin_length = max(distances) / 1000  # convert to km
                            results["hydrology.basin_length_km"] = basin_length
                
                # Calculate basin shape metrics
                if "hydrology.basin_length_km" in results and catchment_area_km2 > 0:
                    # Basin elongation ratio (Schumm, 1956)
                    # Re = diameter of circle with same area as basin / basin length
                    diameter_equivalent_circle = 2 * np.sqrt(catchment_area_km2 / np.pi)
                    elongation_ratio = diameter_equivalent_circle / results["hydrology.basin_length_km"]
                    results["hydrology.basin_elongation_ratio"] = elongation_ratio
                    
                    # Basin form factor (Horton, 1932)
                    # Form factor = Area / (Basin length)²
                    form_factor = catchment_area_km2 / (results["hydrology.basin_length_km"]**2)
                    results["hydrology.basin_form_factor"] = form_factor
                    
                    # Basin circularity ratio
                    # Circularity = 4πA / P² where A is area and P is perimeter
                    perimeter_km = catchment.to_crs(equal_area_crs).geometry.length.sum() / 1000
                    circularity_ratio = (4 * np.pi * catchment_area_km2) / (perimeter_km**2)
                    results["hydrology.basin_circularity_ratio"] = circularity_ratio
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in enhanced river network analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return results

    def enhance_hydrogeological_attributes(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance existing geological attributes with derived hydraulic properties.
        
        Args:
            current_results: Current geological attributes
            
        Returns:
            Dict[str, Any]: Enhanced geological attributes
        """
        results = dict(current_results)  # Create a copy to avoid modifying the original
        
        try:
            # Check if we have log permeability values
            if "geology.log_permeability_mean" in results:
                log_k = results["geology.log_permeability_mean"]
                
                # Convert log permeability to hydraulic conductivity in m/s
                # K [m/s] = 10^(log_k) * (rho*g/mu) where rho*g/mu ≈ 10^7 for water
                hydraulic_conductivity = 10**(log_k) * (10**7)
                results["geology.hydraulic_conductivity_m_per_s"] = hydraulic_conductivity
                
                # Convert to more common units (cm/hr, m/day)
                results["geology.hydraulic_conductivity_cm_per_hr"] = hydraulic_conductivity * 3600 * 100
                results["geology.hydraulic_conductivity_m_per_day"] = hydraulic_conductivity * 86400
                
                # Calculate transmissivity assuming typical aquifer thickness values
                # Shallow aquifer (10m)
                results["geology.transmissivity_shallow_m2_per_day"] = hydraulic_conductivity * 10 * 86400
                
                # Deep aquifer (100m)
                results["geology.transmissivity_deep_m2_per_day"] = hydraulic_conductivity * 100 * 86400
                
                # Calculate hydraulic diffusivity if we have porosity data
                if "geology.porosity_mean" in results:
                    porosity = results["geology.porosity_mean"]
                    if porosity > 0:
                        specific_storage = 1e-6  # Typical value for confined aquifer (1/m)
                        storativity = specific_storage * 100  # For 100m thick aquifer
                        
                        # Calculate specific yield (typically 0.1-0.3 of porosity)
                        specific_yield = 0.2 * porosity
                        results["geology.specific_yield"] = specific_yield
                        
                        # Hydraulic diffusivity (m²/s)
                        diffusivity = hydraulic_conductivity / storativity
                        results["geology.hydraulic_diffusivity_m2_per_s"] = diffusivity
                        
                        # Groundwater response time (days) for 1km travel distance
                        if diffusivity > 0:
                            response_time = (1000**2) / (diffusivity * 86400)  # days
                            results["geology.groundwater_response_time_days"] = response_time
            
            # Derive aquifer properties from soil depth if available
            if "soil.regolith_thickness_mean" in results or "soil.sedimentary_thickness_mean" in results:
                # Use available thickness data or default to a typical value
                regolith_thickness = results.get("soil.regolith_thickness_mean", 0)
                sedimentary_thickness = results.get("soil.sedimentary_thickness_mean", 0)
                
                # Estimate total aquifer thickness
                aquifer_thickness = max(regolith_thickness, sedimentary_thickness)
                if aquifer_thickness == 0:
                    aquifer_thickness = 50  # Default value if no data available
                
                results["geology.estimated_aquifer_thickness_m"] = aquifer_thickness
                
                # If we have hydraulic conductivity, calculate transmissivity
                if "geology.hydraulic_conductivity_m_per_s" in results:
                    transmissivity = results["geology.hydraulic_conductivity_m_per_s"] * aquifer_thickness
                    results["geology.transmissivity_m2_per_s"] = transmissivity
            
            # Extract bedrock depth information from soil data if available
            if "soil.regolith_thickness_mean" in results:
                results["geology.bedrock_depth_m"] = results["soil.regolith_thickness_mean"]
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error enhancing hydrogeological attributes: {str(e)}")
            return current_results  # Return the original results if there was an error

    def calculate_streamflow_signatures(self) -> Dict[str, Any]:
        """
        Calculate streamflow signatures including durations, timing, and distributions.
        
        Returns:
            Dict[str, Any]: Dictionary of streamflow signature attributes
        """
        results = {}
        
        # Load streamflow data
        streamflow_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not streamflow_path.exists():
            self.logger.warning(f"Streamflow data not found: {streamflow_path}")
            return results
        
        try:
            import pandas as pd
            import numpy as np
            from scipy.stats import skew, kurtosis
            
            # Read streamflow data
            streamflow_df = pd.read_csv(streamflow_path, parse_dates=['date'])
            streamflow_df.set_index('date', inplace=True)
            
            # Calculate water year
            streamflow_df['water_year'] = streamflow_df.index.year.where(
                streamflow_df.index.month < 10, 
                streamflow_df.index.year + 1
            )
            
            # Basic statistics
            results["hydrology.daily_discharge_mean"] = streamflow_df['flow'].mean()
            results["hydrology.daily_discharge_std"] = streamflow_df['flow'].std()
            
            # Calculate flow percentiles
            for percentile in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                results[f"hydrology.q{percentile}"] = np.percentile(streamflow_df['flow'].dropna(), percentile)
            
            # Calculate flow duration curve slope
            flow_sorted = np.sort(streamflow_df['flow'].dropna().values)
            if len(flow_sorted) > 0:
                # Replace zeros with a small value to enable log calculation
                flow_sorted[flow_sorted == 0] = 0.001 * results["hydrology.daily_discharge_mean"]
                
                log_flow = np.log(flow_sorted)
                q33_index = int(0.33 * len(log_flow))
                q66_index = int(0.66 * len(log_flow))
                
                if q66_index > q33_index:
                    fdc_slope = (log_flow[q66_index] - log_flow[q33_index]) / (0.66 - 0.33)
                    results["hydrology.fdc_slope"] = fdc_slope
            
            # Calculate high flow and low flow durations
            mean_flow = streamflow_df['flow'].mean()
            median_flow = streamflow_df['flow'].median()
            
            # High flow condition (Q > 9 * median)
            high_flow_threshold = 9 * median_flow
            high_flow = streamflow_df['flow'] > high_flow_threshold
            
            # Low flow condition (Q < 0.2 * mean)
            low_flow_threshold = 0.2 * mean_flow
            low_flow = streamflow_df['flow'] < low_flow_threshold
            
            # No flow condition
            no_flow = streamflow_df['flow'] <= 0
            
            # Calculate durations
            for condition, name in [(high_flow, 'high_flow'), (low_flow, 'low_flow'), (no_flow, 'no_flow')]:
                # Find durations of consecutive True values
                transitions = np.diff(np.concatenate([[False], condition, [False]]).astype(int))
                starts = np.where(transitions == 1)[0]
                ends = np.where(transitions == -1)[0]
                durations = ends - starts
                
                if len(durations) > 0:
                    results[f"hydrology.{name}_duration_mean"] = durations.mean()
                    results[f"hydrology.{name}_duration_median"] = np.median(durations)
                    results[f"hydrology.{name}_duration_max"] = durations.max()
                    results[f"hydrology.{name}_count_per_year"] = len(durations) / streamflow_df['water_year'].nunique()
                    results[f"hydrology.{name}_duration_skew"] = skew(durations) if len(durations) > 2 else np.nan
                    results[f"hydrology.{name}_duration_kurtosis"] = kurtosis(durations) if len(durations) > 2 else np.nan
            
            # Calculate half-flow date
            # Group by water year and find the date when 50% of annual flow occurs
            half_flow_dates = []
            
            for year, year_data in streamflow_df.groupby('water_year'):
                if year_data['flow'].sum() > 0:  # Avoid years with no flow
                    # Calculate cumulative flow for the year
                    year_data = year_data.sort_index()  # Ensure chronological order
                    year_data['cum_flow'] = year_data['flow'].cumsum()
                    year_data['frac_flow'] = year_data['cum_flow'] / year_data['flow'].sum()
                    
                    # Find the first date when fractional flow exceeds 0.5
                    half_flow_idx = year_data['frac_flow'].gt(0.5).idxmax() if any(year_data['frac_flow'] > 0.5) else None
                    
                    if half_flow_idx:
                        half_flow_dates.append(half_flow_idx.dayofyear)
            
            if half_flow_dates:
                results["hydrology.half_flow_day_mean"] = np.mean(half_flow_dates)
                results["hydrology.half_flow_day_std"] = np.std(half_flow_dates)
            
            # Calculate runoff ratio if precipitation data is available
            precip_path = self.project_dir / "forcing" / "basin_averaged_data" / f"{self.config['DOMAIN_NAME']}_precipitation.csv"
            
            if precip_path.exists():
                precip_df = pd.read_csv(precip_path, parse_dates=['date'])
                precip_df.set_index('date', inplace=True)
                
                # Ensure same time period for both datasets
                common_period = streamflow_df.index.intersection(precip_df.index)
                if len(common_period) > 0:
                    streamflow_subset = streamflow_df.loc[common_period]
                    precip_subset = precip_df.loc[common_period]
                    
                    # Calculate runoff ratio (Q/P)
                    total_flow = streamflow_subset['flow'].sum()
                    total_precip = precip_subset['precipitation'].sum()
                    
                    if total_precip > 0:
                        runoff_ratio = total_flow / total_precip
                        results["hydrology.runoff_ratio"] = runoff_ratio
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating streamflow signatures: {str(e)}")
            return results

    def calculate_baseflow_attributes(self) -> Dict[str, Any]:
        """
        Process baseflow separation and calculate baseflow index using the Eckhardt method.
        
        Returns:
            Dict[str, Any]: Dictionary of baseflow attributes
        """
        results = {}
        
        # Load streamflow data
        streamflow_path = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.config['DOMAIN_NAME']}_streamflow_processed.csv"
        
        if not streamflow_path.exists():
            self.logger.warning(f"Streamflow data not found: {streamflow_path}")
            return results
        
        try:
            # Read streamflow data
            import pandas as pd
            import baseflow
            
            # Load streamflow data with pandas
            streamflow_df = pd.read_csv(streamflow_path, parse_dates=['date'])
            
            # Apply baseflow separation using the Eckhardt method
            rec = baseflow.separation(streamflow_df, method='Eckhardt')
            
            # Calculate baseflow index (BFI)
            bfi = rec['Eckhardt']['q_bf'].mean() / rec['Eckhardt']['q_obs'].mean()
            
            results["hydrology.baseflow_index"] = bfi
            results["hydrology.baseflow_recession_constant"] = baseflow.calculate_recession_constant(streamflow_df)
            
            # Calculate seasonal baseflow indices
            seasons = {'winter': [12, 1, 2], 'spring': [3, 4, 5], 
                    'summer': [6, 7, 8], 'autumn': [9, 10, 11]}
            
            for season_name, months in seasons.items():
                seasonal_df = rec['Eckhardt'][rec['Eckhardt'].index.month.isin(months)]
                if not seasonal_df.empty:
                    seasonal_bfi = seasonal_df['q_bf'].mean() / seasonal_df['q_obs'].mean()
                    results[f"hydrology.baseflow_index_{season_name}"] = seasonal_bfi
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error calculating baseflow attributes: {str(e)}")
            return results

    def _process_lakes_data(self, hydrolakes_path: Path) -> Dict[str, Any]:
        """
        Process lake data from HydroLAKES.
        
        Args:
            hydrolakes_path: Path to HydroLAKES data
            
        Returns:
            Dictionary of lake attributes
        """
        results = {}
        
        # Look for HydroLAKES shapefile
        lakes_files = list(hydrolakes_path.glob("**/HydroLAKES_polys*.shp"))
        
        if not lakes_files:
            self.logger.warning(f"No HydroLAKES files found in {hydrolakes_path}")
            return results
        
        lakes_file = lakes_files[0]
        self.logger.info(f"Processing lake data from: {lakes_file}")
        
        try:
            # Read catchment shapefile
            catchment = gpd.read_file(self.catchment_path)
            
            # Create output directory for clipped data
            output_dir = self.project_dir / 'attributes' / 'hydrology' / 'lakes'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a clipped version of HydroLAKES if it doesn't exist
            clipped_lakes_file = output_dir / f"{self.domain_name}_hydrolakes.gpkg"
            
            if not clipped_lakes_file.exists():
                self.logger.info(f"Clipping HydroLAKES to catchment bounding box")
                
                # Parse bounding box coordinates
                bbox_coords = self.config.get('BOUNDING_BOX_COORDS').split('/')
                if len(bbox_coords) == 4:
                    bbox = (
                        float(bbox_coords[1]),  # lon_min
                        float(bbox_coords[2]),  # lat_min
                        float(bbox_coords[3]),  # lon_max
                        float(bbox_coords[0])   # lat_max
                    )
                    
                    # Create a spatial filter for the bbox (to avoid loading the entire dataset)
                    bbox_geom = box(*bbox)
                    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
                    
                    # Read lakes within the bounding box
                    self.logger.info(f"Reading lakes within bounding box")
                    lakes = gpd.read_file(lakes_file, bbox=bbox)
                    
                    # Save the clipped lakes
                    if not lakes.empty:
                        self.logger.info(f"Saving {len(lakes)} lakes to {clipped_lakes_file}")
                        lakes.to_file(clipped_lakes_file, driver="GPKG")
                    else:
                        self.logger.warning("No lakes found within bounding box")
                else:
                    self.logger.error("Invalid bounding box coordinates format")
            
            # Process lakes for the catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if clipped_lakes_file.exists():
                # Read the clipped lakes
                lakes = gpd.read_file(clipped_lakes_file)
                
                if lakes.empty:
                    self.logger.info("No lakes found in the study area")
                    return results
                
                # Ensure CRS match
                if lakes.crs != catchment.crs:
                    lakes = lakes.to_crs(catchment.crs)
                
                # Intersect lakes with catchment
                if is_lumped:
                    # For lumped catchment - intersect with entire catchment
                    lakes_in_catchment = gpd.overlay(lakes, catchment, how='intersection')
                    
                    # Count lakes and calculate area
                    if not lakes_in_catchment.empty:
                        # Convert to equal area projection for area calculations
                        equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                        lakes_equal_area = lakes_in_catchment.to_crs(equal_area_crs)
                        catchment_equal_area = catchment.to_crs(equal_area_crs)
                        
                        # Calculate catchment area in km²
                        catchment_area_km2 = catchment_equal_area.area.sum() / 10**6
                        
                        # Calculate lake attributes
                        num_lakes = len(lakes_in_catchment)
                        lake_area_km2 = lakes_equal_area.area.sum() / 10**6
                        lake_area_percent = (lake_area_km2 / catchment_area_km2) * 100 if catchment_area_km2 > 0 else 0
                        lake_density = num_lakes / catchment_area_km2 if catchment_area_km2 > 0 else 0
                        
                        # Add to results
                        results["hydrology.lake_count"] = num_lakes
                        results["hydrology.lake_area_km2"] = lake_area_km2
                        results["hydrology.lake_area_percent"] = lake_area_percent
                        results["hydrology.lake_density"] = lake_density  # lakes per km²
                        
                        # Calculate statistics for lake size
                        if num_lakes > 0:
                            lakes_equal_area['area_km2'] = lakes_equal_area.area / 10**6
                            results["hydrology.lake_area_min_km2"] = lakes_equal_area['area_km2'].min()
                            results["hydrology.lake_area_max_km2"] = lakes_equal_area['area_km2'].max()
                            results["hydrology.lake_area_mean_km2"] = lakes_equal_area['area_km2'].mean()
                            results["hydrology.lake_area_median_km2"] = lakes_equal_area['area_km2'].median()
                        
                        # Get volume and residence time if available
                        if 'Vol_total' in lakes_in_catchment.columns:
                            # Total volume in km³
                            total_volume_km3 = lakes_in_catchment['Vol_total'].sum() / 10**9
                            results["hydrology.lake_volume_km3"] = total_volume_km3
                        
                        if 'Depth_avg' in lakes_in_catchment.columns:
                            # Area-weighted average depth
                            weighted_depth = np.average(
                                lakes_in_catchment['Depth_avg'], 
                                weights=lakes_equal_area.area
                            )
                            results["hydrology.lake_depth_avg_m"] = weighted_depth
                        
                        if 'Res_time' in lakes_in_catchment.columns:
                            # Area-weighted average residence time
                            weighted_res_time = np.average(
                                lakes_in_catchment['Res_time'], 
                                weights=lakes_equal_area.area
                            )
                            results["hydrology.lake_residence_time_days"] = weighted_res_time
                        
                        # Largest lake attributes
                        if num_lakes > 0:
                            largest_lake_idx = lakes_equal_area['area_km2'].idxmax()
                            largest_lake = lakes_in_catchment.iloc[largest_lake_idx]
                            
                            results["hydrology.largest_lake_area_km2"] = lakes_equal_area['area_km2'].max()
                            
                            if 'Lake_name' in largest_lake:
                                results["hydrology.largest_lake_name"] = largest_lake['Lake_name']
                            
                            if 'Vol_total' in largest_lake:
                                results["hydrology.largest_lake_volume_km3"] = largest_lake['Vol_total'] / 10**9
                            
                            if 'Depth_avg' in largest_lake:
                                results["hydrology.largest_lake_depth_avg_m"] = largest_lake['Depth_avg']
                            
                            if 'Shore_len' in largest_lake:
                                results["hydrology.largest_lake_shoreline_km"] = largest_lake['Shore_len']
                            
                            if 'Res_time' in largest_lake:
                                results["hydrology.largest_lake_residence_time_days"] = largest_lake['Res_time']
                    else:
                        self.logger.info("No lakes found within catchment boundary")
                        results["hydrology.lake_count"] = 0
                        results["hydrology.lake_area_km2"] = 0
                        results["hydrology.lake_area_percent"] = 0
                        results["hydrology.lake_density"] = 0
                else:
                    # For distributed catchment - process each HRU
                    catchment_results = []
                    
                    for idx, hru in catchment.iterrows():
                        hru_id = hru[self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Create a GeoDataFrame with just this HRU
                        hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                        
                        # Intersect with lakes
                        lakes_in_hru = gpd.overlay(lakes, hru_gdf, how='intersection')
                        
                        if not lakes_in_hru.empty:
                            # Convert to equal area projection for area calculations
                            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                            lakes_equal_area = lakes_in_hru.to_crs(equal_area_crs)
                            hru_equal_area = hru_gdf.to_crs(equal_area_crs)
                            
                            # Calculate HRU area in km²
                            hru_area_km2 = hru_equal_area.area.sum() / 10**6
                            
                            # Calculate lake attributes for this HRU
                            num_lakes = len(lakes_in_hru)
                            lake_area_km2 = lakes_equal_area.area.sum() / 10**6
                            lake_area_percent = (lake_area_km2 / hru_area_km2) * 100 if hru_area_km2 > 0 else 0
                            lake_density = num_lakes / hru_area_km2 if hru_area_km2 > 0 else 0
                            
                            # Add to results
                            results[f"{prefix}hydrology.lake_count"] = num_lakes
                            results[f"{prefix}hydrology.lake_area_km2"] = lake_area_km2
                            results[f"{prefix}hydrology.lake_area_percent"] = lake_area_percent
                            results[f"{prefix}hydrology.lake_density"] = lake_density
                            
                            # Calculate statistics for lake size if we have more than one lake
                            if num_lakes > 0:
                                lakes_equal_area['area_km2'] = lakes_equal_area.area / 10**6
                                results[f"{prefix}hydrology.lake_area_min_km2"] = lakes_equal_area['area_km2'].min()
                                results[f"{prefix}hydrology.lake_area_max_km2"] = lakes_equal_area['area_km2'].max()
                                results[f"{prefix}hydrology.lake_area_mean_km2"] = lakes_equal_area['area_km2'].mean()
                                
                                # Add largest lake attributes for this HRU
                                largest_lake_idx = lakes_equal_area['area_km2'].idxmax()
                                largest_lake = lakes_in_hru.iloc[largest_lake_idx]
                                
                                if 'Vol_total' in largest_lake:
                                    results[f"{prefix}hydrology.largest_lake_volume_km3"] = largest_lake['Vol_total'] / 10**9
                                
                                if 'Depth_avg' in largest_lake:
                                    results[f"{prefix}hydrology.largest_lake_depth_avg_m"] = largest_lake['Depth_avg']
                        else:
                            # No lakes in this HRU
                            results[f"{prefix}hydrology.lake_count"] = 0
                            results[f"{prefix}hydrology.lake_area_km2"] = 0
                            results[f"{prefix}hydrology.lake_area_percent"] = 0
                            results[f"{prefix}hydrology.lake_density"] = 0
            else:
                self.logger.warning(f"Clipped lakes file not found: {clipped_lakes_file}")
        
        except Exception as e:
            self.logger.error(f"Error processing lake data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_river_network_data(self, merit_basins_path: Path) -> Dict[str, Any]:
        """
        Process river network data from MERIT Basins.
        
        Args:
            merit_basins_path: Path to MERIT Basins data
            
        Returns:
            Dictionary of river network attributes
        """
        results = {}
        
        # Look for MERIT Basins river network shapefile
        rivers_dir = merit_basins_path / 'MERIT_Hydro_modified_North_America_shapes' / 'rivers'
        basins_dir = merit_basins_path / 'MERIT_Hydro_modified_North_America_shapes' / 'basins'
        
        if not rivers_dir.exists() or not basins_dir.exists():
            self.logger.warning(f"MERIT Basins directories not found: {rivers_dir} or {basins_dir}")
            return results
        
        # Find river network shapefile
        river_files = list(rivers_dir.glob("**/riv_*.shp"))
        basin_files = list(basins_dir.glob("**/cat_*.shp"))
        
        if not river_files or not basin_files:
            self.logger.warning(f"MERIT Basins shapefiles not found in {rivers_dir} or {basins_dir}")
            return results
        
        river_file = river_files[0]
        basin_file = basin_files[0]
        
        self.logger.info(f"Processing river network data from: {river_file}")
        self.logger.info(f"Processing basin data from: {basin_file}")
        
        try:
            # Read catchment shapefile
            catchment = gpd.read_file(self.catchment_path)
            
            # Create output directory for clipped data
            output_dir = self.project_dir / 'attributes' / 'hydrology' / 'rivers'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a clipped version of river network if it doesn't exist
            clipped_river_file = output_dir / f"{self.domain_name}_merit_rivers.gpkg"
            clipped_basin_file = output_dir / f"{self.domain_name}_merit_basins.gpkg"
            
            # Process river network data
            if not clipped_river_file.exists() or not clipped_basin_file.exists():
                self.logger.info(f"Clipping MERIT Basins data to catchment bounding box")
                
                # Parse bounding box coordinates
                bbox_coords = self.config.get('BOUNDING_BOX_COORDS').split('/')
                if len(bbox_coords) == 4:
                    bbox = (
                        float(bbox_coords[1]),  # lon_min
                        float(bbox_coords[2]),  # lat_min
                        float(bbox_coords[3]),  # lon_max
                        float(bbox_coords[0])   # lat_max
                    )
                    
                    # Create a spatial filter for the bbox
                    bbox_geom = box(*bbox)
                    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")
                    
                    # Read rivers within the bounding box
                    self.logger.info(f"Reading rivers within bounding box")
                    rivers = gpd.read_file(river_file, bbox=bbox)
                    
                    # Read basins within the bounding box
                    self.logger.info(f"Reading basins within bounding box")
                    basins = gpd.read_file(basin_file, bbox=bbox)
                    
                    # Save the clipped data
                    if not rivers.empty:
                        self.logger.info(f"Saving {len(rivers)} river segments to {clipped_river_file}")
                        rivers.to_file(clipped_river_file, driver="GPKG")
                    else:
                        self.logger.warning("No river segments found within bounding box")
                    
                    if not basins.empty:
                        self.logger.info(f"Saving {len(basins)} basins to {clipped_basin_file}")
                        basins.to_file(clipped_basin_file, driver="GPKG")
                    else:
                        self.logger.warning("No basins found within bounding box")
                else:
                    self.logger.error("Invalid bounding box coordinates format")
            
            # Process river network for the catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if clipped_river_file.exists() and clipped_basin_file.exists():
                # Read the clipped data
                rivers = gpd.read_file(clipped_river_file)
                basins = gpd.read_file(clipped_basin_file)
                
                if rivers.empty:
                    self.logger.info("No rivers found in the study area")
                    return results
                
                # Ensure CRS match
                if rivers.crs != catchment.crs:
                    rivers = rivers.to_crs(catchment.crs)
                
                if basins.crs != catchment.crs:
                    basins = basins.to_crs(catchment.crs)
                
                # Intersect river network with catchment
                if is_lumped:
                    # For lumped catchment - intersect with entire catchment
                    rivers_in_catchment = gpd.overlay(rivers, catchment, how='intersection')
                    
                    if not rivers_in_catchment.empty:
                        # Convert to equal area projection for length calculations
                        equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                        rivers_equal_area = rivers_in_catchment.to_crs(equal_area_crs)
                        catchment_equal_area = catchment.to_crs(equal_area_crs)
                        
                        # Calculate catchment area in km²
                        catchment_area_km2 = catchment_equal_area.area.sum() / 10**6
                        
                        # Calculate river network attributes
                        num_segments = len(rivers_in_catchment)
                        total_length_km = rivers_equal_area.length.sum() / 1000  # Convert m to km
                        river_density = total_length_km / catchment_area_km2 if catchment_area_km2 > 0 else 0
                        
                        # Add to results
                        results["hydrology.river_segment_count"] = num_segments
                        results["hydrology.river_length_km"] = total_length_km
                        results["hydrology.river_density_km_per_km2"] = river_density
                        
                        # Calculate stream order statistics if available
                        if 'str_order' in rivers_in_catchment.columns:
                            results["hydrology.stream_order_min"] = int(rivers_in_catchment['str_order'].min())
                            results["hydrology.stream_order_max"] = int(rivers_in_catchment['str_order'].max())
                            
                            # Calculate frequency of each stream order
                            for order in range(1, int(rivers_in_catchment['str_order'].max()) + 1):
                                order_segments = rivers_in_catchment[rivers_in_catchment['str_order'] == order]
                                if not order_segments.empty:
                                    order_length_km = order_segments.to_crs(equal_area_crs).length.sum() / 1000
                                    results[f"hydrology.stream_order_{order}_length_km"] = order_length_km
                                    results[f"hydrology.stream_order_{order}_segments"] = len(order_segments)
                        
                        # Calculate slope statistics if available
                        if 'slope' in rivers_in_catchment.columns:
                            # Length-weighted average slope
                            weighted_slope = np.average(
                                rivers_in_catchment['slope'], 
                                weights=rivers_equal_area.length
                            )
                            results["hydrology.river_slope_mean"] = weighted_slope
                            results["hydrology.river_slope_min"] = rivers_in_catchment['slope'].min()
                            results["hydrology.river_slope_max"] = rivers_in_catchment['slope'].max()
                        
                        # Calculate sinuosity if available
                        if 'sinuosity' in rivers_in_catchment.columns:
                            # Length-weighted average sinuosity
                            weighted_sinuosity = np.average(
                                rivers_in_catchment['sinuosity'], 
                                weights=rivers_equal_area.length
                            )
                            results["hydrology.river_sinuosity_mean"] = weighted_sinuosity
                    else:
                        self.logger.info("No river segments found within catchment boundary")
                        results["hydrology.river_segment_count"] = 0
                        results["hydrology.river_length_km"] = 0
                        results["hydrology.river_density_km_per_km2"] = 0
                else:
                    # For distributed catchment - process each HRU
                    for idx, hru in catchment.iterrows():
                        hru_id = hru[self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Create a GeoDataFrame with just this HRU
                        hru_gdf = gpd.GeoDataFrame([hru], geometry='geometry', crs=catchment.crs)
                        
                        # Intersect with rivers
                        rivers_in_hru = gpd.overlay(rivers, hru_gdf, how='intersection')
                        
                        if not rivers_in_hru.empty:
                            # Convert to equal area projection for length calculations
                            equal_area_crs = 'ESRI:102008'  # North America Albers Equal Area Conic
                            rivers_equal_area = rivers_in_hru.to_crs(equal_area_crs)
                            hru_equal_area = hru_gdf.to_crs(equal_area_crs)
                            
                            # Calculate HRU area in km²
                            hru_area_km2 = hru_equal_area.area.sum() / 10**6
                            
                            # Calculate river network attributes for this HRU
                            num_segments = len(rivers_in_hru)
                            total_length_km = rivers_equal_area.length.sum() / 1000
                            river_density = total_length_km / hru_area_km2 if hru_area_km2 > 0 else 0
                            
                            # Add to results
                            results[f"{prefix}hydrology.river_segment_count"] = num_segments
                            results[f"{prefix}hydrology.river_length_km"] = total_length_km
                            results[f"{prefix}hydrology.river_density_km_per_km2"] = river_density
                            
                            # Calculate stream order statistics if available
                            if 'str_order' in rivers_in_hru.columns:
                                results[f"{prefix}hydrology.stream_order_min"] = int(rivers_in_hru['str_order'].min())
                                results[f"{prefix}hydrology.stream_order_max"] = int(rivers_in_hru['str_order'].max())
                            
                            # Calculate slope statistics if available
                            if 'slope' in rivers_in_hru.columns:
                                # Length-weighted average slope
                                weighted_slope = np.average(
                                    rivers_in_hru['slope'], 
                                    weights=rivers_equal_area.length
                                )
                                results[f"{prefix}hydrology.river_slope_mean"] = weighted_slope
                        else:
                            # No rivers in this HRU
                            results[f"{prefix}hydrology.river_segment_count"] = 0
                            results[f"{prefix}hydrology.river_length_km"] = 0
                            results[f"{prefix}hydrology.river_density_km_per_km2"] = 0
            else:
                self.logger.warning(f"Clipped river network files not found")
        
        except Exception as e:
            self.logger.error(f"Error processing river network data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results


    def _process_climate_attributes(self) -> Dict[str, Any]:
        """
        Process climate data including temperature, precipitation, potential evapotranspiration,
        and derived climate indices.
        
        Sources:
        - WorldClim raw data: Monthly temperature, precipitation, radiation, wind, vapor pressure
        - WorldClim derived: PET, moisture index, climate indices, snow
        """
        results = {}
        
        # Find and check paths for climate data
        worldclim_path = self._get_data_path('ATTRIBUTES_WORLDCLIM_PATH', 'worldclim')
        
        self.logger.info(f"WorldClim path: {worldclim_path}")
        
        # Process raw climate variables
        raw_climate = self._process_raw_climate_variables(worldclim_path)
        results.update(raw_climate)
        
        # Process derived climate indices
        derived_climate = self._process_derived_climate_indices(worldclim_path)
        results.update(derived_climate)
        
        return results

    def _process_raw_climate_variables(self, worldclim_path: Path) -> Dict[str, Any]:
        """
        Process raw climate variables from WorldClim.
        
        Args:
            worldclim_path: Path to WorldClim data
            
        Returns:
            Dictionary of climate attributes
        """
        results = {}
        output_dir = self.project_dir / 'attributes' / 'climate'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the raw climate variables to process
        raw_variables = {
            'prec': {'unit': 'mm/month', 'description': 'Precipitation'},
            'tavg': {'unit': '°C', 'description': 'Average Temperature'},
            'tmax': {'unit': '°C', 'description': 'Maximum Temperature'},
            'tmin': {'unit': '°C', 'description': 'Minimum Temperature'},
            'srad': {'unit': 'kJ/m²/day', 'description': 'Solar Radiation'},
            'wind': {'unit': 'm/s', 'description': 'Wind Speed'},
            'vapr': {'unit': 'kPa', 'description': 'Vapor Pressure'}
        }
        
        # Process each climate variable
        for var, var_info in raw_variables.items():
            var_path = worldclim_path / 'raw' / var
            
            if not var_path.exists():
                self.logger.warning(f"WorldClim {var} directory not found: {var_path}")
                continue
            
            self.logger.info(f"Processing WorldClim {var_info['description']} ({var})")
            
            # Find all monthly files for this variable
            monthly_files = sorted(var_path.glob(f"wc2.1_30s_{var}_*.tif"))
            
            if not monthly_files:
                self.logger.warning(f"No {var} files found in {var_path}")
                continue
            
            # Process monthly data
            monthly_values = []
            monthly_attributes = {}
            
            for month_file in monthly_files:
                # Extract month number from filename (assuming format wc2.1_30s_var_MM.tif)
                month_str = os.path.basename(month_file).split('_')[-1].split('.')[0]
                try:
                    month = int(month_str)
                except ValueError:
                    self.logger.warning(f"Could not extract month from filename: {month_file}")
                    continue
                
                # Clip to catchment if not already done
                month_name = os.path.basename(month_file)
                clipped_file = output_dir / f"{self.domain_name}_{month_name}"
                
                if not clipped_file.exists():
                    self.logger.info(f"Clipping {var} for month {month} to catchment")
                    self._clip_raster_to_bbox(month_file, clipped_file)
                
                # Calculate zonal statistics for this month
                if clipped_file.exists():
                    try:
                        stats = ['mean', 'min', 'max', 'std']
                        zonal_out = zonal_stats(
                            str(self.catchment_path),
                            str(clipped_file),
                            stats=stats,
                            all_touched=True
                        )
                        
                        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                        
                        if is_lumped:
                            # For lumped catchment, add monthly values
                            if zonal_out and len(zonal_out) > 0:
                                monthly_values.append(zonal_out[0].get('mean', np.nan))
                                
                                for stat in stats:
                                    if stat in zonal_out[0]:
                                        monthly_attributes[f"climate.{var}_m{month:02d}_{stat}"] = zonal_out[0][stat]
                        else:
                            # For distributed catchment
                            catchment = gpd.read_file(self.catchment_path)
                            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                            
                            # Process each HRU
                            for i, zonal_result in enumerate(zonal_out):
                                if i < len(catchment):
                                    hru_id = catchment.iloc[i][hru_id_field]
                                    prefix = f"HRU_{hru_id}_"
                                    
                                    for stat in stats:
                                        if stat in zonal_result:
                                            monthly_attributes[f"{prefix}climate.{var}_m{month:02d}_{stat}"] = zonal_result[stat]
                                    
                                    # Keep track of monthly values for each HRU for annual calculations
                                    if i >= len(monthly_values):
                                        monthly_values.append([])
                                    if len(monthly_values[i]) < month:
                                        # Fill with NaNs if we skipped months
                                        monthly_values[i].extend([np.nan] * (month - len(monthly_values[i]) - 1))
                                        monthly_values[i].append(zonal_result.get('mean', np.nan))
                                    else:
                                        monthly_values[i][month-1] = zonal_result.get('mean', np.nan)
                    
                    except Exception as e:
                        self.logger.error(f"Error processing {var} for month {month}: {str(e)}")
            
            # Add monthly attributes to results
            results.update(monthly_attributes)
            
            # Calculate annual statistics
            if monthly_values:
                if is_lumped:
                    # For lumped catchment
                    monthly_values_clean = [v for v in monthly_values if not np.isnan(v)]
                    if monthly_values_clean:
                        # Annual mean
                        annual_mean = np.mean(monthly_values_clean)
                        results[f"climate.{var}_annual_mean"] = annual_mean
                        
                        # Annual range (max - min)
                        annual_range = np.max(monthly_values_clean) - np.min(monthly_values_clean)
                        results[f"climate.{var}_annual_range"] = annual_range
                        
                        # Seasonality metrics
                        if len(monthly_values_clean) >= 12:
                            # Calculate seasonality index (standard deviation / mean)
                            if annual_mean > 0:
                                seasonality_index = np.std(monthly_values_clean) / annual_mean
                                results[f"climate.{var}_seasonality_index"] = seasonality_index
                            
                            # For precipitation: calculate precipitation seasonality
                            if var == 'prec':
                                # Walsh & Lawler (1981) seasonality index
                                # 1/P * sum(|pi - P/12|) where P is annual precip, pi is monthly precip
                                annual_total = np.sum(monthly_values_clean)
                                if annual_total > 0:
                                    monthly_diff = [abs(p - annual_total/12) for p in monthly_values_clean]
                                    walsh_index = sum(monthly_diff) / annual_total
                                    results["climate.prec_walsh_seasonality_index"] = walsh_index
                                
                                # Markham Seasonality Index - vector-based measure
                                # Each month is treated as a vector with magnitude=precip, direction=month angle
                                if len(monthly_values_clean) == 12:
                                    month_angles = np.arange(0, 360, 30) * np.pi / 180  # in radians
                                    x_sum = sum(p * np.sin(a) for p, a in zip(monthly_values_clean, month_angles))
                                    y_sum = sum(p * np.cos(a) for p, a in zip(monthly_values_clean, month_angles))
                                    
                                    # Markham concentration index (0-1, higher = more seasonal)
                                    vector_magnitude = np.sqrt(x_sum**2 + y_sum**2)
                                    markham_index = vector_magnitude / annual_total
                                    results["climate.prec_markham_seasonality_index"] = markham_index
                                    
                                    # Markham seasonality angle (direction of concentration)
                                    seasonality_angle = np.arctan2(x_sum, y_sum) * 180 / np.pi
                                    if seasonality_angle < 0:
                                        seasonality_angle += 360
                                    
                                    # Convert to month (1-12)
                                    seasonality_month = round(seasonality_angle / 30) % 12
                                    if seasonality_month == 0:
                                        seasonality_month = 12
                                    
                                    results["climate.prec_seasonality_month"] = seasonality_month
                else:
                    # For distributed catchment
                    for i, hru_values in enumerate(monthly_values):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            hru_values_clean = [v for v in hru_values if not np.isnan(v)]
                            if hru_values_clean:
                                # Annual mean
                                annual_mean = np.mean(hru_values_clean)
                                results[f"{prefix}climate.{var}_annual_mean"] = annual_mean
                                
                                # Annual range
                                annual_range = np.max(hru_values_clean) - np.min(hru_values_clean)
                                results[f"{prefix}climate.{var}_annual_range"] = annual_range
                                
                                # Seasonality metrics
                                if len(hru_values_clean) >= 12:
                                    # Calculate seasonality index
                                    if annual_mean > 0:
                                        seasonality_index = np.std(hru_values_clean) / annual_mean
                                        results[f"{prefix}climate.{var}_seasonality_index"] = seasonality_index
                                    
                                    # For precipitation: calculate precipitation seasonality
                                    if var == 'prec':
                                        # Walsh & Lawler seasonality index
                                        annual_total = np.sum(hru_values_clean)
                                        if annual_total > 0:
                                            monthly_diff = [abs(p - annual_total/12) for p in hru_values_clean]
                                            walsh_index = sum(monthly_diff) / annual_total
                                            results[f"{prefix}climate.prec_walsh_seasonality_index"] = walsh_index
            
            # Calculate derived attributes based on variable
            if var == 'prec' and 'climate.prec_annual_mean' in results:
                # Define aridity zones based on annual precipitation
                aridity_zones = {
                    'hyperarid': (0, 100),
                    'arid': (100, 400),
                    'semiarid': (400, 600),
                    'subhumid': (600, 1000),
                    'humid': (1000, 2000),
                    'superhumid': (2000, float('inf'))
                }
                
                if is_lumped:
                    annual_precip = results['climate.prec_annual_mean']
                    for zone, (lower, upper) in aridity_zones.items():
                        if lower <= annual_precip < upper:
                            results['climate.aridity_zone_precip'] = zone
                            break
                else:
                    for i in range(len(catchment)):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        key = f"{prefix}climate.prec_annual_mean"
                        
                        if key in results:
                            annual_precip = results[key]
                            for zone, (lower, upper) in aridity_zones.items():
                                if lower <= annual_precip < upper:
                                    results[f"{prefix}climate.aridity_zone_precip"] = zone
                                    break
        
        return results

    def _process_derived_climate_indices(self, worldclim_path: Path) -> Dict[str, Any]:
        """
        Process derived climate indices from WorldClim.
        
        Args:
            worldclim_path: Path to WorldClim data
            
        Returns:
            Dictionary of climate indices
        """
        results = {}
        output_dir = self.project_dir / 'attributes' / 'climate' / 'derived'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for derived data directories
        derived_path = worldclim_path / 'derived'
        if not derived_path.exists():
            self.logger.warning(f"WorldClim derived data directory not found: {derived_path}")
            return results
        
        # Define the derived products to process
        derived_products = [
            {
                'name': 'pet',
                'path': derived_path / 'pet',
                'pattern': 'wc2.1_30s_pet_*.tif',
                'description': 'Potential Evapotranspiration',
                'unit': 'mm/day'
            },
            {
                'name': 'moisture_index',
                'path': derived_path / 'moisture_index',
                'pattern': 'wc2.1_30s_moisture_index_*.tif',
                'description': 'Moisture Index',
                'unit': '-'
            },
            {
                'name': 'snow',
                'path': derived_path / 'snow',
                'pattern': 'wc2.1_30s_snow_*.tif',
                'description': 'Snow',
                'unit': 'mm/month'
            },
            {
                'name': 'climate_index_im',
                'path': derived_path / 'climate_indices',
                'pattern': 'wc2.1_30s_climate_index_im.tif',
                'description': 'Humidity Index',
                'unit': '-'
            },
            {
                'name': 'climate_index_imr',
                'path': derived_path / 'climate_indices',
                'pattern': 'wc2.1_30s_climate_index_imr.tif',
                'description': 'Relative Humidity Index',
                'unit': '-'
            },
            {
                'name': 'climate_index_fs',
                'path': derived_path / 'climate_indices',
                'pattern': 'wc2.1_30s_climate_index_fs.tif',
                'description': 'Fraction of Precipitation as Snow',
                'unit': '-'
            }
        ]
        
        # Process each derived product
        for product in derived_products:
            product_path = product['path']
            
            if not product_path.exists():
                self.logger.warning(f"WorldClim {product['description']} directory not found: {product_path}")
                continue
            
            self.logger.info(f"Processing WorldClim {product['description']} ({product['name']})")
            
            # Find files for this product
            product_files = sorted(product_path.glob(product['pattern']))
            
            if not product_files:
                self.logger.warning(f"No {product['name']} files found in {product_path}")
                continue
            
            # Process monthly data for products with monthly files
            if len(product_files) > 1 and any('_01.' in file.name for file in product_files):
                # This is a monthly product
                monthly_values = []
                monthly_attributes = {}
                
                for month_file in product_files:
                    # Extract month number
                    if '_' in month_file.name and '.' in month_file.name:
                        month_part = month_file.name.split('_')[-1].split('.')[0]
                        try:
                            month = int(month_part)
                        except ValueError:
                            self.logger.warning(f"Could not extract month from filename: {month_file}")
                            continue
                    
                        # Clip to catchment if not already done
                        month_name = os.path.basename(month_file)
                        clipped_file = output_dir / f"{self.domain_name}_{month_name}"
                        
                        if not clipped_file.exists():
                            self.logger.info(f"Clipping {product['name']} for month {month} to catchment")
                            self._clip_raster_to_bbox(month_file, clipped_file)
                        
                        # Calculate zonal statistics for this month
                        if clipped_file.exists():
                            try:
                                stats = ['mean', 'min', 'max', 'std']
                                zonal_out = zonal_stats(
                                    str(self.catchment_path),
                                    str(clipped_file),
                                    stats=stats,
                                    all_touched=True
                                )
                                
                                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                                
                                if is_lumped:
                                    # For lumped catchment
                                    if zonal_out and len(zonal_out) > 0:
                                        monthly_values.append(zonal_out[0].get('mean', np.nan))
                                        
                                        for stat in stats:
                                            if stat in zonal_out[0]:
                                                monthly_attributes[f"climate.{product['name']}_m{month:02d}_{stat}"] = zonal_out[0][stat]
                                else:
                                    # For distributed catchment
                                    catchment = gpd.read_file(self.catchment_path)
                                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                                    
                                    # Process each HRU
                                    for i, zonal_result in enumerate(zonal_out):
                                        if i < len(catchment):
                                            hru_id = catchment.iloc[i][hru_id_field]
                                            prefix = f"HRU_{hru_id}_"
                                            
                                            for stat in stats:
                                                if stat in zonal_result:
                                                    monthly_attributes[f"{prefix}climate.{product['name']}_m{month:02d}_{stat}"] = zonal_result[stat]
                            
                            except Exception as e:
                                self.logger.error(f"Error processing {product['name']} for month {month}: {str(e)}")
                
                # Add monthly attributes to results
                results.update(monthly_attributes)
                
                # Calculate annual statistics for monthly products
                if monthly_values and is_lumped:
                    monthly_values_clean = [v for v in monthly_values if not np.isnan(v)]
                    if monthly_values_clean:
                        # Annual statistics
                        results[f"climate.{product['name']}_annual_mean"] = np.mean(monthly_values_clean)
                        results[f"climate.{product['name']}_annual_min"] = np.min(monthly_values_clean)
                        results[f"climate.{product['name']}_annual_max"] = np.max(monthly_values_clean)
                        results[f"climate.{product['name']}_annual_range"] = np.max(monthly_values_clean) - np.min(monthly_values_clean)
                        
                        # Calculate seasonality index for applicable variables
                        if product['name'] in ['pet', 'snow']:
                            annual_mean = results[f"climate.{product['name']}_annual_mean"]
                            if annual_mean > 0:
                                seasonality_index = np.std(monthly_values_clean) / annual_mean
                                results[f"climate.{product['name']}_seasonality_index"] = seasonality_index
            else:
                # This is an annual/static product (like climate indices)
                for product_file in product_files:
                    # Clip to catchment if not already done
                    file_name = os.path.basename(product_file)
                    clipped_file = output_dir / f"{self.domain_name}_{file_name}"
                    
                    if not clipped_file.exists():
                        self.logger.info(f"Clipping {product['name']} to catchment")
                        self._clip_raster_to_bbox(product_file, clipped_file)
                    
                    # Calculate zonal statistics
                    if clipped_file.exists():
                        try:
                            stats = ['mean', 'min', 'max', 'std']
                            zonal_out = zonal_stats(
                                str(self.catchment_path),
                                str(clipped_file),
                                stats=stats,
                                all_touched=True
                            )
                            
                            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                            
                            if is_lumped:
                                # For lumped catchment
                                if zonal_out and len(zonal_out) > 0:
                                    for stat in stats:
                                        if stat in zonal_out[0]:
                                            results[f"climate.{product['name']}_{stat}"] = zonal_out[0][stat]
                            else:
                                # For distributed catchment
                                catchment = gpd.read_file(self.catchment_path)
                                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                                
                                # Process each HRU
                                for i, zonal_result in enumerate(zonal_out):
                                    if i < len(catchment):
                                        hru_id = catchment.iloc[i][hru_id_field]
                                        prefix = f"HRU_{hru_id}_"
                                        
                                        for stat in stats:
                                            if stat in zonal_result:
                                                results[f"{prefix}climate.{product['name']}_{stat}"] = zonal_result[stat]
                        
                        except Exception as e:
                            self.logger.error(f"Error processing {product['name']}: {str(e)}")
        
        # Calculate derived climate indicators if we have the necessary data
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        
        if is_lumped:
            # Aridity index (PET/P) - if we have both annual precipitation and PET
            if "climate.prec_annual_mean" in results and "climate.pet_annual_mean" in results:
                precip = results["climate.prec_annual_mean"]
                pet = results["climate.pet_annual_mean"]
                
                if precip > 0:
                    aridity_index = pet / precip
                    results["climate.aridity_index"] = aridity_index
                    
                    # Classify aridity zone based on UNEP criteria
                    if aridity_index < 0.03:
                        results["climate.aridity_zone"] = "hyperhumid"
                    elif aridity_index < 0.2:
                        results["climate.aridity_zone"] = "humid"
                    elif aridity_index < 0.5:
                        results["climate.aridity_zone"] = "subhumid"
                    elif aridity_index < 0.65:
                        results["climate.aridity_zone"] = "dry_subhumid"
                    elif aridity_index < 1.0:
                        results["climate.aridity_zone"] = "semiarid"
                    elif aridity_index < 3.0:
                        results["climate.aridity_zone"] = "arid"
                    else:
                        results["climate.aridity_zone"] = "hyperarid"
            
            # Snow fraction - total annual snow / total annual precipitation
            if "climate.snow_annual_mean" in results and "climate.prec_annual_mean" in results:
                snow = results["climate.snow_annual_mean"] * 12  # Convert mean to annual total
                precip = results["climate.prec_annual_mean"] * 12
                
                if precip > 0:
                    snow_fraction = snow / precip
                    results["climate.snow_fraction"] = snow_fraction
        else:
            # For distributed catchment, calculate indices for each HRU
            catchment = gpd.read_file(self.catchment_path)
            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
            
            for i in range(len(catchment)):
                hru_id = catchment.iloc[i][hru_id_field]
                prefix = f"HRU_{hru_id}_"
                
                # Aridity index
                prec_key = f"{prefix}climate.prec_annual_mean"
                pet_key = f"{prefix}climate.pet_annual_mean"
                
                if prec_key in results and pet_key in results:
                    precip = results[prec_key]
                    pet = results[pet_key]
                    
                    if precip > 0:
                        aridity_index = pet / precip
                        results[f"{prefix}climate.aridity_index"] = aridity_index
                        
                        # Classify aridity zone
                        if aridity_index < 0.03:
                            results[f"{prefix}climate.aridity_zone"] = "hyperhumid"
                        elif aridity_index < 0.2:
                            results[f"{prefix}climate.aridity_zone"] = "humid"
                        elif aridity_index < 0.5:
                            results[f"{prefix}climate.aridity_zone"] = "subhumid"
                        elif aridity_index < 0.65:
                            results[f"{prefix}climate.aridity_zone"] = "dry_subhumid"
                        elif aridity_index < 1.0:
                            results[f"{prefix}climate.aridity_zone"] = "semiarid"
                        elif aridity_index < 3.0:
                            results[f"{prefix}climate.aridity_zone"] = "arid"
                        else:
                            results[f"{prefix}climate.aridity_zone"] = "hyperarid"
                
                # Snow fraction
                snow_key = f"{prefix}climate.snow_annual_mean"
                if snow_key in results and prec_key in results:
                    snow = results[snow_key] * 12
                    precip = results[prec_key] * 12
                    
                    if precip > 0:
                        snow_fraction = snow / precip
                        results[f"{prefix}climate.snow_fraction"] = snow_fraction
        
        return results


    def _process_landcover_vegetation_attributes(self) -> Dict[str, Any]:
        """
        Process land cover and vegetation attributes from multiple datasets,
        including GLCLU2019, MODIS LAI, and others.
        
        Returns:
            Dict[str, Any]: Dictionary of land cover and vegetation attributes
        """
        results = {}
        
        # Process GLCLU2019 land cover classification
        glclu_results = self._process_glclu2019_landcover()
        results.update(glclu_results)
        
        # Process LAI (Leaf Area Index) data
        lai_results = self._process_lai_data()
        results.update(lai_results)
        
        # Process forest height data
        forest_results = self._process_forest_height()
        results.update(forest_results)
        
        return results

    def _process_glclu2019_landcover(self) -> Dict[str, Any]:
        """
        Process land cover data from the GLCLU2019 dataset.
        
        Returns:
            Dict[str, Any]: Dictionary of land cover attributes
        """
        results = {}
        
        # Define path to GLCLU2019 data
        glclu_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/glclu2019/raw")
        main_tif = glclu_path / "glclu2019_map.tif"
        
        # Check if file exists
        if not main_tif.exists():
            self.logger.warning(f"GLCLU2019 file not found: {main_tif}")
            return results
        
        # Create cache directory and define cache file
        cache_dir = self.project_dir / 'cache' / 'landcover'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_glclu2019_results.pickle"
        
        # Check if cached results exist
        if cache_file.exists():
            self.logger.info(f"Loading cached GLCLU2019 results from {cache_file}")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached GLCLU2019 results: {str(e)}")
        
        self.logger.info("Processing GLCLU2019 land cover data")
        
        try:
            # Define the land cover classes from GLCLU2019
            landcover_classes = {
                1: 'true_desert',
                2: 'semi_arid',
                3: 'dense_short_vegetation',
                4: 'open_tree_cover',
                5: 'dense_tree_cover',
                6: 'tree_cover_gain',
                7: 'tree_cover_loss',
                8: 'salt_pan',
                9: 'wetland_sparse_vegetation',
                10: 'wetland_dense_short_vegetation',
                11: 'wetland_open_tree_cover',
                12: 'wetland_dense_tree_cover',
                13: 'wetland_tree_cover_gain',
                14: 'wetland_tree_cover_loss',
                15: 'ice',
                16: 'water',
                17: 'cropland',
                18: 'built_up',
                19: 'ocean',
                20: 'no_data'
            }
            
            # Define broader categories for aggregation
            category_mapping = {
                'forest': [4, 5, 6, 11, 12, 13],  # Tree cover classes
                'wetland': [9, 10, 11, 12, 13, 14],  # Wetland classes
                'barren': [1, 2, 8],  # Desert and semi-arid classes
                'water': [15, 16, 19],  # Water and ice classes
                'agricultural': [17],  # Cropland
                'urban': [18],  # Built-up areas
                'other_vegetation': [3]  # Other vegetation
            }
            
            # Calculate zonal statistics (categorical)
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(main_tif), 
                categorical=True, 
                nodata=20,  # 'no_data' class
                all_touched=True
            )
            
            # Check if we have valid results
            if not zonal_out:
                self.logger.warning("No valid zonal statistics for GLCLU2019")
                return results
            
            # Check if we're dealing with lumped or distributed catchment
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
            
            if is_lumped:
                # For lumped catchment
                if zonal_out and len(zonal_out) > 0:
                    # Calculate total pixels (excluding 'no_data')
                    valid_pixels = sum(count for class_id, count in zonal_out[0].items() 
                                    if class_id is not None and class_id != 20)
                    
                    if valid_pixels > 0:
                        # Calculate fractions for each land cover class
                        for class_id, class_name in landcover_classes.items():
                            if class_id == 20:  # Skip 'no_data'
                                continue
                            
                            pixel_count = zonal_out[0].get(class_id, 0)
                            fraction = pixel_count / valid_pixels if valid_pixels > 0 else 0
                            results[f"landcover.{class_name}_fraction"] = fraction
                        
                        # Calculate broader category fractions
                        for category, class_ids in category_mapping.items():
                            category_count = sum(zonal_out[0].get(class_id, 0) for class_id in class_ids)
                            fraction = category_count / valid_pixels if valid_pixels > 0 else 0
                            results[f"landcover.{category}_fraction"] = fraction
                        
                        # Find dominant land cover class
                        dominant_class_id = max(
                            ((class_id, count) for class_id, count in zonal_out[0].items() 
                            if class_id is not None and class_id != 20),
                            key=lambda x: x[1],
                            default=(None, 0)
                        )[0]
                        
                        if dominant_class_id is not None:
                            results["landcover.dominant_class"] = landcover_classes[dominant_class_id]
                            results["landcover.dominant_class_id"] = dominant_class_id
                            results["landcover.dominant_fraction"] = zonal_out[0][dominant_class_id] / valid_pixels
                        
                        # Calculate land cover diversity (Shannon entropy)
                        shannon_entropy = 0
                        for class_id, count in zonal_out[0].items():
                            if class_id is not None and class_id != 20 and count > 0:
                                p = count / valid_pixels
                                shannon_entropy -= p * np.log(p)
                        results["landcover.diversity_index"] = shannon_entropy
            else:
                # For distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                for i, zonal_result in enumerate(zonal_out):
                    if i < len(catchment):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Calculate total pixels (excluding 'no_data')
                        valid_pixels = sum(count for class_id, count in zonal_result.items() 
                                        if class_id is not None and class_id != 20)
                        
                        if valid_pixels > 0:
                            # Calculate fractions for each land cover class
                            for class_id, class_name in landcover_classes.items():
                                if class_id == 20:  # Skip 'no_data'
                                    continue
                                
                                pixel_count = zonal_result.get(class_id, 0)
                                fraction = pixel_count / valid_pixels if valid_pixels > 0 else 0
                                results[f"{prefix}landcover.{class_name}_fraction"] = fraction
                            
                            # Calculate broader category fractions
                            for category, class_ids in category_mapping.items():
                                category_count = sum(zonal_result.get(class_id, 0) for class_id in class_ids)
                                fraction = category_count / valid_pixels if valid_pixels > 0 else 0
                                results[f"{prefix}landcover.{category}_fraction"] = fraction
                            
                            # Find dominant land cover class
                            dominant_class_id = max(
                                ((class_id, count) for class_id, count in zonal_result.items() 
                                if class_id is not None and class_id != 20),
                                key=lambda x: x[1],
                                default=(None, 0)
                            )[0]
                            
                            if dominant_class_id is not None:
                                results[f"{prefix}landcover.dominant_class"] = landcover_classes[dominant_class_id]
                                results[f"{prefix}landcover.dominant_class_id"] = dominant_class_id
                                results[f"{prefix}landcover.dominant_fraction"] = zonal_result[dominant_class_id] / valid_pixels
                            
                            # Calculate land cover diversity (Shannon entropy)
                            shannon_entropy = 0
                            for class_id, count in zonal_result.items():
                                if class_id is not None and class_id != 20 and count > 0:
                                    p = count / valid_pixels
                                    shannon_entropy -= p * np.log(p)
                            results[f"{prefix}landcover.diversity_index"] = shannon_entropy
            
            # Try to read additional information from the Excel legend if available
            legend_file = glclu_path / "legend_0.xlsx"
            if legend_file.exists():
                try:
                    import pandas as pd
                    legend_df = pd.read_excel(legend_file)
                    self.logger.info(f"Loaded GLCLU2019 legend with {len(legend_df)} entries")
                    # Add any additional processing based on legend information
                except Exception as e:
                    self.logger.warning(f"Error reading GLCLU2019 legend: {str(e)}")
            
            # Cache results
            try:
                self.logger.info(f"Caching GLCLU2019 results to {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except Exception as e:
                self.logger.warning(f"Error caching GLCLU2019 results: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing GLCLU2019 data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_lai_data(self) -> Dict[str, Any]:
        """
        Process Leaf Area Index (LAI) data from MODIS.
        
        Returns:
            Dict[str, Any]: Dictionary of LAI attributes
        """
        results = {}
        
        # Define path to LAI data
        lai_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/lai/monthly_average_2013_2023")
        
        # Check if we should use water-masked LAI (preferred) or non-water-masked
        use_water_mask = self.config.get('USE_WATER_MASKED_LAI', True)
        
        if use_water_mask:
            lai_folder = lai_path / 'monthly_lai_with_water_mask'
        else:
            lai_folder = lai_path / 'monthly_lai_no_water_mask'
        
        # Check if folder exists
        if not lai_folder.exists():
            self.logger.warning(f"LAI folder not found: {lai_folder}")
            # Try alternative folder if preferred one doesn't exist
            if use_water_mask:
                lai_folder = lai_path / 'monthly_lai_no_water_mask'
            else:
                lai_folder = lai_path / 'monthly_lai_with_water_mask'
            
            if not lai_folder.exists():
                self.logger.warning(f"Alternative LAI folder not found: {lai_folder}")
                return results
            
            self.logger.info(f"Using alternative LAI folder: {lai_folder}")
        
        # Create cache directory and define cache file
        cache_dir = self.project_dir / 'cache' / 'lai'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_lai_results.pickle"
        
        # Check if cached results exist
        if cache_file.exists():
            self.logger.info(f"Loading cached LAI results from {cache_file}")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached LAI results: {str(e)}")
        
        self.logger.info("Processing LAI data")
        
        try:
            # Find monthly LAI files
            lai_files = sorted(lai_folder.glob("*.tif"))
            
            if not lai_files:
                self.logger.warning(f"No LAI files found in {lai_folder}")
                return results
            
            # Process each monthly LAI file
            monthly_lai_values = []  # For seasonal analysis
            
            for lai_file in lai_files:
                # Extract month from filename (e.g., "2013_2023_01_MOD_Grid_MOD15A2H_Lai_500m.tif")
                filename = lai_file.name
                if '_' in filename:
                    try:
                        month = int(filename.split('_')[2])
                    except (IndexError, ValueError):
                        self.logger.warning(f"Could not extract month from LAI filename: {filename}")
                        continue
                else:
                    self.logger.warning(f"Unexpected LAI filename format: {filename}")
                    continue
                
                self.logger.info(f"Processing LAI for month {month}")
                
                # Calculate zonal statistics
                zonal_out = zonal_stats(
                    str(self.catchment_path), 
                    str(lai_file), 
                    stats=['mean', 'min', 'max', 'std', 'median', 'count'], 
                    nodata=255,  # Common MODIS no-data value
                    all_touched=True
                )
                
                # Check if we have valid results
                if not zonal_out:
                    self.logger.warning(f"No valid zonal statistics for LAI month {month}")
                    continue
                
                # Get scale and offset
                scale, offset = self._read_scale_and_offset(lai_file)
                if scale is None:
                    scale = 0.1  # Default scale for MODIS LAI (convert to actual LAI values)
                if offset is None:
                    offset = 0
                
                # Check if we're dealing with lumped or distributed catchment
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
                
                if is_lumped:
                    # For lumped catchment
                    if zonal_out and len(zonal_out) > 0 and 'mean' in zonal_out[0]:
                        # Apply scale and offset
                        for stat in ['mean', 'min', 'max', 'std', 'median']:
                            if stat in zonal_out[0] and zonal_out[0][stat] is not None:
                                value = zonal_out[0][stat] * scale + offset
                                results[f"vegetation.lai_month{month:02d}_{stat}"] = value
                        
                        # Store mean LAI for seasonal analysis
                        if 'mean' in zonal_out[0] and zonal_out[0]['mean'] is not None:
                            monthly_lai_values.append((month, zonal_out[0]['mean'] * scale + offset))
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            # Apply scale and offset
                            for stat in ['mean', 'min', 'max', 'std', 'median']:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    value = zonal_result[stat] * scale + offset
                                    results[f"{prefix}vegetation.lai_month{month:02d}_{stat}"] = value
            
            # Calculate seasonal metrics from monthly values
            if is_lumped and monthly_lai_values:
                # Sort by month for correct seasonal analysis
                monthly_lai_values.sort(key=lambda x: x[0])
                
                # Extract LAI values
                lai_values = [v[1] for v in monthly_lai_values if not np.isnan(v[1])]
                
                if lai_values:
                    # Annual LAI statistics
                    results["vegetation.lai_annual_min"] = min(lai_values)
                    results["vegetation.lai_annual_mean"] = sum(lai_values) / len(lai_values)
                    results["vegetation.lai_annual_max"] = max(lai_values)
                    results["vegetation.lai_annual_range"] = max(lai_values) - min(lai_values)
                    
                    # Seasonal amplitude (difference between max and min)
                    results["vegetation.lai_seasonal_amplitude"] = results["vegetation.lai_annual_range"]
                    
                    # Calculate seasonality index (coefficient of variation)
                    if results["vegetation.lai_annual_mean"] > 0:
                        lai_std = np.std(lai_values)
                        results["vegetation.lai_seasonality_index"] = lai_std / results["vegetation.lai_annual_mean"]
                    
                    # Growing season metrics
                    # Define growing season as months with LAI >= 50% of annual range above minimum
                    threshold = results["vegetation.lai_annual_min"] + 0.5 * results["vegetation.lai_annual_range"]
                    growing_season_months = [month for month, lai in monthly_lai_values if lai >= threshold]
                    
                    if growing_season_months:
                        results["vegetation.growing_season_months"] = len(growing_season_months)
                        results["vegetation.growing_season_start"] = min(growing_season_months)
                        results["vegetation.growing_season_end"] = max(growing_season_months)
                    
                    # Phenology timing
                    max_month = [m for m, l in monthly_lai_values if l == results["vegetation.lai_annual_max"]]
                    min_month = [m for m, l in monthly_lai_values if l == results["vegetation.lai_annual_min"]]
                    
                    if max_month:
                        results["vegetation.lai_peak_month"] = max_month[0]
                    if min_month:
                        results["vegetation.lai_min_month"] = min_month[0]
                    
                    # Vegetation density classification
                    if results["vegetation.lai_annual_max"] < 1:
                        results["vegetation.density_class"] = "sparse"
                    elif results["vegetation.lai_annual_max"] < 3:
                        results["vegetation.density_class"] = "moderate"
                    else:
                        results["vegetation.density_class"] = "dense"
            
            # Cache results
            try:
                self.logger.info(f"Caching LAI results to {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except Exception as e:
                self.logger.warning(f"Error caching LAI results: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing LAI data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_forest_height(self) -> Dict[str, Any]:
        """
        Process forest height data.
        
        Returns:
            Dict[str, Any]: Dictionary of forest height attributes
        """
        results = {}
        
        # Define path to forest height data
        forest_dir = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/forest_height/raw")
        
        # Check for 2000 and 2020 forest height files
        forest_2000_file = forest_dir / "forest_height_2000.tif"
        forest_2020_file = forest_dir / "forest_height_2020.tif"
        
        # Create cache directory and define cache file
        cache_dir = self.project_dir / 'cache' / 'forest_height'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.domain_name}_forest_height_results.pickle"
        
        # Check if cached results exist
        if cache_file.exists():
            self.logger.info(f"Loading cached forest height results from {cache_file}")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading cached forest height results: {str(e)}")
        
        self.logger.info("Processing forest height data")
        
        try:
            years = []
            
            # Check which years are available
            if forest_2000_file.exists():
                years.append(("2000", forest_2000_file))
            
            if forest_2020_file.exists():
                years.append(("2020", forest_2020_file))
            
            if not years:
                self.logger.warning("No forest height files found")
                return results
            
            # Process each year's forest height data
            for year_str, forest_file in years:
                self.logger.info(f"Processing forest height for year {year_str}")
                
                # Calculate zonal statistics
                zonal_out = zonal_stats(
                    str(self.catchment_path), 
                    str(forest_file), 
                    stats=['mean', 'min', 'max', 'std', 'median', 'count'], 
                    nodata=-9999,  # Common no-data value for forest height
                    all_touched=True
                )
                
                # Check if we have valid results
                if not zonal_out:
                    self.logger.warning(f"No valid zonal statistics for forest height year {year_str}")
                    continue
                
                # Get scale and offset
                scale, offset = self._read_scale_and_offset(forest_file)
                if scale is None:
                    scale = 1  # Default scale
                if offset is None:
                    offset = 0
                
                # Check if we're dealing with lumped or distributed catchment
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
                
                if is_lumped:
                    # For lumped catchment
                    if zonal_out and len(zonal_out) > 0 and 'mean' in zonal_out[0]:
                        # Apply scale and offset to basic statistics
                        for stat in ['mean', 'min', 'max', 'std', 'median']:
                            if stat in zonal_out[0] and zonal_out[0][stat] is not None:
                                value = zonal_out[0][stat] * scale + offset
                                results[f"forest.height_{year_str}_{stat}"] = value
                        
                        # Calculate forest coverage
                        if 'count' in zonal_out[0] and zonal_out[0]['count'] is not None:
                            # Try to count total pixels in catchment
                            with rasterio.open(str(forest_file)) as src:
                                catchment = gpd.read_file(self.catchment_path)
                                geom = catchment.geometry.values[0]
                                if catchment.crs != src.crs:
                                    # Transform catchment to raster CRS
                                    catchment = catchment.to_crs(src.crs)
                                    geom = catchment.geometry.values[0]
                                
                                # Get the mask of pixels within the catchment
                                mask, transform = rasterio.mask.mask(src, [geom], crop=False, nodata=0)
                                total_pixels = np.sum(mask[0] != 0)
                                
                                if total_pixels > 0:
                                    # Calculate forest coverage percentage (non-zero forest height pixels)
                                    forest_pixels = zonal_out[0]['count']
                                    forest_coverage = (forest_pixels / total_pixels) * 100
                                    results[f"forest.coverage_{year_str}_percent"] = forest_coverage
                        
                        # Calculate height distribution metrics
                        if forest_file.exists():
                            with rasterio.open(str(forest_file)) as src:
                                # Mask the raster to the catchment
                                catchment = gpd.read_file(self.catchment_path)
                                if catchment.crs != src.crs:
                                    catchment = catchment.to_crs(src.crs)
                                
                                # Get masked data
                                masked_data, _ = rasterio.mask.mask(src, catchment.geometry, crop=True, nodata=-9999)
                                
                                # Flatten and filter to valid values
                                valid_data = masked_data[masked_data != -9999]
                                
                                if len(valid_data) > 10:  # Only calculate if we have enough data points
                                    # Calculate additional statistics
                                    results[f"forest.height_{year_str}_skew"] = float(skew(valid_data))
                                    results[f"forest.height_{year_str}_kurtosis"] = float(kurtosis(valid_data))
                                    
                                    # Calculate percentiles
                                    percentiles = [5, 10, 25, 50, 75, 90, 95]
                                    for p in percentiles:
                                        results[f"forest.height_{year_str}_p{p}"] = float(np.percentile(valid_data, p))
                                    
                                    # Calculate canopy density metrics
                                    # Assume forest height threshold of 5m for "canopy"
                                    canopy_threshold = 5.0
                                    canopy_pixels = np.sum(valid_data >= canopy_threshold)
                                    results[f"forest.canopy_density_{year_str}"] = float(canopy_pixels / len(valid_data))
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            # Apply scale and offset to basic statistics
                            for stat in ['mean', 'min', 'max', 'std', 'median']:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    value = zonal_result[stat] * scale + offset
                                    results[f"{prefix}forest.height_{year_str}_{stat}"] = value
                            
                            # Calculate forest coverage for each HRU
                            if 'count' in zonal_result and zonal_result['count'] is not None:
                                # Try to count total pixels in this HRU
                                with rasterio.open(str(forest_file)) as src:
                                    hru_gdf = catchment.iloc[i:i+1]
                                    if hru_gdf.crs != src.crs:
                                        hru_gdf = hru_gdf.to_crs(src.crs)
                                    
                                    # Get the mask of pixels within the HRU
                                    mask, transform = rasterio.mask.mask(src, hru_gdf.geometry, crop=False, nodata=0)
                                    total_pixels = np.sum(mask[0] != 0)
                                    
                                    if total_pixels > 0:
                                        # Calculate forest coverage percentage (non-zero forest height pixels)
                                        forest_pixels = zonal_result['count']
                                        forest_coverage = (forest_pixels / total_pixels) * 100
                                        results[f"{prefix}forest.coverage_{year_str}_percent"] = forest_coverage
            
            # Cache results
            try:
                self.logger.info(f"Caching forest height results to {cache_file}")
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)
            except Exception as e:
                self.logger.warning(f"Error caching forest height results: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing forest height data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return results

    def _process_irrigation_attributes(self) -> Dict[str, Any]:
        """
        Process irrigation data including irrigated area percentages.
        
        Sources:
        - LGRIP30: Global Landsat-based rainfed and irrigated cropland map at 30m resolution
        
        The LGRIP30 dataset classifies areas into:
        0: Non-cropland
        1: Rainfed cropland
        2: Irrigated cropland
        3: Paddy cropland (flooded irrigation, primarily rice)
        """
        results = {}
        
        # Find and check paths for irrigation data
        lgrip_path = self._get_data_path('ATTRIBUTES_LGRIP_PATH', 'lgrip30')
        
        self.logger.info(f"LGRIP path: {lgrip_path}")
        
        # Look for the LGRIP30 agriculture raster
        lgrip_file = lgrip_path / 'raw' / 'lgrip30_agriculture.tif'
        
        if not lgrip_file.exists():
            self.logger.warning(f"LGRIP30 agriculture file not found: {lgrip_file}")
            return results
        
        self.logger.info(f"Processing irrigation data from: {lgrip_file}")
        
        # Create output directory for clipped data
        output_dir = self.project_dir / 'attributes' / 'irrigation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clip irrigation data to catchment if not already done
        clipped_file = output_dir / f"{self.domain_name}_lgrip30_agriculture.tif"
        
        if not clipped_file.exists():
            self.logger.info(f"Clipping irrigation data to catchment bounding box")
            self._clip_raster_to_bbox(lgrip_file, clipped_file)
        
        # Process the clipped irrigation data
        if clipped_file.exists():
            try:
                # LGRIP30 is a categorical raster with these classes:
                lgrip_classes = {
                    0: 'non_cropland',
                    1: 'rainfed_cropland',
                    2: 'irrigated_cropland',
                    3: 'paddy_cropland'
                }
                
                # Calculate zonal statistics (categorical)
                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(clipped_file),
                    categorical=True,
                    all_touched=True
                )
                
                # Process results
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                
                if is_lumped:
                    # For lumped catchment
                    if zonal_out and len(zonal_out) > 0:
                        # Calculate total pixel count (area)
                        total_pixels = sum(count for class_id, count in zonal_out[0].items() 
                                        if class_id is not None)
                        
                        if total_pixels > 0:
                            # Calculate class fractions and add to results
                            for class_id, class_name in lgrip_classes.items():
                                pixel_count = zonal_out[0].get(class_id, 0)
                                fraction = pixel_count / total_pixels
                                results[f"irrigation.{class_name}_fraction"] = fraction
                            
                            # Calculate combined cropland area (all types)
                            cropland_pixels = sum(zonal_out[0].get(i, 0) for i in [1, 2, 3])
                            cropland_fraction = cropland_pixels / total_pixels
                            results["irrigation.cropland_fraction"] = cropland_fraction
                            
                            # Calculate combined irrigated area (irrigated + paddy)
                            irrigated_pixels = sum(zonal_out[0].get(i, 0) for i in [2, 3])
                            irrigated_fraction = irrigated_pixels / total_pixels
                            results["irrigation.irrigated_fraction"] = irrigated_fraction
                            
                            # Calculate irrigation intensity (irrigated area / total cropland)
                            if cropland_pixels > 0:
                                irrigation_intensity = irrigated_pixels / cropland_pixels
                                results["irrigation.irrigation_intensity"] = irrigation_intensity
                            
                            # Calculate dominant agricultural type
                            ag_classes = {
                                1: 'rainfed_cropland',
                                2: 'irrigated_cropland',
                                3: 'paddy_cropland'
                            }
                            
                            # Find the dominant agricultural class
                            dominant_class = max(
                                ag_classes.keys(),
                                key=lambda k: zonal_out[0].get(k, 0),
                                default=None
                            )
                            
                            if dominant_class and zonal_out[0].get(dominant_class, 0) > 0:
                                results["irrigation.dominant_agriculture"] = ag_classes[dominant_class]
                                dom_fraction = zonal_out[0].get(dominant_class, 0) / total_pixels
                                results["irrigation.dominant_agriculture_fraction"] = dom_fraction
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    # Process each HRU
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            # Calculate total pixel count (area) for this HRU
                            total_pixels = sum(count for class_id, count in zonal_result.items() 
                                            if class_id is not None)
                            
                            if total_pixels > 0:
                                # Calculate class fractions and add to results
                                for class_id, class_name in lgrip_classes.items():
                                    pixel_count = zonal_result.get(class_id, 0)
                                    fraction = pixel_count / total_pixels
                                    results[f"{prefix}irrigation.{class_name}_fraction"] = fraction
                                
                                # Calculate combined cropland area (all types)
                                cropland_pixels = sum(zonal_result.get(i, 0) for i in [1, 2, 3])
                                cropland_fraction = cropland_pixels / total_pixels
                                results[f"{prefix}irrigation.cropland_fraction"] = cropland_fraction
                                
                                # Calculate combined irrigated area (irrigated + paddy)
                                irrigated_pixels = sum(zonal_result.get(i, 0) for i in [2, 3])
                                irrigated_fraction = irrigated_pixels / total_pixels
                                results[f"{prefix}irrigation.irrigated_fraction"] = irrigated_fraction
                                
                                # Calculate irrigation intensity
                                if cropland_pixels > 0:
                                    irrigation_intensity = irrigated_pixels / cropland_pixels
                                    results[f"{prefix}irrigation.irrigation_intensity"] = irrigation_intensity
                                
                                # Calculate dominant agricultural type
                                ag_classes = {
                                    1: 'rainfed_cropland',
                                    2: 'irrigated_cropland',
                                    3: 'paddy_cropland'
                                }
                                
                                # Find the dominant agricultural class
                                dominant_class = max(
                                    ag_classes.keys(),
                                    key=lambda k: zonal_result.get(k, 0),
                                    default=None
                                )
                                
                                if dominant_class and zonal_result.get(dominant_class, 0) > 0:
                                    results[f"{prefix}irrigation.dominant_agriculture"] = ag_classes[dominant_class]
                                    dom_fraction = zonal_result.get(dominant_class, 0) / total_pixels
                                    results[f"{prefix}irrigation.dominant_agriculture_fraction"] = dom_fraction
                
                # If we have any irrigation, add a simple binary flag
                if is_lumped:
                    has_irrigation = results.get("irrigation.irrigated_fraction", 0) > 0.01  # >1% irrigated
                    results["irrigation.has_significant_irrigation"] = has_irrigation
                else:
                    for i in range(len(catchment)):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        irr_key = f"{prefix}irrigation.irrigated_fraction"
                        
                        if irr_key in results:
                            has_irrigation = results[irr_key] > 0.01  # >1% irrigated
                            results[f"{prefix}irrigation.has_significant_irrigation"] = has_irrigation
                
            except Exception as e:
                self.logger.error(f"Error processing irrigation data: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
        else:
            self.logger.warning(f"Clipped irrigation file not found or could not be created: {clipped_file}")
        
        return results


    def _get_data_path(self, config_key: str, default_subfolder: str) -> Path:
        """
        Get the path to a data source, checking multiple possible locations.
        
        Args:
            config_key: Configuration key to check for a specified path
            default_subfolder: Default subfolder to use within attribute data directories
            
        Returns:
            Path to the data source
        """
        # Check if path is explicitly specified in config
        path = self.config.get(config_key)
        
        if path and path != 'default':
            return Path(path)
        
        # Check the attribute data directory from config
        attr_data_dir = self.config.get('ATTRIBUTES_DATA_DIR')
        if attr_data_dir and attr_data_dir != 'default':
            attr_path = Path(attr_data_dir) / default_subfolder
            if attr_path.exists():
                return attr_path
        
        # Check the specific North America data directory
        na_path = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial") / default_subfolder
        if na_path.exists():
            return na_path
        
        # Fall back to project data directory
        proj_path = self.data_dir / 'geospatial-data' / default_subfolder
        return proj_path


    def _process_forest_height_attributes(self) -> Dict[str, Any]:
        """
        Acquire, clip, and process forest height data for the catchment.
        """
        results = {}
        
        # Create forest height directory
        forest_dir = self.project_dir / 'attributes' / 'forest_height'
        forest_dir.mkdir(parents=True, exist_ok=True)
        
        # Find source forest height data
        forest_source = self._find_forest_height_source()
        
        if not forest_source:
            self.logger.warning("No forest height data found")
            return results
        
        # Clip forest height data to bounding box if not already done
        clipped_forest = forest_dir / f"{self.domain_name}_forest_height.tif"
        
        if not clipped_forest.exists():
            self.logger.info(f"Clipping forest height data to catchment bounding box")
            self._clip_raster_to_bbox(forest_source, clipped_forest)
        
        # Process the clipped forest height data
        if clipped_forest.exists():
            self.logger.info(f"Processing forest height statistics from {clipped_forest}")
            
            # Calculate zonal statistics
            stats = ['min', 'mean', 'max', 'std', 'median', 'count']
            
            try:
                zonal_out = zonal_stats(
                    str(self.catchment_path),
                    str(clipped_forest),
                    stats=stats,
                    all_touched=True,
                    nodata=-9999  # Common no-data value for forest height datasets
                )
                
                # Process results
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
                
                if is_lumped:
                    # For lumped catchment
                    if zonal_out and len(zonal_out) > 0:
                        for stat, value in zonal_out[0].items():
                            if value is not None:
                                results[f"forest.height_{stat}"] = value
                        
                        # Calculate forest coverage
                        if 'count' in zonal_out[0] and zonal_out[0]['count'] is not None:
                            with rasterio.open(str(clipped_forest)) as src:
                                # Calculate total number of pixels in the catchment
                                total_pixels = self._count_pixels_in_catchment(src)
                                
                                if total_pixels > 0:
                                    # Count non-zero forest height pixels
                                    forest_pixels = zonal_out[0]['count']
                                    # Calculate forest coverage percentage
                                    forest_coverage = (forest_pixels / total_pixels) * 100
                                    results["forest.coverage_percent"] = forest_coverage
                        
                        # Calculate forest height distribution metrics
                        self._add_forest_distribution_metrics(clipped_forest, results)
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    # Process each HRU
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            for stat, value in zonal_result.items():
                                if value is not None:
                                    results[f"{prefix}forest.height_{stat}"] = value
                            
                            # Calculate forest coverage for this HRU
                            if 'count' in zonal_result and zonal_result['count'] is not None:
                                with rasterio.open(str(clipped_forest)) as src:
                                    # Calculate total number of pixels in this HRU
                                    total_pixels = self._count_pixels_in_hru(src, catchment.iloc[i:i+1])
                                    
                                    if total_pixels > 0:
                                        # Count non-zero forest height pixels
                                        forest_pixels = zonal_result['count']
                                        # Calculate forest coverage percentage
                                        forest_coverage = (forest_pixels / total_pixels) * 100
                                        results[f"{prefix}forest.coverage_percent"] = forest_coverage
                            
                            # Calculate forest height distribution metrics for this HRU
                            self._add_forest_distribution_metrics(
                                clipped_forest,
                                results,
                                hru_geometry=catchment.iloc[i:i+1],
                                prefix=prefix
                            )
            
            except Exception as e:
                self.logger.error(f"Error processing forest height statistics: {str(e)}")
        
        return results

    def _find_forest_height_source(self) -> Optional[Path]:
        """
        Find forest height data in the data directory or global datasets.
        Returns the path to the forest height raster if found, None otherwise.
        """
        # Check config for forest height data path - use the correct config key
        forest_path = self.config.get('ATTRIBUTES_FOREST_HEIGHT_PATH')
        
        if forest_path and forest_path != 'default':
            forest_path = Path(forest_path)
            if forest_path.exists():
                self.logger.info(f"Using forest height data from config: {forest_path}")
                return forest_path
        
        # Use the base attribute data directory from config
        attr_data_dir = self.config.get('ATTRIBUTES_DATA_DIR')
        if attr_data_dir and attr_data_dir != 'default':
            attr_data_dir = Path(attr_data_dir)
            
            # Check for forest_height directory within the attribute data directory
            forest_dir = attr_data_dir / 'forest_height' / 'raw'
            if forest_dir.exists():
                # Prefer the 2020 data as it's more recent
                if (forest_dir / "forest_height_2020.tif").exists():
                    self.logger.info(f"Using forest height data from attribute directory: {forest_dir / 'forest_height_2020.tif'}")
                    return forest_dir / "forest_height_2020.tif"
                # Fall back to 2000 data if 2020 isn't available
                elif (forest_dir / "forest_height_2000.tif").exists():
                    self.logger.info(f"Using forest height data from attribute directory: {forest_dir / 'forest_height_2000.tif'}")
                    return forest_dir / "forest_height_2000.tif"
                
                # Look for any TIF file in the directory
                forest_files = list(forest_dir.glob("*.tif"))
                if forest_files:
                    self.logger.info(f"Using forest height data from attribute directory: {forest_files[0]}")
                    return forest_files[0]
        
        # Fall back to project data directory if attribute data dir doesn't have the data
        global_forest_dir = self.data_dir / 'geospatial-data' / 'forest_height'
        if global_forest_dir.exists():
            forest_files = list(global_forest_dir.glob("*.tif"))
            if forest_files:
                self.logger.info(f"Using forest height data from global directory: {forest_files[0]}")
                return forest_files[0]
        
        # Try the hard-coded path as a last resort
        specific_forest_dir = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/forest_height/raw")
        if specific_forest_dir.exists():
            forest_files = list(specific_forest_dir.glob("*.tif"))
            if forest_files:
                self.logger.info(f"Using forest height data from specific directory: {forest_files[0]}")
                return forest_files[0]
        
        self.logger.warning("No forest height data found")
        return None

    def _clip_raster_to_bbox(self, source_raster: Path, output_raster: Path) -> None:
        """
        Clip a raster to the catchment's bounding box.
        """
        try:
            # Parse bounding box coordinates
            bbox_coords = self.config.get('BOUNDING_BOX_COORDS').split('/')
            if len(bbox_coords) == 4:
                bbox = [
                    float(bbox_coords[1]),  # lon_min
                    float(bbox_coords[2]),  # lat_min
                    float(bbox_coords[3]),  # lon_max
                    float(bbox_coords[0])   # lat_max
                ]
            else:
                raise ValueError("Invalid bounding box coordinates format")
            
            # Ensure output directory exists
            output_raster.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a bounding box geometry
            geom = box(*bbox)
            
            # Create a GeoDataFrame with the bounding box
            bbox_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
            
            # Clip the raster
            with rasterio.open(str(source_raster)) as src:
                # Check if coordinate reference systems match
                if src.crs != bbox_gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(src.crs)
                
                # Get the geometry in the raster's CRS
                geoms = [geom for geom in bbox_gdf.geometry]
                
                # Clip the raster
                out_image, out_transform = mask(src, geoms, crop=True)
                
                # Copy the metadata
                out_meta = src.meta.copy()
                
                # Update metadata for clipped raster
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                
                # Write the clipped raster
                with rasterio.open(str(output_raster), "w", **out_meta) as dest:
                    dest.write(out_image)
            
            self.logger.info(f"Clipped raster saved to {output_raster}")
            
        except Exception as e:
            self.logger.error(f"Error clipping raster: {str(e)}")
            raise

    def _count_pixels_in_catchment(self, raster_src) -> int:
        """
        Count the total number of pixels within the catchment boundary.
        """
        # Read the catchment shapefile
        catchment = gpd.read_file(str(self.catchment_path))
        
        # Ensure the shapefile is in the same CRS as the raster
        if catchment.crs != raster_src.crs:
            catchment = catchment.to_crs(raster_src.crs)
        
        # Create a mask of the catchment area
        mask, transform = rasterio.mask.mask(raster_src, catchment.geometry, crop=False, nodata=0)
        
        # Count non-zero pixels (i.e., pixels within the catchment)
        return np.sum(mask[0] != 0)

    def _count_pixels_in_hru(self, raster_src, hru_geometry) -> int:
        """
        Count the total number of pixels within a specific HRU boundary.
        """
        # Ensure the HRU geometry is in the same CRS as the raster
        if hru_geometry.crs != raster_src.crs:
            hru_geometry = hru_geometry.to_crs(raster_src.crs)
        
        # Create a mask of the HRU area
        mask, transform = rasterio.mask.mask(raster_src, hru_geometry.geometry, crop=False, nodata=0)
        
        # Count non-zero pixels (i.e., pixels within the HRU)
        return np.sum(mask[0] != 0)

    def _add_forest_distribution_metrics(self, forest_raster: Path, results: Dict[str, Any], 
                                        hru_geometry=None, prefix="") -> None:
        """
        Calculate forest height distribution metrics like skewness, kurtosis, and percentiles.
        """
        try:
            with rasterio.open(str(forest_raster)) as src:
                if hru_geometry is not None:
                    # For a specific HRU
                    if hru_geometry.crs != src.crs:
                        hru_geometry = hru_geometry.to_crs(src.crs)
                    
                    # Mask the raster to the HRU
                    masked_data, _ = rasterio.mask.mask(src, hru_geometry.geometry, crop=True, nodata=-9999)
                else:
                    # For the entire catchment
                    catchment = gpd.read_file(str(self.catchment_path))
                    
                    if catchment.crs != src.crs:
                        catchment = catchment.to_crs(src.crs)
                    
                    # Mask the raster to the catchment
                    masked_data, _ = rasterio.mask.mask(src, catchment.geometry, crop=True, nodata=-9999)
                
                # Flatten the data and exclude no-data values
                valid_data = masked_data[masked_data != -9999]
                
                # Only calculate statistics if we have enough valid data points
                if len(valid_data) > 10:
                    # Calculate additional statistics
                    results[f"{prefix}forest.height_skew"] = float(skew(valid_data))
                    results[f"{prefix}forest.height_kurtosis"] = float(kurtosis(valid_data))
                    
                    # Calculate percentiles
                    percentiles = [5, 10, 25, 50, 75, 90, 95]
                    for p in percentiles:
                        results[f"{prefix}forest.height_p{p}"] = float(np.percentile(valid_data, p))
                    
                    # Calculate canopy density metrics
                    # Assume forest height threshold of 5m for "canopy"
                    canopy_threshold = 5.0
                    canopy_pixels = np.sum(valid_data >= canopy_threshold)
                    if len(valid_data) > 0:
                        results[f"{prefix}forest.canopy_density"] = float(canopy_pixels / len(valid_data))
        
        except Exception as e:
            self.logger.error(f"Error calculating forest distribution metrics: {str(e)}")

    def _process_soil_attributes(self) -> Dict[str, Any]:
        """
        Process soil properties including texture, hydraulic properties, and soil depth.
        Extracts data from SOILGRIDS and Pelletier datasets.
        
        Returns:
            Dict[str, Any]: Dictionary of soil attributes
        """
        results = {}
        
        # Process soil texture from SOILGRIDS
        texture_results = self._process_soilgrids_texture()
        results.update(texture_results)
        
        # Process soil hydraulic properties derived from texture classes
        hydraulic_results = self._derive_hydraulic_properties(texture_results)
        results.update(hydraulic_results)
        
        # Process soil depth attributes from Pelletier dataset
        depth_results = self._process_pelletier_soil_depth()
        results.update(depth_results)
        
        return results

    def _process_soilgrids_texture(self) -> Dict[str, Any]:
        """
        Process soil texture data from SOILGRIDS.
        
        Returns:
            Dict[str, Any]: Dictionary of soil texture attributes
        """
        results = {}
        
        # Define paths to SOILGRIDS data
        soilgrids_dir = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/soilgrids/raw")
        
        # Define soil components and depths to process
        components = ['clay', 'sand', 'silt']
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        stats = ['mean', 'min', 'max', 'std']
        
        # Process each soil component at each depth
        for component in components:
            for depth in depths:
                # Define file path for mean values
                tif_path = soilgrids_dir / component / f"{component}_{depth}_mean.tif"
                
                # Skip if file doesn't exist
                if not tif_path.exists():
                    self.logger.warning(f"Soil component file not found: {tif_path}")
                    continue
                
                self.logger.info(f"Processing soil {component} at depth {depth}")
                
                # Calculate zonal statistics
                zonal_out = zonal_stats(
                    str(self.catchment_path), 
                    str(tif_path), 
                    stats=stats, 
                    all_touched=True
                )
                
                # Get scale and offset for proper unit conversion
                scale, offset = self._read_scale_and_offset(tif_path)
                
                # Update results
                is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
                
                if is_lumped:
                    for stat in stats:
                        if zonal_out and len(zonal_out) > 0 and stat in zonal_out[0]:
                            value = zonal_out[0][stat]
                            if value is not None:
                                # Apply scale and offset if they exist
                                if scale is not None:
                                    value = value * scale
                                if offset is not None:
                                    value = value + offset
                                results[f"soil.{component}_{depth}_{stat}"] = value
                else:
                    # For distributed catchment
                    catchment = gpd.read_file(self.catchment_path)
                    hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                    
                    for i, zonal_result in enumerate(zonal_out):
                        if i < len(catchment):
                            hru_id = catchment.iloc[i][hru_id_field]
                            prefix = f"HRU_{hru_id}_"
                            
                            for stat in stats:
                                if stat in zonal_result and zonal_result[stat] is not None:
                                    value = zonal_result[stat]
                                    # Apply scale and offset if they exist
                                    if scale is not None:
                                        value = value * scale
                                    if offset is not None:
                                        value = value + offset
                                    results[f"{prefix}soil.{component}_{depth}_{stat}"] = value
        
        # Calculate USDA soil texture class percentages
        self._calculate_usda_texture_classes(results)
        
        return results

    def _derive_hydraulic_properties(self, texture_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive hydraulic properties from soil texture using pedotransfer functions.
        
        Args:
            texture_results: Dictionary containing soil texture information
            
        Returns:
            Dict[str, Any]: Dictionary of derived hydraulic properties
        """
        results = {}
        
        # Define pedotransfer functions for common hydraulic properties
        # These are based on the relationships in Saxton and Rawls (2006)
        
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
        
        if is_lumped:
            # For lumped catchment
            # Calculate weighted average properties across all depths
            depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
            depth_weights = [5, 10, 15, 30, 40, 100]  # Thickness of each layer in cm
            total_depth = sum(depth_weights)
            
            # Get weighted average of sand, clay, and silt content across all depths
            avg_sand = 0
            avg_clay = 0
            avg_silt = 0
            
            for depth, weight in zip(depths, depth_weights):
                sand_key = f"soil.sand_{depth}_mean"
                clay_key = f"soil.clay_{depth}_mean"
                silt_key = f"soil.silt_{depth}_mean"
                
                if sand_key in texture_results and clay_key in texture_results and silt_key in texture_results:
                    avg_sand += texture_results[sand_key] * weight / total_depth
                    avg_clay += texture_results[clay_key] * weight / total_depth
                    avg_silt += texture_results[silt_key] * weight / total_depth
            
            # Convert from g/kg to fraction (divide by 10)
            sand_fraction = avg_sand / 1000
            clay_fraction = avg_clay / 1000 
            silt_fraction = avg_silt / 1000
            
            # Calculate hydraulic properties using pedotransfer functions
            
            # Porosity (saturated water content)
            porosity = 0.46 - 0.0026 * avg_clay
            results["soil.porosity"] = porosity
            
            # Field capacity (-33 kPa matric potential)
            field_capacity = 0.2576 - 0.002 * avg_sand + 0.0036 * avg_clay + 0.0299 * avg_silt
            results["soil.field_capacity"] = field_capacity
            
            # Wilting point (-1500 kPa matric potential)
            wilting_point = 0.026 + 0.005 * avg_clay + 0.0158 * avg_silt
            results["soil.wilting_point"] = wilting_point
            
            # Available water capacity
            results["soil.available_water_capacity"] = field_capacity - wilting_point
            
            # Saturated hydraulic conductivity (mm/h)
            ksat = 10 * (2.54 * (2.778 * (10**-6)) * 10**(3.0 * porosity - 8.5))
            results["soil.ksat"] = ksat
            
        else:
            # For distributed catchment
            catchment = gpd.read_file(self.catchment_path)
            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
            
            for i in range(len(catchment)):
                hru_id = catchment.iloc[i][hru_id_field]
                prefix = f"HRU_{hru_id}_"
                
                # Calculate weighted average properties across all depths
                depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
                depth_weights = [5, 10, 15, 30, 40, 100]  # Thickness of each layer in cm
                total_depth = sum(depth_weights)
                
                # Get weighted average of sand, clay, and silt content across all depths
                avg_sand = 0
                avg_clay = 0
                avg_silt = 0
                
                for depth, weight in zip(depths, depth_weights):
                    sand_key = f"{prefix}soil.sand_{depth}_mean"
                    clay_key = f"{prefix}soil.clay_{depth}_mean"
                    silt_key = f"{prefix}soil.silt_{depth}_mean"
                    
                    if sand_key in texture_results and clay_key in texture_results and silt_key in texture_results:
                        avg_sand += texture_results[sand_key] * weight / total_depth
                        avg_clay += texture_results[clay_key] * weight / total_depth
                        avg_silt += texture_results[silt_key] * weight / total_depth
                
                # Convert from g/kg to fraction (divide by 10)
                sand_fraction = avg_sand / 1000
                clay_fraction = avg_clay / 1000
                silt_fraction = avg_silt / 1000
                
                # Calculate hydraulic properties using pedotransfer functions
                
                # Porosity (saturated water content)
                porosity = 0.46 - 0.0026 * avg_clay
                results[f"{prefix}soil.porosity"] = porosity
                
                # Field capacity (-33 kPa matric potential)
                field_capacity = 0.2576 - 0.002 * avg_sand + 0.0036 * avg_clay + 0.0299 * avg_silt
                results[f"{prefix}soil.field_capacity"] = field_capacity
                
                # Wilting point (-1500 kPa matric potential)
                wilting_point = 0.026 + 0.005 * avg_clay + 0.0158 * avg_silt
                results[f"{prefix}soil.wilting_point"] = wilting_point
                
                # Available water capacity
                results[f"{prefix}soil.available_water_capacity"] = field_capacity - wilting_point
                
                # Saturated hydraulic conductivity (mm/h)
                ksat = 10 * (2.54 * (2.778 * (10**-6)) * 10**(3.0 * porosity - 8.5))
                results[f"{prefix}soil.ksat"] = ksat
        
        return results

    def _calculate_usda_texture_classes(self, results: Dict[str, Any]) -> None:
        """
        Calculate USDA soil texture classes based on sand, silt, and clay percentages.
        Updates the results dictionary in place.
        
        Args:
            results: Dictionary to update with texture class information
        """
        # USDA soil texture classes
        usda_classes = {
            'clay': (0, 45, 0, 40, 40, 100),  # Sand%, Silt%, Clay% ranges
            'silty_clay': (0, 20, 40, 60, 40, 60),
            'sandy_clay': (45, 65, 0, 20, 35, 55),
            'clay_loam': (20, 45, 15, 53, 27, 40),
            'silty_clay_loam': (0, 20, 40, 73, 27, 40),
            'sandy_clay_loam': (45, 80, 0, 28, 20, 35),
            'loam': (23, 52, 28, 50, 7, 27),
            'silty_loam': (0, 50, 50, 88, 0, 27),
            'sandy_loam': (50, 80, 0, 50, 0, 20),
            'silt': (0, 20, 80, 100, 0, 12),
            'loamy_sand': (70, 90, 0, 30, 0, 15),
            'sand': (85, 100, 0, 15, 0, 10)
        }
        
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
        
        # Depths to analyze
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        
        if is_lumped:
            # For each depth, determine texture class percentage
            for depth in depths:
                sand_key = f"soil.sand_{depth}_mean"
                clay_key = f"soil.clay_{depth}_mean"
                silt_key = f"soil.silt_{depth}_mean"
                
                if sand_key in results and clay_key in results and silt_key in results:
                    # Get values and convert to percentages
                    sand_val = results[sand_key] / 10  # g/kg to %
                    clay_val = results[clay_key] / 10  # g/kg to %
                    silt_val = results[silt_key] / 10  # g/kg to %
                    
                    # Normalize to ensure they sum to 100%
                    total = sand_val + clay_val + silt_val
                    if total > 0:
                        sand_pct = (sand_val / total) * 100
                        clay_pct = (clay_val / total) * 100
                        silt_pct = (silt_val / total) * 100
                        
                        # Determine soil texture class
                        for texture_class, (sand_min, sand_max, silt_min, silt_max, clay_min, clay_max) in usda_classes.items():
                            if (sand_min <= sand_pct <= sand_max and
                                silt_min <= silt_pct <= silt_max and
                                clay_min <= clay_pct <= clay_max):
                                
                                # Add texture class to results
                                results[f"soil.texture_class_{depth}"] = texture_class
                                break
        else:
            # For distributed catchment
            catchment = gpd.read_file(self.catchment_path)
            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
            
            for i in range(len(catchment)):
                hru_id = catchment.iloc[i][hru_id_field]
                prefix = f"HRU_{hru_id}_"
                
                # For each depth, determine texture class percentage
                for depth in depths:
                    sand_key = f"{prefix}soil.sand_{depth}_mean"
                    clay_key = f"{prefix}soil.clay_{depth}_mean"
                    silt_key = f"{prefix}soil.silt_{depth}_mean"
                    
                    if sand_key in results and clay_key in results and silt_key in results:
                        # Get values and convert to percentages
                        sand_val = results[sand_key] / 10  # g/kg to %
                        clay_val = results[clay_key] / 10  # g/kg to %
                        silt_val = results[silt_key] / 10  # g/kg to %
                        
                        # Normalize to ensure they sum to 100%
                        total = sand_val + clay_val + silt_val
                        if total > 0:
                            sand_pct = (sand_val / total) * 100
                            clay_pct = (clay_val / total) * 100
                            silt_pct = (silt_val / total) * 100
                            
                            # Determine soil texture class
                            for texture_class, (sand_min, sand_max, silt_min, silt_max, clay_min, clay_max) in usda_classes.items():
                                if (sand_min <= sand_pct <= sand_max and
                                    silt_min <= silt_pct <= silt_max and
                                    clay_min <= clay_pct <= clay_max):
                                    
                                    # Add texture class to results
                                    results[f"{prefix}soil.texture_class_{depth}"] = texture_class
                                    break

    def _process_pelletier_soil_depth(self) -> Dict[str, Any]:
        """
        Process soil depth attributes from Pelletier dataset.
        
        Returns:
            Dict[str, Any]: Dictionary of soil depth attributes
        """
        results = {}
        
        # Define path to Pelletier data
        pelletier_dir = Path("/work/comphyd_lab/data/_to-be-moved/NorthAmerica_geospatial/pelletier/raw")
        
        # Define files to process
        pelletier_files = {
            "upland_hill-slope_regolith_thickness.tif": "regolith_thickness",
            "upland_hill-slope_soil_thickness.tif": "soil_thickness",
            "upland_valley-bottom_and_lowland_sedimentary_deposit_thickness.tif": "sedimentary_thickness",
            "average_soil_and_sedimentary-deposit_thickness.tif": "average_thickness"
        }
        
        stats = ['mean', 'min', 'max', 'std']
        
        for file_name, attribute in pelletier_files.items():
            # Define file path
            tif_path = pelletier_dir / file_name
            
            # Skip if file doesn't exist
            if not tif_path.exists():
                self.logger.warning(f"Pelletier file not found: {tif_path}")
                continue
            
            self.logger.info(f"Processing Pelletier {attribute}")
            
            # Check and set no-data value if needed
            self._check_and_set_nodata_value(tif_path)
            
            # Calculate zonal statistics
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(tif_path), 
                stats=stats, 
                all_touched=True
            )
            
            # Some tifs may have no data because the variable doesn't exist in the area
            zonal_out = self._check_zonal_stats_outcomes(zonal_out, new_val=0)
            
            # Get scale and offset
            scale, offset = self._read_scale_and_offset(tif_path)
            
            # Update results
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD', 'delineate') == 'lumped'
            
            if is_lumped:
                for stat in stats:
                    if zonal_out and len(zonal_out) > 0 and stat in zonal_out[0]:
                        value = zonal_out[0][stat]
                        if value is not None:
                            # Apply scale and offset if they exist
                            if scale is not None:
                                value = value * scale
                            if offset is not None:
                                value = value + offset
                            results[f"soil.{attribute}_{stat}"] = value
            else:
                # For distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                for i, zonal_result in enumerate(zonal_out):
                    if i < len(catchment):
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        for stat in stats:
                            if stat in zonal_result and zonal_result[stat] is not None:
                                value = zonal_result[stat]
                                # Apply scale and offset if they exist
                                if scale is not None:
                                    value = value * scale
                                if offset is not None:
                                    value = value + offset
                                results[f"{prefix}soil.{attribute}_{stat}"] = value
        
        return results

    def _read_scale_and_offset(self, geotiff_path: Path) -> Tuple[Optional[float], Optional[float]]:
        """
        Read scale and offset from a GeoTIFF file.
        
        Args:
            geotiff_path: Path to the GeoTIFF file
            
        Returns:
            Tuple of scale and offset values, potentially None if not set
        """
        try:
            dataset = gdal.Open(str(geotiff_path))
            if dataset is None:
                self.logger.warning(f"Could not open GeoTIFF: {geotiff_path}")
                return None, None
            
            # Get the scale and offset values
            scale = dataset.GetRasterBand(1).GetScale()
            offset = dataset.GetRasterBand(1).GetOffset()
            
            # Close the dataset
            dataset = None
            
            return scale, offset
        except Exception as e:
            self.logger.error(f"Error reading scale and offset: {str(e)}")
            return None, None

    def _check_and_set_nodata_value(self, tif: Path, nodata: int = 255) -> None:
        """
        Check and set no-data value for a GeoTIFF if not already set.
        
        Args:
            tif: Path to the GeoTIFF file
            nodata: No-data value to set
        """
        try:
            # Open the dataset
            ds = gdal.Open(str(tif), gdal.GA_Update)
            if ds is None:
                self.logger.warning(f"Could not open GeoTIFF for no-data setting: {tif}")
                return
            
            # Get the current no-data value
            band = ds.GetRasterBand(1)
            current_nodata = band.GetNoDataValue()
            
            # If no no-data value is set but we need one
            if current_nodata is None:
                # Check if the maximum value in the dataset is the no-data value
                data = band.ReadAsArray()
                if data.max() == nodata:
                    # Set the no-data value
                    band.SetNoDataValue(nodata)
                    self.logger.info(f"Set no-data value to {nodata} for {tif}")
                
            # Close the dataset
            ds = None
        except Exception as e:
            self.logger.error(f"Error checking and setting no-data value: {str(e)}")

    def _check_zonal_stats_outcomes(self, zonal_out: List[Dict], new_val: Union[float, int] = np.nan) -> List[Dict]:
        """
        Check for None values in zonal statistics results and replace with specified value.
        
        Args:
            zonal_out: List of dictionaries with zonal statistics results
            new_val: Value to replace None with
            
        Returns:
            Updated zonal statistics results
        """
        for i in range(len(zonal_out)):
            for key, val in zonal_out[i].items():
                if val is None:
                    zonal_out[i][key] = new_val
                    self.logger.debug(f"Replaced None in {key} with {new_val}")
        
        return zonal_out

    def _process_elevation_attributes(self) -> Dict[str, float]:
        """Process elevation, slope, and aspect attributes."""
        results = {}
        
        try:
            # Find the DEM file
            dem_file = self.find_dem_file()
            
            # Generate slope and aspect
            raster_files = self.generate_slope_and_aspect(dem_file)
            
            # Calculate statistics for each raster
            for attribute_name, raster_file in raster_files.items():
                stats = self.calculate_statistics(raster_file, attribute_name)
                
                # Add statistics with proper prefixes
                for stat_name, value in stats.items():
                    if "." in stat_name:  # If stat_name already has hierarchical structure
                        results[stat_name] = value
                    else:
                        # Extract any HRU prefix if present
                        if stat_name.startswith("HRU_"):
                            hru_part = stat_name.split("_", 2)[0] + "_" + stat_name.split("_", 2)[1] + "_"
                            clean_stat = stat_name.replace(hru_part, "")
                            prefix = hru_part
                        else:
                            clean_stat = stat_name
                            prefix = ""
                        
                        # Remove attribute prefix if present in the stat name
                        clean_stat = clean_stat.replace(f"{attribute_name}_", "")
                        results[f"{prefix}{attribute_name}.{clean_stat}"] = value
        
        except Exception as e:
            self.logger.error(f"Error processing elevation attributes: {str(e)}")
        
        return results

    def _process_landcover_attributes(self) -> Dict[str, Any]:
        """Process land cover attributes including dominant land cover type."""
        results = {}
        
        # Find the land cover raster
        landclass_dir = self.project_dir / 'attributes' / 'landclass'
        if not landclass_dir.exists():
            self.logger.warning(f"Land cover directory not found: {landclass_dir}")
            return results
        
        # Search for land cover raster files
        landcover_files = list(landclass_dir.glob("*.tif"))
        if not landcover_files:
            self.logger.warning(f"No land cover raster files found in {landclass_dir}")
            return results
        
        # Use the first land cover file found
        landcover_file = landcover_files[0]
        self.logger.info(f"Processing land cover raster: {landcover_file}")
        
        try:
            # Get land cover classes and class names if available
            lc_classes = self._get_landcover_classes(landcover_file)
            
            # Calculate zonal statistics with categorical=True to get class counts
            zonal_out = zonal_stats(
                str(self.catchment_path), 
                str(landcover_file), 
                categorical=True, 
                all_touched=True
            )
            
            # Process results
            is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
            
            if is_lumped:
                # For lumped catchment
                if zonal_out:
                    # Calculate class fractions
                    total_pixels = sum(count for count in zonal_out[0].values() if count is not None)
                    
                    if total_pixels > 0:
                        for class_id, count in zonal_out[0].items():
                            if class_id is not None:  # Skip None values
                                # Get class name if available, otherwise use class ID
                                class_name = lc_classes.get(class_id, f"class_{class_id}")
                                fraction = (count / total_pixels) if count is not None else 0
                                results[f"landcover.{class_name}_fraction"] = fraction
                        
                        # Find dominant land cover class
                        dominant_class_id = max(zonal_out[0].items(), key=lambda x: x[1] if x[1] is not None else 0)[0]
                        dominant_class_name = lc_classes.get(dominant_class_id, f"class_{dominant_class_id}")
                        results["landcover.dominant_class"] = dominant_class_name
                        results["landcover.dominant_class_id"] = dominant_class_id
                        results["landcover.dominant_fraction"] = (zonal_out[0][dominant_class_id] / total_pixels) if zonal_out[0][dominant_class_id] is not None else 0
            else:
                # For distributed catchment
                catchment = gpd.read_file(self.catchment_path)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                # Process each HRU
                for i, zonal_result in enumerate(zonal_out):
                    try:
                        hru_id = catchment.iloc[i][hru_id_field]
                        prefix = f"HRU_{hru_id}_"
                        
                        # Calculate class fractions for this HRU
                        total_pixels = sum(count for count in zonal_result.values() if count is not None)
                        
                        if total_pixels > 0:
                            for class_id, count in zonal_result.items():
                                if class_id is not None:  # Skip None values
                                    class_name = lc_classes.get(class_id, f"class_{class_id}")
                                    fraction = (count / total_pixels) if count is not None else 0
                                    results[f"{prefix}landcover.{class_name}_fraction"] = fraction
                            
                            # Find dominant land cover class for this HRU
                            dominant_class_id = max(zonal_result.items(), key=lambda x: x[1] if x[1] is not None else 0)[0]
                            dominant_class_name = lc_classes.get(dominant_class_id, f"class_{dominant_class_id}")
                            results[f"{prefix}landcover.dominant_class"] = dominant_class_name
                            results[f"{prefix}landcover.dominant_class_id"] = dominant_class_id
                            results[f"{prefix}landcover.dominant_fraction"] = (zonal_result[dominant_class_id] / total_pixels) if zonal_result[dominant_class_id] is not None else 0
                    except Exception as e:
                        self.logger.error(f"Error processing HRU {i}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error processing land cover attributes: {str(e)}")
        
        return results

    def _get_landcover_classes(self, landcover_file: Path) -> Dict[int, str]:
        """
        Get land cover class definitions.
        Attempts to determine class names from metadata or uses a default mapping.
        """
        # Default MODIS IGBP land cover classes
        modis_classes = {
            1: 'evergreen_needleleaf',
            2: 'evergreen_broadleaf',
            3: 'deciduous_needleleaf',
            4: 'deciduous_broadleaf',
            5: 'mixed_forest',
            6: 'closed_shrubland',
            7: 'open_shrubland',
            8: 'woody_savanna',
            9: 'savanna',
            10: 'grassland',
            11: 'wetland',
            12: 'cropland',
            13: 'urban',
            14: 'crop_natural_mosaic',
            15: 'snow_ice',
            16: 'barren',
            17: 'water'
        }
        
        # Try to read class definitions from raster metadata or config
        try:
            with rasterio.open(str(landcover_file)) as src:
                # Check if there's land cover class metadata
                if 'land_cover_classes' in src.tags():
                    # Parse metadata
                    class_info = src.tags()['land_cover_classes']
                    # Implement parsing logic based on the format
                    return {}  # Return parsed classes
        except Exception as e:
            self.logger.warning(f"Could not read land cover class metadata: {str(e)}")
        
        # Check if land cover type is specified in config
        lc_type = self.config.get('LAND_COVER_TYPE', '').lower()
        
        if 'modis' in lc_type or 'igbp' in lc_type:
            return modis_classes
        elif 'esa' in lc_type or 'cci' in lc_type:
            # ESA CCI land cover classes
            return {
                10: 'cropland_rainfed',
                20: 'cropland_irrigated',
                30: 'cropland_mosaic',
                40: 'forest_broadleaf_evergreen',
                50: 'forest_broadleaf_deciduous',
                60: 'forest_needleleaf_evergreen',
                70: 'forest_needleleaf_deciduous',
                80: 'forest_mixed',
                90: 'shrubland_mosaic',
                100: 'grassland',
                110: 'wetland',
                120: 'shrubland',
                130: 'grassland_sparse',
                140: 'lichens_mosses',
                150: 'sparse_vegetation',
                160: 'forest_flooded_freshwater',
                170: 'forest_flooded_saline',
                180: 'shrubland_flooded',
                190: 'urban',
                200: 'bare_areas',
                210: 'water',
                220: 'snow_ice'
            }
        
        # Default to generic class names
        return {i: f"class_{i}" for i in range(1, 30)}