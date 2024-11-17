from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd # type: ignore
import numpy as np
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import shutil
from datetime import datetime
import subprocess
import os
import cdo # type: ignore

# For data processing
import dask # type: ignore

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
# Type hints
from typing import TYPE_CHECKING

# Optional imports for parallel processing
try:
    from concurrent.futures import ProcessPoolExecutor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

# Set pandas options for better display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

# Register pint with xarray for unit handling
xr.set_options(keep_attrs=True)

class HYPEPreProcessor:
    """
    HYPE (HYdrological Predictions for the Environment) preprocessor for CONFLUENCE.
    Handles preparation of HYPE model inputs using CONFLUENCE's data structure.
    
    Attributes:
        config (Dict[str, Any]): CONFLUENCE configuration settings
        logger (logging.Logger): Logger for the preprocessing workflow
        project_dir (Path): Project directory path
        domain_name (str): Name of the modeling domain
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize HYPE preprocessor with CONFLUENCE config."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Define HYPE-specific paths
        self.hype_setup_dir = self.project_dir / "settings" / "HYPE"
        self.gistool_output = self.project_dir / "attributes" / "gistool-outputs"
        self.easymore_output = self.project_dir / "forcing" / "easymore-outputs"
        
        # Initialize time parameters
        self.timeshift = -6  # Default timeshift
        self.spinup_days = 274  # Default spinup period
        
        # Create necessary directories
        self.hype_setup_dir.mkdir(parents=True, exist_ok=True)

    def run_preprocessing(self):
        """Execute complete HYPE preprocessing workflow."""
        self.logger.info("Starting HYPE preprocessing")
        
        try:

            # Write forcing files
            self.write_hype_forcing()
            
            # Write geographic data files
            self.write_hype_geo_files()
            
            # Sort geofabric from upstream to downstream
            self.sort_geofabric()

            # Write parameter file
            self.write_hype_par_file()
            
            # Write info and file directory files
            self.write_hype_info_filedir_files(spinup_days=self.spinup_days)
            
            self.logger.info("HYPE preprocessing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during HYPE preprocessing: {str(e)}")
            raise

    def sort_geofabric(self):
        """
        Sort GeoData.txt starting from known outlet (SIM_REACH_ID) and working upstream.
        Ignores any connections downstream of the outlet point.
        """
        self.logger.info("Sorting geofabric from outlet to headwaters")
        
        try:
            geodata_path = self.hype_setup_dir / 'GeoData.txt'
            if not geodata_path.exists():
                raise FileNotFoundError(f"GeoData.txt not found at {geodata_path}")
                
            geodata = pd.read_csv(geodata_path, sep='\t')
            geodata['subid'] = geodata['subid'].astype(int)
            geodata['maindown'] = geodata['maindown'].astype(int)
            
            # Get outlet ID from config
            outlet_id = int(self.config['SIM_REACH_ID'])
            self.logger.info(f"Using basin {outlet_id} as outlet point")
            
            if outlet_id not in geodata['subid'].values:
                raise ValueError(f"Outlet ID {outlet_id} not found in GeoData.txt")
            
            # Build upstream basin dictionary
            upstream_basins = {}  # For each basin, which basins flow into it
            for _, row in geodata.iterrows():
                subid = row['subid']
                maindown = row['maindown']
                
                if maindown not in upstream_basins:
                    upstream_basins[maindown] = set()
                upstream_basins[maindown].add(subid)
            
            # Get all basins that are upstream of our outlet
            def get_all_upstream(basin_id, visited=None):
                if visited is None:
                    visited = set()
                
                # Add current basin
                visited.add(basin_id)
                
                # Add all basins that flow into this one
                for upstream_id in upstream_basins.get(basin_id, set()):
                    if upstream_id not in visited:
                        get_all_upstream(upstream_id, visited)
                
                return visited
            
            # Get all basins in our network (outlet and everything upstream)
            network_basins = get_all_upstream(outlet_id)
            self.logger.info(f"Found {len(network_basins)} basins in network")
            
            # Create list of basins in downstream to upstream order
            ordered_basins = []
            processed = set()
            
            # Start with the outlet
            queue = [outlet_id]
            
            while queue:
                current = queue.pop(0)
                if current not in processed:
                    ordered_basins.append(current)
                    processed.add(current)
                    
                    # Add any unprocessed upstream basins to the queue
                    if current in upstream_basins:
                        queue.extend(sorted(
                            [b for b in upstream_basins[current] if b not in processed],
                            key=lambda x: -float(geodata.loc[geodata['subid'] == x, 'elev_mean'].iloc[0])
                        ))
            
            # Reverse the list to get upstream to downstream order
            ordered_basins.reverse()
            
            # Verify we processed all basins in our network
            if set(ordered_basins) != network_basins:
                missing = network_basins - set(ordered_basins)
                self.logger.error(f"Failed to process all connected basins. Missing: {missing}")
                raise ValueError("Failed to create complete basin ordering")
            
            # Sort geodata according to our ordering
            sort_idx = pd.Series(range(len(ordered_basins)), index=ordered_basins)
            geodata = geodata[geodata['subid'].isin(network_basins)]  # Keep only basins in our network
            geodata['sort_index'] = geodata['subid'].map(sort_idx)
            sorted_geodata = geodata.sort_values('sort_index').drop('sort_index', axis=1)
            
            # Modify any connections downstream of our outlet to point to 0 (external)
            sorted_geodata.loc[sorted_geodata['subid'] == outlet_id, 'maindown'] = 0
            
            # Create backup
            backup_path = geodata_path.with_suffix('.txt.bak')
            if geodata_path.exists():
                shutil.copy2(geodata_path, backup_path)
                self.logger.info(f"Created backup of original GeoData.txt at {backup_path}")
            
            # Format floating point columns
            float_cols = ['area', 'rivlen', 'latitude', 'longitude', 'elev_mean']
            for col in float_cols:
                if col in sorted_geodata.columns:
                    sorted_geodata[col] = sorted_geodata[col].map('{:.6g}'.format)
            
            # Write sorted data
            sorted_geodata.to_csv(geodata_path, sep='\t', index=False)
            
            self.logger.info("Geofabric sorting completed successfully")
            self.logger.debug(f"Final basin ordering: {' -> '.join(map(str, ordered_basins))}")
            
        except Exception as e:
            self.logger.error(f"Error during geofabric sorting: {str(e)}")
            self.logger.error(f"Error details: {str(e.__class__.__name__)}: {str(e)}")
            raise

    def write_hype_forcing(self):
        """Convert CONFLUENCE forcing data to HYPE format efficiently."""
        self.logger.info("Writing HYPE forcing files")
        
        try:
            # Define forcing units mapping (using same structure as your colleague)
            forcing_units = {
                'temperature': {
                    'in_varname': 'RDRS_v2.1_P_TT_09944',
                    'in_units': 'kelvin',
                    'out_units': 'celsius'
                },
                'precipitation': {
                    'in_varname': 'RDRS_v2.1_A_PR0_SFC',  # Make sure this matches exactly
                    'in_units': 'm/hr',
                    'out_units': 'mm/day'
                }
            }
            
            # Find easymore files and convert to string paths
            easymore_files = sorted(str(p) for p in self.easymore_output.glob('*.nc'))
            if not easymore_files:
                raise FileNotFoundError("No easymore output files found")
            
            # Split files into manageable batches
            batch_size = min(20, len(easymore_files))
            files_split = np.array_split(easymore_files, batch_size)
            cdo_obj = cdo.Cdo()
            intermediate_files = []
            
            # Process in batches
            self.logger.info("Merging easymore outputs")
            for i, batch_files in enumerate(files_split):
                batch_output = f'forcing_batch_{i}.nc'
                cdo_obj.mergetime(input=batch_files.tolist(), output=batch_output)
                intermediate_files.append(batch_output)
            
            # Merge all batches
            cdo_obj.mergetime(input=intermediate_files, output='merged_forcing.nc')
            
            # Clean up intermediates
            for f in intermediate_files:
                os.remove(f)
            
            # Open merged dataset
            ds = xr.open_dataset('merged_forcing.nc')
            
            # Debug info
            self.logger.debug(f"Variables in dataset: {list(ds.variables)}")
            self.logger.debug(f"Coordinates in dataset: {list(ds.coords)}")
            
            ds = ds.convert_calendar('standard')
            ds['time'] = ds['time'] + pd.Timedelta(hours=self.timeshift)
            
            # Process temperature
            self._process_forcing_file(
                ds=ds,
                var_name=forcing_units['temperature']['in_varname'],
                out_name='TMAXobs',
                unit_conversion={
                    'in_units': forcing_units['temperature']['in_units'],
                    'out_units': forcing_units['temperature']['out_units']
                },
                stat='max'
            )
            
            self._process_forcing_file(
                ds=ds,
                var_name=forcing_units['temperature']['in_varname'],
                out_name='TMINobs',
                unit_conversion={
                    'in_units': forcing_units['temperature']['in_units'],
                    'out_units': forcing_units['temperature']['out_units']
                },
                stat='min'
            )
            
            self._process_forcing_file(
                ds=ds,
                var_name=forcing_units['temperature']['in_varname'],
                out_name='Tobs',
                unit_conversion={
                    'in_units': forcing_units['temperature']['in_units'],
                    'out_units': forcing_units['temperature']['out_units']
                },
                stat='mean'
            )
            
            # Process precipitation with special handling for daily accumulation
            self._process_forcing_file(
                ds=ds,
                var_name=forcing_units['precipitation']['in_varname'],
                out_name='Pobs',
                unit_conversion={
                    'in_units': forcing_units['precipitation']['in_units'],
                    'out_units': 'mm/hr'  # Convert to mm/hr first
                },
                stat='sum'  # Change to sum for precipitation
            )
            
            # Clean up
            ds.close()
            os.remove('merged_forcing.nc')
            
            self.logger.info("HYPE forcing files written successfully")
            
        except Exception as e:
            self.logger.error(f"Error writing HYPE forcing: {str(e)}")
            raise
 
    def _write_forcing_files(self, results: Dict[str, Dict[str, np.ndarray]]):
        """Write forcing files efficiently using numpy."""
        # Prepare output formats
        output_files = {
            'temperature': [
                ('Tobs.txt', 'mean'),
                ('TMAXobs.txt', 'max'),
                ('TMINobs.txt', 'min')
            ],
            'precipitation': [
                ('Pobs.txt', 'mean')
            ]
        }
        
        # Write files using numpy's efficient I/O
        for var_name, file_specs in output_files.items():
            var_data = results[var_name]
            times = pd.DatetimeIndex(var_data['time'])
            
            for filename, stat_type in file_specs:
                filepath = self.hype_setup_dir / filename
                
                # Create structured data
                data = np.column_stack((
                    times.strftime('%Y-%m-%d'),
                    var_data[stat_type]
                ))
                
                # Write efficiently with numpy
                header = 'time\tvalue'
                np.savetxt(
                    filepath,
                    data,
                    fmt=['%s', '%.3f'],
                    delimiter='\t',
                    header=header,
                    comments=''
                )
                
    def write_hype_geo_files(self):
        """Create HYPE geographic data files using CONFLUENCE shapefiles."""
        self.logger.info("Writing HYPE geographic files")
        
        try:
            # Read CONFLUENCE shapefiles
            basins = gpd.read_file(self.project_dir / "shapefiles" / "river_basins" / 
                                 f"{self.domain_name}_riverBasins_{self.config['DOMAIN_DEFINITION_METHOD']}.shp")
            rivers = gpd.read_file(self.project_dir / "shapefiles" / "river_network" / 
                                 f"{self.domain_name}_riverNetwork_delineate.shp")
            
            # Create and write GeoData.txt
            self._create_geodata(basins, rivers)
            
            # Create and write GeoClass.txt
            self._create_geoclass()
            
            self.logger.info("HYPE geographic files written successfully")
            
        except Exception as e:
            self.logger.error(f"Error writing HYPE geographic files: {str(e)}")
            raise

    @staticmethod
    def _count_downstream(subid: int, downstream_dict: Dict[int, int]) -> int:
        """Count number of downstream subbasins for a given subid."""
        count = 0
        current = downstream_dict.get(subid, 0)
        visited = set()
        
        while current > 0 and current in downstream_dict and current not in visited:
            visited.add(current)
            count += 1
            current = downstream_dict.get(current, 0)
        
        return count

    @staticmethod
    def _count_downstream_wrapper(args):
        """Wrapper function for multiprocessing that unpacks arguments."""
        subid, downstream_dict = args
        return HYPEPreProcessor._count_downstream(subid, downstream_dict)

    def _sort_geodata(self, geodata: pd.DataFrame) -> pd.DataFrame:
        """Sort geodata from upstream to downstream more efficiently."""
        # Create a network dictionary for faster lookups
        downstream_dict = geodata.set_index('subid')['maindown'].to_dict()
        
        # Create argument tuples for parallel processing
        args = [(subid, downstream_dict) for subid in geodata['subid']]
        
        # Use vectorized operations with pandas
        if PARALLEL_AVAILABLE:
            with ProcessPoolExecutor() as executor:
                n_downstream = list(executor.map(self._count_downstream_wrapper, args))
        else:
            n_downstream = [self._count_downstream(subid, downstream_dict) 
                        for subid in geodata['subid']]
        
        geodata['n_ds_subbasins'] = n_downstream
        
        # Sort and clean up
        geodata = geodata.sort_values('n_ds_subbasins', ascending=False)
        return geodata.drop(columns=['n_ds_subbasins'])

    def _create_geodata(self, basins: gpd.GeoDataFrame, rivers: gpd.GeoDataFrame):
        """Create HYPE GeoData.txt file more efficiently."""
        # Create geodata DataFrame in one go
        geodata = pd.DataFrame({
            'subid': basins[self.config['RIVER_BASIN_SHP_RM_GRUID']],
            'maindown': rivers[self.config['RIVER_NETWORK_SHP_DOWNSEGID']],
            'area': basins[self.config['RIVER_BASIN_SHP_AREA']],
            'rivlen': rivers[self.config['RIVER_NETWORK_SHP_LENGTH']]
        })
        
        # Calculate centroids once
        centroids = basins.to_crs(epsg=4326).geometry.centroid
        geodata['latitude'] = centroids.y
        geodata['longitude'] = centroids.x
        
        # Process elevation data efficiently
        elev_stats = pd.read_csv(self.gistool_output / 'domain_stats_elv.csv')
        geodata['elev_mean'] = elev_stats['mean']
        
        # Sort from upstream to downstream
        geodata = self._sort_geodata(geodata)
        
        # Save GeoData.txt efficiently using numpy's savetxt
        header = '\t'.join(geodata.columns)
        np.savetxt(
            self.hype_setup_dir / 'GeoData.txt',
            geodata.values,
            fmt='%g',
            delimiter='\t',
            header=header,
            comments=''
        )

    def _process_forcing_file(self, ds: xr.Dataset, 
                            var_name: str, 
                            out_name: str,
                            unit_conversion: Dict[str, str],
                            stat: str = 'mean',
                            time_diff: int = -7) -> None:
        """Process and write a single forcing file with proper orientation."""
        try:
            self.logger.debug(f"Processing {var_name} to {out_name}")
            
            # Rename GRU_ID to id if present
            if 'GRU_ID' in ds.coords:
                ds = ds.rename({'GRU_ID': 'id'})
            elif 'GRU_ID' in ds:
                ds = ds.rename({'GRU_ID': 'id'})
                
            ds.coords['id'] = ds.coords['id'].astype(int)
            
            # Keep only necessary variables
            variables_to_keep = [var_name, 'time', 'id']
            ds = ds.drop([v for v in ds.variables if v not in variables_to_keep])
            
            # Apply time shift if needed
            if time_diff != 0:
                ds['time'] = ds['time'].roll(time=time_diff)
                if time_diff < 0:
                    ds = ds.isel(time=slice(None, time_diff))
                elif time_diff > 0:
                    ds = ds.isel(time=slice(time_diff, None))
            
            # Convert units before resampling for precipitation
            if out_name == 'Pobs':
                ds[var_name] = ds[var_name].pint.quantify(unit_conversion['in_units'])
                ds[var_name] = ds[var_name].pint.to('mm/hr')
                ds = ds.pint.dequantify()
            
            # Calculate daily statistics
            if stat == 'max':
                ds_daily = ds.resample(time='D').max()
            elif stat == 'min':
                ds_daily = ds.resample(time='D').min()
            elif stat == 'sum':
                ds_daily = ds.resample(time='D').sum() * 24  # Convert hourly sum to daily sum
            else:  # mean is default
                ds_daily = ds.resample(time='D').mean()
                if out_name == 'Pobs':
                    ds_daily = ds_daily * 24  # Convert hourly mean to daily sum for precipitation
            
            # Convert units after resampling if not precipitation
            if out_name != 'Pobs':
                ds_daily[var_name] = ds_daily[var_name].pint.quantify(unit_conversion['in_units'])
                ds_daily[var_name] = ds_daily[var_name].pint.to(unit_conversion['out_units'])
                ds_daily = ds_daily.pint.dequantify()
            
            # Convert to DataFrame and handle MultiIndex
            df = ds_daily[var_name].to_dataframe()
            
            # Reset index to get all columns
            df = df.reset_index()
            
            # Pivot the data to get dates as rows and stations as columns
            df_pivot = df.pivot(index='time', columns='id', values=var_name)
            
            # Convert column names to integers while keeping them as object type
            df_pivot.columns = [str(int(x)) for x in df_pivot.columns]
            
            # Sort rows by date and columns by ID (numerically, but as strings)
            df_pivot = df_pivot.sort_index()
            df_pivot = df_pivot.reindex(sorted(df_pivot.columns, key=lambda x: int(x)), axis=1)
            
            # Write to file with precise formatting
            output_path = self.hype_setup_dir / f'{out_name}.txt'
            df_pivot.to_csv(output_path,
                        sep='\t',
                        float_format='%.3f',
                        date_format='%Y-%m-%d',
                        na_rep='-9999')
            
            self.logger.info(f"Successfully wrote {out_name} file")
            
        except Exception as e:
            self.logger.error(f"Error processing {out_name}: {str(e)}")
            self.logger.error(f"DataFrame head:\n{df.head() if 'df' in locals() else 'DataFrame not created'}")
            raise

    
    def _process_temperature(self, ds: xr.Dataset, units: Dict[str, str]):
        """Process temperature data efficiently."""
        temp = ds[units['in_varname']].copy()
        
        # Vectorized unit conversion
        if units['in_units'] == 'kelvin':
            temp = temp - 273.15
        
        # Calculate daily statistics using xarray's efficient operations
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            tmax = temp.resample(time='D').max()
            tmin = temp.resample(time='D').min()
            tmean = temp.resample(time='D').mean()
        
        return tmax, tmin, tmean

    def _process_precipitation(self, ds: xr.Dataset, units: Dict[str, str]):
        """Process precipitation data efficiently."""
        precip = ds[units['in_varname']].copy()
        
        # Vectorized unit conversion (m/hr to mm/day)
        precip = precip * 24000
        
        # Calculate daily mean efficiently
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            precip_daily = precip.resample(time='D').mean()
        
        return precip_daily

    def write_hype_par_file(self):
        """Write HYPE parameter file following the simple format from documentation"""
        self.logger.info("Writing HYPE parameter file")
        
        try:
            # Read GeoClass.txt
            geoclass_path = self.hype_setup_dir / 'GeoClass.txt'
            if not geoclass_path.exists():
                raise FileNotFoundError("GeoClass.txt not found")
            
            # Read with explicit whitespace separator
            geoclass = pd.read_csv(geoclass_path, sep='\s+', comment='!', header=None,
                                names=['SLC', 'LULC', 'SOIL_TYPE', 'CROP1', 'CROP2', 'CROP_ROT', 
                                    'VEG_TYPE', 'SPECIAL', 'TILE_DEPTH', 'STREAM_DEPTH', 
                                    'N_SOIL_LAYERS', 'DEPTH1', 'DEPTH2', 'DEPTH3'])
            
            # Get unique values
            land_types = sorted(geoclass['LULC'].unique())
            soil_types = sorted(geoclass['SOIL_TYPE'].unique())
            
            output_path = self.hype_setup_dir / 'par.txt'
            
            with open(output_path, 'w') as f:
                # Write header comment
                f.write("!! Parameter file for HYPE model - Generated by CONFLUENCE\n")
                
                # General parameters
                f.write("!! General parameters\n")
                f.write(f"ttpi {1.7083}\n")
                f.write(f"sdnsnew {0.13}\n")
                f.write(f"snowdensdt {0.0016}\n")
                f.write(f"fsceff {1.0}\n")
                f.write(f"cmrefr {0.2}\n")
                f.write(f"lp {0.6613}\n")
                f.write(f"epotdist {4.7088}\n")
                f.write(f"rrcs3 {0.0939}\n")
                f.write(f"deepmem {1000}\n")
                
                # Land use dependent parameters
                f.write("!! Land use dependent parameters\n")
                f.write(f"ttmp {' '.join([f'{v:.4f}' for v in [-0.9253, -1.5960, -0.9620, -2.7121, 2.6945]])}\n")
                f.write(f"cmlt {' '.join([f'{v:.4f}' for v in [9.6497, 9.2928, 9.8897, 5.5393, 2.5333]])}\n")
                f.write(f"cevp {' '.join([f'{v:.4f}' for v in [0.4689, 0.7925, 0.6317, 0.1699, 0.4506]])}\n")
                f.write(f"ttrig {' '.join([f'{v:.4f}' for v in [0.0, 0.0, 0.0, 0.0, 0.0]])}\n")
                f.write(f"treda {' '.join([f'{v:.4f}' for v in [0.84, 0.84, 0.84, 0.84, 0.95]])}\n")
                f.write(f"tredb {' '.join([f'{v:.4f}' for v in [0.4, 0.4, 0.4, 0.4, 0.4]])}\n")
                f.write(f"fepotsnow {' '.join([f'{v:.4f}' for v in [0.8, 0.8, 0.8, 0.8, 0.8]])}\n")
                f.write(f"surfmem {' '.join([f'{v:.4f}' for v in [17.8, 17.8, 17.8, 17.8, 5.15]])}\n")
                f.write(f"depthrel {' '.join([f'{v:.4f}' for v in [1.1152, 1.1152, 1.1152, 1.1152, 2.47]])}\n")
                f.write(f"frost {' '.join([f'{v:.4f}' for v in [2.0, 2.0, 2.0, 2.0, 2.0]])}\n")
                
                # Soil type dependent parameters
                f.write("!! Soil type dependent parameters\n")
                f.write(f"bfroznsoil {' '.join([f'{v:.4f}' for v in [3.7518, 3.2838, 3.7518]])}\n")
                f.write(f"logsatmp {' '.join([f'{v:.4f}' for v in [1.15, 1.15, 1.15]])}\n")
                f.write(f"bcosby {' '.join([f'{v:.4f}' for v in [11.2208, 19.6669, 11.2208]])}\n")
                f.write(f"rrcs1 {' '.join([f'{v:.4f}' for v in [0.4345, 0.5985, 0.4345]])}\n")
                f.write(f"rrcs2 {' '.join([f'{v:.4f}' for v in [0.1201, 0.1853, 0.1201]])}\n")
                f.write(f"wcwp {' '.join([f'{v:.4f}' for v in [0.1171, 0.0280, 0.1171]])}\n")
                f.write(f"wcfc {' '.join([f'{v:.4f}' for v in [0.3771, 0.2009, 0.3771]])}\n")
                f.write(f"wcep {' '.join([f'{v:.4f}' for v in [0.4047, 0.4165, 0.4047]])}\n")
                
                # Lake parameters
                f.write("!! Lake parameters\n")
                f.write(f"ilratk {149.9593}\n")
                f.write(f"ilratp {4.9537}\n")
                f.write(f"illdepth {0.33}\n")
                f.write(f"ilicatch {1.0}\n")
                
                # River parameters
                f.write("!! River parameters\n")
                f.write(f"damp {0.2719}\n")
                f.write(f"rivvel {9.7605}\n")
                f.write(f"qmean {200}\n")
            
            self.logger.info(f"Parameter file written to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error writing parameter file: {str(e)}")
            if 'geoclass' in locals():
                self.logger.error(f"GeoClass data:\n{geoclass.head()}")
            raise


    def write_hype_info_filedir_files(self, spinup_days: int):
        """Write HYPE info and filedir files."""
        self.logger.info("Writing HYPE info and filedir files")
        
        try:
            # Write filedir file
            with open(self.hype_setup_dir / 'filedir.txt', 'w') as f:
                f.write('./')
            
            # Create results directory
            results_dir = self.hype_setup_dir / 'results'
            results_dir.mkdir(exist_ok=True)
            
            # Read Pobs.txt - ensure we parse the time column correctly
            Pobs = pd.read_csv(self.hype_setup_dir / 'Pobs.txt', 
                            sep='\t',
                            index_col='time', 
                            parse_dates=True)
            
            # Get start and end dates from index
            start_date = Pobs.index.min()
            end_date = Pobs.index.max()
            spinup_date = start_date + pd.Timedelta(days=spinup_days)
            
            self.logger.debug(f"Start date: {start_date}")
            self.logger.debug(f"Spinup date: {spinup_date}")
            self.logger.debug(f"End date: {end_date}")
            
            # Write info.txt
            info_content = [
                "!! ----------------------------------------------------------------------------",
                "!!",
                "!! HYPE - Model Agnostic Framework",
                "!!",
                "!! -----------------------------------------------------------------------------",
                "!! Check Indata during first runs (deactivate after first runs)",
                "indatacheckonoff\t2",
                "indatachecklevel\t2",
                "!! -----------------------------------------------------------------------------",
                "!!",
                "!! Simulation settings:",
                "!!",
                "!! -----------------",
                f"bdate\t{start_date.strftime('%Y-%m-%d')}",
                f"cdate\t{spinup_date.strftime('%Y-%m-%d')}",
                f"edate\t{end_date.strftime('%Y-%m-%d')}",
                "resultdir\t./results/",
                "instate\tn",
                "warning\ty",
                "readdaily\ty",
                "submodel\tn",
                "calibration\tn",
                "readobsid\tn",
                "soilstretch\tn",
                "steplength\t1d",
                "!! -----------------------------------------------------------------------------",
                "!!",
                "!! Enable/disable optional input files",
                "!!",
                "!! -----------------",
                "readsfobs\tn\t!! For observed snowfall fractions",
                "readswobs\tn\t!! For observed shortwave radiation",
                "readuobs\tn\t!! For observed wind speeds",
                "readrhobs\tn\t!! For observed relative humidity",
                "readtminobs\tn\t!! For observed min air temperature",
                "readtmaxobs\tn\t!! For observed max air temperature",
                "soiliniwet\tn\t!! Use porosity instead of field capacity",
                "usestop84\tn\t!! Use old return code",
                "!! -----------------------------------------------------------------------------",
                "!!",
                "!! Define model options",
                "!!",
                "!! -----------------",
                "modeloption snowfallmodel\t0",
                "modeloption snowdensity\t0",
                "modeloption snowfalldist\t2",
                "modeloption snowheat\t0",
                "modeloption snowmeltmodel\t0",
                "modeloption snowevapmodel\t1",
                "modeloption snowevaporation\t1",
                "modeloption lakeriverice\t0",
                "modeloption deepground\t0",
                "modeloption glacierini\t1",
                "modeloption floodmodel\t0",
                "modeloption frozensoil\t2",
                "modeloption infiltration\t3",
                "modeloption surfacerunoff\t0",
                "modeloption petmodel\t1",
                "modeloption wetlandmodel\t2",
                "modeloption connectivity\t0",
                "!! -----------------------------------------------------------------------------",
                "!!",
                "!! Define outputs",
                "!!",
                "!! -----------------",
                "timeoutput variable cout",
                "timeoutput variable evap",
                "timeoutput variable snow",
                "timeoutput meanperiod\t1",
                "timeoutput decimals\t3"
            ]
            
            with open(self.hype_setup_dir / 'info.txt', 'w') as f:
                f.write('\n'.join(info_content))
            
            self.logger.info("HYPE info and filedir files written successfully")
            
        except Exception as e:
            self.logger.error(f"Error writing HYPE info files: {str(e)}")
            # Add detailed error info
            self.logger.error(f"Current directory contents: {list((self.hype_setup_dir).glob('*'))}")
            if 'start_date' in locals():
                self.logger.error(f"Start date: {start_date}")
            raise
        
    def _create_geoclass(self):
        """
        Create the GeoClass.txt file with proper NALCMS mapping and all required HYPE columns
        """
        self.logger.info("Creating GeoClass.txt")
        
        try:
            # Read soil and land use statistics
            soil_stats = pd.read_csv(self.gistool_output / 'domain_stats_soil_classes.csv')
            land_stats = pd.read_csv(self.gistool_output / 'domain_stats_NA_NALCMS_landcover_2020_30m.csv')
            
            # NALCMS to HYPE mapping with proper vegetation types
            # Format: (HYPE land use code, vegetation type)
            nalcms_to_hype = {
                1: (1, 2),    # Temperate/subpolar needleleaf forest → Forest, veg=forest
                2: (1, 2),    # Subpolar taiga needleleaf forest → Forest, veg=forest
                6: (1, 2),    # Temperate/subpolar broadleaf deciduous forest → Forest, veg=forest
                8: (1, 2),    # Mixed Forest → Forest, veg=forest
                10: (2, 1),   # Temperate/subpolar grassland → Open ground, veg=open
                14: (2, 1),   # Wetland → Open ground, veg=open
                15: (4, 3),   # Water bodies → Water, veg=water
                16: (3, 1),   # Barren land → Mountain, veg=open
                17: (5, 1),   # Urban → Urban, veg=open
                18: (2, 1),   # Agriculture → Open ground, veg=open
                19: (3, 1),   # Snow/Ice → Mountain, veg=open
            }
            
            # Write file
            with open(self.hype_setup_dir / 'GeoClass.txt', 'w') as f:
                # Write header comments
                f.write("! HYPE GeoClass file - Generated by CONFLUENCE\n")
                f.write("! NALCMS land use classes mapped to HYPE classes\n")
                f.write("! Following columns:\n")
                f.write("! 1:SLC 2:LandUse 3:SoilType 4:Crop1 5:Crop2 6:CropRotation 7:VegType 8:Special\n")
                f.write("! 9:TileDepth 10:StreamDepth 11:SoilLayers 12:SoilDepth1 13:SoilDepth2 14:SoilDepth3\n")
                
                # Process unique combinations
                soil_classes = soil_stats['majority'].unique()
                land_classes = []
                for col in land_stats.columns:
                    if col.startswith('frac_') and any(land_stats[col] > 0.01):
                        land_class = int(col.split('_')[1])
                        if land_class in nalcms_to_hype:
                            land_classes.append(land_class)
                
                # Write data rows
                slc_id = 1
                for soil in soil_classes:
                    for nalcms_class in sorted(land_classes):  # Sort for consistency
                        if nalcms_class in nalcms_to_hype:
                            hype_class, veg_type = nalcms_to_hype[nalcms_class]
                            
                            # Set parameters based on class type
                            if hype_class == 4:  # Water class
                                special_code = 2  # Internal lake
                                n_soil_layers = 1
                                soil_depths = (1.0, 1.0, 1.0)  # As per documentation for water
                                stream_depth = 0.0
                            else:  # Land classes
                                special_code = 0
                                n_soil_layers = 2
                                soil_depths = (0.5, 1.0, 1.0)
                                stream_depth = 2.0
                            
                            # Format: SLC LandUse Soil Crop1 Crop2 CropRot VegType Special TileDepth StreamDepth NSoil Depth1 Depth2 Depth3
                            line = f"{slc_id}  {hype_class}  {int(soil)}  0  0  0  {veg_type}  {special_code}  0.0  {stream_depth}  {n_soil_layers}  {soil_depths[0]}  {soil_depths[1]}  {soil_depths[2]}\n"
                            f.write(line)
                            slc_id += 1
                
            self.logger.info(f"Created GeoClass.txt with {slc_id-1} SLC classes")
            
        except Exception as e:
            self.logger.error(f"Error creating GeoClass.txt: {str(e)}")
            raise


class HYPERunner:
    """
    Runner class for the HYPE model within CONFLUENCE.
    Handles model execution and run-time management.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings
        logger (logging.Logger): Logger instance
        project_dir (Path): Project directory path
        domain_name (str): Name of the modeling domain
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize HYPE runner."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Set up HYPE paths
        self.hype_dir = self._get_hype_path()
        self.setup_dir = self.project_dir / "settings" / "HYPE"
        self.output_dir = self._get_output_path()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_hype(self) -> Optional[Path]:
        """
        Run the HYPE model simulation.
        
        Returns:
            Optional[Path]: Path to output directory if successful, None otherwise
        """
        self.logger.info("Starting HYPE model run")
        
        try:
            
            # Create run command
            cmd = self._create_run_command()
            
            # Set up logging
            log_file = self._setup_logging()
            
            # Execute HYPE
            self.logger.info(f"Executing command: {' '.join(map(str, cmd))}")
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.setup_dir
                )
            
            # Check execution success
            if result.returncode == 0 and self._verify_outputs():
                self.logger.info("HYPE simulation completed successfully")
                return self.output_dir
            else:
                self.logger.error("HYPE simulation failed")
                self._analyze_log_file(log_file)
                return None
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"HYPE execution failed: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error running HYPE: {str(e)}")
            raise

    def _get_hype_path(self) -> Path:
        """Get HYPE installation path."""
        hype_path = self.config.get('HYPE_INSTALL_PATH')
        if hype_path == 'default' or hype_path is None:
            return Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'hype'
        return Path(hype_path)

    def _get_output_path(self) -> Path:
        """Get path for HYPE outputs."""
        if self.config.get('EXPERIMENT_OUTPUT_HYPE') == 'default':
            return (self.project_dir / "simulations" / 
                   self.config.get('EXPERIMENT_ID') / "HYPE")
        return Path(self.config.get('EXPERIMENT_OUTPUT_HYPE'))


    def _create_run_command(self) -> List[str]:
        """Create HYPE execution command."""
        hype_exe = self.hype_dir / self.config.get('HYPE_EXE', 'hype')
        
        cmd = [
            str(hype_exe),
            str(self.setup_dir) + '/'  # HYPE requires trailing slash
        ]
        print(cmd)
        return cmd

    def _setup_logging(self) -> Path:
        """Set up HYPE run logging."""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return log_dir / f'hype_run_{current_time}.log'

    def _verify_outputs(self) -> bool:
        """Verify HYPE output files exist."""
        required_outputs = [
            'timeCOUT.txt',  # Computed discharge
            'timeROUT.txt',  # Routed discharge
            'timeEVAP.txt',  # Evaporation
            'timeSNOW.txt'   # Snow water equivalent
        ]
        
        missing_files = []
        for output in required_outputs:
            if not (self.output_dir / output).exists():
                missing_files.append(output)
        
        if missing_files:
            self.logger.error(f"Missing HYPE output files: {', '.join(missing_files)}")
            return False
        return True

    def _analyze_log_file(self, log_file: Path):
        """Analyze HYPE log file for errors."""
        error_patterns = [
            "ERROR",
            "Error",
            "Failed",
            "FAILED",
            "Invalid",
            "Cannot open"
        ]
        
        try:
            with open(log_file, 'r') as f:
                log_content = f.readlines()
            
            for line in log_content:
                if any(pattern in line for pattern in error_patterns):
                    self.logger.error(f"HYPE error found: {line.strip()}")
        except Exception as e:
            self.logger.error(f"Error analyzing log file: {str(e)}")


class HYPEPostProcessor:
    """
    Postprocessor for HYPE model outputs within CONFLUENCE.
    Handles output extraction, processing, and analysis.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings
        logger (logging.Logger): Logger instance
        project_dir (Path): Project directory path
        domain_name (str): Name of the modeling domain
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        """Initialize HYPE postprocessor."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Setup paths
        self.sim_dir = (self.project_dir / "simulations" / 
                       self.config.get('EXPERIMENT_ID') / "HYPE")
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_results(self) -> Dict[str, Path]:
        """
        Extract and process all HYPE results.
        
        Returns:
            Dict[str, Path]: Paths to processed result files
        """
        self.logger.info("Extracting HYPE results")
        
        try:
            results = {}
            
            # Process streamflow
            streamflow_path = self.extract_streamflow()
            if streamflow_path:
                results['streamflow'] = streamflow_path
            
            # Process water balance
            wb_path = self.extract_water_balance()
            if wb_path:
                results['water_balance'] = wb_path
            
            # Process performance metrics
            perf_path = self.analyze_performance()
            if perf_path:
                results['performance'] = perf_path
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error extracting HYPE results: {str(e)}")
            raise

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract and process streamflow results.
        Handles unit conversions and data organization.
        """
        try:
            self.logger.info("Processing HYPE streamflow results")
            
            # Read computed and routed discharge
            cout = pd.read_csv(self.sim_dir / "timeCOUT.txt", 
                             sep='\t', parse_dates=['time'])
            rout = pd.read_csv(self.sim_dir / "timeROUT.txt", 
                             sep='\t', parse_dates=['time'])
            
            # Create results DataFrame
            results = pd.DataFrame(index=cout['time'])
            
            # Get catchment areas from GeoData
            geodata = pd.read_csv(self.sim_dir / "_settings_backup" / "GeoData.txt", 
                                sep='\t')
            
            # Process each subbasin
            for col in cout.columns:
                if col != 'time':
                    area_km2 = geodata.loc[geodata['subid'] == int(col), 'area'].iloc[0] / 1e6
                    
                    # Add simulated discharge (m3/s and mm/day)
                    results[f'HYPE_discharge_cms_{col}'] = cout[col]
                    results[f'HYPE_discharge_mmday_{col}'] = cout[col] * 86.4 / area_km2
                    
                    # Add observed discharge if available
                    if col in rout.columns:
                        results[f'Obs_discharge_cms_{col}'] = rout[col]
                        results[f'Obs_discharge_mmday_{col}'] = rout[col] * 86.4 / area_km2
            
            # Save results
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_streamflow.csv"
            results.to_csv(output_file)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            return None

    def extract_water_balance(self) -> Optional[Path]:
        """
        Extract and process water balance components.
        """
        try:
            self.logger.info("Processing HYPE water balance")
            
            # Define components to process
            components = {
                'PREC': 'timePREC.txt',  # Precipitation
                'EVAP': 'timeEVAP.txt',  # Evaporation
                'SNOW': 'timeSNOW.txt',  # Snow water equivalent
                'COUT': 'timeCOUT.txt'   # Discharge
            }
            
            # Read all components
            wb_data = {}
            for comp, filename in components.items():
                file_path = self.sim_dir / filename
                if file_path.exists():
                    wb_data[comp] = pd.read_csv(file_path, sep='\t', parse_dates=['time'])
            
            # Create water balance DataFrame
            results = pd.DataFrame(index=wb_data['COUT']['time'])
            
            # Process each component
            for comp, df in wb_data.items():
                for col in df.columns:
                    if col != 'time':
                        results[f'{comp}_{col}'] = df[col]
            
            # Calculate storage change (P - E - Q)
            for col in wb_data['COUT'].columns:
                if col != 'time':
                    P = wb_data['PREC'][col] if 'PREC' in wb_data else 0
                    E = wb_data['EVAP'][col] if 'EVAP' in wb_data else 0
                    Q = wb_data['COUT'][col]
                    results[f'dS_{col}'] = P - E - Q
            
            # Save results
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_water_balance.csv"
            results.to_csv(output_file)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting water balance: {str(e)}")
            return None

    def analyze_performance(self) -> Optional[Path]:
        """
        Analyze model performance using various metrics.
        """
        try:
            self.logger.info("Analyzing HYPE performance")
            
            metrics = {}
            
            # Get simulated and observed discharge
            cout = pd.read_csv(self.sim_dir / "timeCOUT.txt", 
                             sep='\t', parse_dates=['time'])
            rout = pd.read_csv(self.sim_dir / "timeROUT.txt", 
                             sep='\t', parse_dates=['time'])
            
            # Calculate metrics for each subbasin
            for col in cout.columns:
                if col != 'time' and col in rout.columns:
                    sim = cout[col]
                    obs = rout[col]
                    
                    # Calculate various performance metrics
                    metrics[col] = {
                        'NSE': self._calc_nse(sim, obs),
                        'RMSE': self._calc_rmse(sim, obs),
                        'KGE': self._calc_kge(sim, obs),
                        'Bias': self._calc_bias(sim, obs)
                    }
            
            # Create performance summary DataFrame
            summary = pd.DataFrame(metrics).T
            
            # Save results
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_performance.csv"
            summary.to_csv(output_file)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            return None

    @staticmethod
    def _calc_nse(sim: np.ndarray, obs: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency."""
        return 1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)

    @staticmethod
    def _calc_rmse(sim: np.ndarray, obs: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(np.mean((sim - obs) ** 2))

    @staticmethod
    def _calc_kge(sim: np.ndarray, obs: np.ndarray) -> float:
        """Calculate Kling-Gupta Efficiency."""
        r = np.corrcoef(sim, obs)[0, 1]
        alpha = sim.std() / obs.std()
        beta = sim.mean() / obs.mean()
        return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    @staticmethod
    def _calc_bias(sim: np.ndarray, obs: np.ndarray) -> float:
        """
        Calculate mean bias between simulated and observed values.
        
        Args:
            sim (np.ndarray): Simulated values
            obs (np.ndarray): Observed values
            
        Returns:
            float: Mean bias as percentage
        """
        return 100 * (sim.mean() - obs.mean()) / obs.mean()

    def compile_summary_report(self) -> Optional[Path]:
        """
        Create a comprehensive summary report of HYPE results.
        
        Returns:
            Optional[Path]: Path to summary report file
        """
        try:
            self.logger.info("Compiling HYPE summary report")
            
            report_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_hype_summary.txt"
            
            with open(report_file, 'w') as f:
                # Write header
                f.write(f"HYPE Model Results Summary\n")
                f.write(f"========================\n\n")
                f.write(f"Domain: {self.domain_name}\n")
                f.write(f"Experiment ID: {self.config['EXPERIMENT_ID']}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Add performance metrics summary
                perf_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_performance.csv"
                if perf_file.exists():
                    perf_df = pd.read_csv(perf_file)
                    f.write("Performance Metrics Summary\n")
                    f.write("-----------------------\n")
                    f.write(f"Mean NSE: {perf_df['NSE'].mean():.3f}\n")
                    f.write(f"Mean KGE: {perf_df['KGE'].mean():.3f}\n")
                    f.write(f"Mean Bias: {perf_df['Bias'].mean():.3f}%\n\n")
                
                # Add water balance summary
                wb_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_water_balance.csv"
                if wb_file.exists():
                    wb_df = pd.read_csv(wb_file)
                    f.write("Water Balance Summary\n")
                    f.write("--------------------\n")
                    # Calculate annual averages
                    wb_df['year'] = pd.to_datetime(wb_df['time']).dt.year
                    annual_means = wb_df.groupby('year').mean()
                    f.write(f"Annual Average Precipitation: {annual_means.filter(like='PREC').mean().mean():.1f} mm\n")
                    f.write(f"Annual Average Evaporation: {annual_means.filter(like='EVAP').mean().mean():.1f} mm\n")
                    f.write(f"Annual Average Discharge: {annual_means.filter(like='COUT').mean().mean():.1f} mm\n\n")
                
                # Add simulation notes
                f.write("Simulation Notes\n")
                f.write("----------------\n")
                sim_success = all(file.exists() for file in [perf_file, wb_file])
                f.write(f"Simulation Status: {'Successful' if sim_success else 'Incomplete'}\n")
                if not sim_success:
                    f.write("Missing output files detected - check simulation logs\n")
            
            self.logger.info(f"Summary report saved to {report_file}")
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error creating summary report: {str(e)}")
            return None

    def plot_results(self) -> Dict[str, Path]:
        """
        Create standard visualization plots for HYPE results.
        
        Returns:
            Dict[str, Path]: Dictionary mapping plot types to file paths
        """
        try:

            
            self.logger.info("Creating HYPE results plots")
            plots = {}
            
            # Set up plotting style
            plt.style.use('seaborn')
            
            # Streamflow plot
            streamflow_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_streamflow.csv"
            if streamflow_file.exists():
                df = pd.read_csv(streamflow_file, parse_dates=['time'])
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sim_cols = [col for col in df.columns if 'HYPE_discharge_cms' in col]
                obs_cols = [col for col in df.columns if 'Obs_discharge_cms' in col]
                
                for sim, obs in zip(sim_cols, obs_cols):
                    ax.plot(df['time'], df[sim], label='Simulated', alpha=0.8)
                    ax.plot(df['time'], df[obs], label='Observed', alpha=0.8)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Discharge (m³/s)')
                ax.set_title('HYPE Streamflow Comparison')
                ax.legend()
                
                plot_path = self.results_dir / f"{self.config['EXPERIMENT_ID']}_streamflow_plot.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots['streamflow'] = plot_path
            
            # Performance metrics plot
            perf_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_performance.csv"
            if perf_file.exists():
                df = pd.read_csv(perf_file)
                
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.boxplot(data=df[['NSE', 'KGE', 'Bias']], ax=ax)
                ax.set_title('HYPE Performance Metrics Distribution')
                ax.set_ylabel('Value')
                
                plot_path = self.results_dir / f"{self.config['EXPERIMENT_ID']}_performance_plot.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots['performance'] = plot_path
            
            return plots
            
        except ImportError:
            self.logger.warning("Plotting libraries not available - skipping plot generation")
            return {}
        except Exception as e:
            self.logger.error(f"Error creating plots: {str(e)}")
            return {}