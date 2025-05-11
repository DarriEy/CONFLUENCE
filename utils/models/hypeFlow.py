# Load needed packages
import xarray as xr
import pint_xarray
import pint
import glob
import netCDF4 as nc4
import os
import cdo
import pandas as pd
from   easymore import Easymore
import numpy       as      np
import geopandas   as      gpd
import sys
from   itertools   import  product
import datetime
from alive_progress import alive_bar #progress bar
import shutil

# sort geodata from upstream to downstream
def sort_geodata(geodata):
    """Sort sub-basins from upstream to downstream using topological sorting.
    Handles cycles by breaking them at the highest downstream point."""
    
    import networkx as nx
    
    # Create directed graph from subid -> maindown relationships
    G = nx.DiGraph()
    for _, row in geodata.iterrows():
        if row['maindown'] > 0:  # Only add valid downstream connections
            G.add_edge(row['subid'], row['maindown'])
    
    # Find and break cycles if they exist
    cycles = list(nx.simple_cycles(G))
    if cycles:
        print(f"Warning: Found {len(cycles)} circular reference(s) in the network")
        for cycle in cycles:
            # Find the node in the cycle with the most downstream connections
            # and break the cycle there
            max_downstream = max(cycle, 
                key=lambda n: len(list(nx.descendants(G, n))))
            # Find the edge pointing to this node within the cycle
            cycle_idx = cycle.index(max_downstream)
            from_node = cycle[cycle_idx-1]
            # Remove this edge to break the cycle
            G.remove_edge(from_node, max_downstream)
            print(f"Breaking cycle at edge: {from_node} -> {max_downstream}")
    
    try:
        # Find all nodes with no incoming edges (headwaters)
        headwaters = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        # For each headwater, find all downstream nodes and their distances
        ordered_subids = []
        visited = set()
        
        def traverse_downstream(node, depth=0):
            """Recursively traverse downstream, tracking depth"""
            if node in visited:
                return
            visited.add(node)
            # Get all downstream nodes
            downstream = list(G.successors(node))
            if downstream:
                # Recursively process downstream nodes
                for next_node in downstream:
                    traverse_downstream(next_node, depth + 1)
            # Add node to ordered list with its depth
            ordered_subids.append((node, depth))
        
        # Process each headwater
        for hw in headwaters:
            traverse_downstream(hw)
        
        # Sort by depth (upstream to downstream)
        ordered_subids.sort(key=lambda x: x[1])
        ordered_subids = [x[0] for x in ordered_subids]  # Extract just the subids
        
        # Handle nodes that weren't reached (isolated nodes)
        missing_subids = geodata[~geodata['subid'].isin(ordered_subids)]['subid'].tolist()
        
        # Add missing subids at the start
        final_order = missing_subids + ordered_subids
        
        # Create a mapping from subid to desired position
        position_map = {subid: pos for pos, subid in enumerate(final_order)}
        
        # Sort geodata based on the position map
        geodata['sort_idx'] = geodata['subid'].map(position_map)
        geodata = geodata.sort_values('sort_idx', ignore_index=True)
        geodata = geodata.drop(columns=['sort_idx'])
        
        # Verify the sorting
        for i, row in geodata.iterrows():
            if row['maindown'] > 0:
                downstream_idx = geodata.index[geodata['subid'] == row['maindown']].tolist()
                if downstream_idx and downstream_idx[0] < i:
                    print(f"Warning: Basin {row['subid']} appears before its downstream basin {row['maindown']}")
        
        return geodata
        
    except Exception as e:
        print(f"Error during sorting: {str(e)}")
        return geodata  # Return unsorted data if we can't resolve the ordering
#---------------------------------------------------------------
#---------------------------------------------------------------

# write HYPE forcing from easymore nc files
def write_hype_forcing(easymore_output, timeshift, forcing_units, geofabric_mapping, path_to_save, cache_path):
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)
    # function to get daily values from hourly timeseries
    def convert_hourly_to_daily (input_file_name,
                                 variable_in,
                                 variable_out,
                                 variable_out_long_name = None,
                                 var_unit_conversion = None,
                                 var_time = 'time',
                                 var_id = 'id',
                                 time_diff = 0,
                                 stat = 'max', 
                                 output_file_name_nc = None,
                                 output_file_name_txt = None,
                                 Fill_value = -9999.0): # 'max', 'min', 'mean'

        # read the input houtly nc file
        ds = xr.open_dataset(input_file_name)
        # set id as integer
        ds.coords[var_id] = ds.coords[var_id].astype(int)

        # drop all the other variables except the mentioned varibale, time and id
        variables_to_keep = [variable_in, var_time]
        if not var_id is None:
            variables_to_keep.append(var_id)

        # Drop all variables except the specified ones
        ds = ds.drop([v for v in ds.variables if v not in variables_to_keep])

        # roll the time based on hour of difference to have more accurate
        if time_diff !=0:
            ds[var_time] = ds[var_time].roll(time=time_diff)
            # Remove the first or last roll_steps time steps
            if time_diff < 0:
                ds = ds.isel(time=slice( None, time_diff))
            elif time_diff > 0:
                ds = ds.isel(time=slice( time_diff, None))

        # to create the xarray dataframe with daily time
        if stat == 'max':
            ds_daily = ds.resample(time='D').max()
        elif stat == 'min':
            ds_daily = ds.resample(time='D').min()
        elif stat == 'mean':
            ds_daily = ds.resample(time='D').mean()
        elif stat == 'sum':
            ds_daily = ds.resample(time='D').sum()
        else:
            sys.exit('input stat should be max, min, mean or sum')

        # conversion of units based on provided conversion unit
        ds_daily[variable_in] = ds_daily[variable_in].pint.quantify(var_unit_conversion['in_unit'])
        ds_daily[variable_in] = ds_daily[variable_in].pint.to(var_unit_conversion['out_unit'])
        ds_daily = ds_daily.pint.dequantify()

        # drop the vairiable in
        ds_daily = ds_daily.rename({variable_in: variable_out})

        # add long name
        if not variable_out_long_name is None:
            ds_daily[variable_out].attrs['long_name'] = variable_out_long_name

        # transpose the variable
        ds_daily[variable_out] = ds_daily[variable_out].transpose()

        # this section is written to avoid issues with netcdf and HYPE!
        # I could not find what is the issue, however, when the data is 
        # transferred to df, tranfer back to xarray and saved, the issue
        # with HYPE is resolved. this need closer look. Also HYPE netcdf
        # is in its initial stage of developement and can have issue as
        # well
        df = ds_daily[variable_out].to_dataframe()
        df = df.unstack()
        df = df.T
        df = df.droplevel(level=0, axis=0)
        df.columns.name = None
        df.index.name = var_time
        if not output_file_name_txt is None:
            df.to_csv(output_file_name_txt,\
                      sep='\t', na_rep='', index_label='time', float_format='%.3f')
        esmr = Easymore()
        ds_daily = esmr.dataframe_to_netcdf_xr(df,
                                         data_frame_DateTime_column = var_time,
                                         variable_name = variable_out,
                                         variable_dim_name = 'id',
                                         unit_of_variable = var_unit_conversion['out_unit'],
                                         variable_long_name = variable_out_long_name,
                                         Fill_value = Fill_value)

        # save the file if path is provided
        if not output_file_name_nc is None:
            if os.path.isfile(output_file_name_nc):
                os.remove(output_file_name_nc)
            ds_daily.to_netcdf(output_file_name_nc,\
                               encoding = {variable_out:{'_FillValue':Fill_value}})

        # return
        return ds_daily
    ############
    print('Merging easymore outputs to one NetCDF file \n')
    # Replace with your file path pattern
    easymore_nc_files = sorted(glob.glob(easymore_output+'/*.nc'))
    # split the files in batches as cdo cannot mergetime long list of file names
    batch_size = 20
    # avoid splitting files if their number is too small
    if(len(easymore_nc_files) < batch_size):
        batch_size = len(easymore_nc_files)
    files_split = np.array_split(easymore_nc_files,batch_size)
    cdo_obj = cdo.Cdo()  # CDO object
    intermediate_files = []

    # split files into intermediate files
    # Combine in batches
    with alive_bar(batch_size, force_tty=True) as bar:
        for i in range(batch_size):
            # print(f'Processing easymore outputs: batch no {i+1} out of {batch_size} \n')
            batch_files = files_split[i].tolist()
            batch_output = f"{cache_path}/forcing_batch_{i}.nc"
            cdo_obj.mergetime(input=batch_files, output=batch_output)
            intermediate_files.append(batch_output)
            bar()

    # Combine intermediate results into one netcdf file
    cdo_obj.mergetime(input=intermediate_files, output=f"{cache_path}/merged_forcing.nc")

    # Clean up intermediate files if needed
    for f in intermediate_files:
        os.remove(f)

    # open the forcing file
    forcing = xr.open_dataset(f"{cache_path}/merged_forcing.nc")
    # convert calendar to 'standard'
    forcing = forcing.convert_calendar('standard')
    # The data are in UTC time and they need to be shifted by "timeshift" to local time
    forcing['time'] = forcing['time'] + pd.Timedelta(hours=timeshift)
    # write to netcdf
    forcing.to_netcdf(f"{cache_path}/merged_forcing.nc")
    forcing.close()
    ############
    print('Get average daily values for HYPE \n')
    basinID = geofabric_mapping['basinID']['in_varname']
    ds1= convert_hourly_to_daily(f"{cache_path}/merged_forcing.nc",
                                forcing_units['temperature']['in_varname'],
                                'TMAXobs',
                                var_unit_conversion = {'in_unit':forcing_units['temperature']['in_units'],'out_unit':forcing_units['temperature']['out_units']},
                                var_time = 'time',
                                var_id = basinID,
                                time_diff = -7,
                                stat = 'max',
                                # output_file_name_nc = path_to_save+'TMAXobs.nc',
                                output_file_name_txt = path_to_save+'TMAXobs.txt')

    ds2= convert_hourly_to_daily(f"{cache_path}/merged_forcing.nc",
                                forcing_units['temperature']['in_varname'],
                                'TMINobs',
                                var_unit_conversion = {'in_unit':forcing_units['temperature']['in_units'],'out_unit':forcing_units['temperature']['out_units']},
                                var_time = 'time',
                                var_id = basinID,
                                time_diff = -7,
                                stat = 'min',
                                # output_file_name_nc = path_to_save+'TMINobs.nc',
                                output_file_name_txt = path_to_save+'TMINobs.txt')

    ds3= convert_hourly_to_daily(f"{cache_path}/merged_forcing.nc",
                                forcing_units['temperature']['in_varname'],
                                'Tobs',
                                var_unit_conversion = {'in_unit':forcing_units['temperature']['in_units'],'out_unit':forcing_units['temperature']['out_units']},
                                var_time = 'time',
                                var_id = basinID,
                                time_diff = -7,
                                stat = 'mean',
                                # output_file_name_nc = path_to_save+'Tobs.nc',
                                output_file_name_txt = path_to_save+'Tobs.txt')

    ds4= convert_hourly_to_daily(f"{cache_path}/merged_forcing.nc",
                                forcing_units['precipitation']['in_varname'],
                                'Pobs',
                                var_unit_conversion = {'in_unit':forcing_units['precipitation']['in_units'],'out_unit':forcing_units['precipitation']['out_units']},
                                var_time = 'time',
                                var_id = basinID,
                                time_diff = -7,
                                stat = 'mean',
                                # output_file_name_nc = path_to_save+'Pobs.nc',
                                output_file_name_txt = path_to_save+'Pobs.txt')
    
    # remove the merged netcdf file
    shutil.rmtree(f"{cache_path}")

#---------------------------------------------------------------
#---------------------------------------------------------------

# write GeoData and GeoClass files
def write_hype_geo_files(gistool_output, subbasins_shapefile, rivers_shapefile, frac_threshold, geofabric_mapping, path_to_save):
    # Extract mapping variables
    basinID = geofabric_mapping['basinID']['in_varname']
    NextDownID = geofabric_mapping['nextDownID']['in_varname']
    
    # 1. Start with river network as our base - it defines the topology
    riv = gpd.read_file(rivers_shapefile)
    
    # 2. Create base dataframe with correct topology
    base_df = pd.DataFrame({
        'subid': riv[basinID],
        'maindown': riv[NextDownID]
    })
    
    # 3. Add river properties
    rivlen_name = geofabric_mapping['rivlen']['in_varname']
    ureg = pint.UnitRegistry()
    lengthm = riv[rivlen_name].values * ureg(geofabric_mapping['rivlen']['in_units'])
    base_df['rivlen'] = lengthm.to(geofabric_mapping['rivlen']['out_units']).magnitude
    base_df['slope_mean'] = riv['Slope']
    
    # 4. Add catchment properties
    cat = gpd.read_file(subbasins_shapefile)
    cat_props = pd.DataFrame({
        basinID: cat[basinID],
        'area': cat[geofabric_mapping['area']['in_varname']].values * ureg(geofabric_mapping['area']['in_units']).to(geofabric_mapping['area']['out_units']).magnitude,
        'latitude': cat.centroid.y,
        'longitude': cat.centroid.x
    }).set_index(basinID)
    
    # 5. Add soil, landcover and elevation data
    soil_type = pd.read_csv(gistool_output+'modified_domain_stats_soil_classes.csv').set_index(basinID)
    landcover_type = pd.read_csv(gistool_output+'modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv').set_index(basinID)
    elevation_mean = pd.read_csv(gistool_output+'modified_domain_stats_elv.csv').set_index(basinID)
    
    # 6. Calculate SLC combinations
    combinations_set_all = set()
    for basin_id in landcover_type.index:
        # Get fractions above threshold for this basin
        lc_fracs = [col for col in landcover_type.columns if col.startswith('frac_') 
                    and landcover_type.loc[basin_id, col] > frac_threshold]
        lc_values = [int(col.split('_')[1]) for col in lc_fracs]
        
        # Get soil type for this basin
        soil_value = [soil_type.loc[basin_id, 'majority']]
        
        # Add all combinations
        combinations_set_all.update(product(lc_values, soil_value))
    
    # Create SLC mapping
    slc_df = pd.DataFrame(combinations_set_all, columns=['landcover', 'soil'])
    slc_df['SLC'] = range(1, len(slc_df) + 1)
    
    # 7. Calculate SLC fractions for each basin
    slc_columns = [f'SLC_{i}' for i in range(1, len(slc_df) + 1)]
    for basin_id in base_df['subid']:
        for slc_idx, (lc, soil) in enumerate(zip(slc_df['landcover'], slc_df['soil']), 1):
            lc_frac = landcover_type.loc[basin_id, f'frac_{lc}'] if f'frac_{lc}' in landcover_type.columns else 0
            if lc_frac > frac_threshold and soil_type.loc[basin_id, 'majority'] == soil:
                base_df.loc[base_df['subid'] == basin_id, f'SLC_{slc_idx}'] = lc_frac
            else:
                base_df.loc[base_df['subid'] == basin_id, f'SLC_{slc_idx}'] = 0
    
    # 8. Add remaining properties
    base_df = base_df.join(cat_props, on='subid')
    base_df['elev_mean'] = base_df['subid'].map(elevation_mean['mean'])
    
    # 9. Normalize SLC fractions
    slc_cols = [col for col in base_df.columns if col.startswith('SLC_')]
    base_df[slc_cols] = base_df[slc_cols].div(base_df[slc_cols].sum(axis=1), axis=0)
    
    # 10. Sort and save
    sorted_df = sort_geodata(base_df)
    sorted_df.to_csv(path_to_save+'GeoData.txt', sep='\t', index=False)
    
    # Write GeoClass file (similar to before)
    write_geoclass(slc_df, path_to_save)

#---------------------------------------------------------------
#---------------------------------------------------------------

# write par.txt file
def write_hype_par_file(path_to_save):

    output_file = path_to_save+'par.txt'

    if os.path.isfile(output_file):
        os.remove(output_file)
    par_file = """!!	=======================================================================================================									
!! Parameter file for:										
!! HYPE -- Generated by the Model Agnostic Framework (hypeflow)									
!!	=======================================================================================================									
!!										
!!	------------------------									
!!										
!!	=======================================================================================================									
!!	"SNOW - MELT, ACCUMULATION, AND DISTRIBUTION; sublimation is sorted under Evapotranspiration"									
!!	-----									
!!	"General snow accumulation and melt related parameters (baseline values from SHYPE, unless noted otherwise)"									
ttpi	1.7083	!! width of the temperature interval with mixed precipitation								
sdnsnew	0.13	!! density of fresh snow (kg/dm3)								
snowdensdt	0.0016	!! snow densification parameter								
fsceff	1	!! efficiency of fractional snow cover to reduce melt and evap								
cmrefr	0.2	"!! snow refreeze capacity (fraction of degreeday melt factor) - baseline value from HBV (pers comm Barbro Johansson, but also in publications)"								
!!	-----									
!!	Landuse dependent snow melt parameters									
!!LUSE:	LU1	LU2	LU3	LU4	LU5					
ttmp	 -0.9253	 -1.5960	 -0.9620	 -2.7121	  2.6945    -0.9253	 -1.5960	 -0.9620	 -2.7121	  2.6945    -0.9253	 -1.5960	 -0.9620	 -2.7121	  2.6945    -0.9253	 -1.5960	 -0.9620	 -2.7121	  2.6945    !! Snowmelt threshold temperature (deg), baseline zero for all landuses"				
cmlt	   9.6497	   9.2928	   9.8897	   5.5393	   2.5333   9.6497	   9.2928	   9.8897	   5.5393	   2.5333   9.6497	   9.2928	   9.8897	   5.5393	   2.5333   9.6497	   9.2928	   9.8897	   5.5393	   2.5333	!! Snowmelt degree day coef (mm/deg/timestep)							
!!	-----									
!!	=======================================================================================================									
!!	EVAPOTRANSPIRATION PARAMETERS									
!!	-----									
!!	General evapotranspiration parameters									
lp	    0.6613	!! Threshold for water content reduction of transpiration (fraction of field capacity) - baseline value from SHYPE because its more realistic with a value slightly below field capacity								
epotdist	   4.7088	!! Coefficient in exponential function for potential evapotranspiration's depth dependency - baseline from EHYPE and/or SHYPE (very similar)																					
!!	-----									
!!										
!!LUSE:	LU1	LU2	LU3	LU4	LU5					
cevp	  0.4689	  0.7925	  0.6317	  0.1699	  0.4506    0.4689	  0.7925	  0.6317	  0.1699	  0.4506    0.4689	  0.7925	  0.6317	  0.1699	  0.4506    0.4689	  0.7925	  0.6317	  0.1699	  0.4506
ttrig	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	!! Soil temperature threshold to allow transpiration - disabled if treda is set to zero				
treda	0.84	0.84	0.84	0.84	0.95	0.84	0.84	0.84	0.84	0.95	0.84	0.84	0.84	0.84	0.95	0.84	0.84	0.84	0.84	0.95	"!! Coefficient in soil temperature response function for root water uptake, default value from gren et al, set to zero to disable the function"				
tredb	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	0.4	"!! Coefficient in soil temperature response fuction for root water uptake, default value from gren et al"				
fepotsnow	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	0.8	!! Fraction of potential evapotranspiration used for snow sublimation				
!!										
!! Frozen soil infiltration parameters										
!! SOIL:	S1	S2								
bfroznsoil  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838  3.7518  3.2838								
logsatmp	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15	1.15								
bcosby	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669	    11.2208	    19.6669								
!!	=======================================================================================================									
!!	"SOIL/LAND HYDRAULIC RESPONSE PARAMETERS - recession coef., water retention, infiltration, macropore, surface runoff; etc."									
!!	-----									
!!	Soil-class parameters									
!!	S1	S2								
rrcs1   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985   0.4345  0.5985	!! recession coefficients uppermost layer (fraction of water content above field capacity/timestep)							
rrcs2   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853   0.1201  0.1853	!! recession coefficients bottom layer (fraction of water content above field capacity/timestep)							
rrcs3	    0.0939	!! Recession coefficient (upper layer) slope dependance (fraction/deg)								
sfrost  1   1  1   1  1   1  1   1  1   1  1   1  1   1  1   1  1   1  1   1	!! frost depth parameter (cm/degree Celsius) soil-type dependent							
wcwp    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280    0.1171  0.0280	!! Soil water content at wilting point (volume fraction)											
wcfc    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009    0.3771  0.2009	!! Field capacity, layerOne (additional to wilting point) (volume fraction)"										
wcep    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165    0.4047  0.4165	!! Effective porosity, layerOne (additional to wp and fc) (volume fraction)"							
!!	-----									
!!	Landuse-class parameters	parameters								
!!LUSE:	LU1	LU2	LU3	LU4	LU5					
srrcs   0.0673  0.1012  0.1984  0.0202  0.0202   0.0673  0.1012  0.1984  0.0202  0.0202   0.0673  0.1012  0.1984  0.0202  0.0202   0.0673  0.1012  0.1984  0.0202  0.0202	!! Runoff coefficient for surface runoff from saturated overland flow of uppermost soil layer (fraction/timestep)				
!!	-----									
!!	Regional groundwater outflow									
rcgrw	0	!! recession coefficient for regional groundwater outflow from soil layers								
!!	=======================================================================================================									
!!	SOIL TEMPERATURE AND SOIL FROST DEPT									
!!	-----									
!!	General									
deepmem	1000	!! temperature memory of deep soil (days)								!! temperature memory of deep soil (days)							
!!-----										
!!LUSE:	LU1	LU2	LU3	LU4	LU5					
surfmem 17.8	17.8	17.8	17.8	5.15 17.8	17.8	17.8	17.8	5.15 17.8	17.8	17.8	17.8	5.15 17.8	17.8	17.8	17.8	5.15	!! upper soil layer soil temperature memory (days)				
depthrel	1.1152	1.1152	1.1152	1.1152	2.47    1.1152	1.1152	1.1152	1.1152	2.47	1.1152	1.1152	1.1152	1.1152	2.47	1.1152	1.1152	1.1152	1.1152	2.47	!! depth relation for soil temperature memory (/m)				
frost	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	2	!! frost depth parameter (cm/degree Celsius) soil-type dependent				
!!	-----									
!!	=======================================================================================================									
!!	LAKE DISCHARGE									
!!	-----									
!!	-----									
!!	"ILAKE and OLAKE REGIONAL PARAMETERS (1 ilakeregions , defined in geodata)"									
!!	ILAKE parameters																	
!! ilRegion	PPR 1									
ilratk  149.9593						
ilratp  4.9537						
illdepth    0.33					
ilicatch    1.0								
!!										
!!	=======================================================================================================									
!!	RIVER ROUTING									
!!	-----									
damp	   0.2719	!! fraction of delay in the watercourse which also causes damping								
rivvel	     9.7605	!! celerity of flood in watercourse (rivvel>0)								
qmean 	200	!! initial value for calculation of mean flow (mm/yr) - can also be given in LakeData								"""

    with open(output_file, 'w') as file:
            file.write(par_file)
    
################################################################
# write info and filedir files
def write_hype_info_filedir_files(path_to_save, spinup_days, hype_results_dir):
    # write filedir file
    output_file = path_to_save+'filedir.txt'

    if os.path.isfile(output_file):
        os.remove(output_file)

    with open(output_file, 'w') as file:
            file.write('./')

    # create results directory
    #if not os.path.isdir(hype_results_dir+'/results'):
    #    os.makedirs(hype_results_dir+'/results')
        
    ###########

    # Output par to a .txt file

    output_file = path_to_save+'info.txt'
    if os.path.isfile(output_file):
        os.remove(output_file)


    # define start time, end time, based on input forcing
    # spinup period is a user defined inputs

    Pobs = pd.read_csv(path_to_save+'Pobs.txt', sep='\t', parse_dates=['time'])
    Pobs['time'] = Pobs['time'].dt.date
    start_date = Pobs['time'].iloc[0]
    end_date = Pobs['time'].iloc[-1]
    spinup_date = start_date + pd.Timedelta(days=spinup_days)


    # write out first text section
    s1= [
    """!! ----------------------------------------------------------------------------							
!!							
!! HYPE - Model Agnostic Framework
!!							
!! -----------------------------------------------------------------------------							
!! Check Indata during first runs (deactivate after first runs) 
indatacheckonoff 	2						
indatachecklevel	2		
!! -----------------------------------------------------------------------------							"""
    ]

    # write s1 in output file
    with open(output_file, 'w') as file:
        # Write the commented lines
        for line in s1:
            file.write(line + '\n')

    # write out first text section
    s2= [
    """!!
!! -----------------------------------------------------------------------------							
!!						
!! Simulation settings:							
!!							
!! -----------------	 """
    ]

    # write s2 in output file
    with open(output_file, 'a') as file:
        # Write the commented lines
        for line in s2:
            file.write(line + '\n')

    # create df2
    df2_row=['bdate','cdate','edate','resultdir','instate', 'warning']
    df2_val=[start_date,spinup_date,end_date, hype_results_dir, 'n','y']
    df2=pd.DataFrame(df2_val, index=df2_row, columns=None)

    # append df2
    with open(output_file, 'a') as file:
        # Write the DataFrame to the file
        df2.to_csv(file, sep='\t', index=True, header=False)#, line_terminator='\n')

    # write out s3
    s3= [
    """readdaily 	y						
submodel 	n						
calibration	n						
readobsid   n							
soilstretch	n						
!! Soilstretch enable the use of soilcorr parameters (strech soildepths in layer 2 and 3)
steplength	1d							
!! -----------------------------------------------------------------------------							
!!							
!! Enable/disable optional input files
!!							
!! -----------------							
readsfobs	n	!! For observed snowfall fractions in SFobs.txt							
readswobs	n	!! For observed shortwave radiation in SWobs.txt
readuobs	n	!! For observed wind speeds in Uobs.txt
readrhobs	n	!! For observed relative humidity in RHobs.txt					
readtminobs	y	!! For observed min air temperature in TMINobs.txt				
readtmaxobs	y	!! For observed max air temperature in TMAXobs.txt
soiliniwet	n	!! initiates soil water to porosity instead of field capacity which is default (N). Set Y to use porosity.
usestop84	n	!! flag to use the old return code 84 for a successful run					
!! -----------------------------------------------------------------------------							
!!							
!! Define model options (optional)
!!							
!! -----------------							
!!snowfallmodel:								
!!                  0 threshold temperature model							
!!                  1 inputdata (SFobs.txt)							
!!snowmeltmodel:							
!!                  0,1 temperature index             (with/without snowcover scaling)							
!!                  2   temperature + radiation index (with/without snowcover scaling)							
!!							
!!  snowevapmodel   0 off							
!!                  1 on							
!!                   							
!!  petmodel:  (potential evapotranspiration) (is shown in geodata for WWH)							
!!                  0 original HYPE temperature model (with Xobs epot replacement)							
!!                  1 original HYPE temperature model (without Xobs epot replacement)							
!!                  2 Modified Jensen-Haise 							
!!                  3 Modified Hargreaves-Samani							
!!                  4 Priestly-Taylor							
!!                  5 FAo Penman-Monteith							
!!							
!! lakeriverice:							
!!                  0 off							
!!                  1 on, old (simple) air-water heat exchange              (requires T2 water temperature model)							
!!                  2 on, new heatbalance model for air-water heat exchange (requires T2 water temperature model)							
!!							
!! substance T2     switching on the new water temperature trace model							
!!							
!! deepground       0   off    Deep groundwater (Aquifer) model options							
!!                  1,2 on
!! Glacierini	0 off 1 on	(1 used for statefile preparation)	
!! Floodplain		0, 1, 2, 3 (3 used for WWH)					
!! -----------------							
modeloption snowfallmodel	0						
modeloption snowdensity	0
modeloption snowfalldist	2
modeloption snowheat	0
modeloption snowmeltmodel	0	
modeloption	snowevapmodel	1				
modeloption snowevaporation	1					
modeloption lakeriverice	0									
modeloption deepground	0 	
modeloption glacierini	1
modeloption floodmodel 0
modeloption frozensoil 2
modeloption infiltration 3
modeloption surfacerunoff 0
modeloption petmodel	1
modeloption wetlandmodel 2		
modeloption connectivity 0					
!! ------------------------------------------------------------------------------------							
!!							
!! Define outputs
!!							
!! -----------------							
!! meanperiod 1=daymean, 2=weekmean, 3=monthmean, 4=yearmean, 5=total period mean							
!! output variables: see http://www.smhi.net/hype/wiki/doku.php?id=start:hype_file_reference:info.txt:variables 
!! -----------------							
!! BASIN outputs 
!! The present basins are some large rivers distributed over different continents
!! -----------------
!! basinoutput variable	rout	cout	cilv	evap	fnca	fcon
!! basinoutput meanperiod	1						
!! basinoutput decimals	3
!! basinoutput subbasin	basinID1    basinID2    basinID3		
!! -----------------							
!! TIME outputs 
!! -----------------	
timeoutput variable cout	evap	snow
timeoutput meanperiod	1
timeoutput decimals	3					
!! -----------------							
!! MAP outputs
!! -----------------							
!! mapoutput variable	cout cprc ctmp
!! mapoutput decimals	3						
!! mapoutput meanperiod	5						
!! ------------------------------------------------------------------------------------							
!!							
!! Select criteria for model evaluation and automatic calibration
!!							
!! -----------------							
!! General settings
!! -----------------			
!! crit meanperiod	1
!! crit datalimit	30
!! crit subbasin	outletBasinID
!! -----------------			
!! Criterion-specific settings
!! -----------------				
!! crit 1 criterion	MKG
!! crit 1 cvariable	cout
!! crit 1 rvariable	rout
!! crit 1 weight	1"""
    ]

    # write s3 in output file
    with open(output_file, 'a') as file:
        # Write the commented lines
        for line in s3:
            file.write(line + '\n')

def write_geoclass(slc_df, path_to_save):
    """Write GeoClass.txt file for HYPE model with full metadata and specific formatting."""
    
    # Prepare the combination DataFrame with all required columns
    combination = slc_df.copy()
    combination = combination.rename(columns={'landcover': 'LULC', 'soil': 'SOIL TYPE'})
    combination = combination[['SLC', 'LULC', 'SOIL TYPE']]
    
    # Add additional required columns with default values
    combination['Main crop cropid'] = 0
    combination['Second crop cropid'] = 0
    combination['Crop rotation group'] = 0
    combination['Vegetation type'] = 1
    combination['Special class code'] = 0
    combination['Tile depth'] = 0
    combination['Stream depth'] = 2.296
    combination['Number of soil layers'] = 3
    combination['Soil layer depth 1'] = 0.091
    combination['Soil layer depth 2'] = 0.493
    combination['Soil layer depth 3'] = 2.296

    # Define the metadata header with landcover and soil descriptions
    commented_lines = [
    """! MODIS landcover													
! Add legend (raster value) and discription													
!	original legend (raster_value)	description											
!   1: 'Temperate or sub-polar needleleaf forest',
!   2: 'Sub-polar taiga needleleaf forest',
!   3: 'Tropical or sub-tropical broadleaf evergreen forest',
!   4: 'Tropical or sub-tropical broadleaf deciduous forest',
!   5: 'Temperate or sub-polar broadleaf deciduous forest',
!   6: 'Mixed forest',
!   7: 'Tropical or sub-tropical shrubland',
!   8: 'Temperate or sub-polar shrubland',
!   9: 'Tropical or sub-tropical grassland',
!   10: 'Temperate or sub-polar grassland',
!   11: 'Sub-polar or polar shrubland-lichen-moss',
!   12: 'Sub-polar or polar grassland-lichen-moss',
!   13: 'Sub-polar or polar barren-lichen-moss',
!   14: 'Wetland',
!   15: 'Cropland',
!   16: 'Barren lands',
!   17: 'Urban',
!   18: 'Water',
!   19: 'Snow and Ice',											
!													
!													
!													
! ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------													
!	SoilGrid V1												
!		original legend (raster_value)	description										
!	 C 	    1	 clay										
!	 SIC 	2	 silty clay										
!	 SC 	3	 sandy clay										
!	 CL 	4	 clay loam										
!	 SICL 	5	 silty clay loam										
!	 SCL 	6	 sandy clay loam										
!	 L   	7	 loam										
!	 SIL 	8	 silty loam										
!	 SL 	9	 sandy loam										
!	 SI 	10	 silt										
!	 LS 	11	 loamy sand										
!	 S  	12	 sand										
!													
!													
! ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	"""]

    # Write the file
    with open(path_to_save+'GeoClass.txt', 'w') as file:
        # Write the metadata header
        for line in commented_lines:
            file.write(line + '\n')

    # Reclassify landcover and soil values
    for i in ['LULC', 'SOIL TYPE']:
        # Create sequential numbering for unique values
        unique_values = pd.unique(combination[i])
        value_mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
        
        # Apply new numbering
        combination[i] = combination[i].map(value_mapping)
        
        # Write mapping information to file
        mapping_df = pd.DataFrame(list(value_mapping.items()), columns=['Old Value', 'New Value'])
        with open(path_to_save+'GeoClass.txt', 'a') as f:
            f.write('! changes (reclassification) to '+i+'\n')
            for _, row in mapping_df.iterrows():
                f.write(f"! {row['Old Value']} -> {row['New Value']}\n")

    # Write the final data with column headers
    with open(path_to_save+'GeoClass.txt', 'a') as file:
        file.write("""! ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	
!          SLC	LULC	SOIL TYPE	Main crop cropid	Second crop cropid	Crop rotation group	Vegetation type	Special class code	Tile depth	Stream depth	Number of soil layers	Soil layer depth 1	Soil layer depth 2	Soil layer depth 3 \n""")
        combination.to_csv(file, sep='\t', index=False, header=False)