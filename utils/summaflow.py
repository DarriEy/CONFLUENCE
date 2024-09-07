# summaflow: a collection of functions to create summa input files

# load the needed packages
# packages are loaded
import xarray as xr
import pint_xarray
import cdo
import pint
import glob
import netCDF4 as nc4
import os
import pandas as pd
import numpy as np
import geopandas   as      gpd
from alive_progress import alive_bar #progress bar

############################
# sort geofabric from upstream to downstream
def sort_geofabric(geofabric, basinID, nextDownID):
    # find the subbasin order
    geofabric['n_ds_subbasins'] = 0
    for index, row in geofabric.iterrows():
        n_ds_subbasins = 0
        nextID = row[nextDownID]
        nextID_index = np.where(geofabric[basinID]==nextID)[0]

        while (len(nextID_index>0) or nextID > 0) :
            n_ds_subbasins += 1
            nextID = geofabric[nextDownID][nextID_index[0]]
            nextID_index = np.where(geofabric[basinID]==nextID)[0]
            
        geofabric.loc[index, 'n_ds_subbasins']= n_ds_subbasins

    # Sort the DataFrame by 'n_ds_subbasins' in descending order
    geofabric = geofabric.sort_values(by='n_ds_subbasins', ascending=False, ignore_index=True)
    return geofabric

############################

# write summa attribute and mizuRoute domain files
def write_summa_attribute(path_to_save, subbasins_shapefile, rivers_shapefile, gistool_output, frac_threshold, hru_discr, 
                          geofabric_mapping, landcover_mapping, soil_mapping, write_mizuroute_domain):
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)
        
    # get geofabric fields    
    basinID = geofabric_mapping['basinID']['in_varname']
    nextDownID = geofabric_mapping['nextDownID']['in_varname']
    area_name = geofabric_mapping['area']['in_varname']
    area_in_units = geofabric_mapping['area']['in_units']
    area_out_units = geofabric_mapping['area']['out_units']
    rivlen_name = geofabric_mapping['rivlen']['in_varname']
    length_in_units = geofabric_mapping['rivlen']['in_units']
    length_out_units = geofabric_mapping['rivlen']['out_units']

    # prepare data by merging the spatial fields into one dataframe
    # read geofabric
    # read rivers
    riv = gpd.read_file(rivers_shapefile)

    # read catchments
    cat = gpd.read_file(subbasins_shapefile)

    # read elevation stats
    elev_stats = pd.read_csv(os.path.join(gistool_output, 'elevation/domain_Bow_at_Banff_stats_elv.csv'))

    #rename columns except basinID
    elev_stats = elev_stats.rename(columns=lambda x: x + '_elev' if x != basinID else x)

    # merge riv, cat, and elev_stat in one dataframe on basinID
    print(riv,cat,elev_stats)
    geofabric = riv.merge(cat, on=basinID)
    geofabric = geofabric.merge(elev_stats, on=basinID)
    geofabric = geofabric.drop(columns=['geometry_x', 'hillslope_x', 'hillslope_y', 'geometry_y'])

    # sort geofabric from upstream to downstream
    geofabric = sort_geofabric(geofabric, basinID, nextDownID)

    # make necessary unit conversion

    # Initialize a unit registry
    ureg = pint.UnitRegistry()
    # Quantify the DataFrame column with the original units
    lengthm = geofabric[rivlen_name].values * ureg(length_in_units)
    # Convert to the desired units
    geofabric['rivlength_m'] = lengthm.to(length_out_units).magnitude

    # Initialize a unit registry
    ureg = pint.UnitRegistry()

    # Quantify the DataFrame column with the original units
    area_m2 = geofabric[area_name].values * ureg(area_in_units)

    # Convert to the desired units
    geofabric['area_m2'] = area_m2.to(area_out_units).magnitude

    # read soil and landcover data
    soil_stats = pd.read_csv(os.path.join(gistool_output, 'modified_domain_stats_soil_classes.csv'))

    # reorder df to follow the same order as the geofabric df
    soil_stats = soil_stats.set_index(basinID).loc[geofabric[basinID]].reset_index()

    # rename majority column
    soil_stats = soil_stats.rename(columns={'majority': 'soil_majority'})

    landuse_stats = pd.read_csv(os.path.join(gistool_output, 'modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv'))
    # reorder df to follow the same order as the geofabric df
    landuse_stats = landuse_stats.set_index(basinID).loc[geofabric[basinID]].reset_index()
    # rename majority column
    landuse_stats = landuse_stats.rename(columns={'majority': 'landuse_majority'})

    soil_landuse_stats = landuse_stats.merge(soil_stats, on=basinID)
    
    # get the fraction for land cover for each row
    hru_info = pd.DataFrame(columns=['hru2gruId', 'hruId'])
    k = 1
    
    for index, row in soil_landuse_stats.iterrows():
        # hru discretization is based on landcover
        if(hru_discr=='landcover'):
            fractions = [col for col in soil_landuse_stats.columns if col.startswith('frac') and row[col] > frac_threshold]
            frac_vals= row[fractions].to_list()
            # remove frac_ from the list
            fractions = [col.split('_')[1] for col in fractions]
            fractions = [int(name) for name in fractions]
            frac_number = fractions

        # check if the user wants 1 hru per gru (no spatial discretization )
        if(hru_discr == '1hru_gru'):
            # use max fraction as the dominant landcover
            frac_vals = [1]
            frac_number = [soil_landuse_stats['landuse_majority'][index]]

        # Normalize the array so that the sum of its values equals 1
        normalized_frac_vals = frac_vals / np.sum(frac_vals)
        nhru =len(frac_number)
        
        # construct the gru/hru df
        hru_info = pd.concat([hru_info, pd.DataFrame({
                                        'hru2gruId': np.repeat(soil_landuse_stats[basinID][index], nhru),
                                        'hruId': np.arange(nhru)+k,
                                        'downHRUindex': np.zeros(nhru),
                                        'HRUarea': normalized_frac_vals * geofabric['area_m2'][index],
                                        'vegTypeIndex': list(map(landcover_mapping.get, frac_number)),
                                        'soilTypeIndex': np.repeat(list(map(soil_mapping.get, [soil_landuse_stats['soil_majority'][index]])), nhru),
                                        'slopeTypeIndex': np.ones(nhru),
                                        'elevation': np.repeat(geofabric['mean_elev'][index], nhru),
                                        'contourLength': normalized_frac_vals * geofabric['rivlength_m'][index],
                                        'latitude': np.repeat(soil_landuse_stats['lat'][index], nhru),
                                        'longitude': np.repeat(soil_landuse_stats['lon'][index], nhru),
                                        'mHeight': np.ones(nhru)*40,
                                        'tan_slope': np.repeat(geofabric['slope'][index], nhru),
                                        })], ignore_index=True)
        k += nhru

    # hru_info
    # Create a new xarray dataset
    attr = xr.Dataset()

    # prepare for the summa attr file
    attr ['hruId']          = xr.DataArray(hru_info['hruId'].values, dims=('hru'), 
                                            attrs={'long_name': 'Index of hydrological response units (HRU)', 'units': '-'})

    attr ['gruId']          = xr.DataArray(geofabric[basinID].values, dims=('gru'),
                                            attrs={'long_name': 'Index of group of response unit (GRU)', 'units': '-'})

    attr ['hru2gruId']      = xr.DataArray(hru_info['hru2gruId'].values, dims=('hru'),
                                            attrs={'long_name': 'Index of GRU to which the HRU belongs', 'units': '-'})

    attr ['downHRUindex']   = xr.DataArray(hru_info['downHRUindex'].values, dims=('hru'),
                                            attrs={'long_name': 'Index of downslope HRU (0 = basin outlet)', 'units': '-'}).astype(int)

    attr ['elevation']      = xr.DataArray(hru_info['elevation'].values, dims=('hru'),
                                            attrs={'long_name': 'Elevation of HRU\'s centriod point', 'units': 'm'})

    attr ['HRUarea']        = xr.DataArray(hru_info['HRUarea'].values, dims=('hru'),
                                            attrs={'long_name': 'Area of each HRU', 'units': 'm^2'})

    attr ['tan_slope']      = xr.DataArray(hru_info['tan_slope'].values, dims=('hru'),
                                            attrs={'long_name': 'Average tangent slope of HRU', 'units': 'm/m'})

    attr ['contourLength']  = xr.DataArray(hru_info['contourLength'].values, dims=('hru'),
                                            attrs={'long_name': 'ContourLength of HRU', 'units': 'm'})

    attr ['slopeTypeIndex'] = xr.DataArray(hru_info['slopeTypeIndex'].values, dims=('hru'),
                                            attrs={'long_name': 'Index defining slope', 'units': '-'}).astype(int)

    attr ['soilTypeIndex']  = xr.DataArray(hru_info['soilTypeIndex'].values, dims=('hru'),
                                            attrs={'long_name': 'Index defining soil type - ROSETTA', 'units': '-'}).astype(int)

    attr ['vegTypeIndex']   = xr.DataArray(hru_info['vegTypeIndex'].values, dims=('hru'),
                                            attrs={'long_name': 'Index defining vegetation type - USGS', 'units': '-'}).astype(int)

    attr ['mHeight']        = xr.DataArray(hru_info['mHeight'].values, dims=('hru'),
                                            attrs={'long_name': 'Measurement height above bare ground', 'units': 'm'})

    attr ['longitude'] = xr.DataArray(hru_info['longitude'].values, dims=('hru'),
                                        attrs={'long_name': 'Longitude of HRU\'s centriod point', 'units': 'decimal degree east'})

    attr ['latitude'] = xr.DataArray(hru_info['latitude'].values, dims=('hru'),
                                        attrs={'long_name': 'Latitude of HRU\'s centriod point', 'units': 'decimal degree north'})

    # write attribute file
    if os.path.isfile(path_to_save+'summa_zLocalAttributes.nc'):
        os.remove(path_to_save+'summa_zLocalAttributes.nc')

    attr.to_netcdf(path_to_save+'summa_zLocalAttributes.nc')
    ########
    # write mizuroute domain file
    if(write_mizuroute_domain):

        if not os.path.isdir(path_to_save+'mizuroute/'):
            os.makedirs(path_to_save+'mizuroute/')
        
        if not os.path.isdir(path_to_save+'mizuroute/mizuroute_results/'):
            os.makedirs(path_to_save+'mizuroute/mizuroute_results/')
        
        # Create a new xarray dataset
        mizuroute = xr.Dataset()

        # define gru dimension
        mizuroute ['gru']           = xr.DataArray(geofabric[basinID].values, dims=('gru'))
        # prepare other topo information
        mizuroute ['gruId']          = xr.DataArray(geofabric[basinID].values, dims=('gru'),
                                                attrs={'long_name': 'Index of group of response unit (GRU)', 'units': '-'})
    
        mizuroute ['rivlength']  = xr.DataArray(geofabric['rivlength_m'].values, dims=('gru'),
                                            attrs={'long_name': 'river segment length of GRU', 'units': 'm'})
        
        mizuroute ['GRUarea']        = xr.DataArray(geofabric['area_m2'].values, dims=('gru'),
                                            attrs={'long_name': 'Area of each GRU', 'units': 'm^2'})

        mizuroute ['tan_slope']      = xr.DataArray(geofabric['slope'].values, dims=('gru'),
                                            attrs={'long_name': 'Average slope of GRU', 'units': 'm/m'})
        
        mizuroute ['NextDownID']      = xr.DataArray(geofabric[nextDownID].values, dims=('gru'),
                                            attrs={'long_name': 'Next downstream gruId', 'units': '-'})
        # write attribute file
        if os.path.isfile(path_to_save+'mizuroute/'+'mizuroute_domain.nc'):
            os.remove(path_to_save+'mizuroute/'+'mizuroute_domain.nc')

        mizuroute.to_netcdf(path_to_save+'mizuroute/'+'mizuroute_domain.nc')

        # copy setting files
        os.system("cp -r setting_files/mizuroute/* "+ path_to_save+'/mizuroute/')

    # return the attribute file
    return(attr)
#################################

# write summa forcing from MESH forcing
def write_summa_forcing(path_to_save, timeshift, forcing_units, easymore_output, attr, geofabric_mapping):
    # get basin ID field from the geofabric_mapping
    basinID = geofabric_mapping['basinID']['in_varname']

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
            batch_files = files_split[i].tolist()
            batch_output = f'forcing_batch_{i}.nc'
            cdo_obj.mergetime(input=batch_files, output=batch_output)
            intermediate_files.append(batch_output)
            bar()

    # Combine intermediate results into one netcdf file
    cdo_obj.mergetime(input=intermediate_files, output='merged_forcing.nc')

    # Clean up intermediate files if needed
    for f in intermediate_files:
        os.remove(f)
    # open the forcing file
    forcing = xr.open_dataset('merged_forcing.nc')
    # convert calendar to 'standard'
    forcing = forcing.convert_calendar('standard')
    # The data are in UTC time and they need to be shifted -6h to local time
    forcing['time'] = forcing['time'] + pd.Timedelta(hours=timeshift)

    # do the units

    # Create the new dictionary
    inputForcingUnits = {v['in_varname']: v['in_units'] for v in forcing_units.values()}

    # SUMMA's units
    summa_units = {v['in_varname']: v['out_units'] for v in forcing_units.values()}

    # Define the unit registries
    forcing = forcing.pint.quantify(inputForcingUnits)  # Add unit attributes of the original RDRS
    forcing = forcing.pint.to(summa_units)  # convert the units to summa units
    # remove pint units as they conflict with writing
    forcing = forcing.pint.dequantify()
    # rename the dictionary
    # Create the new dictionary
    rename_dictionary = {v['in_varname']: k for k, v in forcing_units.items()}

    forcing = forcing.rename_vars(rename_dictionary)
    # time step of the data in seconds
    forcing['data_step'] = (forcing['time'][1].values - forcing['time'][0].values).astype('timedelta64[s]').astype(int)
    
    # create hru forcing
    forcing = forcing.sel({basinID: attr['hru2gruId'].values})
    forcing = forcing.rename_dims({basinID: 'hru'})
    forcing = forcing.rename_vars({basinID: 'hru'})
    # set values based on the actual hruId
    forcing.coords['hru'] = attr['hruId'].values
    # Select the first time step (index 0) for lon and lat variables if time is in the dims
    if 'time' in forcing['longitude'].dims:
        forcing['longitude'] = forcing['longitude'].isel(time=0).drop_vars('time')
        forcing['latitude'] = forcing['latitude'].isel(time=0).drop_vars('time')
    # save to file
    forcing.to_netcdf(path_to_save+'summa_forcing.nc')
    # close the netcdf file
    forcing.close()
    
    # replace T in the time unit with space

    ncid = nc4.Dataset(path_to_save + 'summa_forcing.nc', 'r+')
    
    # Access the 'units' attribute and replace 'T' with a space
    units_attribute = ncid['time'].units
    units_attribute = units_attribute.replace('T', ' ')
    
    # Update the 'units' attribute in the netCDF file
    ncid['time'].setncattr('units', units_attribute)
    
    # Close the netCDF file
    ncid.close()
    # remove RDRS forcing
    os.remove('merged_forcing.nc')
    return(xr.open_dataset(path_to_save + 'summa_forcing.nc'))
######################
def write_summa_paramtrial(attr, path_to_save):
    
    # Define dimensions
    hru_size = len(attr['hruId'].values)
    
    # Create a new xarray dataset
    ds = xr.Dataset()
    
    # Add dimensions to the dataset
    ds['hru'] = xr.DataArray(attr['hruId'].values, dims=('hru'), attrs={'units': '-'})
    
    # Add variables to the dataset
    ds['hruId'] = xr.DataArray(attr['hruId'].values, dims=('hru'), attrs={'units': '-', 'long_name': 'Index of hydrological response unit (HRU)'})
    ds
    if os.path.isfile(path_to_save+'summa_zParamTrial.nc'):
        os.remove(path_to_save+'summa_zParamTrial.nc')
    
    ds.to_netcdf(path_to_save+'summa_zParamTrial.nc')
    return(ds)

#################################
def write_summa_initial_conditions(attr, soil_mLayerDepth, path_to_save):
    
    # Define dimensions
    hru_size = len(attr['hruId'].values)
    scalarv_size = 1
    midSoil_size = midToto_size = len(soil_mLayerDepth)
    ifcToto_size = midSoil_size + 1

    # Create a new xarray dataset
    ds = xr.Dataset()
    
    # Add dimensions to the dataset
    ds['hru'] = xr.DataArray(attr['hruId'].values, dims=('hru'), attrs={'units': '-'})
    ds['midSoil'] = xr.DataArray(range(midSoil_size), dims=('midSoil'))
    ds['midToto'] = xr.DataArray(range(midToto_size), dims=('midToto'))
    ds['ifcToto'] = xr.DataArray(range(ifcToto_size), dims=('ifcToto'))
    ds['scalarv'] = xr.DataArray(range(scalarv_size), dims=('scalarv'))
    
    # Add variables to the dataset
    ds['hruId'] = xr.DataArray(attr['hruId'].values, dims=('hru'), attrs={'units': '-', 'long_name': 'Index of hydrological response unit (HRU)'})
    ds['dt_init'] = xr.DataArray([[3600.0] * hru_size], dims=('scalarv', 'hru'))
    ds['nSoil'] = xr.DataArray([[midSoil_size] * hru_size], dims=('scalarv', 'hru'))
    ds['nSnow'] = xr.DataArray([[0] * hru_size], dims=('scalarv', 'hru'))
    ds['scalarCanopyIce'] = xr.DataArray([[0] * hru_size], dims=('scalarv', 'hru'))
    ds['scalarCanopyLiq'] = xr.DataArray([[0] * hru_size], dims=('scalarv', 'hru'))
    ds['scalarSnowDepth'] = xr.DataArray([[0] * hru_size], dims=('scalarv', 'hru'))
    ds['scalarSWE'] = xr.DataArray([[0] * hru_size], dims=('scalarv', 'hru'))
    ds['scalarSfcMeltPond'] = xr.DataArray([[0] * hru_size], dims=('scalarv', 'hru'))
    ds['scalarAquiferStorage'] = xr.DataArray([[1.0] * hru_size], dims=('scalarv', 'hru'))
    ds['scalarSnowAlbedo'] = xr.DataArray([[0] * hru_size], dims=('scalarv', 'hru'))
    ds['scalarCanairTemp'] = xr.DataArray([[283.16] * hru_size], dims=('scalarv', 'hru'))
    ds['scalarCanopyTemp'] = xr.DataArray([[283.16] * hru_size], dims=('scalarv', 'hru'))
    ds['mLayerTemp'] = xr.DataArray([[283.16] * hru_size ] * midToto_size , dims=('midToto', 'hru'))
    ds['mLayerVolFracIce'] = xr.DataArray([[0] * hru_size] * midToto_size, dims=('midToto', 'hru'))
    ds['mLayerVolFracLiq'] = xr.DataArray([[0.2] * hru_size] * midToto_size, dims=('midToto', 'hru'))
    ds['mLayerMatricHead'] = xr.DataArray([[-1] * hru_size] * midToto_size, dims=('midSoil', 'hru'))
    ds['iLayerHeight'] = xr.DataArray(np.transpose([[0]+list(np.cumsum(soil_mLayerDepth))] * hru_size) , dims=('ifcToto', 'hru'))
    ds['mLayerDepth'] = xr.DataArray(np.transpose([soil_mLayerDepth] * hru_size), dims=('midToto',  'hru'))
    
    
    if os.path.isfile(path_to_save+'summa_zInitialCond.nc'):
        os.remove(path_to_save+'summa_zInitialCond.nc')
    
    ds.to_netcdf(path_to_save+'summa_zInitialCond.nc')
    
    return(ds)

##################################
def write_summa_filemanager(path_to_save, forcing):

    # Convert to pandas datetime if not already in that format
    time_values = pd.to_datetime(forcing['time'].values)
    # Filter to get only the midnight times
    midnight_times = time_values[time_values.time == pd.Timestamp('00:00:00').time()]
    # Find the first and last midnight values
    start_date = midnight_times.min().strftime('%Y-%m-%d %H:%M')
    end_date = midnight_times.max().strftime('%Y-%m-%d %H:%M')

    template_string="""controlVersion       'SUMMA_FILE_MANAGER_V3.0.0' !  fman_ver 
simStartTime         '{start_date}' ! 
simEndTime           '{end_date}' ! 
tmZoneInfo           'localTime' ! 
settingsPath         './' !  setting_path
forcingPath          './' !  input_path
outputPath           './summa_results/' !  output_path
decisionsFile        './summa_zDecisions.txt' !  decision
outputControlFile    './Model_Output.txt' !  OUTPUT_CONTROL
globalHruParamFile   './summa_zLocalParamInfo.txt' !  local_par
globalGruParamFile   './summa_zBasinParamInfo.txt' !  basin_par
attributeFile        './summa_zLocalAttributes.nc' !  local_attr
trialParamFile       './summa_zParamTrial.nc' !  para_trial
forcingListFile      './summa_zForcingFileList.txt' !  forcing_list
initConditionFile    './summa_zInitialCond.nc' !  initial_cond
outFilePrefix        'myTest' !  output_prefix
vegTableFile         'VEGPARM.TBL' ! 
soilTableFile        'SOILPARM.TBL' ! 
generalTableFile     'GENPARM.TBL' ! 
noahmpTableFile      'MPTABLE.TBL' ! 
! history written by summaflow
"""

    # Replace the placeholder with the actual start date using an f-string
    fileManager = template_string.format(start_date=start_date, end_date = end_date)


    # Optionally, write the formatted string to a file
    output_file = path_to_save+'summa_fileManager.txt'
    with open(output_file, 'w') as file:
        file.write(fileManager)

###############################
def copy_summa_static_files(path_to_save):
    os.system("cp -r setting_files/summa/* "+ path_to_save)
    # create results firectory
    if not os.path.isdir(path_to_save+'summa_results/'):
        os.makedirs(path_to_save+'summa_results/')