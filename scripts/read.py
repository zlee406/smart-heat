import time, os, sys, pathlib, pickle
import pandas as pd
import numpy as np
import concurrent.futures
from geopy.geocoders import Nominatim
import geopy.distance
import datetime
sys.path.append(f'{pathlib.Path(os.path.abspath("")).parents[0]}')

DATA_DIR = f'{pathlib.Path(os.path.abspath("")).parents[1]}/data'

# User Packages
try:
    import process
except Exception as e:
    from . import process

def get_counties():
    """
    Deprecated function used for adding county names to meta_data.csv. Originally used to match design day temperatures
    to thermostat runtime data.
    :return:
    """

    DATA_DIR = f'{pathlib.Path(__file__).parents[2]}/data'
    meta_data = pd.read_csv(f'{DATA_DIR}/meta_data.csv', header=0)
    meta_data['County'] = ''
    geolocator = Nominatim(user_agent="Your_Name")

    for n, ind in enumerate(meta_data.index):
        print('.', end="")
        if meta_data.loc[ind, 'Country'] == 'US':
            if meta_data.loc[ind, 'ProvinceState'] == 'NY':
                if meta_data.loc[ind, 'ProvinceState'] == 'NY':
                    address = {'city': meta_data.loc[ind, 'City'], 'state': 'New York'}

                else:
                    address = {'city': meta_data.loc[ind, 'City'], 'state': meta_data.loc[ind, 'ProvinceState']}

                location = geolocator.geocode(address, timeout=60, addressdetails=True)
                try:
                    meta_data.loc[ind, 'County'] = location.raw['address']['county']
                except:
                    try:
                        if address['city'].lower() == 'new york':
                            manual_county = 'New York'
                        elif address['city'].lower() == 'brooklyn':
                            manual_county = 'Kings'
                        elif address['city'].lower() == 'staten island':
                            manual_county = 'Richmond'
                        elif address['city'].lower() == 'bronx':
                            manual_county = 'Bronx'
                        elif address['city'].lower() in ['sunnyside','elmhurst', 'ozone park']:
                            manual_county = 'Queens'
                        else:
                            # manual_county = input(f'\nEnter County for {address}: ')
                            meta_data.loc[ind, 'County'] = ''
                    except:
                        meta_data.loc[ind, 'County'] = ''
        if n % 100 == 0:
            print(f'{n}/{len(meta_data.index)}')
    return meta_data

def get_solar_dfs(location, index):
    """
    Reads in timeseries data from files contained in <DATA_DIR>/Weather/<location> and collates them into a list of
    pandas dataframes. Files should be obtained from the NSRDB database for various points in your focus region.
    :param location: Data directory for focus region
    :param index: a pandas datetime index representing the timesteps desired
    :return: solar_dfs_loc: A list of dataframes containing processed solar information.
    """
    solar_dfs_orig = [pd.read_csv(f'{DATA_DIR}/Weather/{location}/{f}', header=2) for f in
                      os.listdir(f'{DATA_DIR}/Weather/{location}') if 'wind' not in f]
    solar_dfs = []
    for df in solar_dfs_orig:
        timestamp = []
        for i in range(len(df)):
            timestamp.append(
                datetime.datetime(year=df['Year'][i], month=df['Month'][i], day=df['Day'][i], hour=df['Hour'][i],
                                  minute=df['Minute'][i]))
        df['timestamp'] = pd.to_datetime(timestamp)
        df.index = df['timestamp']
        df = df.iloc[:, 5:].resample('5T').interpolate()
        df['GHI'] = df['GHI'] / 1000
        df = df.rename(
            columns={'GHI': 'GHI_(kW/m2)'})
        solar_dfs.append(df)

    solar_dfs_loc = [pd.read_csv(f'{DATA_DIR}/Weather/{location}/{f}').loc[0, ['Latitude', 'Longitude']] for f
                     in os.listdir(f'{DATA_DIR}/Weather/{location}') if 'wind' not in f]
    solar_dfs_loc = [pd.to_numeric(w) for w in solar_dfs_loc]
    return solar_dfs_loc, solar_dfs

def get_wind_dfs(location, index):
    """
    Gets data from wind speed files contained in  <DATA_DIR>/Weather/<location> to be used for comparing heating demand
    to 100m wind speed.
    :param location: Data directory for focus region
    :param index:  a pandas datetime index representing the timesteps desired
    :return:
    """
    wind_df = pd.read_csv(f'{DATA_DIR}/Weather/{location}/wind_speed.csv', index_col=0, parse_dates=True)
    wind_df = wind_df.loc[wind_df.index.isin(index), :]

    wind_df_loc = [wind_df.loc[:, ['latitude', 'longitude']].iloc[i] for i in range(len(wind_df.loc[:, ['latitude', 'longitude']].drop_duplicates()))]

    return wind_df_loc, wind_df

def create_wind_data(path_to_grib, location):
    """
    Creates a timeseries csv from the ERA5 grib files for 100m wind speed.
    :param path_to_grib:
    :param location: Location directory to save the wind_speed csv
    :return:
    """
    wind_ds = xr.load_dataset(path_to_grib, engine='cfgrib')
    wind_df = wind_ds.to_dataframe()
    wind_df['100m_Wind_Speed_(m/s)'] = (wind_df['u100'] ** 2 + wind_df['v100'] ** 2) ** .5
    wind_df.to_csv(f'{DATA_DIR}/Weather/{location}/wind_speed.csv')


def import_grouped_data(location: str,
                        max_files: int,
                        hp_only: bool = False,
                        parallel: bool = False,
                        reduce_size: bool = True,
                        season: str = 'winter'):
    """
    Imports data from individual files and combines them into a list of processed pandas dataframes
    :param location: path to folder containing individual thermostat files
    :param max_files: maximum number of files to process
    :param hp_only: Whether to include heat pump or non-heat pump homes
    :param parallel: Whether to run import in parallel
    :param reduce_size: Whether to import the full data or reduce it to only important columns
    :param season: ['winter', 'summer'] Season to analyze
    :return:
    """


    DIR = f'{DATA_DIR}/{location}'

    # Get metadata from data directory
    meta_data = pd.read_csv(f'{DATA_DIR}/meta_data.csv', header=0,
                            usecols=['City', 'ProvinceState',
                                    'Identifier', 'Floor Area [ft2]', 'Style', 'installedHeatStages',
                                     'Number of Floors', 'Age of Home [years]', 'Number of Occupants',
                                     'Has Electric', 'Has a Heat Pump', 'Auxilliary Heat Fuel Type'])
    max_files = min(max_files, len(os.listdir(DIR)))
    meta_data_list = list(meta_data['Identifier'])
    filenames = [f'{DIR}/{file}' for file in os.listdir(DIR) if file.rstrip('.csv') in meta_data_list][:max_files]

    # Slice based on season
    if season == 'winter':
        grouped_index = list(pd.date_range(start='01/01/2019', end='3/21/2019', freq='5T'))
        grouped_index += (list(pd.date_range(start='11/21/2019', end='12/31/2019', freq='5T')))
    elif season == 'summer':
        grouped_index = list(pd.date_range(start='6/21/2019', end='9/21/2019', freq='5T'))
    grouped_index = pd.DatetimeIndex(grouped_index)

    start = time.time()

    # Columns containing numbers to interpolate
    interp_cols = ['T_stp_cool', 'T_stp_heat', 'Humidity',
                   'auxHeat1', 'auxHeat2', 'auxHeat3',
                   'compHeat1', 'compHeat2', 'fan',
                   'Thermostat_Temperature', 'T_out', 'RH_out']

    # Get Weather Dfs
    solar_dfs_loc, solar_dfs = get_solar_dfs(location, grouped_index)
    wind_df_loc, wind_df = get_wind_dfs(location, grouped_index)


    def import_data(i, file, reduce_size):
        """
        Helper function for parallelization. Imports files and adds them to a list of dataframes
        :param i: file number
        :param file: filename
        :param reduce_size: Whether to conserve space by limiting data input
        :return:
        """
        # Time Count
        if i % 5 == 0 and i != 0:
            print(
                f'\nProgress: {i}/{max_files} Time: {(time.time() - start) / 60:.1f} min/{(time.time() - start) / i * max_files / 60:.1f} min',
                end=" ")
        else:
            print('.', end="")

        # Get Metadata
        key = file[-44:-4] # parses out the identifier from filename
        building_meta_data = meta_data.loc[meta_data['Identifier'] == key]

        # Read In csv
        df_t = time.time()
        df = pd.read_csv(file, parse_dates=['DateTime'], usecols=['DateTime', 'HvacMode', 'Event', 'Schedule', 'T_ctrl',
                                                                  'T_stp_cool', 'T_stp_heat', 'Humidity',
                                                                  'auxHeat1', 'auxHeat2', 'auxHeat3',
                                                                  'compHeat1', 'compHeat2', 'fan',
                                                                  'Thermostat_Temperature', 'T_out', 'RH_out',
                                                                  ],
                         dtype={'HvacMode': 'category', 'Event': 'category', 'Schedule': 'category', 'T_ctr': np.float32,
                                  'T_stp_cool': np.float32, 'T_stp_heat': np.float32, 'Humidity': np.float32,
                                  'auxHeat1': np.float32, 'auxHeat2': np.float32, 'auxHeat3': np.float32,
                                  'compHeat1': np.float32, 'compHeat2': np.float32, 'fan': np.float32,
                                  'Thermostat_Temperature': np.float32, 'T_out': np.float32, 'RH_out': np.float32,
        }
                         ).set_index('DateTime')
        # Get only <season> days
        df = df.loc[df.index.isin(grouped_index), :]

        # print(f'import df: {time.time() - df_t:.2f}')
        start_t = time.time()

        # Only read in heat pump data if hp_only is true
        if hp_only:
            if (df['compHeat2'] > 0).any():
                READ_IN = True
            elif (df['compHeat1'] > 0).any():
                READ_IN = True
            else:
                READ_IN = False
        else:
            if (df['compHeat2'] > 0).any():
                READ_IN = False
            elif (df['compHeat1'] > 0).any():
                READ_IN = False
            else:
                READ_IN = True
        # print(f'check if hp only: {time.time() - start_t:.2f} s')

        # Do not read in data if it is empty
        if df.shape[0] == 0:
            READ_IN = False

        read_t = time.time()
        if READ_IN:
            # Fill some NaNs
            t0 = time.time()
            df[interp_cols] = df[interp_cols].interpolate(method='linear', limit=3)
            df['T_ctrl_C'] = (df['T_ctrl'] - 32)*5/9
            df['T_out_C'] = (df['T_out'] - 32)*5/9

            # Get Runtime for multistage devices
            t1 = time.time()
            df = process.get_effective_runtime(df)
            # print(f'effective runtime: {time.time() - t1:.2f} s')

            # Estimate Effective Power Consumption
            t2 = time.time()
            df = process.get_effective_power(df)
            # print(f'effective power: {time.time() - t2:.2f} s')
            # print(f'Process time: {time.time() - read_t:.2f} s')

            # Get Location-Based Weather Data
            if building_meta_data['ProvinceState'].iloc[0] == 'NY':
                address = {'city': building_meta_data['City'].iloc[0], 'state': 'New York'}
            else:
                address = {'city': building_meta_data['City'].iloc[0], 'state': building_meta_data['ProvinceState'].iloc[0]}
            geolocator = Nominatim(user_agent="Your_Name")
            location = geolocator.geocode(address, timeout=60, addressdetails=True)
            try:
                solar_distances = [geopy.distance.distance((location.latitude, location.longitude), (lat, long)) for lat, long in solar_dfs_loc]
                weather_distances = [geopy.distance.distance((location.latitude, location.longitude), (lat, long)) for lat, long in wind_df_loc]
                df['Lat'] = location.latitude
                df['Long'] = location.longitude
            except:
                solar_distances = [0] #default to NYC if no address found
                weather_distances = [476] #default to NYC if no address found
                df['Lat'] = solar_dfs_loc[0]['Latitude']
                df['Long'] = solar_dfs_loc[0]['Longitude']

            solar_df = solar_dfs[np.argmin(solar_distances)].resample('5T').interpolate()
            wind_df_to_merge = wind_df.loc[(wind_df['latitude'] == wind_df_loc[np.argmin(weather_distances)][0]) & (wind_df['longitude'] == wind_df_loc[np.argmin(weather_distances)][1])].resample('5T').interpolate()

            df = pd.merge(df, solar_df[['DHI', 'DNI', 'GHI_(kW/m2)','Wind Speed', 'Temperature']], left_index=True, right_index=True)
            df = pd.merge(df, wind_df_to_merge[['100m_Wind_Speed_(m/s)']], left_index=True, right_index=True)

            df['Nearest_Lat'] = solar_dfs_loc[np.argmin(solar_distances)]['Latitude'].astype(np.float16)
            df['Nearest_Long'] = solar_dfs_loc[np.argmin(solar_distances)]['Longitude'].astype(np.float16)

            df['Identifier'] = key

            # Get floor area, parse bad data
            floor_area = building_meta_data['Floor Area [ft2]'].iloc[0]
            if floor_area == 0:
                floor_area = 2000 # Median floor area
            if building_meta_data['Floor Area [ft2]'].iloc[0] in ['row house',
                                                                  'rowhouse',
                                                                  'townhouse',
                                                                  'multi-plex',
                                                                  'multiplex',
                                                                  'loft']:
                floor_area = max(floor_area, 4000)
            df['Floor Area [ft2]'] = int(floor_area)

            df.index.rename('DateTime', inplace=True)

            if reduce_size:
                df = df[['HvacMode', 'Event', 'Schedule', 'T_stp_cool', 'T_stp_heat',
                         'Humidity', 'RH_out',
                         'T_ctrl_C', 'T_out_C',
                         # 'GHI_(kW/m2)', 'Wind Speed', '100m_Wind_Speed_(m/s)',
                         'effectiveHeat', 'effectiveElectricPower', 'fan',
                         # 'Nearest_Lat'
                         ]]

            return df

    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as e:
            df_list_parallel = [e.submit(import_data, i, f, reduce_size) for i, f in enumerate(filenames)]
        df_list = []
        for i in range(len(filenames)):
            df_list.append(df_list_parallel[i].result())
    else:
        df_list = []
        for i, file in enumerate(filenames):
            df = import_data(i, file, reduce_size)
            df_list.append(df)


    return df_list

def import_load_data(year, location): #TODO
    DATA_DIR = f'{pathlib.Path(__file__).parents[2]}/Data Files'
    df_list = []
    for file in os.listdir(f'{DATA_DIR}/NYISO Load'):
        if file[0] == 'O':
            df = pd.read_csv(f'{DATA_DIR}/NYISO Load/{file}', index_col=0, parse_dates=True).drop(columns=['Zone PTID', 'Zone Name'])
            df = df.resample('5T').mean()
            df_list.append(df)
    df_tot = pd.concat(df_list).groupby(by=['RTD End Time Stamp']).sum()
    df_tot = df_tot.rename(columns={df_tot.columns[0]: 'Total Load'})

    return df_tot

def main(location, data_dir, reduce_size, season, hp_only):
    size = 'small' if reduce_size else 'large'
    # Heat Pump Only
    if hp_only:
        df_list_hp = import_grouped_data(location, max_files=10000, hp_only=True, parallel=True, season=season, reduce_size=reduce_size)
        df_list_hp = [x for x in df_list_hp if x is not None]
        pickle.dump(df_list_hp, open(f'{DATA_DIR}/df_Lists/df_list_hp_{location}_{size}_{season}.sav', 'wb'))
        df_list_hp = pickle.load(open(f'{DATA_DIR}/df_Lists/df_list_hp_{location}_{size}_{season}.sav', 'rb'))
        grouped_df_hp = process.group_dfs(df_list_hp, size)
        pickle.dump(grouped_df_hp, open(f'{DATA_DIR}/df_Lists/grouped_df_hp_{location}_{size}_{season}.sav', 'wb'))
        grouped_loc_df_hp = process.group_dfs_by_location(df_list_hp, size)
        pickle.dump(grouped_loc_df_hp, open(f'{DATA_DIR}/df_Lists/grouped_loc_df_hp_{location}_{size}_{season}.sav', 'wb'))
    else:
        # Gas Only
        df_list_gas = import_grouped_data(location, max_files=10000, hp_only=False, parallel=False, reduce_size=reduce_size)
        df_list_gas = [x for x in df_list_gas if x is not None]
        pickle.dump(df_list_gas, open(f'{DATA_DIR}/df_Lists/df_list_gas_{location}_{size}.sav', 'wb'))
        df_list_gas = pickle.load(open(f'{DATA_DIR}/df_Lists/df_list_gas_{location}_{size}.sav', 'rb'))
        grouped_df_gas = process.group_dfs(df_list_gas, size)
        pickle.dump(grouped_df_gas, open(f'{DATA_DIR}/df_Lists/grouped_df_gas_{location}_{size}.sav', 'wb'))
        grouped_loc_df_gas = process.group_dfs_by_location(df_list_gas, size)
        pickle.dump(grouped_loc_df_gas, open(f'{DATA_DIR}/df_Lists/grouped_loc_df_gas_{location}_{size}.sav', 'wb'))

if __name__ == '__main__':
    location = 'NY'
    data_dir = f'{pathlib.Path(os.path.abspath("")).parents[1]}/data'
    reduce_size = True
    season='winter'
    hp_only = False
    main(location, data_dir, reduce_size, season, hp_only)