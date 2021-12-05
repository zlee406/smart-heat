import time, os, sys, pathlib, pickle
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
import zipfile
import concurrent.futures
from geopy.geocoders import Nominatim
import geopy.distance
import datetime
sys.path.append(f'{pathlib.Path(os.path.abspath("")).parents[0]}')

# User Packages
try:
    import process
except Exception as e:
    from . import process

def get_counties():
    DATA_DIR = f'{pathlib.Path(__file__).parents[2]}/Data Files'
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


def import_grouped_data(location, NUM_FILES, HP_ONLY=False, parallel=False, size='small', season='winter'):
    DATA_DIR = f'{pathlib.Path(os.path.abspath("")).parents[1]}/Data Files'
    if location == 'NY':
        DIR = f'{DATA_DIR}/2019 NY'
    else:
        DIR = f'{DATA_DIR}/2019 NE'


    meta_data = pd.read_csv(f'{DATA_DIR}/meta_data.csv', header=0,
                            usecols=['City', 'ProvinceState',
                                    'Identifier', 'Floor Area [ft2]', 'Style', 'installedHeatStages',
                                     'Number of Floors', 'Age of Home [years]', 'Number of Occupants',
                                     'Has Electric', 'Has a Heat Pump', 'Auxilliary Heat Fuel Type'])
    meta_data_design = pd.read_csv(f'{DATA_DIR}/meta_data_design_fixed.csv', header=0,
                            usecols=['City', 'ProvinceState',
                                     'Identifier', 'Floor Area [ft2]', 'Style', 'installedHeatStages',
                                     'Number of Floors', 'Age of Home [years]', 'Number of Occupants',
                                     'Has Electric', 'Has a Heat Pump', 'Auxilliary Heat Fuel Type', '99%heating'])
    NUM_FILES = min(NUM_FILES, len(os.listdir(DIR)))
    meta_data_list = list(meta_data['Identifier'])
    filenames = [f'{DIR}/{file}' for file in os.listdir(DIR) if file.rstrip('.csv') in meta_data_list][:NUM_FILES]

    if season == 'winter':
        grouped_index = list(pd.date_range(start='01/01/2019', end='3/21/2019', freq='5T'))
        grouped_index += (list(pd.date_range(start='11/21/2019', end='12/31/2019', freq='5T')))
    elif season == 'summer':
        grouped_index = list(pd.date_range(start='6/21/2019', end='9/21/2019', freq='5T'))
    grouped_index = pd.DatetimeIndex(grouped_index)

    start = time.time()

    interp_cols = ['T_stp_cool', 'T_stp_heat', 'Humidity',
                   'auxHeat1', 'auxHeat2', 'auxHeat3',
                   'compHeat1', 'compHeat2', 'fan',
                   'Thermostat_Temperature', 'T_out', 'RH_out']

    # Get Weather Dfs
    solar_dfs_orig = [pd.read_csv(f'{DATA_DIR}/Weather/NY Weather NSRDB/{f}', header=2) for f in
                   os.listdir(f'{DATA_DIR}/Weather/NY Weather NSRDB')]
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
    solar_dfs = [solar_df.loc[solar_df.index.isin(grouped_index), :] for solar_df in solar_dfs]

    solar_dfs_loc = [pd.read_csv(f'{DATA_DIR}/Weather/NY Weather NSRDB/{f}').loc[0, ['Latitude', 'Longitude']] for f
                       in os.listdir(f'{DATA_DIR}/Weather/NY Weather NSRDB')]
    solar_dfs_loc = [pd.to_numeric(w) for w in solar_dfs_loc]


    wind_df = pd.read_csv(f'{DATA_DIR}/Weather/NY Weather Wind/NY_100m_wind_speed.csv', index_col=0, parse_dates=True)
    wind_df = wind_df.loc[wind_df.index.isin(grouped_index), :]

    wind_df_loc = [wind_df.loc[:, ['latitude', 'longitude']].iloc[i] for i in range(len(wind_df.loc[:, ['latitude', 'longitude']].drop_duplicates()))]

    def import_data(i, file, size):
        # Time Count
        if i % 50 == 0 and i != 0:
            print(
                f'\nProgress: {i}/{NUM_FILES} Time: {(time.time() - start) / 60:.1f} min/{(time.time() - start) / i * NUM_FILES / 60:.1f} min',
                end=" ")
        else:
            print('.', end="")

        # Get Metadata
        key = file[-44:-4]
        building_meta_data = meta_data.loc[meta_data['Identifier'] == key]
        building_meta_data_design = meta_data_design.loc[meta_data_design['Identifier'] == key]

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
        # Get only winter days
        df = df.loc[df.index.isin(grouped_index), :]

        # print(f'import df: {time.time() - df_t:.2f}')
        start_t = time.time()
        if HP_ONLY:
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

        if df.shape[0] == 0:
            READ_IN = False

        read_t = time.time()
        if READ_IN:

            # Fill some NaNs
            t0 = time.time()
            df[interp_cols] = df[interp_cols].interpolate(method='linear', limit=3)
            df['T_ctrl_C'] = (df['T_ctrl'] - 32)*5/9
            df['T_out_C'] = (df['T_out'] - 32)*5/9
            df['Design_Temp'] = building_meta_data_design['99%heating'].astype(np.float32).iloc[0]
            df['Design_Temp_C'] = (df['Design_Temp']-32)*5/9
            # print(f'bfill: {time.time() - t0:.2f} s')
            # Get Runtime for multistage devices
            t1 = time.time()
            df = process.get_effective_runtime(df, building_meta_data)
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

            # solar_df = solar_dfs[np.argmin(solar_distances)].resample('5T').interpolate()
            # wind_df_to_merge = wind_df.loc[(wind_df['latitude'] == wind_df_loc[np.argmin(weather_distances)][0]) & (wind_df['longitude'] == wind_df_loc[np.argmin(weather_distances)][1])].resample('5T').interpolate()
            #
            # df = pd.merge(df, solar_df[['DHI', 'DNI', 'GHI_(kW/m2)','Wind Speed', 'Temperature']], left_index=True, right_index=True)
            # df = pd.merge(df, wind_df_to_merge[['100m_Wind_Speed_(m/s)']], left_index=True, right_index=True)
            #
            # df['Nearest_Lat'] = solar_dfs_loc[np.argmin(solar_distances)]['Latitude'].astype(np.float16)
            # df['Nearest_Long'] = solar_dfs_loc[np.argmin(solar_distances)]['Longitude'].astype(np.float16)

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

            if size == 'small':
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
            df_list_parallel = [e.submit(import_data, i, f, size) for i, f in enumerate(filenames)]
        df_list = []
        for i in range(len(filenames)):
            df_list.append(df_list_parallel[i].result())
    else:
        df_list = []
        for i, file in enumerate(filenames):
            df = import_data(i, file, size)
            df_list.append(df)


    return df_list

def import_load_data(year, location):
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

if __name__ == '__main__':
    location = 'NY'
    DATA_DIR = f'{pathlib.Path(os.path.abspath("")).parents[1]}/Data Files'
    size='small'
    season='summer'
    # Heat Pump Only
    df_list_hp = import_grouped_data(location, NUM_FILES=10000, HP_ONLY=True, parallel=True, season=season)
    df_list_hp = [x for x in df_list_hp if x is not None]
    pickle.dump(df_list_hp, open(f'{DATA_DIR}/DF Lists/df_list_hp_{location}_{size}_{season}.sav', 'wb'))
    df_list_hp = pickle.load(open(f'{DATA_DIR}/DF Lists/df_list_hp_{location}_{size}_{season}.sav', 'rb'))
    grouped_df_hp = process.group_dfs(df_list_hp, size)
    pickle.dump(grouped_df_hp, open(f'{DATA_DIR}/DF Lists/grouped_df_hp_{location}_{size}_{season}.sav', 'wb'))
    grouped_loc_df_hp = process.group_dfs_by_location(df_list_hp, size)
    pickle.dump(grouped_loc_df_hp, open(f'{DATA_DIR}/DF Lists/grouped_loc_df_hp_{location}_{size}_{season}.sav', 'wb'))

    # Gas Only
    # df_list_gas = import_grouped_data(location, NUM_FILES=10000, HP_ONLY=False, parallel=False, size=size)
    # df_list_gas = [x for x in df_list_gas if x is not None]
    # pickle.dump(df_list_gas, open(f'{DATA_DIR}/DF Lists/df_list_gas_{location}_{size}.sav', 'wb'))
    # df_list_gas = pickle.load(open(f'{DATA_DIR}/DF Lists/df_list_gas_{location}_{size}.sav', 'rb'))
    # grouped_df_gas = process.group_dfs(df_list_gas, size)
    # pickle.dump(grouped_df_gas, open(f'{DATA_DIR}/DF Lists/grouped_df_gas_{location}_{size}.sav', 'wb'))
    # grouped_loc_df_gas = process.group_dfs_by_location(df_list_gas, size)
    # pickle.dump(grouped_loc_df_gas, open(f'{DATA_DIR}/DF Lists/grouped_loc_df_gas_{location}_{size}.sav', 'wb'))