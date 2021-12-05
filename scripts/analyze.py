import pandas as pd
import numpy as np
import os, sys, pathlib
import time
import concurrent.futures

import pickle
sys.path.append(f'{pathlib.Path(os.path.abspath("")).parents[0]}')

DATA_DIR = f'{pathlib.Path(os.path.abspath("")).parents[0]}/data'

def import_ecobee_data(location, size, fuel_type):
    out_dict = {}
    out_dict['df_list'] = pickle.load(open(f'{DATA_DIR}/pickled_data/df_list_{fuel_type}_{location}_{size}.sav', 'rb'))
    out_dict['grouped_df'] = pickle.load(open(f'{DATA_DIR}/pickled_data/grouped_df_{fuel_type}_{location}_{size}.sav', 'rb'))
    out_dict['grouped_df_loc'] = pickle.load(open(f'{DATA_DIR}/pickled_data/grouped_loc_df_{fuel_type}_{location}_{size}.sav', 'rb'))
    out_dict['grouped_daily'] = out_dict['grouped_df'].groupby(out_dict['grouped_df'].index.map(lambda t: t.minute + 60*t.hour)).mean()
    out_dict['grouped_daily'].index = pd.date_range(start='01/01/08 00:00', freq='5T', periods=len(out_dict['grouped_daily'].index))

    return out_dict

def import_other_heat_data(source, location=None, normalize=True, daily=True):
    if source == 'when2heat':
        pd.options.mode.chained_assignment = None  # default='warn'
        when2heat_total = pd.read_csv(f'{DATA_DIR}/heat_loads/when2heat/when2heat_transpose.csv', index_col=0,
                                      parse_dates=['1'])
        when2heat_daily = pd.DataFrame(index=pd.date_range(start='01/01/08 00:00', freq='60T', periods=24),
                                       columns=when2heat_total['Country'].unique())
        when2heat_year = {}
        for country in when2heat_total['Country'].unique():
            country_df = when2heat_total[when2heat_total['Country'] == country]
            country_df.index = pd.to_datetime(country_df.index)
            country_df = country_df[(country_df.index.month <= 3)]
            #     country_df.index = country_df.index.tz_localize(None)
            when2heat_daily[country] = \
            country_df.groupby(country_df.index.map(lambda t: t.minute + 60 * t.hour)).mean()[
                'heat_profilespace_SFH'].values
            years = country_df.index.year.unique()[1:6]
            when2heat_year[country] = {2009 + n: year_df for n, year_df in
                                       enumerate([country_df[country_df.index.year == year] for year in years])}
            for year in years:
                when2heat_year[country][year].loc[:, 'heat_profilespace_SFHNorm'] = when2heat_year[country][year][
                                                                                        'heat_profilespace_SFH'] / np.mean(
                    when2heat_year[country][year]['heat_profilespace_SFH'])

        when2heat_daily = when2heat_daily.resample('5T').interpolate().loc[:, location]
        if normalize:
            when2heat_daily = when2heat_daily/when2heat_daily.mean()
        if daily:
            return when2heat_daily
        else:
            if normalize:
                return {year: when2heat_year[location][year]['heat_profilespace_SFHNorm'] for year in years}
            else:
                return {year: when2heat_year[location][year]['heat_profilespace_SFH'] for year in years}

    elif source == 'synpro':
        # SynPro
        synPRO = pd.read_csv(f'{DATA_DIR}/heat_loads/synPRO/synPRO.csv', index_col=0, parse_dates=['time'])
        synPRO = synPRO.resample('5T').interpolate()['heat']
        if normalize:
            synPRO = synPRO/synPRO.mean()
        if daily:
            return synPRO
        else:
            raise(LookupError('Synpro does not support non-daily output'))

    elif source == 'nrel':
        nrel = pd.read_csv(f'{DATA_DIR}/heat_loads/OpenEI/{location}.csv', index_col=0, parse_dates=['Date/Time'])
        nrel.index = pd.date_range(start='01/01/19 01:00', freq='60T', periods=8760)
        nrel = nrel[nrel.index.month <= 3][:-1]
        nrel_daily = nrel.groupby(nrel.index.map(lambda t: t.minute + 60 * t.hour)).mean()[
            'Heating:Gas [kW](Hourly)']
        nrel_daily.index = pd.date_range(start='01/01/08 00:00', freq='60T', periods=24)
        nrel_daily = nrel_daily.resample('5T').interpolate()
        if normalize:
            nrel_daily = nrel_daily/nrel_daily.mean()
            nrel = nrel['Heating:Gas [kW](Hourly)']/nrel['Heating:Gas [kW](Hourly)'].mean()
        if daily:
            return nrel_daily
        else:
            return nrel

def get_daily_peaks(grouped_df):
    if type(grouped_df) == dict:
        peaks = {
            key: df.loc[df.groupby(pd.Grouper(freq='D')).idxmax()] for key, df in
            zip(grouped_df.keys(), grouped_df.values())}
        years = list(grouped_df.keys())
        peaks = pd.concat(peaks[year] for year in years)
    else:
        grouped_df = grouped_df[grouped_df.index.month <= 3]
        peaks = grouped_df.loc[grouped_df.groupby(pd.Grouper(freq='D')).idxmax()]
    return peaks

def renewable_correlation(grouped_loc_df, resample_freq='15T', wind_thresh=6, solar_thresh=.125, demand_thresh=2):

    grouped_loc_df_resample = grouped_loc_df.groupby(pd.Grouper(freq=resample_freq, level=-2)).mean()

    grouped_loc_df_resample['time_of_day'] = (grouped_loc_df_resample.index.hour*60 + grouped_loc_df_resample.index.minute)/60

    grouped_loc_df_resample['solar_bins'] = pd.cut(grouped_loc_df_resample['GHI_(kW/m2)'],
                                                       np.arange(-.0008, .9, .125), precision=2)
    grouped_loc_df_resample['wind_bins'] = pd.cut(grouped_loc_df_resample['100m_Wind_Speed_(m/s)'],
                                                      np.arange(-.0008, 20, 3), precision=2)
    grouped_loc_df_resample['heat_bins'] = pd.cut(grouped_loc_df_resample['effectiveHeatNorm'],
                                                  np.arange(-.0008, 4, .25), precision=2)


    high_demand = grouped_loc_df_resample.loc[grouped_loc_df_resample['effectiveHeatNorm'] >= demand_thresh]
    low_resource_high_demand = high_demand.loc[
        (high_demand['GHI_(kW/m2)'] <= solar_thresh) & (high_demand['100m_Wind_Speed_(m/s)'] <= wind_thresh)]
    print('High Demand, Low Resource: ', len(low_resource_high_demand))
    print('High Demand', len(high_demand))
    print('Percent: ', len(low_resource_high_demand)/len(high_demand))

    return grouped_loc_df_resample, low_resource_high_demand

def get_metadata(metadata, f):

    DIR = 'Data Files/2019 NY'
    try:
        df = pd.read_csv(f'{DIR}/{f}',
                         usecols=['auxHeat1', 'auxHeat2', 'auxHeat3', 'compCool1', 'compCool2',
                                  'compHeat1', 'compHeat2',
                                  'Schedule', 'Event'
                                  ],
                         dtype={'auxHeat1': np.float16, 'auxHeat2': np.float16, 'auxHeat3': np.float16,
                                'compCool1': np.float16, 'compCool2': np.float16,
                                'compHeat1': np.float16, 'compHeat2': np.float16,
                                'Schedule': 'string', 'Event': 'string'}
                         )
        # Get Compressor Stages
        if (df['compHeat2'] > 0).any():
            compStage = 2
        elif (df['compHeat1'] > 0).any():
            compStage = 1
        else:
            compStage = 0

        # Get Aux Stages
        if (df['auxHeat2'] > 0).any() and (df['auxHeat3'] > 0).any():
            auxStage = 3
        elif (df['auxHeat2'] > 0).any():
            auxStage = 2
        elif (df['auxHeat1'] > 0).any():
            auxStage = 1
        else:
            auxStage = 0

        # Get Schedule Pct
        schedule = pd.get_dummies(df['Schedule'])
        away_pct = 0
        home_pct = 0
        sleep_pct = 0
        custom_sch_pct = 0
        awake_pct = 0
        tou_pct = 0
        for c in schedule.columns:
            if c == 'Away':
                away_pct = np.mean(schedule[c])
            elif c == 'Home':
                home_pct = np.mean(schedule[c])
            elif c == 'Sleep':
                sleep_pct = np.mean(schedule[c])
            elif c == 'Awake':
                awake_pct = np.mean(schedule[c])
            elif c == 'TOU':
                tou_pct = np.mean(schedule[c])
            else:
                assert c[0] == 'c', 'sch' + c
            custom_sch_pct = 1 - (away_pct + home_pct + sleep_pct + awake_pct + tou_pct)

        # Get Event Pct
        event = pd.get_dummies(df['Event'])
        smart_away_pct = 0
        smart_home_pct = 0
        dr_pct = 0
        hold_pct = 0
        custom_pct = 0
        tou_event_pct = 0
        for c in event.columns:
            if c == 'Hold':
                hold_pct = np.mean(event[c])
            elif c == 'Smart Home':
                smart_home_pct = np.mean(event[c])
            elif c == 'Smart Away':
                smart_away_pct = np.mean(event[c])
            elif c == 'Demand Response Event':
                dr_pct = np.mean(event[c])
            elif c == 'TOU':
                tou_event_pct = np.mean(event[c])
            else:
                assert c[0] == 'c', 'event' + c
        custom_event_pct = 1 - (hold_pct + smart_home_pct + smart_away_pct + dr_pct + tou_event_pct)

        # Get Other

        metadata_row = metadata[metadata['filename'] == f]
        state = metadata_row['ProvinceState'].values[0]
        sq_ft = metadata_row['Floor Area [ft2]'].values[0]
        age = metadata_row['Age of Home [years]'].values[0]
        style = metadata_row['Style'].values[0]
        floors = metadata_row['Number of Floors'].values[0]
        city = metadata_row['City'].values[0]

        return (f, compStage, auxStage, away_pct, home_pct, sleep_pct, awake_pct, tou_pct, custom_sch_pct,
                smart_away_pct, smart_home_pct, dr_pct, hold_pct, tou_event_pct, custom_event_pct,
                sq_ft, floors, age, style, state, city)
    except Exception as e:
        print(f'{f} is invalid because of {e}')
        return (np.nan,) * 21

def process_metadata(province_list, data_dir, file_name='metadata_process.csv'):
    metadata = pd.read_csv('Data Files/meta_data.csv').drop_duplicates()
    metadata = metadata[metadata['ProvinceState'].isin(province_list)]

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=22) as executor:
        results = [executor.submit(get_metadata, metadata, f) for f in os.listdir(data_dir)]

    processed_metadata = pd.DataFrame(index=np.arange(len(os.listdir(data_dir))), columns=['filename', 'compStages', 'auxStages',
                                                                                           'Away %', 'Home %', 'Sleep %', 'Awake %',
                                                                                           'TOU %', 'Custom Schedule %',
                                                                                           'Smart Away %', 'Smart Home %', 'DR %',
                                                                                           'Hold %', 'TOU Event %',
                                                                                           'Custom Event %', 'Sq. Ft.', 'Floors',
                                                                                           'Age', 'Style', 'State', 'City'])
    for n, f in enumerate(os.listdir(data_dir)):
        r = results[n].result()
        processed_metadata.iloc[n] = r
    print('Process End Time: ', time.time() - start, ' s')
    processed_metadata.to_csv(f'data/{file_name}')