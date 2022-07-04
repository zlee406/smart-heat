import pandas as pd
import numpy as np
import os, sys, pathlib
import matplotlib.style
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Helvetica')
import  pickle
sys.path.append(f'{pathlib.Path(os.path.abspath("")).parents[0]}')
from . import read, cop, process

DATA_DIR = f'{pathlib.Path(os.path.abspath("")).parents[0]}/Data Files'

def import_ecobee_data(location, size, fuel_type, DATA_DIR=None):
    """
    Function to import all data from precomputed df_lists. Run read.main() to process these and add save them.
    Args:
        location:
        size:
        fuel_type:
        DATA_DIR:

    Returns:

    """
    if DATA_DIR is None:
        DATA_DIR = f'{pathlib.Path(os.path.abspath("")).parents[0]}/Data Files'
    out_dict = {}
    out_dict['df_list'] = pickle.load(open(f'{DATA_DIR}/DF Lists/df_list_{fuel_type}_{location}_{size}.sav', 'rb'))
    out_dict['grouped_df'] = pickle.load(open(f'{DATA_DIR}/DF Lists/grouped_df_{fuel_type}_{location}_{size}.sav', 'rb'))
    out_dict['grouped_df_loc'] = pickle.load(open(f'{DATA_DIR}/DF Lists/grouped_loc_df_{fuel_type}_{location}_{size}.sav', 'rb'))
    out_dict['grouped_daily'] = out_dict['grouped_df'].groupby(out_dict['grouped_df'].index.map(lambda t: t.minute + 60*t.hour)).mean()
    out_dict['grouped_daily'].index = pd.date_range(start='01/01/08 00:00', freq='5T', periods=len(out_dict['grouped_daily'].index))

    return out_dict

def import_other_heat_data(source, location=None, normalize=True, daily=True, DATA_DIR=None):
    """
    Imports other heat load data from existing literature used for comparisons.
    Args:
        source:
        location:
        normalize:
        daily:
        DATA_DIR:

    Returns:

    """
    if DATA_DIR is None:
        DATA_DIR = f'{pathlib.Path(os.path.abspath("")).parents[0]}/Data Files'
    if source == 'when2heat':
        pd.options.mode.chained_assignment = None  # default='warn'
        when2heat_total = pd.read_csv(f'{DATA_DIR}/Heat Loads/when2heat/when2heat_transpose.csv', index_col=0,
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
        synPRO = pd.read_csv(f'{DATA_DIR}/Heat Loads/synPRO/synPRO.csv', index_col=0, parse_dates=['time'])
        synPRO = synPRO.resample('5T').interpolate()['heat']
        if normalize:
            synPRO = synPRO/synPRO.mean()
        if daily:
            return synPRO
        else:
            raise(LookupError('Synpro does not support non-daily output'))

    elif source == 'nrel':
        nrel = pd.read_csv(f'{DATA_DIR}/Heat Loads/OpenEI/{location}.csv', index_col=0, parse_dates=['Date/Time'])
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
    """
    Groups the aggregated timeseries data into daily peak.
    Args:
        grouped_df:

    Returns:

    """
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
    """
    Calculates the correlation with renewable energy. Creates bins based on renewable energy resources (wind/solar).
    Also calculates low_resource, high demand periods that can be plotted. These are defined by both renewable sources
    being under wind_thresh and solar_thresh with a heating demand above demand_thresh.
    Args:
        grouped_loc_df: timeseries dataframe of each clustered region separated into a multi index of [time, location]
        resample_freq: Period of time to average over
        wind_thresh: Threshold for wind speed (m/s)
        solar_thresh: Threshold for solar GHI (kW/m2)
        demand_thresh: Threshold for normalized heating demand

    Returns:

    """

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
