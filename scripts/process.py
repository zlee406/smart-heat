import time, os, sys, pathlib
import pandas as pd
import dask.dataframe as dd
import numpy as np
sys.path.append(f'{pathlib.Path(os.path.abspath("")).parents[0]}')

# User Packages
try:
    import cop
except Exception as e:
    from . import cop


def get_effective_runtime(df, building_meta_data):

    if (df['compHeat2'] > 0).any():
        df['effectiveCompRun'] = df[['compHeat1', 'compHeat2']].mean(axis=1)
        df['countCompRun'] = df['compHeat1'].notna() & df['compHeat2'].notna()
        has_hp=True
    elif (df['compHeat1'] > 0).any():
        df['effectiveCompRun'] = df['compHeat1']
        df['countCompRun'] = df['compHeat1'].notna()
        has_hp=True
    else:
        has_hp=False

    if has_hp:
        # Get Auxilliary Gas Heat Runtime
        if (df['auxHeat2'] > 0).any() and (df['auxHeat3'] > 0).any():
            df['effectiveAuxRun'] = df[['auxHeat1', 'auxHeat2', 'auxHeat3']].mean(axis=1)
            df['countAuxRun'] = df['auxHeat1'].notna() & df['auxHeat2'].notna() & df['auxHeat3'].notna()
        elif (df['auxHeat2'] > 0).any():
            df['effectiveAuxRun'] = df[['auxHeat1', 'auxHeat2']].mean(axis=1)
            df['countAuxRun'] = df['auxHeat1'].notna() & df['auxHeat2'].notna()
        else:
            df['effectiveAuxRun'] = df['auxHeat1']
    else:
        # # Get Regular Aux Heat Runtime
        if (df['auxHeat2'] > 0).any():
            df['effectiveGasRun'] = df[['auxHeat1', 'auxHeat2']].mean(axis=1)
            df['countGasRun'] = df['auxHeat1'].notna() & df['auxHeat2'].notna()
        elif (df['auxHeat2'] > 0).any() and (df['auxHeat3'] > 0).any():
            df['effectiveGasRun'] = df[['auxHeat1', 'auxHeat2', 'auxHeat3']].mean(axis=1)
            df['countGasRun'] = df['auxHeat1'].notna() & df['auxHeat2'].notna() & df['auxHeat3'].notna()
        else:
            df['effectiveGasRun'] = df['auxHeat1']
            df['countGasRun'] = df['auxHeat1'].notna()

    return df

def get_effective_power(df):

    # Get Estimated COP at each time step
    HSPF = 10
    df['cop'] = cop.calc_cop_v(df['T_ctrl_C'], df['T_out_C'], HSPF)

    if 'effectiveGasRun' in df.columns:
        df['gasElectricalDemandAsHP'] = df['effectiveGasRun'] / df['cop']
        df['effectiveElectricPower'] = np.nan
        df['effectiveElectricPower_w/out_aux'] = np.nan
        df['effectiveHeat'] = df['effectiveGasRun']
    elif 'effectiveCompRun' in df.columns:
        df['eta'] = cop.calc_cop(21.11, df['Design_Temp_C'][0]) * cop.calc_power(df['Design_Temp_C'][0], HSPF)
        df['effectiveCompPower'] = cop.calc_power_v(df['T_out_C'], HSPF) * df['effectiveCompRun']
        df['effectiveAuxPower'] = df['eta'] * df['effectiveAuxRun']
        df['effectiveCompHeat'] = df['effectiveCompPower'] * df['cop']
        df['effectiveAuxHeat'] = df['effectiveAuxPower']
        df['effectiveHeat'] = df['effectiveCompRun'] + df['effectiveAuxRun'] #FIXME Not right but for testing
        df['effectiveElectricPower'] = df['effectiveCompPower'] + df['effectiveAuxPower']
        df['effectiveElectricPowerNoAux'] = df['effectiveCompPower'] + df['effectiveAuxPower'] / df['cop']

        # for n in [1, 2, 4, 6, 8, 10]:
        #     df[f'effectiveElectricPower_n={n}'] = df['effectiveCompHeat'] + n*df['effectiveAuxHeat']
        #     df[f'effectiveElectricPower_w/out_aux_n={n}'] = df['effectiveCompHeat'] + n * df[
        #         'effectiveAuxHeat'] / df['COP']
        #     df['gasElectricalDemandAsHP'] = np.nan
        #     df[f'effectiveHPHeat_n={n}'] = df['effectiveCompHeat'] * df['COP'] + n*df['effectiveAuxHeat']

    return df


def group_dfs(df_list, size='small'):
    df_list = [x for x in df_list if x is not None]

    if size == 'small':
        dtype_dict = {'T_ctrl_C': np.float16, 'T_out_C': np.float16,
                      # 'GHI_(kW/m2)': np.float32, 'Wind Speed': np.float32,
                      'effectiveHeat': np.float32, 'effectiveElectricPower': np.float32, 'fan': np.float32,
                      # 'Nearest_Lat': np.float32, '100m_Wind_Speed_(m/s)': np.float32
                      }
    else:
        dtype_dict = {'T_stp_cool': np.float32, 'T_stp_heat': np.float32, 'Humidity': np.float32,
                      'auxHeat1': np.float32, 'auxHeat2': np.float32, 'auxHeat3': np.float32,
                      'compHeat1': np.float32, 'compHeat2': np.float32, 'fan': np.float32,
                      'Thermostat_Temperature': np.float32, 'T_out': np.float32, 'RH_out': np.float32,
                      'T_ctrl': np.float32, 'T_ctrl_C': np.float32, 'T_out_C': np.float32,
                      # 'effectiveGasHeat': np.float32, 'effectiveGasPower': np.float32
                      }

    grouped_df = dd.concat(df_list).astype(dtype_dict)

    grouped_df = grouped_df.groupby('DateTime').mean().compute()

    grouped_df['effectiveElectricPowerNorm'] = grouped_df['effectiveElectricPower'] / np.mean(
        grouped_df['effectiveElectricPower'])
    grouped_df['effectiveHeatNorm'] = grouped_df['effectiveHeat']/np.mean(grouped_df['effectiveHeat'])

    return grouped_df

def group_dfs_by_location(df_list, size='small'):
    for df in df_list:
        df.index.rename('DateTime', inplace=True)

    if size == 'small':
        dtype_dict = {'T_ctrl_C': np.float16, 'T_out_C': np.float16,
                      # 'GHI_(kW/m2)': np.float16, 'Wind Speed': np.float16,
                      'effectiveHeat': np.float32, 'effectiveElectricPower': np.float16, 'fan': np.float16,
                      # '100m_Wind_Speed_(m/s)': np.float32,
                      # 'Nearest_Lat': np.float32
                      }
    else:
        dtype_dict = {'T_stp_cool': np.float16, 'T_stp_heat': np.float16, 'Humidity': np.float16,
                      'auxHeat1': np.float32, 'auxHeat2': np.float32, 'auxHeat3': np.float32,
                      'compHeat1': np.float32, 'compHeat2': np.float32, 'fan': np.float16,
                      'Thermostat_Temperature': np.float32, 'T_out': np.float32, 'RH_out': np.float16,
                      'T_ctrl': np.float32, 'T_ctrl_C': np.float32, 'T_out_C': np.float32,
                      'Nearest_Lat': np.float32, 'Nearest_Long': np.float32, 'Lat': np.float32, 'Long': np.float32
                      # 'effectiveGasHeat': np.float32, 'effectiveGasPower': np.float32
                      }

    grouped_df = dd.concat(df_list).astype(dtype_dict)
    grouped_df = grouped_df.groupby(['DateTime', 'Nearest_Lat']).mean().compute()

    # Get Normalized Heating
    grouped_df['effectiveHeatNorm'] = np.nan
    mean_per_lat = grouped_df.groupby(level=1).mean()
    for idx in mean_per_lat.index:
        grouped_df.loc[grouped_df.index.get_level_values(1) == idx, 'effectiveHeatNorm'] = \
            grouped_df.loc[grouped_df.index.get_level_values(1) == idx, 'effectiveHeat']/mean_per_lat.loc[idx, 'effectiveHeat']

    lats = []
    for df in df_list:
        lats.append(df['Nearest_Lat'].iloc[0])
    a, b = np.unique(lats, return_counts=True)
    lat_counts = dict(zip(a, b))

    include_lats = []
    for lat in lat_counts.keys():
        if lat_counts[lat] > 100:
            include_lats.append(lat)

    grouped_df = grouped_df.loc[pd.IndexSlice[:, include_lats], :]

    grouped_df['effectiveElectricPowerNorm'] = grouped_df['effectiveElectricPower'] / np.mean(
        grouped_df['effectiveElectricPower'])

    return grouped_df

def get_additional_energy(grouped_df):

    grouped_df['additionalElectricPower'] = grouped_df['effectiveElectricPower'] - grouped_df['effectiveElectricPower_w/out_aux']
    grouped_df['potentialElectricPowerPctDecrease'] = (-grouped_df['effectiveElectricPower']
                                                        + grouped_df['effectiveElectricPower_w/out_aux']
                                                        )/grouped_df['effectiveElectricPower']

    return grouped_df


def process_df_list(df_list):

    grouped_df = group_dfs(df_list)

    grouped_df = get_additional_energy(grouped_df)

    return grouped_df
