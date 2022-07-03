import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import matplotlib.tri as mtri
import matplotlib.ticker as mtick
from matplotlib import cm
import numpy as np
import seaborn as sns
import pandas as pd

def plot_daily_heating_demand(data_list, label_list, color_list=None, style_list=None):
    """
    Plots the daily heating demand profile for each df in the data_list. data_list should be a grouded_daily_df of
    only 24 hours.
    Args:
        data_list: data_list should be a grouded_daily_df of only 24 hours.
        label_list: labels to put in the legend
        color_list: colors for each data entry
        style_list: linestyle for each data entry

    Returns:

    """
    if color_list is None:
        color_list = ['navy', 'red', 'purple', 'green', 'green']
    if style_list is None:
        style_list = ['-', '-', '-', '-', '--']

    f, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    for n, df in enumerate(data_list):
        ax1.plot(df.index,
                 df,
                 label=label_list[n],
                 color=color_list[n],
                 linewidth=3,
                 linestyle=style_list[n],
                 )
        print(label_list[n], df.max())

    ax1.axhline(1, linestyle='--', color='k', linewidth=3, label='Mean Demand')

    handles, labels = ax1.get_legend_handles_labels()
    idxs = [3, 2, 0, 1] # order for legend labels
    handles, labels = [handles[idx] for idx in idxs], [labels[idx] for idx in idxs]
    # sort both labels and handles by labels
    ax1.legend(handles, labels, fontsize=20)
    ax1.set_ylim(0, 2.25)
    ax1.set_ylabel('Normalized Heating\n Demand', fontsize=24)

    daysFmt = mdates.DateFormatter('%m/%d/%y')
    hourFmt = mdates.DateFormatter('%H:%M')
    ax1.tick_params(axis='both', labelsize=22)
    #     ax1.set_ylim(0, .75)
    ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=range(4, 23, 4)))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(hourFmt)
    ax1.set_xlabel('Time of Day', fontsize=24)
    ax1.set_xlim((df.index[0], df.index[-1]))
    ax1.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01)
    plt.savefig('plots/normalized_all.pdf', dpi=300)
    plt.show()

    return f, ax1

def plot_daily_peaks(data_list, label_list, color_list=None):
    """
    Plots the histogram of daily peaks
    Args:
        data_list: data_list should be a grouded_df of the full analyzed time period
        label_list: labels to put in the legend
        color_list: colors for each data entry

    Returns:

    """
    if color_list is None:
        color_list = ['green', 'purple', 'navy']
    with matplotlib.style.context('seaborn-paper'):
        f, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        for n, data in enumerate(data_list):
            weights = 1/(len(data)/90)
            g = sns.histplot(x=data, ax=axs[n], color=color_list[n], binwidth=.25, binrange=[.25, 3.5], weights=weights,
                             legend=False)
            sns.histplot(np.arange(-3, -1), color=color_list[n], ax=axs[1], label=label_list[n])
            axs[n].axvline(data.mean(), color='k', linestyle='--', linewidth=3, label='Mean')
            print(label_list[n], data.mean())

        for ax in axs:
            ax.tick_params(labelsize=16)
            # ax1.tick_params(labelsize=12, which='minor')
            # ax1.set_ylim(0, .5)
            ax.grid(True)
            ax.set_xlim(0.1, 3.6)
            ax.set_ylim(0, 29)
            ax.set_ylabel('')

        axs[1].set_ylabel('Number of Days per Year', fontsize=20)
        axs[-1].set_xlabel('Daily Peak Normalized Heating Demand', fontsize=20)

        handles, labels = axs[1].get_legend_handles_labels()
        idxs = [1, 2, 3, 0] # order for legend labels
        handles, labels = [handles[idx] for idx in idxs], [labels[idx] for idx in idxs]
        # sort both labels and handles by labels
        axs[1].legend(handles, labels, fontsize=12)

        # axs[1].legend(fontsize=12)
        plt.savefig('plots/daily_peak_distribution.pdf', dpi=300)
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()
        return f, axs

def plot_renewable_correlation(grouped_loc_df, low_resource_high_demand):
    """

    Args:
        grouped_loc_df:
        low_resource_high_demand:

    Returns:

    """
    grouped_loc_df_clipped = grouped_loc_df.drop(index = low_resource_high_demand.index)

    with matplotlib.style.context('seaborn-paper'):
        f, (ax1) = plt.subplots(1, 1, figsize=(8, 4))
        for ax in [ax1]:
            ax.tick_params(labelsize=16)
            # ax1.tick_params(labelsize=12, which='minor')
            # ax1.set_ylim(0, .5)
            ax.grid(True)
            ax.set_ylabel('Normalized Heating\n Demand', fontsize=20)

        box_palette = sns.color_palette('Blues_d', 7)
        strip_palette = sns.color_palette('Blues_d', 7, desat=.5)
        jitter = True
        alpha = .35

        sns.boxplot(data=grouped_loc_df, x='solar_bins', y='effectiveHeatNorm',
                    ax=ax1, fliersize=0, palette=box_palette)
        sns.stripplot(data=grouped_loc_df_clipped, x='solar_bins', y='effectiveHeatNorm',
                      ax=ax1, dodge=False, jitter=jitter,
                      alpha=alpha, palette=strip_palette)
        sns.stripplot(data=low_resource_high_demand, x='solar_bins', y='effectiveHeatNorm',
                      ax=ax1, dodge=False, jitter=jitter,
                      alpha=.5, color='indianred')
        # ax1.scatter(0, -1, label='High Demand, Low Resource Periods', color='indianred')
        ax1.scatter(0, -1, label='15 Minute Time Periods', color='navy', alpha=alpha)

        ax1.set_xticks(ticks=[-.5, .5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        ax1.set_xticklabels(labels=np.arange(0, 1, .125))
        ax1.set_ylim((-.1, 3.5))
        ax1.set_ylabel('Normalized Heating\nDemand')
        ax1.set_xlabel('Local Global Horizontal Irradiance (GHI) (kW/m2)', fontsize=20)
        ax1.grid(True)
        ax1.legend(fontsize=14,framealpha=.5 )

        plt.tight_layout()
        plt.savefig('plots/demand_vs_solar.png', dpi=300)

        f, (ax2) = plt.subplots(1, 1, figsize=(8, 4))
        for ax in [ax1, ax2]:
            ax.tick_params(labelsize=16)
            # ax1.tick_params(labelsize=12, which='minor')
            # ax1.set_ylim(0, .5)
            ax.grid(True)
            ax.set_ylabel('Normalized Heating\n Demand', fontsize=20)

        sns.boxplot(data=grouped_loc_df, x='wind_bins', y='effectiveHeatNorm',
                    ax=ax2, fliersize=0, palette=box_palette)
        sns.stripplot(data=grouped_loc_df_clipped, x='wind_bins', y='effectiveHeatNorm',
                      ax=ax2, dodge=False, jitter=jitter,
                      alpha=alpha, palette=strip_palette)
        sns.stripplot(data=low_resource_high_demand, x='wind_bins', y='effectiveHeatNorm',
                      ax=ax2, dodge=False, jitter=jitter,
                      alpha=.5, color='indianred')
        # ax2.scatter(0, -1, label='High Demand, Low Resource Periods',
        #             color='indianred')
        ax2.scatter(0, -1, label='15 Minute Time Periods', color='navy', alpha=alpha)
        ax2.legend(fontsize=14, framealpha=.5, loc='upper right')


        ax2.set_xticks(ticks=[-.5, .5, 1.5, 2.5, 3.5, 4.5, 5.5])
        # ax2.set_xticklabels(labels=np.arange(0, 5.5, .75))
        # ax2.set_xticks(ticks=np.arange(-.5, 21, 3))
        ax2.set_xticklabels(labels=np.arange(0., 19., 3.))
        ax2.set_ylim((-.1, 3.5))
        ax2.set_ylabel('Normalized Heating\nDemand')
        ax2.set_xlabel('Local 100m Wind Speed (m/s)', fontsize=20)
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig('plots/demand_vs_wind.png', dpi=300)
        return f, ax2

def plot_renewable_correlation_loc(grouped_loc_df, low_resource_high_demand): #TODO
    grouped_loc_df_clipped = grouped_loc_df.drop(index = low_resource_high_demand.index)

    with matplotlib.style.context('seaborn-paper'):
        f, (ax1) = plt.subplots(1, 1, figsize=(8, 4))
        for ax in [ax1]:
            ax.tick_params(labelsize=16)
            # ax1.tick_params(labelsize=12, which='minor')
            # ax1.set_ylim(0, .5)
            ax.grid(True)
            ax.set_ylabel('Normalized Heating\n Demand', fontsize=20)

        box_palette = sns.color_palette('Blues_d', 7)
        strip_palette = sns.color_palette('Blues_d', 7, desat=.5)
        jitter = True
        alpha = .35

        sns.boxplot(data=grouped_loc_df, x='solar_bins', y='effectiveHeatNorm',
                    ax=ax1, fliersize=0, palette=box_palette)
        sns.stripplot(data=grouped_loc_df_clipped, x='solar_bins', y='effectiveHeatNorm',
                      ax=ax1, dodge=False, jitter=jitter,
                      alpha=alpha, palette=strip_palette)
        sns.stripplot(data=low_resource_high_demand, x='solar_bins', y='effectiveHeatNorm',
                      ax=ax1, dodge=False, jitter=jitter,
                      alpha=.5, color='indianred')
        # ax1.scatter(0, -1, label='High Demand, Low Resource Periods', color='indianred')
        ax1.scatter(0, -1, label='15 Minute Time Periods', color='navy', alpha=alpha)

        ax1.set_xticks(ticks=[-.5, .5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        ax1.set_xticklabels(labels=np.arange(0, 1, .125))
        ax1.set_ylim((-.1, 3.5))
        ax1.set_ylabel('Normalized Heating\nDemand')
        ax1.set_xlabel('Local Global Horizontal Irradiance (GHI) (kW/m2)', fontsize=20)
        ax1.grid(True)
        ax1.legend(fontsize=14,framealpha=.5 )

        plt.tight_layout()
        plt.savefig('plots/demand_vs_solar.png', dpi=300)

        f, (ax2) = plt.subplots(1, 1, figsize=(8, 4))
        for ax in [ax1, ax2]:
            ax.tick_params(labelsize=16)
            # ax1.tick_params(labelsize=12, which='minor')
            # ax1.set_ylim(0, .5)
            ax.grid(True)
            ax.set_ylabel('Normalized Heating\n Demand', fontsize=20)

        sns.boxplot(data=grouped_loc_df, x='wind_bins', y='effectiveHeatNorm',
                    ax=ax2, fliersize=0, palette=box_palette)
        sns.stripplot(data=grouped_loc_df_clipped, x='wind_bins', y='effectiveHeatNorm',
                      ax=ax2, dodge=False, jitter=jitter,
                      alpha=alpha, palette=strip_palette)
        sns.stripplot(data=low_resource_high_demand, x='wind_bins', y='effectiveHeatNorm',
                      ax=ax2, dodge=False, jitter=jitter,
                      alpha=.5, color='indianred')
        # ax2.scatter(0, -1, label='High Demand, Low Resource Periods',
        #             color='indianred')
        ax2.scatter(0, -1, label='15 Minute Time Periods', color='navy', alpha=alpha)
        ax2.legend(fontsize=14, framealpha=.5, loc='upper right')


        ax2.set_xticks(ticks=[-.5, .5, 1.5, 2.5, 3.5, 4.5, 5.5])
        # ax2.set_xticklabels(labels=np.arange(0, 5.5, .75))
        # ax2.set_xticks(ticks=np.arange(-.5, 21, 3))
        ax2.set_xticklabels(labels=np.arange(0., 19., 3.))
        ax2.set_ylim((-.1, 3.5))
        ax2.set_ylabel('Normalized Heating\nDemand')
        ax2.set_xlabel('Local 100m Wind Speed (m/s)', fontsize=20)
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig('plots/demand_vs_wind.png', dpi=300)
        return f, ax2


def plot_peak_time(data_list, label_list, color_list=None): #TODO
    if color_list is None:
        color_list = ['green', 'purple', 'navy']
    with matplotlib.style.context('seaborn-paper'):
        f, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        #     box_palette=sns.color_palette('Blues_d', 7)
        #     strip_palette=sns.color_palette('Blues_d', 7, desat=.5)
        #     jitter=True
        #     alpha=.35
        for n, data in enumerate(data_list):
            weights = 1/(len(data)/90)
            g = sns.histplot(x=data, ax=axs[n], color=color_list[n], binwidth=.25, binrange=[.25, 3.5], weights=weights,
                             legend=False)
            sns.histplot(np.arange(-3, -1), color=color_list[n], ax=axs[1], label=label_list[n])
            axs[n].axvline(data.mean(), color='k', linestyle='--', linewidth=3, label='Mean')

        #
        # sns.histplot(ecobee_peaks['effectiveHeatNorm'], ax=ax1, color='g', binwidth=.25, binrange=[.25, 3.5])
        # ax1.axvline(ecobee_peaks['effectiveHeatNorm'].mean(),
        #             color='k', linestyle='--', linewidth=3)
        # sns.histplot(openEI_NYC_peaks['Heating:Gas [kW](Hourly)'] / openEI_NYC['Heating:Gas [kW](Hourly)'].mean(),
        #              ax=ax2, color='purple', binwidth=.25, binrange=[.25, 3.5])
        # ax2.axvline(
        #     (openEI_NYC_peaks['Heating:Gas [kW](Hourly)'] / openEI_NYC['Heating:Gas [kW](Hourly)'].mean()).mean(),
        #     color='k', linestyle='--', linewidth=3)
        # sns.histplot(x=pd.concat(when2heat_DE_peaks_all[year]['heat_profilespace_SFHNorm'] for year in years),
        #              ax=ax3, weights=1 / len(years), color='navy', binwidth=.25, binrange=[.25, 3.5])
        # ax3.axvline((when2heat_DE_peaks_all[year]['heat_profilespace_SFHNorm']).mean(),
        #             color='k', linestyle='--', linewidth=3)

        for ax in axs:
            ax.tick_params(labelsize=16)
            # ax1.tick_params(labelsize=12, which='minor')
            # ax1.set_ylim(0, .5)
            ax.grid(True)
            ax.set_xlim(0.1, 3.6)
            ax.set_ylim(0, 29)
            ax.set_ylabel('')

        axs[1].set_ylabel('Number of Days per Year', fontsize=20)
        axs[-1].set_xlabel('Daily Peak Normalized Heating Demand', fontsize=20)

        handles, labels = axs[1].get_legend_handles_labels()
        idxs = [1, 2, 3, 0]
        handles, labels = [handles[idx] for idx in idxs], [labels[idx] for idx in idxs]
        # sort both labels and handles by labels
        axs[1].legend(handles, labels, fontsize=12)

        # axs[1].legend(fontsize=12)
        plt.savefig('plots/daily_peak_distribution.pdf', dpi=300)
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()
        return f, axs


def plot_renewable_time(grouped_loc_df, low_resource_high_demand, color_list=None): #TODO
    if color_list is None:
        color_list = ['navy', 'orange', 'purple', 'green']
    df_mean = grouped_loc_df.groupby('heat_bins').mean()
    with matplotlib.style.context('seaborn-paper'):
        f, ax1 = plt.subplots(1, 1, figsize=(8, 4),
                              # subplot_kw={"projection": "3d"}
                              )
        N = df_mean.shape[0]
        width=.45
        ind = np.arange(N)
        ax1.bar(ind-width/2, df_mean['100m_Wind_Speed_(m/s)'], width,
                color='navy', alpha=.75,
                label='Wind',
                linewidth=3)
        ax1.bar(ind, 0, 0, label='GHI',
                color='green', alpha=.75,
                )
        ax1_2 = ax1.twinx()
        ax1_2.bar(ind+width/2, df_mean['GHI_(kW/m2)'], width,
                  color='green', alpha=.75,
                  linewidth=3)

        ax1.set_ylabel('Mean Local 100m\nWind Speed (m/s)', fontsize=16)
        ax1_2.set_ylabel('Mean Local GHI (kW/m2)', fontsize=16)

        ax1.tick_params(labelsize=16)
        ax1_2.tick_params(labelsize=16)
        ax1.set_xlabel('Normalized Heating Demand', fontsize=20)
        ax1.set_xticks(range(0, 14, 2))
        ax1.set_xticklabels(labels=[0 + .5 * i for i in range(7)])
        ax1_2.set_yticks([0, .05, .10, .15, .20])
        ax1.set_yticks([0., 2.5, 5.0, 7.5, 10.0])
        ax1.set_ylim(0, 11.25)
        ax1_2.set_ylim(0, .225)
        ax1.legend(ncol=2, fontsize=16)
        ax1.grid(True)
    plt.tight_layout()
    plt.savefig('plots/wind_and_solar.pdf', dpi=300)

    return f, ax1


def plot_setpoints(grouped_df, day): #TODO
    filler = np.zeros(288)  # FIXME
    if day == 'weekday':
        grouped_df = grouped_df[grouped_df.index.dayofweek <= 4]
        filler[216:228] = .071
    else:
        grouped_df = grouped_df[grouped_df.index.dayofweek > 4]
        filler[252:] = -.071 + np.linspace(0, .071, len(filler)-252)
    grouped_df_daily = grouped_df.groupby(grouped_df.index.map(lambda t: t.minute + 60 * t.hour)).mean()
    grouped_df_daily.index = pd.date_range(start='01/01/08 00:00', freq='5T', periods=len(grouped_df_daily.index))



    with matplotlib.style.context('seaborn-whitegrid'):
        linesize = 2.5
        f, ax1 = plt.subplots(1, 1, figsize=(10, 5.4))
        ax1.plot(grouped_df_daily.index,
                 (grouped_df_daily.loc[:, 'T_stp_heat'] - 32) * 5 / 9,
                 color='k', linewidth=linesize, label='Setpoint')
        ax1.plot(grouped_df_daily.index,
                 grouped_df_daily.loc[:, 'T_ctrl_C'] +filler,
                 color='navy', linewidth=linesize, label='Indoor Temperature')
        # ax1.plot(grouped_df_daily.index,
        #          (grouped_df_daily.loc[:, 'T_stp_heat'] - 32) * 5 / 9,
        #          color='darkgreen', linewidth=linesize, label='Setpoint')
        # ax1.plot(grouped_df_daily.index,
        #          grouped_df_daily.loc[:, 'T_ctrl_C'],
        #          color='darkgreen', linewidth=linesize, label='Indoor Temperature', linestyle='--')
        #     ax1.axhline(1, linestyle='--', color='k', linewidth=3, label='Mean Demand')
        ax1.legend(fontsize=24, ncol=2)
        ax1.set_ylabel('Average\nTemperature ($^\circ$C)', fontsize=24)
        ax1.tick_params(axis='both', labelsize=20)
        daysFmt = mdates.DateFormatter('%m/%d/%y')
        hourFmt = mdates.DateFormatter('%H:%M')
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.xaxis.set_major_formatter(hourFmt)
        ax1.set_xlabel('Time of Day', fontsize=24)
        ax1.set_ylim((18, 20))
        plt.tight_layout()
        plt.savefig(f'plots/average_setpoint_{day}.pdf', dpi=300)

    return f, ax1