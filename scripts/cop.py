import numpy as np

def calc_cop(indoor_temp, outdoor_temp, HSPF):
    '''

    :param indoor_temp: Indoor temperature in Celcius
    :param outdoor_temp: Outdoor Temperature in Celcius
    :param HSPF: Rating to calculate COP for in [9, 10, 14,]
    :return:
    '''
    # From NEEP Data: Params are for Farenheit so must convert temperatures
    if np.isnan(indoor_temp) or np.isnan(outdoor_temp):
        return np.nan

    if HSPF == 9:
        indoor_temp = indoor_temp * 9 / 5 + 32
        outdoor_temp = outdoor_temp * 9 / 5 + 32
        params = [4.4029, -.033447]
    elif HSPF == 10:
        indoor_temp = indoor_temp * 9 / 5 + 32
        outdoor_temp = outdoor_temp * 9 / 5 + 32
        params = [4.4693, -.035504]
    elif HSPF == 14:
        indoor_temp = indoor_temp * 9/5 + 32
        outdoor_temp = outdoor_temp * 9/5 + 32
        params = [5.5930, -.05187]
    else:
        raise(ValueError(f'{HSPF} not a valid HSPF. Possible values are: [9, 10, 14]'))

    delta_t = indoor_temp - outdoor_temp

    cop = sum(params[i] * delta_t ** i for i in range(len(params)))
    if cop <= 1: cop = 1

    return cop

def calc_power(outdoor_temp, HSPF):

    # From NEEP Data: Params are for Farenheit so must convert temperatures
    if np.isnan(outdoor_temp):
        return np.nan

    if HSPF == 9:
        outdoor_temp = outdoor_temp * 9/5 + 32
        params = [3.6317, 0.00744944]
    elif HSPF == 10:
        outdoor_temp = outdoor_temp * 9/5 + 32
        params = [2.8739, -0.00026471]
    elif HSPF == 14:
        outdoor_temp = outdoor_temp * 9/5 + 32
        params = [1.8214, -0.00105342]
    else:
        raise(ValueError(f'{HSPF} not a valid HSPF. Possible values are: [9, 10, 14]'))

    power = sum(params[i] * outdoor_temp ** i for i in range(len(params))) # in kW

    return power


calc_cop_v = np.vectorize(calc_cop)
calc_power_v = np.vectorize(calc_power)