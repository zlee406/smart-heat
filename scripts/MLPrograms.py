import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True,categories=False,auxcats=False):


    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list, df, or NumPy array.
        n_in: (int) Number of lag observations as input (X).
        n_out: (int) Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
        categories: (list) Categories to keep
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    df_orig = pd.DataFrame(data)
    if categories is not False:
        df = df_orig[categories]
    else:
        df = df_orig
    n_vars = 1 if type(data) is list else df.shape[1]
    cols, names = list(), list()
    # past sequence (t, t-1, ... t-n)
    for i in range(n_in,0,-1):
        cols.append(df.shift(i))
        if i == 0:
            names += [(df.columns[j] + '(t)') for j in range(n_vars)]
        else:
            names += [(df.columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        names += [(df.columns[j]+'(t+%d)' % (i)) for j in range(n_vars)]
    # aux columns
    if auxcats is not False:
        for cat in auxcats:
            cols.append(df_orig[cat])
            names += [cat]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg, n_vars

def supervised_df_to_array(data, n_in=1, n_out=1, n_vars=6,y_var=[0], x_aux_var=[0],shuffle=False,normalize=False):


    """
    Separates An Autoregressed data frame into features and labels
    Arguments:
        data: (pandas df) dataframe input
        n_in: (int) number of lags
        n_out: (int) number of leads
        n_vars: (int) number of variables
        y_var: (int) column of label
        x_aux_var: (List) column of auxilliary predictions (predictions of data in future timesteps)
        normalize: (BOOLEAN) whether or not to normalize the data
    Returns:
        x: (numpy array) All past values
        x_aux: (numpy array)Predicted Future Values
        y: (numpy array) Future Labels
    """

    if shuffle:
        data = data.sample(frac=1)
    if normalize:
        import tensorflow as tf
        data_norm = pd.DataFrame(
            tf.keras.utils.normalize(np.array(data),
                                     axis=1, order=2))
    else:
        data_norm = data

    x = np.array(data_norm.iloc[:, 0:n_in*n_vars])
    x = x.reshape((data_norm.shape[0], n_in, n_vars))
    x = np.flip(x,axis=1)


    x_aux = np.array(data_norm.iloc[:,n_in*n_vars+x_aux_var[0]])
    x_aux = x_aux.reshape((x.shape[0],-1,1))
    for i in range(1,n_out):
        x_aux2 = np.array(data_norm.iloc[:,n_in*n_vars+x_aux_var[0] + n_vars*i]).reshape((x_aux.shape[0],-1,1))
        x_aux = np.concatenate((x_aux, x_aux2),axis=2)
    for j in range(1,len(x_aux_var)):
        for i in range(0,n_out):
            x_aux2 = np.array(data_norm.iloc[:,n_in*n_vars + n_vars*i+x_aux_var[j]]).reshape((x_aux.shape[0],-1,1))
            x_aux = np.concatenate((x_aux, x_aux2),axis=2)


    y = np.array(data.iloc[:,n_in*n_vars+y_var[0]])
    y = y.reshape((x.shape[0],-1,1))
    for i in range(1,n_out):
        y2 = np.array(data.iloc[:,n_in*n_vars+y_var[0] + n_vars*i]).reshape((y.shape[0],-1,1))
        y = np.concatenate((y, y2),axis=2)
    for j in range(1,len(y_var)):
        for i in range(0,n_out):
            y2 = np.array(data.iloc[:,n_in*n_vars + n_vars*i+y_var[j]]).reshape((y.shape[0],-1,1))
            y = np.concatenate((y, y2),axis=2)

    x_aux = x_aux.reshape((x_aux.shape[0],n_out,len(x_aux_var)),order='F')
    y = y.reshape((y.shape[0],n_out,len(y_var)),order='F')

    return x, x_aux, y



def plot_history(history):

    """
    Plots training progress, currently only valid for using mse as a metric
    :param history:
    :return plot of validation and training mean squared error:
    """

    plt.figure(figsize=(7,5))
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (kWh)')
    nonval_mse = history.history['main_output_mean_squared_error']
    last_nonval_mse = nonval_mse[len(nonval_mse)-1]
    val_mse = history.history['val_main_output_mean_squared_error']
    last_val_mse = val_mse[len(val_mse)-1]
    plt.plot(history.epoch, history.history['main_output_mean_squared_error'],
             label='Train MSE')
    plt.plot(history.epoch, history.history['val_main_output_mean_squared_error'],
             label = 'Val MSE')
    plt.legend()
    plt.ylim([0, .5])
    last_val_mse = val_mse[len(val_mse)-1]
    plt.title('Bias: '+str(np.round(last_nonval_mse,3))+' / Var: '+str(np.round(last_val_mse,3)))
