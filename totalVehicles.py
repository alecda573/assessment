import pandas as pd
pd.set_option('display.float_format', lambda x: '%.1f' % x)   # display only 1 decimal
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import statsmodels.api as api
import statsmodels.tsa.stattools as stattools
import statsmodels.tsa.seasonal as seasonal
import statsmodels.graphics.tsaplots as tsaplots
import scipy.stats as stats
from sklearn import metrics
import os
import argparse
from sktime.performance_metrics.forecasting import MeanSquaredPercentageError


def main(args):
    labels = args.labels.split(',')

    df = pd.read_csv(args.data, parse_dates=True, index_col='DATE', sep=',')
    df.index.freq = 'MS'
    df.columns = ['y']

    if args.genPlots:
        df.plot(figsize=(20, 4))
        plt.title("Monthly Vehicle Sales in the USA", size=30)
        plt.ylabel('Total Vehicle Sales')
        plt.xlabel('Month')
        plt.savefig(args.save + "MVS.png")
        if args.showPlots:
            plt.show()

    df['month_index'] = df.index.month
    if args.genPlots:
        plt.figure(figsize=(10, 5))
        ax = seaborn.boxplot(x="month_index", y="y", data=df)
        plt.title("Box plot of Monthly Vehicle Sales")
        ax.set_ylabel('Total Vehicle Sales')
        ax.set_xlabel('Month')
        plt.savefig(args.save + "MVS_box.png")
        if args.showPlots:
            plt.show()

    df.drop('month_index', inplace=True, axis=1)
    decomp = seasonal.seasonal_decompose(df, period=12)

    if args.genPlots:
        figure = plt.figure()
        figure = decomp.plot()
        figure.set_size_inches(13, 12);
        figure.savefig(args.save + "decomp.png")
        if args.showPlots:
            figure.show()

    diff_month = (df - df.shift(12)).dropna()
    if args.genPlots:
        diff_month.plot()
        plt.ylabel('Total Vehicle Sales Monthly Difference')
        plt.xlabel('Month')
        plt.savefig(args.save + "vehicles_monthly_difference.png")
        if args.showPlots:
            plt.show()

    diff_month_first = (diff_month - diff_month.shift(1)).dropna()
    if args.genPlots:
        diff_month_first.plot()
        plt.ylabel('Total Vehicle Sales Monthly + First Difference')
        plt.xlabel('Month')
        plt.savefig(args.save + "vehiles_monthly_first_diff.png")
        if args.showPlots:
            plt.show()

    # Perform Dickey-Fuller Test to assess stationarity on seasonal + first difference
    df_test_result = stattools.adfuller(diff_month_first)
    print("Dickey Fuller Test results on month + first difference data: ")
    for val, lab in zip(df_test_result, labels):
        print(lab + " {}".format(val))

    if args.genPlots:
        # generate without first difference
        tsaplots.plot_acf(diff_month)
        plt.title("Autocorrelation")
        plt.savefig(args.save + "autocorrelation.png")
        if args.showPlots:
            plt.show()
        tsaplots.plot_pacf(diff_month)
        plt.title("Partial Autocorrelation")
        plt.savefig(args.save + "partial_autocorrelation.png")
        if args.showPlots:
            plt.show()

        tsaplots.plot_acf(diff_month_first)
        plt.title("Autocorrelation")
        plt.savefig(args.save + "autocorrelation.png")
        if args.showPlots:
            plt.show()
        tsaplots.plot_pacf(diff_month_first)
        plt.title("Partial Autocorrelation")
        plt.savefig(args.save + "partial_autocorrelation.png")
        if args.showPlots:
            plt.show()

    model = api.tsa.statespace.SARIMAX(df, order=(4, 1, 1), seasonal_order=(1, 1, 1, 12))
    ARIMAresult = model.fit()
    print(ARIMAresult.summary())
    df['forecast'] = ARIMAresult.predict(start=515, end=527, dynamic=True, full_results=True)
    if args.genPlots:
        df.plot();

    if args.genDiagnostics:
        ARIMAresult.plot_diagnostics(figsize=(20, 10))
        plt.savefig(args.save + "diagnostics.png")
        if args.showPlots:
            plt.show()

        plt.savefig(args.save + "forecast.png")
        plt.xlim('2015-01-01', '2020-01-01');
        if args.showPlots:
            plt.show()

    print('MSE - mean squared error is {}'.format(round(metrics.mean_squared_error(df['y'][-12:], df['forecast'][-12:]), 1)))
    print('MAE - mean absolute error is {}'.format(round(metrics.mean_absolute_error(df['y'][-12:], df['forecast'][-12:]), 1)))
    print('MAPE with sklean is {}'.format(round(metrics.mean_absolute_percentage_error(df['y'][-12:], df['forecast'][-12:]) * 100, 1)))
    print("MSPE with sklearn is {}".format(round(MeanSquaredPercentageError()(df['y'][-12:], df['forecast'][-12:]) * 100, 1)))

    forecast = ARIMAresult.get_forecast(13).summary_frame(alpha=0.1)

    sales2019 = (df['y'][-11:].sum() + forecast['mean'][:1]).values[0]
    forecast2020 = forecast['mean'][1::].sum()

    print('Total sales in 2019 {}'.format(round(sales2019, 1)))
    print('Total sales in 2020 {}'.format(round(forecast2020, 1)))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/TOTALNSA.csv")
    parser.add_argument("--save", default=os.getcwd() + "/plots/")
    parser.add_argument("--genPlots", default=True)
    parser.add_argument("--showPlots", default=False)
    parser.add_argument("--labels", default='ADF Statistic,p-value,No. of Lags Used,Number of Observations Used')
    parser.add_argument("--genDiagnostics", default=True)

    args = parser.parse_args()

    main(args)