import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

class IndexCalculation:
  def __init__(self, dates):
    print(dates)
    self.dates = dates
    self.rating_data = {}
    self.data_index = pd.DataFrame()
    self.compare_indexes_data = {}

  def analyze_rating_data(self, rating_data_path):
    rating_data = pd.read_csv(rating_data_path)

    rating_results = {}
    for i, date in enumerate(self.dates):
      if i == 0:
        data_to_timestamp = rating_data[rating_data['Data_dokumentu'] <= date]
      else:
        data_to_timestamp = rating_data[(rating_data['Data_dokumentu'] <= date) & (rating_data['Data_dokumentu'] > self.dates[i-1])]

      selected_values = data_to_timestamp.groupby(['Spółka', 'Dom_maklerski', 'Rekomendacja_obecna'], as_index=False)['Poprawa_rekomendacji'].sum()
      selected_values = selected_values[selected_values['Poprawa_rekomendacji'] > 0]
      best_ratings = selected_values.groupby('Spółka')['Poprawa_rekomendacji'].idxmax()
      rating_result = selected_values.loc[best_ratings].reset_index(drop=True)

      print(f'Do daty {date}, znaleziono {rating_result.shape[0]} spółek z rekomendacjami giełdowymi.')

      rating_results[date] = rating_result
      del data_to_timestamp, selected_values, best_ratings, rating_result

    self.rating_data = rating_results

  def get_close_data_simple(self):
    my_index_data = []
    for i, date in enumerate(self.dates):
      if i == 0:
        start_date = '2022-12-30'
        end_date = str((datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
      else:
        start_date = str(self.dates[i-1])
        end_date = str((datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))

      stock_list = list(self.rating_data[date]['Spółka'].unique())
      stock_list = [stock for stock in stock_list if stock != 'CBF.WA']

      print(stock_list)
      data_yf = yf.download(stock_list, start_date, end_date, auto_adjust=False)['Adj Close']
      print(f'Pobrane notowania za okres: od {min(data_yf.index)} do {max(data_yf.index)}')
      print('----------------------------------------------------------------------------------------')

      # YF nie ogarnął zmiany nazwy z R22 na Cyberfolks - dane ze stooq
      start_date_cbf = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y%m%d')
      end_date_cbf = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y%m%d')
      url_cbf = f'https://stooq.pl/q/d/l/?s=cbf&f={start_date_cbf}&t={end_date_cbf}&i=d'
      data_cbf = pd.read_csv(url_cbf, parse_dates=['Data']).set_index('Data')
      data_cbf_close = data_cbf[['Zamkniecie']]

      data_my_index = pd.merge(data_yf, data_cbf_close, right_index=True, left_index=True)
      assert data_yf.shape[0] == data_my_index.shape[0], print('Uups, coś nie tak!')

      my_index_data.append(data_my_index)
    self.data_index = my_index_data

  def get_close_data_wide(self, stock_list):
    start_date = '2022-12-30'
    end_date = '2025-05-01'
    data_yf = yf.download(stock_list, start_date, end_date, auto_adjust=False)['Adj Close']

    start_date_cbf = '20221230'
    end_date_cbf = '20250501'
    url_cbf = f'https://stooq.pl/q/d/l/?s=cbf&f={start_date_cbf}&t={end_date_cbf}&i=d'
    data_cbf = pd.read_csv(url_cbf, parse_dates=['Data']).set_index('Data')
    data_cbf_close = data_cbf[['Zamkniecie']].rename(columns={'Zamkniecie': 'CBF.WA'})

    data_my_index = pd.merge(data_yf, data_cbf_close, right_index=True, left_index=True)
    print(data_yf.shape[0])
    print(data_my_index.shape[0])
    assert data_yf.shape[0] == data_my_index.shape[0], print('Uups, coś nie tak!')

    self.data_index = data_my_index

  def weights_setting(self, strategy='constant', ratio=0.0):
    weights = {}
    stock_count = len(self.data_index.index)
    for date, rating in self.rating_data.items():
      if strategy == 'constant':
        rating_stock_count = int(rating['Spółka'].nunique())
        rating_count_ratio = (rating_stock_count/stock_count) * 1.5
        print(f'Udział spółek z rekomendacjami w oknie czasowym do wszystkich spółek z kwalifikującymi się rekomendacjami: {rating_count_ratio}')

        rating['Udział_rekomendacji'] = rating['Poprawa_rekomendacji']/rating['Poprawa_rekomendacji'].sum()
        rating_weights = {var['Spółka']: var['Udział_rekomendacji'] * rating_count_ratio for idx, var in rating.iterrows()}

        others = 1 - rating_count_ratio
        print(others)

      elif strategy == 'premium':
         sorted_ratings = rating.sort_values(by='Poprawa_rekomendacji', ascending=False)
         gainers = sorted_ratings[sorted_ratings['Poprawa_rekomendacji'] > 0]['Spółka']
         weight_per_gainers = ratio / len(gainers)

         rating_weights = {gainer: weight_per_gainers for gainer in gainers}
         others = 1 - ratio # podawane apriori

      weights[date] = (rating_weights, others)

    return weights

  def get_other_indexes(self, list_indexes):
    compare_indexes = {}
    for index in list_indexes:
      index = index.lower()

      # Sztywne daty
      start_date = '20221230'
      end_date = '20250501'
      url = f'https://stooq.com/q/d/l/?s={index}&f={start_date}&t={end_date}&i=d'

      data = pd.read_csv(url, parse_dates=['Date']).set_index('Date')
      compare_indexes[index] = data[['Close']]

    self.compare_indexes_data = compare_indexes

  def index_calculator(self, weights=None):
    print('Uruchomiono funkcję wyliczającą indeks!')
    if weights is None:
      my_index = None
      print('Wyliczanie indeksu bez podanych wag...')
      for i, data in enumerate(self.data_index):
        stock_number = data.shape[1]
        data_change = data.pct_change()
        data_change = data_change.iloc[1:]
        data_change['Indeks'] = (data_change[data_change.columns].sum(axis=1))/stock_number
        data_change = data_change.drop(columns=[col for col in data_change.columns if col != 'Indeks'])

        if my_index is None:
          my_index = data_change.copy()
        else:
          my_index = pd.concat([my_index, data_change], axis=0)

    else:
      my_index = None
      print('Wyliczanie indeksu z podanymi wagami...')
      for i, (date, (rating_weight, others)) in enumerate(weights.items()):
        current_date = date
        previous_date = list(weights.keys())[i-1]
        if i == 0:
          part_data = self.data_index[self.data_index.index <= current_date]

        else:
          part_data = self.data_index[(self.data_index.index <= current_date) & (self.data_index.index > previous_date)]
        data_change = part_data.pct_change()
        data_change = data_change.iloc[1:]

        data_weightening = data_change.copy()
        rating_stocks = []
        for rating_stock, weight in rating_weight.items():
          data_weightening[rating_stock] = data_weightening[rating_stock] * weight
          rating_stocks.append(rating_stock)

        data_weightening['Indeks_z_rekomendacji'] = data_weightening[rating_stocks].sum(axis=1)

        others_stock = [stock for stock in data_change.columns if stock not in list(rating_weight.keys())]
        data_weightening['Indeks_bez_rekomendacji'] = (data_weightening[others_stock].sum(axis=1) * others)/len(others_stock)

        data_weightening['Indeks'] = data_weightening['Indeks_z_rekomendacji'] + data_weightening['Indeks_bez_rekomendacji']
        data_weightening = data_weightening.drop(columns=[col for col in data_weightening.columns if col != 'Indeks'])

        if my_index is None:
          my_index = data_weightening.copy()
        else:
          my_index = pd.concat([my_index, data_weightening], axis=0)

    return my_index

  def make_chart(self, my_index, colors_dict, save_name):
    my_index['Indeks_skumulowany'] = ((1 + my_index['Indeks']).cumprod() - 1) * 100
    print(my_index.loc[my_index.index[-1], 'Indeks_skumulowany'])

    assert len(self.compare_indexes_data.keys()) > 0

    indexes = {}
    for name, index in self.compare_indexes_data.items():
      index['Zmiana'] = index['Close'].pct_change()
      index = index.iloc[1:]
      index['Zmiana_skumulowana'] = ((1 + index['Zmiana']).cumprod() - 1) * 100
      indexes[name] = index

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(my_index.index, my_index['Indeks_skumulowany'], color='darkkhaki', ls='--', label='Mój index')
    for name, df in indexes.items():
      ax.plot(df.index, df['Zmiana_skumulowana'], color=color_dict[name], ls='-.', label=name.upper())

    print(f'Na wykresie widać {len(ax.lines)} linie')
    ax.set_xlabel('Data')
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_ylabel('Zmiana w %')
    ax.set_title(f'Porównanie "Mój Indeks" vs {[name.upper() for name in indexes.keys()]}')
    ax.legend(loc='best', frameon=False)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_name}.png')
    return plt.show()

## USTAWIENIA OGÓLNE
### KOLORY
color_dict = {
    'wig': (0/255,0/255,0/255),
    'wig_poland': (162/255,45/255,26/255),
    'wig20': (166/255,128/255,185/255),
    'mwig40': (245/255,133/255,36/255),
    'swig80': (79/255,147/255,93/255)
}

rating_data_path = 'rating_data_to_index_calculator.csv'

## REBALANCING GPW
real_rebalancing_dates = ['2023-03-17', '2023-06-16', '2023-09-15', '2023-12-15', '2024-03-15', '2024-06-21', '2024-09-20', '2024-12-20', '2025-03-21']
last_quotation_day = ['2025-05-01']
dates_to_index_real = real_rebalancing_dates + last_quotation_day

### Wywoływanie indeksów:
#### 1) Pierwszy indeks:
f_indexes_object_real = IndexCalculation(dates_to_index_real)
f_indexes_object_real.analyze_rating_data(rating_data_path)
f_indexes_object_real.get_close_data_simple()

list_of_indexes = ['WIG', 'WIG_Poland', 'WIG20tr', 'MWIG40tr', 'SWIG80tr']

f_indexes_object_real.get_other_indexes(list_of_indexes)
f_p_index_real = f_indexes_object_real.index_calculator()

f_p_real_savename = f'pierwszy_indeks_real_{date_now}'
print(f_indexes_object_real.make_chart(f_p_index_real, color_dict, f_p_real_savename))

#### 2) Drugi indeks:
s_indexes_object_real = IndexCalculation(dates_to_index_real)
s_indexes_object_real.analyze_rating_data(path)
s_indexes_object_real.get_close_data_wide(stock_list_wo_cbf)
weights = s_indexes_object_real.weights_setting()

list_of_indexes = ['WIG', 'WIG_Poland']

s_indexes_object_real.get_other_indexes(list_of_indexes)
s_p_index_real = s_indexes_object_real.index_calculator(weights)

s_p_real_savename = f'drugi_indeks_real_{date_now}'
print(s_indexes_object_real.make_chart(s_p_index_real, color_dict, s_p_real_savename))

#### 3) Trzeci indeks:
th_indexes_object_real = IndexCalculation(dates_to_index_real)
th_indexes_object_real.analyze_rating_data(path)
th_indexes_object_real.get_close_data_wide(stock_list_wo_cbf)
weights = th_indexes_object_real.weights_setting(strategy='premium', ratio=0.8)

list_of_indexes = ['WIG', 'WIG_Poland']

th_indexes_object_real.get_other_indexes(list_of_indexes)
th_p_index_real = th_indexes_object_real.index_calculator(weights)

th_p_real_savename = f'trzeci_indeks_real_{date_now}'
print(th_indexes_object_real.make_chart(th_p_index_real, color_dict, th_p_real_savename))

## REBALANCING AUTOENKODER
rebalance_dates_list_nn = ['2023-06-01', '2023-06-22', '2023-08-22', '2023-09-01', '2024-03-01', '2024-07-23', '2025-02-28', '2025-03-03', '2025-04-24']
last_quotation_day = ['2025-05-01']
dates_to_index_nn = rebalance_dates_list_nn + last_quotation_day

### Wywoływanie indeksów:
#### 1) Pierwszy indeks:
f_indexes_object_nn = IndexCalculation(dates_to_index_nn)
f_indexes_object_nn.analyze_rating_data(rating_data_path)
f_indexes_object_nn.get_close_data_simple()

list_of_indexes = ['WIG', 'WIG_Poland', 'WIG20TR', 'MWIG40TR', 'SWIG80TR']

f_indexes_object_nn.get_other_indexes(list_of_indexes)
f_p_index_nn = f_indexes_object_nn.index_calculator()

f_p_nn_savename = f'pierwszy_indeks_test_{date_now}'
print(f_indexes_object_nn.make_chart(f_p_index_nn, color_dict, f_p_nn_savename))

#### 2) Drugi indeks:
stock_list_wo_cbf = [stock for stock in list(data['Spółka'].unique()) if stock != 'CBF.WA']
print(len(stock_list_wo_cbf))

s_indexes_object_nn = IndexCalculation(dates_to_index_nn)
s_indexes_object_nn.analyze_rating_data(path)
s_indexes_object_nn.get_close_data_wide(stock_list_wo_cbf)
weights = s_indexes_object_nn.weights_setting()

list_of_indexes = ['WIG', 'WIG_Poland']

s_indexes_object_nn.get_other_indexes(list_of_indexes)
s_p_index_nn = s_indexes_object_nn.index_calculator(weights)

s_p_nn_savename = f'drugi_indeks_test_{date_now}'
print(s_indexes_object_nn.make_chart(s_p_index_nn, color_dict, s_p_nn_savename))

#### 3) Trzeci indeks:
th_indexes_object_nn = IndexCalculation(dates_to_index_nn)
th_indexes_object_nn.analyze_rating_data(path)
th_indexes_object_nn.get_close_data_wide(stock_list_wo_cbf)
weights = th_indexes_object_nn.weights_setting(strategy='premium', ratio=0.8)

list_of_indexes = ['WIG', 'WIG_Poland']

th_indexes_object_nn.get_other_indexes(list_of_indexes)
th_p_index_nn = th_indexes_object_nn.index_calculator(weights)

th_p_nn_savename = f'trzeci_indeks_test_{date_now}'
print(th_indexes_object_nn.make_chart(th_p_index_nn, color_dict, th_p_nn_savename))

## REBALANCING K-MEANS
rebalance_dates_list_kmeans = ['2023-05-31', '2023-09-26', '2023-11-14', '2023-12-15', '2024-02-14', '2024-03-15', '2024-09-26', '2024-11-14', '2025-04-04']
dates_to_index_kmeans = rebalance_dates_list_kmeans + last_quotation_day

### Wywoływanie indeksów:
#### 1) Pierwszy indeks:
f_indexes_object_kmeans = IndexCalculation(dates_to_index_kmeans)
f_indexes_object_kmeans.analyze_rating_data(rating_data_path)
f_indexes_object_kmeans.get_close_data_simple()

list_of_indexes = ['WIG', 'WIG_Poland', 'WIG20tr', 'MWIG40tr', 'SWIG80tr']

f_indexes_object_kmeans.get_other_indexes(list_of_indexes)
f_p_index_kmeans = f_indexes_object_kmeans.index_calculator()

f_p_kmeans_savename = f'pierwszy_indeks_kmeans_{date_now}'
print(f_indexes_object_kmeans.make_chart(f_p_index_kmeans, color_dict, f_p_kmeans_savename))

#### 2) Drugi indeks:
s_indexes_object_kmeans = IndexCalculation(dates_to_index_kmeans)
s_indexes_object_kmeans.analyze_rating_data(path)
s_indexes_object_kmeans.get_close_data_wide(stock_list_wo_cbf)
weights = s_indexes_object_kmeans.weights_setting()

list_of_indexes = ['WIG', 'WIG_Poland']

s_indexes_object_kmeans.get_other_indexes(list_of_indexes)
s_p_index_kmeans = s_indexes_object_kmeans.index_calculator(weights)

s_p_kmeans_savename = f'drugi_indeks_kmeans_{date_now}'
print(s_indexes_object_kmeans.make_chart(s_p_index_kmeans, color_dict, s_p_kmeans_savename))

#### 3) Trzeci indeks:
th_indexes_object_kmeans = IndexCalculation(dates_to_index_kmeans)
th_indexes_object_kmeans.analyze_rating_data(path)
th_indexes_object_kmeans.get_close_data_wide(stock_list_wo_cbf)
weights = th_indexes_object_kmeans.weights_setting(strategy='premium', ratio=0.8)

list_of_indexes = ['WIG', 'WIG_Poland']

th_indexes_object_kmeans.get_other_indexes(list_of_indexes)
th_p_index_kmeans = th_indexes_object_kmeans.index_calculator(weights)

th_p_kmeans_savename = f'trzeci_indeks_kmeans_{date_now}'
print(th_indexes_object_kmeans.make_chart(th_p_index_kmeans, color_dict, th_p_kmeans_savename))
