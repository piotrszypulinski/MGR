import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

rebalancing_dates_path = ''

with open(rebalancing_dates_path, 'r') as file:
    rebalance_dates_list = file.read().splitlines()

index_date_end = ['2025-05-01'] # Koniec okresu trwania badania 
dates_to_index = rebalance_dates_list + index_date_end

class IndexCalculation:
    def __init__(self, dates):
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
            start_date = '2023-01-01'
            end_date = str((datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
          else:
            start_date = str(self.dates[i-1])
            end_date = str((datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))

          stock_list = list(self.rating_data[i]['Spółka'].unique())
          stock_list = [stock for stock in stock_list if stock != 'CBF.WA']

          print(stock_list)
          data_yf = yf.download(stock_list, start_date, end_date, auto_adjust=False)['Adj Close']
          print(f'Pobrane notowania za okres: od {min(data_yf.index)} do {max(data_yf.index)}')
          print('----------------------------------------------------------------------------------------')

          # Yf nie ogarnął zmiany nazwy z R22 na Cyberfolks - dane ze stooq
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
        start_date = '2023-01-01'
        end_date = '2025-05-01'
        data_yf = yf.download(stock_list, start_date, end_date, auto_adjust=False)['Adj Close']

        start_date_cbf = '20230101'
        end_date_cbf = '20250501'
        url_cbf = f'https://stooq.pl/q/d/l/?s=cbf&f={start_date_cbf}&t={end_date_cbf}&i=d'
        data_cbf = pd.read_csv(url_cbf, parse_dates=['Data']).set_index('Data')
        data_cbf_close = data_cbf[['Zamkniecie']].rename(columns={'Zamkniecie': 'CBF.WA'})

        data_my_index = pd.merge(data_yf, data_cbf_close, right_index=True, left_index=True)
        assert data_yf.shape[0] == data_my_index.shape[0], print('Uups, coś nie tak!')

        self.data_index = data_my_index

    def weights_setting(self):
        weights = {}
        stock_count = len(self.data_index.index)
        for date, rating in self.rating_data.items():
          rating_stock_count = int(rating['Spółka'].nunique())
          rating_count_ratio = (rating_stock_count/stock_count) * 1.5
          print(f'Udział spółek z rekomendacjami we wszystkich: {rating_count_ratio}')

          rating['Udział_rekomendacji'] = rating['Poprawa_rekomendacji']/rating['Poprawa_rekomendacji'].sum()
          rating_weights = {var['Spółka']: var['Udział_rekomendacji'] * rating_count_ratio for idx, var in rating.iterrows()}

          others = 1 - rating_count_ratio
          print(others)

          weights[date] = (rating_weights, others)

        return weights

    def get_other_indexes(self, list_indexes):
        compare_indexes = {}
        for index in list_indexes:
          index = index.lower()

          # Sztywne daty
          start_date = '20230101'
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
                data_change = data.pct_change() * 100
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
            data_change = part_data.pct_change() * 100
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

    def make_chart(self, my_index, colors_dict, output_path):
        my_index['Indeks_skumulowany'] = my_index['Indeks'].cumsum()

        assert len(self.compare_indexes_data.keys()) > 0

        try: 
            indexes = {}
            for name, index in self.compare_indexes_data.items():
                index['Zmiana'] = index['Close'].pct_change() * 100
                index = index.iloc[1:]
                index['Zmiana_skumulowana'] = index['Zmiana'].cumsum()
                indexes[name] = index

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(my_index.index, my_index['Indeks_skumulowany'], color='darkkhaki', ls='--', label='Mój index')
            for name, df in indexes.items():
                ax.plot(df.index, df['Zmiana_skumulowana'], color=colors_dict[name], ls='-.', label=name.upper())

            print(f'Na wykresie widać {len(ax.lines)} linie')
            ax.set_xlabel('Data')
            labels = np.unique([datetime.strftime(x, '%Y-%m') for x in df.index])
            ticks = pd.to_datetime(labels, format='%Y-%m')
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_ylabel('Wartość indeksów')
            ax.set_title(f'Porównanie zwrotów indeksów "Mój Indeks" względem {[name.upper() for name in indexes.keys()]}')
            ax.legend(loc='best', frameon=False)
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(f'{output_path}')
            return plt.show()

        except ValueError as e:
            print('Nie ma indeksów do porównania, więc nie ma wykresu...')


## USTAWIENIA OGÓLNE
### KOLORY
color_dict = {
    'wig': (0/255,0/255,0/255),
    'wig_poland': (162/255,45/255,26/255),
    'wig20': (166/255,128/255,185/255),
    'mwig40': (245/255,133/255,36/255),
    'swig80': (79/255,147/255,93/255)
}

### BAZA DANYCH - REKOMENDACJE
ratings_base = ''

## Wywoływanie indeksów: 
#  1) wyłącznie rekomendacje z danych zakresów czasowych

indexes_object = IndexCalculation(dates_to_index)
indexes_object.analyze_rating_data(ratings_base)
indexes_object.get_close_data_simple()

list_of_indexes = ['WIG', 'WIG_Poland']

indexes_object.get_other_indexes(list_of_indexes)
p_index = indexes_object.index_calculator()

output_path_f = ''
print(indexes_object.make_chart(p_index, color_dict, output_path_f))

# 2) przy wyliczaniu indeksu zawsze uwzględnione spółki, które dostały jakąkolwiek rekomendacje giełdową -> ważone bez przywilejów
stock_list_wo_cbf = []


indexes_object = IndexCalculation(dates_to_index)
indexes_object.analyze_rating_data(ratings_base)
indexes_object.get_close_data_wide(stock_list_wo_cbf)
weights = indexes_object.weights_setting()

list_of_indexes = ['WIG', 'WIG_Poland']

indexes_object.get_other_indexes(list_of_indexes)
p_index = indexes_object.index_calculator(weights)

output_path_s = ''
print(indexes_object.make_chart(p_index, color_dict, output_path_s)) 