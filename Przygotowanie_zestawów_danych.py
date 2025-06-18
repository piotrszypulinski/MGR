import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import yfinance as yf
import time

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Iterable, Tuple

with open('slownik_ticker_spolka.json') as file:
  TICKER_SPOLKA = json.load(file)

def download_data(stock_list, start_date, end_date):
  stock_data = yf.download(stock_list, start=start_date, end=end_date, auto_adjust=False)[['Adj Close', 'Volume']]

  data_close = stock_data['Adj Close'].rename_axis('', axis='columns')
  data_volume = stock_data['Volume'].rename_axis('', axis='columns')

  return data_close, data_volume

def check_row_length_with_benchmark(data, holidays_dates=False):
  fixed_holidays_dates = ['01.01', '01.06', '05.01', '05.03', '08.15', '11.01', '11.11', '12.24', '12.25', '12.26', '12.31']
  fixed_holidays_dates_in_data = []

  for idx in data.index:
    full_date = datetime.strftime(idx, '%Y.%m.%d')
    straight_date = datetime.strftime(idx, '%m.%d')
    if straight_date in fixed_holidays_dates:
      fixed_holidays_dates_in_data.append(full_date)

  print(f'W danych znajduje się {len(fixed_holidays_dates_in_data)} obserwacji z datą będącą stałym Świętem w Polsce.')

  if holidays_dates:
    changeable_holidays_dates = []
    for key, val in holidays_dates.items():
      for v in val:
        dates = key + '.' + v
        changeable_holidays_dates.append(dates)

    all_holidays = fixed_holidays_dates_in_data + changeable_holidays_dates

  else:
    all_holidays = fixed_holidays_dates_in_data

  holidays_index = [x for x in all_holidays if x in data.index]

  return holidays_index

def make_unificate_dict(formal_dict, data_dict, reverse=False):
  new_dict = {}

  if reverse:
    formal_dict = {value: key for key, value in formal_dict.items()}

  for name, date in data_dict.items():
    if name in list(formal_dict.keys()):
      ticker = formal_dict[name]
      new_dict[ticker] = date

  return new_dict

def find_suspect_dates(data):
  error_dates = {}
  error_dates_next = {}
  for idx, row in data.iterrows():
    for col in data.columns:
      if row[col] == 0:
        error_dates[col] = str(idx)
        error_dates_next[col] = str(idx + timedelta(days=1))

  return error_dates, error_dates_next

def archive_structure(url):
  time.sleep(2)
  try:
    response = requests.get(url, headers={"Accept-Encoding": "identity"}, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    td_elements = soup.find_all('td', class_='text-right')

    if len(td_elements) > 0:
      change_price = td_elements[5].text.strip()
      volume = td_elements[6].text.strip()
      return change_price, volume

    else:
      return ()

  except ValueError as e:
    print(e)

def get_data_from_archive(dates_dict):#, kind='volume'):
  archive_data = {}
  data_to_get = {}
  for stock in dates_dict.keys():
    print(stock)
    archive_data[stock] = str(dates_dict[stock])
    format_gpw = datetime.strftime(pd.to_datetime(archive_data[stock]), '%d-%m-%Y')
    format_uni = datetime.strftime(pd.to_datetime(archive_data[stock]), '%Y-%m-%d')
    archive_data[stock] = (archive_structure(f'https://www.gpw.pl/archiwum-notowan-full?type=10&instrument={stock}&date={format_gpw}'))

  return archive_data


def replace_data_in_dataset(data, error_dates, archive_data, kind='volume'):
  archive_data = make_unificate_dict(TICKER_SPOLKA, archive_data, reverse=True)

  for ticker, date in error_dates.items():
    date = pd.to_datetime(date.split(' ')[0], format='%Y-%m-%d')
    if date < min(data.index):
      continue
    if len(archive_data[ticker]) > 0:
      if kind == 'volume':
        volume = archive_data[ticker][1]
        volume = int(volume.replace(' ', ''))

        if volume != 0:
          data.loc[date, ticker] = volume
        else:
          continue

      elif kind == 'price':
        price = archive_data[ticker][0]
        price = float(price.replace(',', '.'))

        if data.loc[date, ticker] == price:
          continue
        else:
          data.loc[date, ticker] = price

    else:
      continue

  return data


## TRAIN 
start_date = '2011-12-30'
end_date = '2023-01-01'

train_close, train_volume = download_data(stock_to_nn_training, start_date, end_date)

holiday_dates_dict = {
    '2012': ['04.06', '04.09', '06.07'],
    '2013': ['03.29', '04.01', '04.16', '05.30'], #16.04 zamknięta giełda
    '2014': ['04.18', '04.21', '06.19'],
    '2015': ['04.03', '04.06', '06.04'],
    '2016': ['03.25', '03.28', '05.26'],
    '2017': ['04.14', '04.17', '06.15'],
    '2018': ['01.02', '03.30', '04.02', '05.31', '11.12'],
    '2019': ['04.19', '04.22', '06.20'],
    '2020': ['04.10', '04.13', '06.11']
}

holidays_in_data = check_row_length_with_benchmark(train_volume, holiday_dates_dict)

train_close = train_close.drop(index=holidays_in_data)
train_volume = train_volume.drop(index=holidays_in_data)

date_from_archive_t = potential_error_dates.copy()
date_from_archive_ta = make_unificate_dict(TICKER_SPOLKA, date_from_archive_t)
archive_data_t = get_data_from_archive(date_from_archive_ta)

train_volume_copy = train_volume.copy()
train_volume_adjust = replace_data_in_dataset(train_volume_copy, date_from_archive_t, archive_data_t, kind='volume')

date_from_archive_t_1 = potential_error_dates_next.copy()
date_from_archive_t_1a = make_unificate_dict(TICKER_SPOLKA, date_from_archive_t_1)
archive_data_t_1 = get_data_from_archive(date_from_archive_t_1a)

train_volume_adjust_1 = replace_data_in_dataset(train_volume_adjust, date_from_archive_t_1, archive_data_t_1, kind='volume')

train_close_pct = train_close.apply(lambda x: x.pct_change(1)*100)
train_close_pct = train_close_pct.iloc[1:]

### T0
train_close_pct_copy = train_close_pct.copy()
train_close_adjust = replace_data_in_dataset(train_close_pct_copy, date_from_archive_t, archive_data_t, kind='price')

### T1
train_close_adjust_1 = replace_data_in_dataset(train_close_adjust, date_from_archive_t_1, archive_data_t_1, kind='price')

## TEST
start_date_test = '2022-12-30'
end_date_test = '2025-05-01'

test_close, test_volume = download_data(stock_to_nn_training, start_date_test, end_date_test)

holidays_in_data = check_row_length_with_benchmark(train_volume)
print(holidays_in_data)

test_close = test_close.drop(index=holidays_in_data)
test_volume = test_volume.drop(index=holidays_in_data)

potential_error_dates_test, potential_error_dates_next_test = find_suspect_dates(test_volume)

date_from_archive_t_test = potential_error_dates_test.copy()
date_from_archive_ta_test = make_unificate_dict(TICKER_SPOLKA, date_from_archive_t_test)
archive_data_t_test = get_data_from_archive(date_from_archive_ta_test)

test_volume_copy = test_volume.copy()
test_volume_adjust = replace_data_in_dataset(test_volume_copy, date_from_archive_t_test, archive_data_t_test, kind='volume')

date_from_archive_t_1_test = potential_error_dates_next_test.copy()
date_from_archive_t_1a_test = make_unificate_dict(TICKER_SPOLKA, date_from_archive_t_1_test)
archive_data_t_1_test = get_data_from_archive(date_from_archive_t_1a_test)

test_volume_adjust_1 = replace_data_in_dataset(test_volume_adjust, date_from_archive_t_1_test, archive_data_t_1_test, kind='volume')

test_close_pct = test_close.apply(lambda x: x.pct_change(1)*100)
test_close_pct = test_close_pct.iloc[1:]

### T0
test_close_pct_copy = test_close_pct.copy()
test_close_adjust = replace_data_in_dataset(test_close_pct_copy, date_from_archive_t_test, archive_data_t_test, kind='price')

### T1
test_close_adjust_1 = replace_data_in_dataset(test_close_adjust, date_from_archive_t_1_test, archive_data_t_1_test, kind='price')
