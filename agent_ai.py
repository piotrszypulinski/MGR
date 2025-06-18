#pip install openai

import getpass
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json
import time
import re

from datetime import datetime, date
from collections import defaultdict
from bs4 import BeautifulSoup
from io import StringIO
import pandas as pd
import requests

OPENAI_API_KEY = getpass.getpass('OpenAI API Key:')
client = OpenAI(api_key=OPENAI_API_KEY)

## Scrapowanie nagłówków artykułów
# Funkcja do znajdowania tytułów artykułów z rekomendacjami giełdowymi
# Z racji tego, że artykuły na stronie bankiera są posortowane malejąco, sprawdzamy, czy data najnowszego artykułu jest wyższa lub równa 2022-01-01,
# jeżeli tak to dopiero wtedy oceniamy czy mamy do czynienia z rekomendacją giełdową
def scrape_page(main_url: str) -> List[Dict[str, str]]:
  response = requests.get(main_url)
  soup = BeautifulSoup(response.content, 'html.parser')

  article_date = soup.find_all('time', pubdate=True)
  pubdates = [time_tag.get("datetime") for time_tag in article_date]
  clear_dates = [pd.to_datetime(article_date[i].text.strip()) for i in range(len(pubdates))]
  dates_over_22 = [date for date in clear_dates if date >= datetime(2022, 1, 1)]
  print('\nDaty \"ważnych\" artykułów:', dates_over_22)

  anchors = soup.find_all('a', attrs={'rel': 'bookmark'})
  valid_anchors = anchors[:len(dates_over_22)]

  valid_articles = []
  for i, a in enumerate(valid_anchors):
    title = a.get('title') or a.get_text(strip=True)
    link = a.get("href")
    date = dates_over_22[i]
    valid_articles.append({"title": title, "link": f"https://bankier.pl{link}", "article_date": date})

  return valid_articles

## Wyciąganie tekstów z artykułu 
def get_data_from_article(article_url: str) -> str:
  soup = BeautifulSoup(requests.get(article_url).content, 'html.parser')

  article = soup.find("article")
  paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
  text = " ".join(paragraphs)
  doc = {'text': text}

  tables = soup.find_all('table')
  if len(tables) != 0:
    html_table = str(tables[0])
    table = pd.read_html(html_table, thousands=None, keep_default_na=False)[0]
    json_lines = table.to_json(orient='records', lines=True, force_ascii=False)
    doc['table'] = json_lines

  return doc

def agent_filter_articles(articles: List[Dict[str, str]], model_name: str) -> List[str]:
    system_prompt = (
        '''
        Jako model jesteś ekspertem w przeszukiwaniu informacji w tekście.
        Twoje zadanie polega na oznaczeniu dla każdego tekstu, który otrzymasz czy dotyczy on rekomendacji giełdowej bądź potencjalnej ceny docelowej dla akcji danej spółki. Takie rekomendacje wydawane są przez domy maklerskie oraz banki inwestycyjne.
        Poniżej znajdują się kryteria, które wskazują, że dany tekst dotyczy rekomendacji giełdowej:
        - zawiera słowa klucze tj. 'rekomendacja', 'kupuj/zwiększaj/trzymaj/sprzedaj/redukuj', 'cena docelowa akcji', 'obniża/podnosi'
        - i/lub znajduje się w nim nazwa domu maklerskiego lub banku inwestycyjnego. (Dla przykładu: DM BDM, BM mBank, DM BOŚ)
        - jasno wskazuje, że chodzi o nową lub zmianę rekomendacji/oceny akcji danej spółki.

        Jeżeli tekst nie zawiera słów kluczowych, ani w żadnym stopniu nie wspomina o nowej lub aktualnej ocenie akcji przez dom maklerski czy bank inwestycyjny lub jest niejasny, zwróć is_relevant=false.

        Zwróć swoją odpowiedź w formie JSON, gdzie:
        - 'link' (url jako string)
        - 'is_relevant' (jako boolean)

        Przykłady:
        - "BM mBanku obniża cenę docelową akcji 11 bit studios" -> is_relevant = true
        - "Popławski (BM Pekao): Na gaming należy patrzeć ostrożnie, koszty głównym ryzykiem" -> is_relevant = false (artykuł nie dotyczy rekomendacji, a opinii o sytuacji branży rynkowej)
        - "Noble Securities obniżył cenę docelową akcji 11 bit studios do 423 zł, podtrzymuje" -> is_relevant = true
        - "Zapraszamy do uczestnictwa w webinarze poświęconym analizie technicznej indeksów i spółek z GPW." -> is_relevant = false (artykuł dotyczy analizy technicznej)
        - "Moody's potwierdził długoterminowe i krótkoterminowe ratingi depozytowe PKO BP na poziomie A2/P-1" -> is_relevant = false (artykuł dotyczy oceny ryzyka kredytowego)
        - "DM BOŚ wskazał nowe TOP PICKS" -> is_relevant = false (artykuł dotyczy zmian w portfelu spółek)
        - "Zarząd Erbudu rekomenduje wypłatę 0,84 zł dywidendy na akcję za '22" -> is_relevant = false (artykuł dotyczy wypłaty dywidendy)
        '''
    )

    article_lines = []
    for i, art in enumerate(articles, start=1):
        article_lines.append(
            f"{i})Title: {art['title']} | Link: {art['link']} | Article date: {art['article_date']}"
        )

    user_prompt = (
        '''
        Przekazuję Ci listę tekstów. Dla każdego z nich zwróć JSON, w którym zawrzesz klucze:
        "link": <link do artykułu>,
        "is_relevant": <boolean>,
        "article_date": <datę artykułu>.
        '''
    )

    user_prompt += "\n".join(article_lines)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.0, top_p=0.05)
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)

        relevant_links = {}
        for item in data:
            if item.get("is_relevant") is True:
                relevant_links[item['link']] = item['article_date']

        return relevant_links

    except Exception as e:
        return None
    
def parse_article_with_llm(link: str, title: str, article_date: pd.Timestamp, article_text: dict, model_name: str) -> Dict[str, any]:
    system_prompt = (
        '''
        Jesteś ekspertem w wydobywaniu ustrukturyzowanych danych dotyczących rekomendacji giełdowych dla danej spółki.
        Twoje zadanie polega na wyciągnięciu odpowiednich informacji tj. publish_date, stock_name, securities_house, present_rating, last_rating, at_price, target_price, last_price dla danej spółki, bazując na otrzymanym tekście i tabeli (o ile się pojawi).
        Wymagane kryteria dla zmiennych:
        - publish_date to data kiedy dom maklerski/bank inwestycyjny opublikował oficjalnie rekomendację giełdową, często jest to zapisywane w formie DD miesiąc YYYY;
        - securities_house to nazwa dom maklerskiego/banku inwestycyjnego, który wykonał i opublikował rekomendację;
        - present_rating to obecny status rekomendacji w momencie jej wydawania;
        - last_rating to informacja gdy rekomendacja odnosi się do poprzedniej wersji rekomendacji, taka informacja występuje w artykułach, ale nie we wszystkich, często jest przedstawiana jako porównanie do obecnej sytuacji;
        - at_price to wartość akcji danej spółki w momencie publikacji rekomendacji;
        - target_price to potencjalna przyszła wartość, którą szacują domy maklerskie/banki inwestycyjne dla akcji spółki. Opisywana jako cena docelowa.
        - last_price to cena, którą poprzednio wyznaczył dom maklerski/bank inwestycyjny i w nowej rekomendacji rozszerza/poprawia/zmienia tę rekomendację. Jest to poprzednia cena docelowa.

        Dodatkowe uwagi:
        - Zarówno present_rating and last_rating should be one word.
        - Usuń elementy interpunkcji oraz elementy niesłowne z pól: present_rating, last_rating.
        - Zmienne: at_price, target_price, last_price powinny zawierać wyłącznie liczby, w szczególności nie powinny zawierać fragmentów tj. zł/akcję.

        Zwróć TYLKO ważnego JSON'a bez żadnych dodatkowych tekstów ani wyjaśnień. Wykorzystaj wiedzę z funckji find_ratings_data jako wzór formatu wyjściowego.

        Przykład:
        "DM BOŚ, w raporcie z 11 kwietnia, podtrzymał rekomendację "kupuj" dla 11 bit studios i obniżył cenę docelową akcji spółki do 730 zł z 800 zł wcześniej. Raport wydano przy kursie 517 zł.
        Potwierdzamy nasze pozytywne nastawienie do spółki – szczególnie teraz, gdy kurs akcji 11 bit studios spadł około 30 proc. względem maksymalnej wyceny" - napisano w raporcie.
        Nasza 12-miesięczna wycena docelowa spada o 9 proc. do 730 zł na akcję (poprzednio 800 zł) pod wpływem obniżenia prognoz finansowych dla segmentu wydawniczego, niewielkiego obniżenia założeń kursu USD/PLN oraz nieco niższego WACC" - dodano.
        Analitycy uważają, że spółka znalazła się w przededniu dynamicznego wzrostu, który może przynieść zwielokrotnienie wyników finansowych."

        Wypełnione parametry powinny wyglądać tak:
        - publish_date: 2024-04-11
        - securities_house: DM BOŚ
        - present_rating: kupuj
        - last_rating: podtrzymał
        - at_price: 517
        - target_price: 730
        - last_price: 800
        '''
    )

    table_part = (f"Table in article {article_text['table']}\n you should also use it to find data." if article_text.get('table') else "")

    user_prompt = (
        f'''
        Article link: {link}
        Article date: {article_date}
        Article title: {title}
        Article text: {article_text['text']}
        {table_part}
        Bądź uważny podczas dostarczania wyników. Nazwa spółki jest zawsze przechowywana w linku, który dostajesz w tym prompcie. Jest zakodowana jako element url w miejscu XYZ w /akcje/XYZ/wiadomosci/.
        Zastanów się dwa razy, zanim przyporządkujesz wartość do zmiennej.
        '''
    )

    try:
        tools = [{
        "type": "function",
        "name": "find_ratings_data",
        "description": "Zwróć wartości znalezione w tekście (w niektórych przypadkach również z dołączonej tabeli) z promptu.",
        "parameters": {
            "type": "object",
            "properties": {
                "article_date": {
                    "type": "string",
                    "description": "Dzień, w którym opublikowany został artykuł o rekomendacji. Otrzymasz tę wartość w user_prompt."
                },
                "publish_date": {
                    "type": "string",
                    "description": "Dzień, w którym dom maklerski opublikował rekomendację giełdową dla spółki."
                    },
                "stock_name": {
                    "type": "string",
                    "description": "Nazwa spółki, która jest obecnie analizowana. Informację otrzymujesz w user_prompt w zmiennej url (nazwa spółki zakodowana w linku w miejscu  /akcje/XYZ/wiadomosci/)"
                },
                "securities_house": {
                    "type": "string",
                    "description": "Nazwa domu maklerskiego, który wystawił rekomendację giełdową."
                    },
                "present_rating": {
                    "type": "string",
                    "description": "Obecny status rekomendacji giełdowej dla danej spółki."
                    },
                "last_rating": {
                    "type": "string",
                    "description": "Poprzedni status rekomendacji giełdowej dla danej spółki."
                    },
                "at_price": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Wartość akcji spółki w momencie publikacji rekomendacji giełdowej."
                },
                "target_price": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Cena docelowa dla akcji danej spółki. Przyszła wycena wydawana przez dom maklerski."
                },
                "last_price": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Case when securities house/investment bank adjust/correct/remake their rating."
                }
            },
            "required": ["article_date", "publish_date", "securities_house", "present_rating", "last_rating", "at_price", "target_price", "last_price"]
        }}]

        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], tools=tools)

        data = json.loads(response.output[0].arguments)

        for key in ["title", "article_date", "publish_date", "stock_name", "securities_house", "present_rating", "last_rating", "at_price", "target_price", "last_price"]:
            if key not in data:
                data[key] = None
        return data

    except Exception as e:
        print("Błąd w funkcji znajdywania wartości w artykule.", e)
        pass

def scrape_and_extract_recommendations(page_url: str, model_name: str) -> List[Dict[str, any]]:
  articles = scrape_page(page_url)
  if len(articles) == 0:
    print(f"Brak artykułów na stronie: {page_url}")
    return None

  else:
    print(f"Znalezione artykuły ({len(articles)}): {articles}")

    relevant_links = agent_filter_articles(articles, model_name=model_name)
    if not relevant_links:
        print('Model nie znalazł żadnych artykułów, które dotyczyłyby rekomendacji giełdowych')
        return []

    print('Linki:', list(relevant_links.keys()))

    extracted_results = []
    for art in articles:
        if art['link'] in relevant_links.keys():
            print(f"\nPrzetwarzanie artykułu: {art['link']}")
            article_text = get_data_from_article(art['link'])
            print(f"Tekst artykułu ma ({len(article_text['text'])} znaków): {article_text['text'][:100]}...")

        if len(article_text['text']) <= 0:
            print("Tekst nie został wydobyty z artykułu!")

        data = parse_article_with_llm(
            link=art['link'],
            title=art['title'],
            article_date=art['article_date'],
            article_text=article_text,
            model_name=model_name
        )

        if data:
            print(f"Dane z artykułu: {data}")
            data['link'] = art['link']
            extracted_results.append(data)

        else:
            print(f"Brak danych z modelu artykułu: {art['link']}")

    return extracted_results
  
def write_results_to_dataframe(articles_data: List[Dict[str, Any]]) -> pd.DataFrame:
    data = [item for sublist in articles_data if sublist for item in sublist]
    results = pd.DataFrame(data)
    results = results.drop(columns=['title', 'link'])
    results = results.rename(columns={
        "article_date": "Data_artykułu",
        "publish_date": "Data_dokumentu",
        "stock_name": "Spółka",
        "securities_house": "Dom_maklerski",
        "present_rating": "Rekomendacja_obecna",
        "last_rating": "Rekomendacja_poprzednia",
        "at_price": "Cena_przy_rekomendacji",
        "target_price": "Cena_docelowa",
        "last_price": "Cena_poprzednia"
    })

    if results is None:
        return print('Brak danych do zapisu')

    else:
        print(f"DataFrame utworzony: ")
        return results
    
def main(stock_list):
    model_name = 'gpt-4.1-mini-2025-04-14'
    data = []
    for stock in stock_list:
        page = 1
        while True:
            print('-------------------------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------------------------')
            time.sleep(5)
            url = f'https://www.bankier.pl/gielda/notowania/akcje/{stock}/wiadomosci/{page}'
            print('Obecny adres to:', url)
            try:
                response = requests.get(url)
                if response.status_code == 404:
                    print('Nie ma żądanej strony:', url)
                    pass

            except Exception:
                print('Nie ma więcej stron z wiadomościami dla spółki:', stock, '\nLiczba stron wyniosła:', page)
                pass

            articles_data = scrape_and_extract_recommendations(url, model_name)
            print(articles_data)
            page += 1

            if articles_data is None:
                break

            elif len(articles_data) <= 0:
                pass

            else:
                data.append(articles_data)

        continue

    data = [item for item in data if len(item) > 0]
    df = write_results_to_dataframe(data)

    return df


if __name__ == "__main__":
    filepath = ''
    with open(filepath, 'r') as file:
        stocks_names = file.readlines()

    stocks_names = [stock.replace('\n', '') for stock in stocks_names]
    parts = {}
    size = 15 # Parametr dzielący listę spółek na grupy/serie

    for i in range(0, len(stocks_names), size):
        right_end = i + size
        parts[f'{i}:{right_end}'] = stocks_names[i:right_end]


    df = main(stocks_names)
    print(df.head())
