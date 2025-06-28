import pandas as pd
import time
import sys, os
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
from io import StringIO

start_time = time.time()

link = "https://live.euronext.com/pd_es/data/stocks/download?mics=XOSL%2CMERK%2CXOAS&initialLetter=&fe_type=csv&fe_decimal_separator=,%2C&fe_date_format=d%2Fm%2FY"
resp = requests.get(link)
df = pd.read_csv(StringIO(resp.text), delimiter=";")


df = df.iloc[3:]


wiki_res = requests.get(
    "https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_Oslo_Stock_Exchange"
)
soup = BeautifulSoup(wiki_res.text, "html.parser")

res = soup.select_one("table.wikitable.sortable")

companies = []

i = 0

for row in res.select("tr"):
    cols = row.find_all("td")
    if len(cols) < 3:
        continue
    i += 1
    cols = [ele.text.strip() for ele in cols]
    exchange, ticker = cols[1].split(":")

    industries = cols[3].split("&")
    cleaned_industries = []
    for industry in industries:
        cleaned_industries.append(industry.strip())

    companies.append(
        [ticker.strip(), cols[0], exchange.strip(), cols[2], cleaned_industries]
    )
    # print([ticker.strip(), cols[0], exchange.strip(), cols[2], cleaned_industries])

companies = pd.DataFrame(
    columns=["Ticker", "Name", "Exchange", "Listing Date", "Industry"], data=companies
)

merged_df = pd.merge(
    df,
    companies[["Ticker", "Industry"]],
    left_on="Symbol",
    right_on="Ticker",
    how="left",  # or "inner" if you only want matches
)
merged_df.drop(columns=["Ticker"], inplace=True)

for row in merged_df.iterrows():
    row = row[1]
    print(row)
    # print(row[1], row[])
    # print(row["Name"], row["Market"], row["Industry"])

merged_df.to_csv("oslo_stock_exchange_companies.csv", index=False)


""

# Temporary way to get the Industry for Oslo stock exchange, should be swapped with a more robust/direct solution
""" 

soup = BeautifulSoup(resp.text, "html.parser")

res = soup.select_one("table.wikitable.sortable")

companies = []

i = 0

for row in res.select("tr"):
    cols = row.find_all("td")
    if len(cols) < 3:
        continue
    i += 1
    cols = [ele.text.strip() for ele in cols]
    exchange, ticker = cols[1].split(":")

    industries = cols[3].split("&")
    cleaned_industries = []
    for industry in industries:
        cleaned_industries.append(industry.strip())

    companies.append(
        [ticker.strip(), cols[0], exchange.strip(), cols[2], cleaned_industries]
    )
    # print([ticker.strip(), cols[0], exchange.strip(), cols[2], cleaned_industries])


companies = pd.DataFrame(
    columns=["Ticker", "Name", "Exchange", "Listing Date", "Industry"], data=companies
)

companies.to_csv("oslo_stock_exchange_companies.csv", index=False)

print(
    f"Retrieved {len(companies)} companies in {time.time() - start_time:.2f} seconds."
)
 """
