import pandas_datareader.data as web
import datetime

start = datetime.datetime(2023, 3, 25)
end = datetime.datetime(2025, 4, 20)

cpi = web.DataReader("CPIAUCSL", "fred", start, end)
cpi.to_csv("data/cpi_monthly.csv")

