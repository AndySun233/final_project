import yfinance as yf
import pandas as pd
#eia oil stocks https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCRSTUS1&f=W
#cpi 
# 定义起止时间
start_date = "2023-05-01"
end_date = "2025-03-31"

# 商品价格数据（30分钟）
wti = yf.download("CL=F", interval="1h", start=start_date, end=end_date)
gold = yf.download("GC=F", interval="1h", start=start_date, end=end_date)

# 宏观经济数据（日线）
dxy = yf.download("DX-Y.NYB", interval="1d", start=start_date, end=end_date)
vix = yf.download("^VIX", interval="1d", start=start_date, end=end_date)

# saved as CSV
wti.to_csv("data/wti_1h_2yr.csv")
gold.to_csv("data/gold_1h_2yr.csv")
dxy.to_csv("data/dxy_daily.csv")
vix.to_csv("data/vix_daily.csv")
