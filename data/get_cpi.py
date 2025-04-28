import pandas_datareader.data as web
import datetime

# 设置时间范围
start = datetime.datetime(2023, 3, 25)
end = datetime.datetime(2025, 4, 20)

# 只抓取原始的 CPI（月度数据），不补成小时频率
cpi = web.DataReader("CPIAUCSL", "fred", start, end)
cpi.to_csv("data/cpi_monthly.csv")

print("✅ 成功抓取 CPI 原始数据（按月）并保存为 CSV")
