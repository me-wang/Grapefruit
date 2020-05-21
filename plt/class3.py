# Line Graph
import matplotlib.pyplot as plt
import pandas as pd

gas = pd.read_csv('./dataCsv/gas_prices.csv')
plt.figure(figsize=(8, 5))
plt.plot(gas.Year, gas.USA, 'r.-', label='USA')
plt.plot(gas.Year, gas.Canada, 'b.-', label='Canada')

plt.title('USA vs Canada Gas Prices')
plt.legend()
plt.savefig('./GraphClass/LineGraph.png')
plt.show()