from __future__ import absolute_import, division, print_function, unicode_literals

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

import backtrader as bt


class MyStrategy(bt.Strategy):
    def next(self):
        if self.data.close[0] > self.data.close[-1]:
            self.buy()


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    """ data = bt.feeds.GenericCSVData(dataname="your_data.csv")
    cerebro.adddata(data) """

    cerebro.addstrategy(MyStrategy)

    cerebro.run()

    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
