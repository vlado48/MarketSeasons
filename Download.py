import duka.app.app as imp_ticks
from duka.core.utils import TimeFrame
import datetime

start_date = datetime.date(2003, 1, 1)
end_date = datetime.date(2020, 9, 25)

pairs = ['EURUSD']

imp_ticks(pairs, start_date, end_date, 1, TimeFrame.D1, ".", True)
