import os, datetime, pytz, csv
from itertools import zip_longest

IST = pytz.timezone('Asia/Jakarta')

def logPerson(x, empty, empyt1):
	datetimee = [datetime.datetime.now(IST)]
	data_arr = [datetimee, empyt1, empty, x]
	export_data = zip_longest(*data_arr, fillvalue='')

	with open(os.path.join('log.csv'), 'w', newline='') as logfile:
		wr = csv.writer(logfile, quoting=csv.QUOTE_ALL)
		wr.writerow(('End Time', 'In', 'Out', 'Total Inside'))
		wr.writerows(export_data)
