from datetime import datetime, timedelta

time_string = "00:02:30"
timedelta_obj = datetime.strptime(time_string, "%H:%M:%S") - datetime(1900, 1, 1)

sec = int(timedelta_obj.total_seconds())

print(sec)