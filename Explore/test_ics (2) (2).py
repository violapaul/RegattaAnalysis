import itertools as it
import icalendar

icalendar.Calenda

from ics import Calendar
import arrow

cal_file = "/Users/viola/Downloads/Peer Gynt Racing Calendar_shvns4ogqbfc2eslavfdsoojtg@group.calendar.google.com.ics"

with open(cal_file, 'r') as fs:
    cal_text = fs.read()

cal = icalendar.Calendar.from_ical(cal_text)

    
cal = Calendar(cal_text)

start = arrow.get("2021-01-01")

sorted_events = sorted([e for e in cal.events if start < e.begin])

for event in it.islice(sorted_events, 10000):
    print(event.name)


for event in it.islice(sorted_events, 10000):
    if start < event.begin:
        print(event.name)
        date = event.begin
        while True:
            print("   ", date.format("MMM, DD"))
            date = date.shift(days=1)
            if date >= event.end:
                break
              
