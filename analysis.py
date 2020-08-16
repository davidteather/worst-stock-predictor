import json

with open("results.json", 'r') as raw:
    data = json.loads(raw.read())

lowest = 99999999
highest = -1
total_good = 0
total_bad = 0
for x in data['data']:
    if lowest > x['bad_cash'] and x['bad_cash'] > 0:
        lowest = x['bad_cash']

    if highest < x['good_cash']:
        highest = x['good_cash']

    total_good += x['good_cash']
    total_bad += x['bad_cash']


print("Lowest Cash: " + str(lowest))
print("Highest Cash: " + str(highest))

print("\n")

print("Avg Good: {}".format(total_good/len(data['data'])))
print("Avg Bad: {}".format(total_bad/len(data['data'])))
