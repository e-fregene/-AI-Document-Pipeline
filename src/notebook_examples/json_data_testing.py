import json
import pandas as pd
from pandas import json_normalize



"""Activity"""
with open("src/data/sample.json", "r") as act_file:
    json_actvity = json.load(act_file)

json_orerd_ids= json_actvity["orders"][0]
json_orerd_totals= json_actvity["orders"][0][2]

print("Order IDS", json_orerd_ids, "Order totals", json_orerd_totals)


#normalize_activity = json_normalize(json_actvity,)