"""
Owner: Ethan

Extract data from the SymMap website, which contains information about the relationships between symptoms, 
syndromes, and herbs in Traditional Chinese Medicine (TCM). 

The data is stored in JSON format on the website, and we used regexes to extract it. 

We used this to create Symptom_Syndrome_Edges.csv and Syndrome_Herb_Edges.csv.
"""

import json
import requests
import pandas as pd
import time
import re

# we care primarily about the edges
symptom_syndrome_edges = []
syndrome_herb_edges = []

syndrome_ids = [f"SMSY{str(i).zfill(5)}" for i in range(1, 234)] # to get all the syndrome ids, start with 0s before the number

# now need a regex pattern to extract from the json file data = []
# re.DOTALL so it can capture across multiple lines if the data is formatted that way
json_pattern = re.compile(r'var\s+data\s*=\s*(\[.*)', re.DOTALL) # this splits up the data array into a group that we can extract

for syndrome_id in syndrome_ids:
    # file is always named after the syndrome id
    url = f"http://www.symmap.org/network_sym/{syndrome_id}_sym"

    try:
        # Add headers to prevent blocks
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)

        # If file exists, extract the data
        if response.status_code == 200:
            match = json_pattern.search(response.text)

            if match:
                raw_text = match.group(1) # extract the data array from the matched group

                try:
                    data, _ = json.JSONDecoder().raw_decode(raw_text) # decode the JSON data

                    for item in data: 
                        if item.get('group') == 'edges':
                            source = item['data']['source'] # the source node of the edge
                            target = item['data']['target'] # the target node of the edge

                            if 'SMTS' in source and 'SMSY' in target: # if the edge is between a symptom and a syndrome
                                symptom_syndrome_edges.append({'TCM_symptom_id': source, 'Syndrome_id': target})
                            elif 'SMSY' in source and 'SMTS' in target: # if the edge is between a syndrome and a symptom
                                symptom_syndrome_edges.append({'TCM_symptom_id': target, 'Syndrome_id': source})
                        
                            # we also want to capture the edges between syndromes and herbs
                            elif 'SMSY' in source and 'SMHB' in target: # if the edge is between a syndrome and a herb
                                syndrome_herb_edges.append({'Syndrome_id': source, 'Herb_id': target})
                            elif 'SMHB' in source and 'SMSY' in target: # if the edge is between a herb and a syndrome
                                syndrome_herb_edges.append({'Syndrome_id': target, 'Herb_id': source})

                    print(f"Processed {syndrome_id}")
                except json.JSONDecodeError as je:
                    print(f"JSON decoding error for {syndrome_id}: {je}")
            else: 
                print(f"No data found for {syndrome_id}")
        
        time.sleep(1) # to avoid overwhelming the server
    
    except Exception as e:
        print(f"Error processing {syndrome_id}: {e}")
    
# save outputs to dataframe
pd.DataFrame(symptom_syndrome_edges).drop_duplicates().to_csv("Symptom_Syndrome_Edges.csv", index=False)
pd.DataFrame(syndrome_herb_edges).drop_duplicates().to_csv("Syndrome_Herb_Edges.csv", index=False)

print("Did it!")
