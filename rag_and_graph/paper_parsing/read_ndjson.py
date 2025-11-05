# import gzip
# import json

# with gzip.open('/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_.ndjson.gz', mode='rt', encoding='utf-8') as f:
#     data = [json.loads(line) for line in f]

# # 'data' will be a list of JSON objects (dictionaries)
# print(f"Loaded {len(data)} records from the NDJSON file.")
# print("First record:", data[0])



import json

with open('/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_metadata.ndjson', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]


# for key, value in data[200]['metadata'].items():  # Print first 5 records as a sample
for i in range(len(data)):
    # print(f"{key}: {value}")
    # print("-" * 40)
    # print(f"{key}: {value}")
    print(i)
    # if "Distraction-based neural networks for modeling documents" in data[i]['metadata']['title']:
    #     print(data[i])
    #     break
    if "1506.02617" in data[i]['metadata']['id']:
        print(data[i])
        break
    # if "1706.03762" in data[i]['metadata']['id']:
    #     print(data[i])
    #     break

# keys_set = set()

# with open("arxiv_metadata.ndjson", "r") as file:
#     for line in file:
#         data = json.loads(line.strip())
#         # Recursive function to collect keys from nested dicts
#         def collect_keys(d):
#             if isinstance(d, dict):
#                 for k, v in d.items():
#                     keys_set.add(k)
#                     collect_keys(v)
#             elif isinstance(d, list):
#                 for item in d:
#                     collect_keys(item)

#         collect_keys(data)

# all_keys = sorted(keys_set)
# print(all_keys)


# import json

# seen = set()
# duplicates = []

# with open('/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_metadata.ndjson', 'r') as f:
#     for line in f:
#         obj = json.loads(line)
#         identifier = obj['oai_header']['identifier']
#         if identifier in seen:
#             duplicates.append(identifier)  # or log line index
#         else:
#             seen.add(identifier)
#         print(f"Found {len(duplicates)} duplicate identifiers.")
#         print("Duplicates:", duplicates)
#         print("-" * 100)
#         if len(duplicates) == 10:
#             break
#         # print(identifier)