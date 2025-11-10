import json
from collections import Counter, defaultdict

# with gzip.open('/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_.ndjson.gz', mode='rt', encoding='utf-8') as f:
#     data = [json.loads(line) for line in f]

# # 'data' will be a list of JSON objects (dictionaries)
# print(f"Loaded {len(data)} records from the NDJSON file.")
# print("First record:", data[0])

# with open('/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson', 'r', encoding='utf-8') as f:
#     data = [json.loads(line) for line in f]

# # for key, value in data[200]['metadata'].items():  # Print first 5 records as a sample
# for i in range(len(data)):
#     # print(f"{key}: {value}")
#     # print("-" * 40)
#     # print(f"{key}: {value}")
#     print(i)
#     # if "Distraction-based neural networks for modeling documents" in data[i]['metadata']['title']:
#     #     print(data[i])
#     #     break
#     if "610877" in data[i]['metadata']['id']:
#         print(data[i])
#         break
#     # if "Evolutionary Policy Optimization" in data[i]['metadata']['title']:
#     #     print(data[i])
#     # if "1706.03762" in data[i]['metadata']['id']:
#     #     print(data[i])
#     #     break








# keys_set = set()

# with open("/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson", "r") as file:
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






# def merge_dicts(dict1, dict2):
#     """Merge two dictionaries recursively, combining keys."""
#     for key, value in dict2.items():
#         if key in dict1:
#             if isinstance(dict1[key], dict) and isinstance(value, dict):
#                 merge_dicts(dict1[key], value)
#             # If values are not both dicts, keep the existing one (or handle conflicts as needed)
#         else:
#             dict1[key] = value
#     return dict1

# def get_full_structure_from_ndjson(file_path):
#     combined_structure = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             try:
#                 json_obj = json.loads(line)
#                 combined_structure = merge_dicts(combined_structure, json_obj)
#             except json.JSONDecodeError:
#                 pass  # or handle error
#     return combined_structure

# # Usage:
# file_path = '/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson'
# structure = get_full_structure_from_ndjson(file_path)
# print(json.dumps(structure, indent=4))






# title_counts = Counter()
# file_path = '/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson'
# with open(file_path, 'r') as f:
#     for line in f:
#         if line.strip():
#             data = json.loads(line)
#             title = data.get('metadata', {}).get('title')
#             id_ = data.get('metadata', {}).get('id')
#             if title and id_:
#                 title_counts[(title, id_)] += 1

# # Aggregate counts by title
# title_aggregate = defaultdict(int)
# for (title, _id), count in title_counts.items():
#     title_aggregate[title] += count
# # Filter titles that have duplicates (appear more than once regardless of id)
# duplicate_titles = {title: count for title, count in title_aggregate.items() if count > 1}
# print("Duplicate titles and their counts:")
# for title, total_count in duplicate_titles.items():
#     print(f"Title: {title} => Total count: {total_count}")
#     # Show all ids and counts for this title
#     for (t, id_), count in title_counts.items():
#         if t == title:
#             print(f"    ID: {id_}, Count: {count}")







def print_duplicate_titles_structure(ndjson_path):
    seen_titles = {}
    duplicates_found = 0
    with open(ndjson_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            title = record.get("metadata", {}).get("title", None)
            if title:
                if title in seen_titles:
                    # Print the first encountered duplicate structure (the original and the duplicate)
                    print("Original entry:")
                    print(json.dumps(seen_titles[title], indent=4))
                    print("\nDuplicate entry:")
                    print(json.dumps(record, indent=4))
                    duplicates_found += 1
                    if duplicates_found == 2:
                        break
                else:
                    seen_titles[title] = record
# Usage:
print_duplicate_titles_structure("/home/mccarryster/very_big_work_ubuntu/ML_projects/arxiv_research_helper/arxiv_paper_metadata/arxiv_metadata.ndjson")