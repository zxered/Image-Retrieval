import json
import numpy as np


def evaluate_accuracy(query_pos: np.array, database_pos: np.array):
    """Is the predicted node located within 1.0 m from the ground-truth nearest node?"""
    step = np.linalg.norm(query_pos - database_pos)
    result = step <= 10

    return bool(result)

evaluation_sheet_file = "./evaluation_sheet.json"
answer_file = "./answer.txt"

with open(evaluation_sheet_file, "r") as f:
    evaluation_sheet = json.load(f)

with open("./answer.txt", "r") as f:
    answer_sheet = f.readlines()

query_size = len(answer_sheet)
accuracy_list = []

for answer in answer_sheet:
    answer_split = answer.split('.jpg')
    query_id = answer_split[0][-6:]
    database_id = answer_split[1][-6:]

    query_pos = np.array(evaluation_sheet[f"{query_id}_query"])
    database_pos = np.array(evaluation_sheet[f"{database_id}_database"])

    accuracy = evaluate_accuracy(query_pos, database_pos)
    accuracy_list.append(accuracy)

print("Recall@1: ", sum(accuracy_list) / query_size)

