#####################################
# q2_script.py                      #
# Aditya Tapshalkar                 #
# Georgia Institute of Technology   #
# Summer/Fall 2021                  #
#####################################


import os
import json
import sys

import numpy as np

# Assigning GTSDB classes to prediction classes
redRoundSign_classes    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16]
pg_classes              = [13]
ps_classes              = [14]
pne_classes             = [17]
pn_classes              = [43, 44]
# other_classes           = [11, 12, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
#                            39, 40, 41, 42]

total_stats = {
    "True Positives"    : 0,
    "False Negatives"   : 0,
    "False Positives"   : 0,
}

average_precisions = []

# Retrieving JSON data
with open("GTSDB.json", "r") as data_json:
    pred_data = json.load(data_json)

# Retrieving ground truths
with open("gt.txt", "r") as gt_txt:
    all_gt = gt_txt.readlines()
    for idx, line in enumerate(all_gt):
        all_gt[idx] = line.strip().split(";")

sys.stdout = open("model_evaluation_noappend.txt", "w")

def evaluate_prediction(image):

    print("Image:\t\t\t\t\t\t", image)

    try:
        # Retrieving ground truths
        gt_sublist = []
        for idx, line in enumerate(all_gt):
            if line[0] == image:
                gt_sublist.append(line)

        # Retrieving predictions
        json_signs = [frame for frame in pred_data["output"]["frames"] if frame["frame_number"] == image]
        test_validate = json_signs[0]["signs"]
    except:
        print("--- Skipped by detector ---")
        print("__________________________________________________\n")
        return

    # Counting ground truths
    gt_sign_count = {
        "RedRoundSign"  : 0,
        "pg"            : 0,
        "ps"            : 0,
        "pne"           : 0,
        "pn"            : 0,
        # "other"         : 0
    }

    # Counting predicted truths
    pred_sign_count = {
        "RedRoundSign"  : 0,
        "pg"            : 0,
        "ps"            : 0,
        "pne"           : 0,
        "pn"            : 0,
        # "other"         : 0
    }

    for idx, gt_sign in enumerate(gt_sublist):
        gt_sign_class = int(gt_sign[5])
        if gt_sign_class in redRoundSign_classes:
            gt_sign_count["RedRoundSign"] += 1
        elif gt_sign_class in pg_classes:
            gt_sign_count["pg"] += 1
        elif gt_sign_class in ps_classes:
            gt_sign_count["ps"] += 1
        elif gt_sign_class in pne_classes:
            gt_sign_count["pne"] += 1
        elif gt_sign_class in pn_classes:
            gt_sign_count["pn"] += 1
        # else:
        #     gt_sign_count["other"] += 1

    print("Ground truth sign count:\t", gt_sign_count)

    try:
        for idx, pred_sign in enumerate(json_signs[0]["signs"]):
            pred_sign_class = pred_sign["class"]
            pred_sign_count[pred_sign_class] += 1
    except IndexError:          # No signs detected by detector
        pass

    print("Predicted sign count:\t\t", pred_sign_count)
    print()

    # Counting TP/FP/FN
    compare_dict = {
        "True Positives"    : 0,
        "False Negatives"   : 0,
        "False Positives"   : 0,
    }

    # Tracking precisions for classes
    img_precisions = []
    img_recalls = []

    for dict_class in gt_sign_count.keys():
        img_tp = min(gt_sign_count[dict_class], pred_sign_count[dict_class])
        compare_dict["True Positives"] += img_tp
        total_stats["True Positives"] += img_tp

        ################################################################
        ############# IF TRUE AND FALSE POSITIVES ARE ZERO #############
        if gt_sign_count[dict_class] == 0 and pred_sign_count[dict_class] == 0:
            # img_precisions.append(1)
            # img_recalls.append(1)
            continue
        ################################################################

        diff = gt_sign_count[dict_class] - pred_sign_count[dict_class]

        if diff >= 0:           # Presence of false negatives (detector missed signs)
            compare_dict["False Negatives"] += diff
            total_stats["False Negatives"] += diff
            try:
                img_precisions.append(img_tp / (img_tp))
            except:             # If true positives and false positives are both 0
                img_precisions.append(1)
            img_recalls.append(img_tp / (img_tp + diff))

        elif diff < 0:          # Presence of false positives (detector predicted non-signs)
            compare_dict["False Positives"] -= diff
            total_stats["False Positives"] -= diff
            img_precisions.append(img_tp / (img_tp - diff))
            img_recalls.append(img_tp / (-diff))

    print(compare_dict)

    if len(img_precisions) > 0:
        avg_img_precision = np.average(img_precisions)
        if not np.isnan(avg_img_precision):
            average_precisions.append(avg_img_precision)
        avg_img_recall = np.average(img_recalls)
        print("Average precision:\t", avg_img_precision)
        print("Average recall:\t", avg_img_recall)


    # recall = compare_dict["True Positives"] / (compare_dict["True Positives"] + compare_dict["False Negatives"])
    # precision = compare_dict["True Positives"] / (compare_dict["True Positives"] + compare_dict["False Positives"])

    print("__________________________________________________\n")





folder_path = "../../data/FullIJCNN2013/FullIJCNN2013/"
files = []

for file in os.listdir(folder_path):
    if file.endswith(".ppm"):
        files.append(file)
        evaluate_prediction(file)

print("\n#######################################################")
print(total_stats)
print("Mean Average Precision:\t\t\t", np.average(average_precisions))
total_precision = total_stats["True Positives"] / (total_stats["True Positives"] + total_stats["False Positives"])
print("Average Precision (Total):\t\t", total_precision)
total_recall = total_stats["True Positives"] / (total_stats["True Positives"] + total_stats["False Negatives"])
print("Average Recall (Total):\t\t\t", total_recall)
print("F1 Score (Total):\t\t\t\t", 2 * total_precision * total_recall / (total_precision + total_recall))




sys.stdout.close()