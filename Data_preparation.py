
# Imports:

from pathlib import Path
import json
import pandas as pd
import unicodedata
import html
import os

# Functions:

def folders_iteration_looking_for_jsons(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"The path was not found: {p}")

    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.rglob("*.json"))
    else:
        raise ValueError(f"The path is neither file nor dir: {p}")

    jsons = []
    for file in files:
        if file.suffix.lower() != ".json":
            print(f"[WARN] I omitted {file}: the file suffix is not '.json'")
            continue

        try:
            with open(file, "r", encoding="utf-8") as f:
                jsons.append(json.load(f))
        except Exception as e:
            print(f"[WARN] I omitted {file}: {e}")

    return jsons

def I_YA_extraction(data, items_list=None):
    if items_list is None:
        items_list = []
    next_YA = False
    next_next_YA = False
    for i in data["AIF"]["nodes"]:
        if i["type"] == "I":
            I = i["text"]
            next_YA = True
            continue
        if i["type"] == "YA" and next_YA == True:
            next_YA = False
            next_next_YA = True
            continue
        if i["type"] == "YA" and next_next_YA == True:
            YA = i["text"]
            items_list.append([I, YA])
            next_next_YA = False
    return items_list

def convert_list2D_to_df_and_write_as_csv_on_desktop(items_list2D, x_categories):
    df = pd.DataFrame(items_list2D, columns = ["text", "annotation"])
    df["text"] = df["text"].apply(lambda x: unicodedata.normalize("NFC", x))
    df["text"] = df["text"].apply(html.unescape)
    df["text"] = (
        df["text"]
            .str.replace(r"@\S+", "<USER>", regex=True)
            .str.replace(r"(https?://\S+|www\.\S+)", "<URL>", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.lower()
    )
    examples_before_drop_duplicates = len(df)
    df = df.drop_duplicates()
    examples_after_drop_duplicates = len(df)
    removed_duplicates = examples_before_drop_duplicates - examples_after_drop_duplicates
    counts = df["annotation"].value_counts()
    top_categories = counts.head(x_categories).index
    if list(top_categories) != [
        "Asserting", "Analysing", "Emotion-appealing", "Emotion-expressing",
        "Rhetorical Questioning", "Pure Questioning", "Phatic-maintaining", "Assertive Questioning"
        ]:
        print("[WARN] Top categories are not equall to predefined list and model train on this data will learn different classes!")
    no_top_categories = counts.index.difference(top_categories)
    removed_classes = len(no_top_categories)
    removed_classes_examples = int(counts.loc[no_top_categories].sum())
    df = df[df["annotation"].isin(top_categories)]
    examples_before_drop_conflicts = len(df)
    conflicts = (
        df.groupby("text")["annotation"]
            .nunique()
    )
    conflicts = conflicts[conflicts > 1]
    conflicting_texts = conflicts.index
    df = df[~df["text"].isin(conflicting_texts)]
    examples_after_drop_conflicts = len(df)
    removed_conflicts = examples_before_drop_conflicts - examples_after_drop_conflicts
    df = df.sample(frac=1, random_state=1)
    df = df.reset_index(drop=True)
    duplicates = df["text"].duplicated().sum()
    if duplicates == 0:
        lack_of_duplicates = True
    else:
        lack_of_duplicates = duplicates
    classes = df["annotation"].nunique()
    if classes == x_categories:
        correct_number_of_classes = True
    else:
        correct_number_of_classes = classes
    examples = len(df)
    shape = df.shape
    if shape == (examples, 2):
        correct_shape = True
    else:
        correct_shape = shape
    class_stats = df["annotation"].value_counts().to_frame("count")
    class_stats["percent"] = df["annotation"].value_counts(normalize=True) * 100
    class_stats.loc["all"] = [
        examples,
        100.0
    ]
    class_stats = class_stats.loc[["all"] + class_stats.index.drop("all").tolist()]
    df["length"] = df["text"].str.len()
    length_describe = df["length"].describe()
    df = df.drop(columns=["length"])
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    df.to_csv(os.path.join(desktop, "Training_data.csv"), index=False, encoding="utf-8")
    return {
        "lack_of_duplicates": lack_of_duplicates,
        "correct_number_of_classes": correct_number_of_classes,
        "correct_shape": correct_shape,
        "removed_duplicates": removed_duplicates,
        "removed_classes": removed_classes,
        "removed_classes_examples": removed_classes_examples,
        "removed_conflicts": removed_conflicts,
        "class_stats": class_stats,
        "length_describe": length_describe,
        "success": "Program ended succesfull."
        }

# Pipeline:

data = folders_iteration_looking_for_jsons(r"C:\Users\matis\Desktop\Training_materials")
items_list = []
for json_data in data:
    I_YA_extraction(json_data, items_list)
x_categories = 8
result = convert_list2D_to_df_and_write_as_csv_on_desktop(items_list, x_categories)

print(result)

