import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


chunk_size = 10_000
all_df = pd.read_json("payloads.jsonl", orient="records", lines=True, chunksize=chunk_size)

timestamps = []
unique_labels = set()
submitters = defaultdict(int)

for df in tqdm(all_df):
    df['update_date_ts'] = pd.to_datetime(df.update_date, format="%Y-%m-%d").view(
        'int64') // 10 ** 9
    timestamps.extend(df.update_date_ts.to_list())
    df['labels'] = df.categories.apply(str.split)

    for label_list in df.labels.values:
        for label in label_list:
            unique_labels.add(label)

    for submitter, count in df.submitter.value_counts().iteritems():
        submitters[submitter] += count

    with open("payloads.jsonl", "a") as fp:
        df.to_json(fp, orient="records", lines=True)

for df in tqdm(all_df):
    timestamps.extend(df.update_date_ts.to_list())
    all_labels = df.labels
    for label_list in all_labels.values:
        for label in label_list:
            unique_labels.add(label)

    for submitter, count in df.submitter.value_counts().iteritems():
        submitters[submitter] += count

np_timestamps = np.array(timestamps)
submitters = pd.Series(submitters)

q25 = np.quantile(np_timestamps, 0.25)
q75 = np.quantile(np_timestamps, 0.75)

final = {
    "timestamp_range": {"q25": int(q25), "q75": int(q75)},
    "top_submitters": submitters[submitters > 25].index.to_list(),
    "labels": list(unique_labels),
}

with open("filters.json", "w") as fp:
    json.dump(final, fp, indent=2)
