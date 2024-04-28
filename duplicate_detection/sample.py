
import pandas as pd
import csv

data1 = pd.read_csv("huggingface_model_card_filtered-paragraph-final.csv")

data2 = pd.read_csv("github_model_card_filtered-paragraph-final.csv")
data3 = pd.read_csv("github_readme_filtered-paragraph-final.csv")

sample1 = []
sample2 = []
sample3 = []

# for cluster column, sample one sample from each cluster except for cluster -1
for cluster in data1["cluster"].unique():
    if cluster == -1:
        continue
    sample1.append(data1[data1["cluster"] == cluster].sample(1))

for cluster in data2["cluster"].unique():
    if cluster == -1:
        continue
    sample2.append(data2[data2["cluster"] == cluster].sample(1))

for cluster in data3["cluster"].unique():
    if cluster == -1:
        continue
    sample3.append(data3[data3["cluster"] == cluster].sample(1))

# write the samples to a new file
with open("sample_hf_mc.csv", "w") as f:
    writer = csv.writer(f)
    for i in range(len(sample1)):
        writer.writerow([sample1[i]["repo_name"].values[0]])

with open("sample_gh_mc.csv", "w") as f:
    writer = csv.writer(f)
    for i in range(len(sample2)):
        writer.writerow([sample2[i]["repo_name"].values[0]])

with open("sample_gh_rm.csv", "w") as f:
    writer = csv.writer(f)
    for i in range(len(sample3)):
        writer.writerow([sample3[i]["repo_name"].values[0]])
            
