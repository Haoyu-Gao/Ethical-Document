import numpy as np
import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

lemmatizer = WordNetLemmatizer()

keywords = set()

with open("../mine_repositories/keyword-base.txt") as f:
    for line in f.readlines():
        keywords.add(line.strip())

with open("../mine_repositories/extra_keywords_paragraph.txt") as f:
    for line in f.readlines():
        keywords.add(line.strip())


def construct_ethical_paragraphs(filtered_data_path, source_folder, to_folder):
    data = pd.read_csv(filtered_data_path)
    data['keywords'] = data.apply(lambda x: [], axis=1)

    for i in range(len(data)):
        if data.iloc[i]['keyword-filtered'] == True:
            paragraph_list = []
            repo_name = data.iloc[i]['repo_name']

            document = repo_name.replace("/", "@_@")
            file_name = f"../mine_repositories/{source_folder}/{document}.md"

            if os.path.exists(file_name):
                with open(file_name) as f:
                    file = f.read()

                # extract the paragraphs with keywords, and append them to the paragraph_list
                paragraphs = file.split("\n\n")
                for paragraph in paragraphs:
                    paragraph = paragraph.lower()
                    lemma_paragraph_tokens = [lemmatizer.lemmatize(token) for token in word_tokenize(paragraph)]
                    for keyword in keywords:
                        if keyword in lemma_paragraph_tokens:
                            paragraph_list.append(paragraph)
                            break
                
                # write the paragraphs to a new file
                with open(f"{to_folder}/{document}.md", "w") as f:
                    for paragraph in paragraph_list:
                        f.write(paragraph + "\n\n")

def compute_similarity_matrix(folder_name):
    files = os.listdir(folder_name)
    num_files = len(files)
    similarity_matrix = np.zeros((num_files, num_files))

    vectorizer = TfidfVectorizer()
    contents = []
    for i in range(num_files):
        with open(f"{folder_name}/{files[i]}") as f:
            content = f.read()
            contents.append(content)
    X = vectorizer.fit_transform(contents)

    similarity_matrix = cosine_similarity(X, X)

    return similarity_matrix, files


def write_cluster(clusters, files, filtered_file, to_file):
    
    data = pd.read_csv(filtered_file)
    data['cluster'] = -1
    for file, cluster in zip(files, clusters):

        file_name = file.replace("@_@", "/").split(".md", 1)[0]
        data["cluster"][data["repo_name"]==file_name] = cluster
    
    data.to_csv(to_file, index=False)

def read_cluster_files(filtered_file, cluster_num):
    data = pd.read_csv(filtered_file)
    # get the documents in this cluster
    cluster_data = data[data["cluster"] == cluster_num]
    
    # write the documents to a new file
    with open(f"cluster_{cluster_num}.txt", "w") as f:
        for i in range(len(cluster_data)):
            repo_name = cluster_data.iloc[i]["repo_name"]
            document = repo_name.replace("/", "@_@")
            file_name = f"huggingface_model_card/{document}.md"
            with open(file_name) as f1:
                content = "******" + repo_name + "******" + "\n" + f1.read()

            f.write(content + "\n\n")



if __name__ == "__main__":
    # construct_ethical_paragraphs(filtered_data_path="../mine_repositories/Github_model_card_filtered-paragraph-final.csv",
    #     source_folder="github_model_documents", to_folder="github_model_readme")
    similarity_matrix, files = compute_similarity_matrix("github_readme")

    # save similarity matrix and files 
    np.save("similarity_matrix_gh_rm.npy", similarity_matrix)
    with open("files_gh_rm.txt", "w") as f:
        for file in files:
            f.write(file + "\n")

    
    similarity_matrix = np.load("similarity_matrix_gh_rm.npy")
    with open("files_gh_rm.txt") as f:
        files = f.readlines()
        files = [file.strip() for file in files]

    distance_matrix = 1 - similarity_matrix
  

    similarity_matrix1 = np.load("similarity_matrix_gh_mc.npy")
    similarity_matrix2 = np.load("similarity_matrix.npy")

    distance_matrix1 = 1 - similarity_matrix1
    distance_matrix2 = 1 - similarity_matrix2

    # using agglomerative clustering to cluster the documents
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1, affinity='precomputed', linkage='average')
    clustering.fit(distance_matrix)
    clusters = clustering.labels_

    write_cluster(clusters, files, "../mine_repositories/Github_readme_filtered-paragraph-final.csv",
                  "github_readme_filtered-paragraph-final.csv")
    
    # print(clustering.n_clusters_)
    # for i, cluster_id in enumerate(clusters):
    #     print(f"Document {i+1} belongs to cluster {cluster_id}")

    