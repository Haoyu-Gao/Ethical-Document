import pandas as pd
import os
import re
import markdown2
from keybert import KeyBERT
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

import nltk

from nltk.corpus import words

lemmatizer = WordNetLemmatizer()

def keyword_filter(data_source, keyword_set, folder_name):

    data_source['keyword-filtered'] = False
    for i in range(len(data_source)):
        if data_source.iloc[i]['filtered'] == True:
            repo_name = data_source.iloc[i]["repo_name"]
            file_name = repo_name.replace("/", "@_@")
            file_name = file_name + '.md'
            file_path = os.path.join(folder_name, file_name)

            with open(file_path) as f:
                content = f.read().lower()
                tokens = word_tokenize(content)
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
                # bigram_tokens = list(bigrams(lemmatized_tokens))    
                for token in lemmatized_tokens:
                    if token in keyword_set:
                        data_source.loc[i, 'keyword-filtered'] = True
                        break
        
    return data_source




def base_filter():
    # load the base keywords, composing of ethics, FAccT, and keywords in one literature review.
    base_keywords = set()
    with open('keyword-base.txt') as f:
        for line in f.readlines():
            base_keywords.add(line.strip())
    
    with open('extra_keywords_paragraph2.txt') as f:
        for line in f.readlines():
            base_keywords.add(line.strip())
    
    # load the data source
    gh_card = pd.read_csv("Github_model_card_filtered-paragraph.csv")
    gh_readme = pd.read_csv("Github_readme_filtered-paragraph.csv")
    hf_card = pd.read_csv("Huggingface_model_card_filtered-paragraph.csv")

    # filter the data source using only base keywords
    gh_card_filtered = keyword_filter(gh_card, base_keywords, "github_model_card_documents")
    gh_readme_filtered = keyword_filter(gh_readme, base_keywords, "github_readme_documents")
    hf_card_filtered = keyword_filter(hf_card, base_keywords, "huggingface_model_card_documents")

    gh_card_filtered.to_csv("Github_model_card_filtered-paragraph-final.csv")
    gh_readme_filtered.to_csv("Github_readme_filtered-paragraph-final.csv")
    hf_card_filtered.to_csv("Huggingface_model_card_filtered-paragraph-final.csv")
    # expand the keywords using the base keywords
    pass

def keyword_expand():
    gh_mc = pd.read_csv("Github_model_card_filtered-paragraph.csv")
    gh_rm = pd.read_csv("Github_readme_filtered-paragraph.csv")
    hf_mc = pd.read_csv("Huggingface_model_card_filtered-paragraph.csv")

            
    base_keywords = []
    with open('keyword-base.txt') as f:
        for line in f.readlines():
            base_keywords.append(line.strip())
    
    kw_model = KeyBERT()

    extra_keywords = []
    for i in range(len(gh_mc)):
        if gh_mc.iloc[i]['manual'] == True:
            repo_name = gh_mc.iloc[i]["repo_name"]
            file_name = repo_name.replace("/", "@_@") + '.md'

            file_path = os.path.join("github_model_card_documents", file_name)
            with open(file_path) as f:
                content = f.read().lower()
                paragraphs = content.split('\n\n')
                for paragraph in paragraphs:
                    tokens = word_tokenize(paragraph)
                    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
                    # bigram_tokens = list(bigrams(lemmatized_tokens))
                    flag = False
                    for token in lemmatized_tokens:
                        if token in base_keywords:
                            flag = True
                            break
            
                    if flag:
                        keywords = kw_model.extract_keywords(paragraph, keyphrase_ngram_range=(1, 1))
                        keywords = [k[0] for k in keywords]
                        extra_keywords += keywords
    

    for i in range(len(gh_rm)):
        if gh_rm.iloc[i]['manual'] == True:
            repo_name = gh_rm.iloc[i]["repo_name"]
            file_name = repo_name.replace("/", "@_@") + '.md'

            file_path = os.path.join("github_readme_documents", file_name)
            with open(file_path) as f:
                content = f.read().lower()
                paragraphs = content.split('\n\n')
                for paragraph in paragraphs:
                    tokens = word_tokenize(paragraph)
                    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
                    # bigram_tokens = list(bigrams(lemmatized_tokens))
                    flag = False
                    for token in lemmatized_tokens:
                        if token in base_keywords:
                            flag = True
                            break
                    
                    if flag:
                        keywords = kw_model.extract_keywords(paragraph, keyphrase_ngram_range=(1, 1))
                        keywords = [k[0] for k in keywords]
                        extra_keywords += keywords


    for i in range(len(hf_mc)):
        if hf_mc.iloc[i]['manual'] == True:
            repo_name = hf_mc.iloc[i]["repo_name"]
            file_name = repo_name.replace("/", "@_@") + '.md'

            file_path = os.path.join("huggingface_model_card_documents", file_name)
            with open(file_path) as f:
                content = f.read().lower()
                paragraphs = content.split('\n\n')
                for paragraph in paragraphs:
                    tokens = word_tokenize(paragraph)
                    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
                    # bigram_tokens = list(bigrams(lemmatized_tokens))
                    flag = False
                    for token in lemmatized_tokens:
                        if token in base_keywords:
                            flag = True
                            break
                    
                    if flag:
                        keywords = kw_model.extract_keywords(paragraph, keyphrase_ngram_range=(1, 1))
                        keywords = [k[0] for k in keywords]
                        extra_keywords += keywords


    extra_keywords = set(extra_keywords)
    print(extra_keywords)
    # remove non-english keywords or keywords that are not all english characters
    for keyword in extra_keywords.copy():
        if not re.search('[a-zA-Z]', keyword):
            extra_keywords.remove(keyword)
            
        elif not keyword.isalpha():
            extra_keywords.remove(keyword)

    # remove keywords that are in the base keywords
    for keyword in extra_keywords.copy():
        if keyword in base_keywords:
            extra_keywords.remove(keyword)
    
    # remove clear outliers: i.e., keywords that have less than 3 characters
    for keyword in extra_keywords.copy():
        if len(keyword) <= 3:
            extra_keywords.remove(keyword)

    for keyword in extra_keywords.copy():
        if keyword not in words.words():
            extra_keywords.remove(keyword)

    # save the keywords
    extra_keywords_lemmatised = [lemmatizer.lemmatize(keyword) for keyword in extra_keywords]
    extra_keywords_lemmatised = set(extra_keywords_lemmatised)
    with open("extra_keywords_paragraph1.txt", 'w') as f:
        for keyword in extra_keywords_lemmatised:
            f.write(keyword + '\n')



if __name__ == '__main__':
    base_filter()

    keyword_expand()
