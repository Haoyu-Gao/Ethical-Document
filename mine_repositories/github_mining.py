from github import Github, RateLimitExceededException, GithubException
import csv
import pandas as pd
import time
import requests



g = Github(ACCESS_TOKEN)

def mine_model_card_repos():
    file_name = "model_card.md"
    query = f"filename:{file_name}"

    # note we are mining through search API, it will reflect the accurate number of results
    # but if the number exceeds 1000, it will only return 1000 results
    results = g.search_code(query)

    for result in results:
        with open("Github_model_card_list.csv", "a") as f:
            writer = csv.writer(f)
            name = result.repository.full_name
            stars = result.repository.stargazers_count
            writer.writerow([name, stars])

    file_name = "model-card.md"
    query = f"filename:{file_name}"

    # note we are mining through search API, it will reflect the accurate number of results
    # but if the number exceeds 1000, it will only return 1000 results
    results = g.search_code(query)

    print(results.totalCount)
    for result in results:
        with open("Github_model_card_list.csv", "a") as f:
            writer = csv.writer(f)
            name = result.repository.full_name
            stars = result.repository.stargazers_count
            writer.writerow([name])



    query = '"model card" filename:README.md path:/'
    results = g.search_code(query=query)
    print(results.totalCount)
    for result in results:
        with open("Github_readme_list1.csv", "a") as f:
            writer = csv.writer(f)
            name = result.repository.full_name
            # stars = result.repository.stargazers_count
            time.sleep(0.5)
            writer.writerow([name])


def remove_duplicate():
    df = pd.read_csv("Github_readme_list.csv", header=None)
    # df.columns = ["name", "stars", "filtered"]
    df.columns = ['name']
    df = df.drop_duplicates(subset="name", keep="first")
    df.to_csv("Github_model_card_list_haha.csv", index=False, header=False)

def filter_github_repos():
    df = pd.read_csv("Github_readme_list.csv", header=None)
    df.columns = ["name", "stars"]
    # filter out the ones with less than 10 stars and filter out the ones that are fork, indicated to a new column with true or false
    for index, row in df.iterrows():
        repo = g.get_repo(row["name"])
        if repo.stargazers_count < 10 or repo.fork:
            df.loc[index, "filter"] = False
        else:
            df.loc[index, "filter"] = True
    
    df.to_csv("Github_readme_list_filtered.csv", index=False, header=False)

def get_documents():
    df = pd.read_csv("Github_model_card_list_haha.csv", header=None)
    df.columns = ["name", "stars", "filter"]
    for index, row in df.iterrows():
        if row['filter'] == True:
            try:
                repo = g.get_repo(row["name"])
                # get model card for this repo
                model_card = repo.get_contents("model-card.md")
                with open(f"github_model_card_documents1/{row['name'].replace('/', '@_@')}.md", "w") as f:
                    f.write(model_card.decoded_content.decode())
                print(f"mining {index}")
            except RateLimitExceededException:
                print("rate limit exceeded")
                break
            except GithubException:
                print("github exception")
                continue
            except:
                print("other exception")
                continue


if __name__ == "__main__":
    mine_model_card_repos()
    remove_duplicate()

    filter_github_repos()
