from dotenv import load_dotenv
import os
import requests
import base64

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

def search_github_repos(query, per_page=10):
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page
    }

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    return r.json().get("items", [])

def get_readme(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        return None

    data = r.json()

    content = data.get("content")
    encoding = data.get("encoding")

    if not content:
        return None

    if encoding == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        except Exception:
            return None

    return None

queries = [
    "web crawler markdown llm",
    "llm web scraper",
    "markdown scraper",
    "ai web crawling",
    "crawler extraction markdown"
]

all_repos = {}

for q in queries:
    repos = search_github_repos(q, per_page=10)
    for repo in repos:
        all_repos[repo["full_name"]] = repo

print(f"\nUnique repos found: {len(all_repos)}\n")

repo_list = list(all_repos.values())[:10]

for i, repo in enumerate(repo_list, 1):
    full_name = repo["full_name"]
    owner = repo["owner"]["login"]
    name = repo["name"]

    print(f"{i}. {full_name}")
    print(f"   Description: {repo.get('description')}")
    print(f"   Stars: {repo['stargazers_count']}")
    print(f"   URL: {repo['html_url']}")

    readme = get_readme(owner, name)

    if readme:
        print("   README FOUND")
        print("   README PREVIEW:")
        preview = readme[:700].replace("\n", " ")
        print(f"   {preview}")
    else:
        print("   README NOT FOUND")

    print("-" * 100)