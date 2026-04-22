---
name: web-search
description: Fetch and search data from the internet, returning relevant web results or summaries.
---

# Web Search Skill

This skill allows you to fetch current information from the internet and search for data across websites. It is useful for gathering the latest news, research, trends, and public information that is not available in local datasets.

## Instructions
1. Use the `execute` tool to run `python skills/web-search/scripts/search.py <query>` for your search needs.
2. The main argument (`<query>`) should be the user's search terms, e.g., `python skills/web-search/scripts/search.py latest AI advancements`.
3. The script fetches search results from DuckDuckGo (or other sources if customized) and prints top results as links with snippets.
4. Read and present the results to the user. Optionally, offer to summarize linked pages (future extension).

## Dependencies
- Python 3
- `requests` and `beautifulsoup4` modules

## Example Usage
- To search "best hybrid cars 2024":
  - `python skills/web-search/scripts/search.py best hybrid cars 2024`

---
Feel free to customize the implementation or add more advanced summarization as needed.
