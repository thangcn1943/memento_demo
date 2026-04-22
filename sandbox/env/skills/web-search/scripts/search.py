import sys
import requests
from bs4 import BeautifulSoup


def duckduckgo_search(query):
    url = 'https://html.duckduckgo.com/html/?q=' + requests.utils.quote(query)
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for result in soup.find_all('div', class_='result'):  # DuckDuckGo lite HTML layout
        link = result.find('a', class_='result__a')
        snippet = result.find('a', class_='result__snippet')
        if link:
            results.append({
                'title': link.get_text(strip=True),
                'href': link['href'],
                'snippet': snippet.get_text(strip=True) if snippet else ''
            })
        if len(results) >= 5:
            break
    return results


def main():
    if len(sys.argv) < 2:
        print('Usage: python search.py <query>')
        sys.exit(1)
    query = ' '.join(sys.argv[1:])
    results = duckduckgo_search(query)
    if not results:
        print('No results found.')
        return
    for i, entry in enumerate(results, 1):
        print(f"{i}. {entry['title']}\n{entry['href']}\nSnippet: {entry['snippet']}\n")


if __name__ == '__main__':
    main()
