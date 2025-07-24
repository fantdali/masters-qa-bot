import requests
from bs4 import BeautifulSoup
import os


def fetch_and_parse(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup


def save_page_text(soup, filename):
    text = soup.get_text(separator="\n", strip=True)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    urls = {
        "ai": "https://abit.itmo.ru/program/master/ai",
        "ai_product": "https://abit.itmo.ru/program/master/ai_product",
    }
    for key, url in urls.items():
        soup = fetch_and_parse(url)
        # Сохраняем весь текст страницы
        save_page_text(soup, f"./data/{key}_page.txt")
        # Скачиваем PDF учебного плана
        # download_pdf(soup, url, f"./data/{key}_plan.pdf")
        # print(f"Saved {key} page text and PDF")


if __name__ == "__main__":
    main()
