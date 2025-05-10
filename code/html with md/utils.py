import os
import requests
import csv
from urllib.parse import urljoin
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.aoshuku.com/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1"
}
DEFAULT_HEADERS.update({
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Sec-Fetch-Dest": "image"
})


def fetch_html_content(url, headers=None):
    final_headers = DEFAULT_HEADERS.copy()
    if headers:
        final_headers.update(headers)

    response = requests.get(
        url,
        headers=final_headers,
        timeout=10
    )
    response.encoding = 'utf-8'
    response.raise_for_status()
    return response.text

def is_url(input_source):
    return input_source.startswith("http://") or input_source.startswith("https://")

def is_file(input_source):
    return os.path.isfile(input_source)

def get_file_extension(input_source):
    return os.path.splitext(input_source)[1].lower()


def get_question_links(base_url):
    html = fetch_html_content(base_url)
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for li in soup.select('.right-ul > li'):
        link = li.find('a', class_='text-dark')['href']
        full_link = urljoin(base_url, link)
        links.append(full_link)
    return links


def create_question_csv(data, filename="fc/aoshuku_get_balls.csv", mode='a'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 检测是否需要写表头
    write_header = not os.path.exists(filename) or mode == 'w'

    if not isinstance(data, list):
        raise ValueError("数据必须为列表")
    if data and not isinstance(data[0], dict):
        raise ValueError("列表元素必须为字典")

    with open(filename, mode, newline="", encoding="utf_8_sig") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "question", "process", "answer"])
        if write_header:
            writer.writeheader()
        writer.writerows(data)