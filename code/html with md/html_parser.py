from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin

from ocr_parser import extract_text_from_image
from utils import DEFAULT_HEADERS


def fetch_html_content(url):
    response = requests.get(url)
    response.encoding = 'utf-8'  # 强制设置响应编码
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"请求失败，状态码：{response.status_code}")


def parse_html(html_content, base_url):
    #解析题目列表页并处理答案跳转
    soup = BeautifulSoup(html_content, "html.parser")
    results = []

    # 解析所有题目项
    for item in soup.find_all("li", class_="item-answer"):
        # 提取基础信息
        problem = item.find("div", class_="text").a.get_text(strip=True)
        answer_span = item.find("span", class_="da")

        # 判断是否需要跳转详情页
        if answer_span and "略,具体答案请查看详情" in answer_span.text:
            detail_url = urljoin(base_url, item.find("a", class_="watch")["href"])
            answer = parse_detail_answer(detail_url)
        else:
            answer = answer_span.get_text(strip=True) if answer_span else "无答案"

        results.append({
            "type": "addition",
            "question": problem,
            "process": "本题暂无解题过程",
            "answer": answer
        })
    next_page_url = get_next_page(soup, base_url)

    return results, next_page_url

def get_next_page(soup, base_url):
    # 检测是否存在下一页
    next_page_tag = soup.find('a', text='下一页')
    if next_page_tag and 'href' in next_page_tag.attrs:
        return urljoin(base_url, next_page_tag['href'])
    return None

def parse_detail_answer(detail_url):
    # 解析答案详情页
    try:
        response = requests.get(detail_url, headers=DEFAULT_HEADERS, timeout=10)
        response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")

        # 查找答案区域
        answer_div = soup.find("div", id="answer")
        if not answer_div:
            return "（未找到答案内容）"

        # 优先处理文字答案
        if text_answer := answer_div.get_text(" ", strip=True):
            if text_answer not in ["", "答案与解析"]:
                return text_answer

        # 处理图片答案
        if answer_img := answer_div.find("img"):
            img_url = urljoin(detail_url, answer_img["src"])
            return extract_text_from_image(img_url)

        return "（无有效答案内容）"


    except Exception as e:
        print(f"详情页解析异常：{str(e)}")
        return "（答案获取失败）"
