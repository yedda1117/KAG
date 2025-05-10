from html_parser import parse_html
from utils import is_url, is_file, get_file_extension, fetch_html_content, create_question_csv, DEFAULT_HEADERS


def parse_content(input_source):
    if is_url(input_source):
        # 请求头
        html_content = fetch_html_content(input_source, headers={
            "Referer": "https://www.aoshuku.com/shuxueti/san/"  # 动态更新来源
        })
        print("检测到输入为URL，按HTML解析...")
        html_content = fetch_html_content(input_source)
        questions = parse_html(html_content)
        create_question_csv(questions)
    else:
        print("输入类型无法识别，请输入URL。")
        return

if __name__ == "__main__":
    url = ("https://www.aoshuku.com/shuxueti/san/456"
           "/")
    parse_content(url)