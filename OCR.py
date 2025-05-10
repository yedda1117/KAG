from mistralai import Mistral
from pathlib import Path
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk, OCRResponse
import json

# ====================== 只需修改这里 ======================
PDF_PATH = f"pdf/吉米多维奇数学分析习题集题解1 (费定晖 周学圣) (Z-Library).pdf"  # 每次运行只需修改这个文件路径
# ========================================================

# 初始化客户端
api_key = "2sXVvUP14wFlJQjXw49W6PwdCYLpwKdo"
client = Mistral(api_key=api_key)

# 检查文件存在性
pdf_file = Path(PDF_PATH)
assert pdf_file.is_file(), f"文件 {PDF_PATH} 不存在"
pdf_stem = pdf_file.stem  # 自动获取文件名（不带扩展名）

# 上传并处理文件
uploaded_file = client.files.upload(
    file={
        "file_name": pdf_stem,
        "content": pdf_file.read_bytes(),
    },
    purpose="ocr",
)

signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

pdf_response = client.ocr.process(
    document=DocumentURLChunk(document_url=signed_url.url),
    model="mistral-ocr-latest",
    include_image_base64=False,
)


# 生成合并后的Markdown
def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"")
    return markdown_str


def get_combined_markdown(ocr_response: OCRResponse) -> str:
    markdowns: list[str] = []
    for page in pdf_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    return "\n\n".join(markdowns)


combined_markdown = get_combined_markdown(pdf_response)

# 保存结果（自动使用同文件名）
output_path = f"md/{pdf_stem}.md"  # 自动生成同名.md文件
with open(output_path, "w", encoding="utf-8") as f:
    f.write(combined_markdown)

print(f"处理完成！结果已保存至: {output_path}")
