from collections import defaultdict
from ragflow_sdk import RAGFlow as OfficialRAGFlow
from pydantic import BaseModel, Field
from typing import Optional, Union, Tuple
import gradio as gr
import requests
import json
import os
from datetime import datetime
import time

import logging
from typing import List, Dict, Optional, Generator
from pydantic import BaseModel
import re
import pandas as pd
import tempfile
from typing import List, Dict, Optional, Generator, Any, Union
import sqlite3
import re


def validate_latex(content: str) -> str:
    """自动修复常见 LaTeX 语法错误"""
    # 闭合未关闭的公式块
    content = re.sub(r"\\\[(.*?)(?=\\])", r"\[\1\]", content, flags=re.DOTALL)

    # 修正变量下标错误（如 a_r -> a_5）
    content = re.sub(r"a_([^0-9{])", r"a_{5}", content)

    # 移除多余分隔符
    content = content.replace("\#\#", "")  # 处理示例中的 ##4 等标记

    return content


# 在返回回答前调用

# 初始化配置

# 配置全局参数
API_KEY = "ragflow-Q0NDI5ODllMTNiNTExZjBhMWNjMDI0Mm"
BASE_URL = "http://workspace.featurize.cn:27455"
# BASE_URL = "http://127.0.0.1"
CHAT_ID = "f14d1f6a259811f098e60242ac120006"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "123456"
LOCAL_ENVIRONMENT = True

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

doc_list = gr.DataFrame(
    value=pd.DataFrame(columns=["ID", "文件名", "状态", "上传时间"]),
    type="pandas",
    label="文档列表",
    interactive=True,
)


def create_db():
    # 创建数据库（如果不存在）
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """
    )

    conn.commit()
    conn.close()


class ParserConfig(BaseModel):
    chunk_token_num: Optional[int] = None
    delimiter: Optional[str] = None
    html4excel: Optional[bool] = None
    layout_recognize: Optional[bool] = None
    raptor: Optional[Dict] = None
    entity_types: Optional[List[str]] = None


class DataSet(BaseModel):
    id: str
    name: str
    avatar: Optional[str] = Field(default="")  # 允许空值并设置默认值
    description: Optional[str] = Field(default="")
    embedding_model: str
    permission: str
    chunk_method: str
    parser_config: dict
    create_time: str


class Document(BaseModel):
    id: str
    display_name: str
    chunk_method: str
    parser_config: Optional[ParserConfig] = None


class RAGFlow:
    def __init__(self, api_key: str, base_url: str, chat_id: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.chat_id = chat_id
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.sdk = OfficialRAGFlow(api_key=api_key, base_url=base_url)

    def _get_question_by_id(self, question_id: str) -> tuple:
        """返回 (问题内容, 相关文档列表)"""
        try:
            cleaned_id = question_id.strip().replace("-", "_")
            keyword = f"problem_id: {cleaned_id}"
            dataset = self.sdk.list_datasets(name="高中代数数据集")[0]
            related_docs = []
            question_content = ""

            for document in dataset.list_documents():
                chunks = document.list_chunks(keywords=keyword)
                if chunks:
                    related_docs.append(document.name)
                    # 从第一个匹配的chunk提取问题内容
                    if not question_content:
                        clean_content = self.clean_html(chunks[0].content)
                        question_content = self._extract_question_from_content(
                            clean_content
                        )

            return question_content, related_docs
        except Exception as e:
            logger.error(f"获取问题失败: {e}")
            return "", []

    def clean_html(self, content: str) -> str:
        # 去掉HTML标签
        clean_content = re.sub(r"<.*?>", "", content)
        return clean_content

    def _extract_question_from_content(self, content: str) -> str:
        try:
            clean_content = self.clean_html(content)  # 清理HTML标签
            # 更通用的正则表达式，匹配 `question` 后的内容
            match = re.search(r"[Qq]uestion\s*(.*?)(?:[；;。]|$)", clean_content)
            if match:
                return match.group(1).strip()
        except Exception as e:
            logger.warning(f"提取 question 失败: {e}")
        return ""

    def get_reference_files(self, question_id: str) -> list:
        """返回 [问题内容，相关文档，检索结果]"""
        # 获取问题内容和关联文档
        question_content, related_docs = self._get_question_by_id(question_id)

        # 检索相关文档
        retrieved_chunks = self.retrieve(
            question=question_content,  # 使用提取的问题内容进行检索
            dataset_ids=["03d5fbe4259311f0870b0242ac120006"],
        )

        # 构建结果表格
        file_scores = {}
        for chunk in retrieved_chunks:
            file_name = (
                chunk.get("document_name", "")
                if isinstance(chunk, dict)
                else getattr(chunk, "document_name", "")
            )
            score = (
                chunk.get("score", 0)
                if isinstance(chunk, dict)
                else getattr(chunk, "score", 0)
            )
            if file_name:  # 添加文件名非空校验
                file_scores[file_name] = max(file_scores.get(file_name, 0), score)

        # 按分数排序后仅保留文件名
        sorted_files = [
            name
            for name, _ in sorted(
                file_scores.items(), key=lambda x: x[1], reverse=True
            )[
                :10
            ]  # 展示前10个结果
        ]

        # 构建二维数组格式（Gradio DataFrame要求）
        table_data = [[name] for name in sorted_files]

        return [
            ", ".join(related_docs) if related_docs else "未找到直接关联文档",
            table_data,
        ]

    def retrieve(
        self,
        question: str,
        dataset_ids: list[str],
        document_ids: list[str] = None,
        page: int = 1,
        page_size: int = 30,
        similarity_threshold: float = 0.2,
        vector_similarity_weight: float = 0.3,
        top_k: int = 1024,
        rerank_id: str = None,
        keyword: bool = False,
    ) -> list[dict]:
        processed_chunks = []  # 初始化 processed_chunks 以防止 UnboundLocalError
        try:
            # 调用 SDK 的 retrieve 方法
            chunks = self.sdk.retrieve(
                question=question,
                dataset_ids=dataset_ids,
                document_ids=document_ids,
                page=page,
                page_size=page_size,
                similarity_threshold=similarity_threshold,
                vector_similarity_weight=vector_similarity_weight,
                top_k=top_k,
                rerank_id=rerank_id,
                keyword=keyword,
            )

            if chunks:
                first_chunk = chunks[0]
                print("Chunk对象类型:", type(first_chunk))
                print("Chunk对象属性:", dir(first_chunk))
                if hasattr(first_chunk, "content"):
                    print("内容示例:", first_chunk.content[:50])

            # 转换 Chunk 对象为字典
            for chunk in chunks:
                # 确保 chunk.content 是字符串类型，避免 .lower() 错误
                chunk_content = (
                    str(chunk.content)
                    if not isinstance(chunk.content, str)
                    else chunk.content
                )

                # 直接通过对象访问属性
                file_name = getattr(
                    chunk, "document_name", "未知文件"
                )  # 使用 getattr 访问 document_name 属性

                chunk_data = {
                    "content": chunk_content,
                    "score": getattr(chunk, "score", 0.0),
                    "metadata": {
                        "file_name": file_name,  # 使用从 chunk 中获取的文件名
                        "problem_id": self._parse_problem_id(
                            chunk_content
                        ),  # 传入已处理的 chunk_content
                    },
                }
                processed_chunks.append(chunk_data)

            return processed_chunks

        except Exception as e:
            logger.error(f"检索失败：{str(e)}")

            # 异常时返回部分已处理的结果
            return [
                {
                    "序号": idx + 1,
                    "文件名": chunk_data["metadata"]["file_name"],
                    "内容摘要": chunk_data["content"][:50] + "...",  # 返回内容摘要
                    "相关度": chunk_data["score"],
                    "问题ID": chunk_data["metadata"]["problem_id"],
                }
                for idx, chunk_data in enumerate(processed_chunks)
            ]

    def _parse_problem_id(self, content: str) -> str:
        """从内容中提取问题ID"""
        match = re.search(r"problem_id\s*[:：]\s*(\d+)", content)
        return match.group(1) if match else ""

    def chat_completions(
        self,
        messages: List[Dict],
        model: str = "qwen2.5:3b",  # 指定默认模型
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Dict:
        """聊天补全接口"""
        endpoint = f"/api/v1/chats_openai/{self.chat_id}/chat/completions"
        url = f"{self.base_url}{endpoint}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            response = requests.post(
                url, headers=self.headers, json=payload, stream=stream, timeout=30
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream_response(response)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Chat completions error: {str(e)}")
            raise

    def _handle_stream_response(self, response) -> Generator[Dict, None, None]:
        """处理流式响应"""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    chunk = decoded_line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        yield json.loads(chunk)
                    except json.JSONDecodeError:
                        continue

    def get_related_questions(self, question: str, history: list = None) -> List[str]:
        """自主生成相关问题（通过大模型）"""
        try:
            # 构造提示词模板
            prompt = f"""你是一个专业的问题推荐引擎。根据以下对话历史和最新问题，生成3个用户可能关心的后续问题。
    要求：
    1. 问题需基于上下文且开放可讨论
    2. 使用中文提问
    3. 每个问题用数字编号开头
    4. 不要使用Markdown格式

    示例：
    用户问：机器学习是什么？
    相关问题：
    1. 监督学习和无监督学习有什么区别？
    2. 常见的机器学习算法有哪些？
    3. 如何评估模型性能？

    当前对话历史（最近3轮）：
    {self._format_history(history[-3:] if history else [])}

    最新问题：{question}
    相关问题："""

            # 调用聊天接口
            response = self.chat_completions(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # 提高创造性
                max_tokens=300,
            )

            # 解析生成结果
            return self._parse_questions(response["choices"][0]["message"]["content"])

        except Exception as e:
            logger.error(f"生成相关问题失败：{str(e)}")
            return []

    def _format_history(self, history: list) -> str:
        """格式化对话历史，只处理 dict 类型的历史记录"""
        result = []
        for item in history:
            # item 必须是 dict 且包含 'role' 和 'content'
            if isinstance(item, dict) and "role" in item and "content" in item:
                result.append(f"{item['role']}：{item['content']}")
            else:
                logger.warning(f"跳过无效历史项：{item}")
        return "\n".join(result)

    def _parse_questions(self, text: str) -> List[str]:
        """解析生成的文本"""
        questions = []
        for line in text.split("\n"):
            line = line.strip()
            # 支持多种编号格式：1. 1) • 等
            cleaned = re.sub(r"^(\d+[\.\)]?|[\•\-])\s*", "", line)
            if 10 <= len(cleaned) <= 100:  # 过滤有效问题
                questions.append(cleaned)
        return questions[:3]  # 最多返回3个

    # 参考文件接口
    # 数据集管理（完整实现）
    def create_dataset(
        self,
        name: str,
        chunk_method: str = "naive",
        embedding_model: str = "nomic-embed-text:latest",
        description: str = "",  # 新增描述参数
        permission: str = "me",  # 新增权限参数
        **kwargs,
    ) -> DataSet:
        """创建数据集"""
        payload = {
            "name": name,
            "avatar": kwargs.get("avatar", ""),
            "description": description,  # 使用传入参数
            "embedding_model": embedding_model,
            "permission": permission,  # 使用传入参数
            "chunk_method": chunk_method,
            "parser_config": self._get_parser_config(chunk_method, kwargs),
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/datasets", headers=self.headers, json=payload
            )
            response.raise_for_status()

            # 提取实际数据字段
            response_data = response.json().get("data", {})
            if not response_data:
                raise ValueError("服务端返回数据为空")

            return DataSet(
                id=response_data["id"],
                name=response_data["name"],
                embedding_model=response_data["embedding_model"],
                permission=response_data["permission"],
                chunk_method=response_data["chunk_method"],
                parser_config=response_data["parser_config"],
                create_time=datetime.fromtimestamp(
                    response_data["create_time"] / 1000
                ).isoformat(),
                avatar=response_data.get("avatar", ""),
                description=response_data.get("description", ""),
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Create dataset error: {str(e)}")
            raise

    def _get_parser_config(self, chunk_method: str, params: Dict) -> Dict:
        """生成解析器配置"""
        config_map = {
            "naive": {
                "chunk_token_num": 128,
                "delimiter": "\\n",
                "html4excel": False,
                "layout_recognize": True,
                "raptor": {"user_raptor": False},
            },
            "knowledge_graph": {
                "chunk_token_num": 128,
                "delimiter": "\\n",
                "entity_types": ["organization", "person", "location", "event", "time"],
            },
            # 其他分块方法的默认配置...
        }
        return params.get("parser_config", config_map.get(chunk_method, {}))

    def delete_datasets(self, dataset_ids):
        """删除数据集"""
        url = f"{self.base_url}/api/v1/datasets"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # 请求体格式
        payload = {"ids": dataset_ids}

        # 发送DELETE请求
        response = requests.delete(url, headers=headers, json=payload)

        # 判断是否删除成功
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 0:
                return True  # 删除成功
            else:
                raise Exception(f"删除失败: {data.get('message', '未知错误')}")
        else:
            raise Exception(f"请求失败: {response.status_code}, {response.text}")

    def update_dataset(self, dataset_id: str, update_data: Dict) -> DataSet:
        """更新数据集配置"""
        try:
            response = requests.put(
                f"{self.base_url}/api/v1/datasets/{dataset_id}",
                headers=self.headers,
                json=update_data,
            )
            response.raise_for_status()
            return DataSet(**response.json()["data"])
        except requests.exceptions.RequestException as e:
            logger.error(f"Update dataset error: {str(e)}")
            raise

    def get_dataset_info(self, dataset_id: str) -> dict:
        """获取单个数据集详情（修复data类型检查）"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/datasets/{dataset_id}", headers=self.headers
            )
            response.raise_for_status()
            response_json = response.json()

            # 添加响应结构验证
            if "data" not in response_json:
                logger.error("响应缺少data字段")
                return {}

            data = response_json["data"]

            # 类型检查
            if not isinstance(data, dict):
                logger.error(f"数据格式错误，期望字典，实际得到：{type(data)}")
                return {}

            return {
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "embedding_model": data.get("embedding_model", ""),
                "chunk_method": data.get("chunk_method", ""),
            }
        except Exception as e:
            logger.error(f"获取数据集信息失败: {str(e)}")
            return {}

    def _delete_selected_datasets(self, dataset_table):
        """从表格中提取选中的数据集并调用后端接口删除"""
        try:
            # 表头是 ["选中", "ID", "名称", "描述", "嵌入模型", "创建时间"]
            selected_ids = [
                row[1]  # ID在第2列（索引1）
                for row in dataset_table
                if str(row[0]).strip().lower() in ["true", "1", "yes"]  # 选中列
            ]
            if not selected_ids:
                raise gr.Error("请至少选择一个要删除的数据集。")

            self.delete_datasets(selected_ids)
            return self._refresh_datasets()
        except Exception as e:
            logger.error(f"删除数据集失败: {e}")
            raise gr.Error(f"删除失败：{e}")

    def wait_for_document_ready(
        self, dataset_id: str, document_id: str, timeout: int = 300
    ) -> bool:
        """等待文档状态变成 UNSTART，可以开始解析"""
        for i in range(timeout):
            docs = self.list_documents(dataset_id, document_id=document_id)["docs"]
            if not docs:
                logger.debug(f"文档{document_id}还没查到，继续等待...")
                time.sleep(1)
                continue
            run_status = docs[0].get("run", "")
            logger.debug(f"当前文档{document_id} run状态: {run_status}")
            print(f"Timeout left: {timeout - i} second(s).")
            if run_status in ("UNSTART", "DONE"):
                return True
            time.sleep(1)
        logger.error(f"文档{document_id}在{timeout}秒内未准备好(run状态={run_status})")
        return False

    def list_datasets(
        self,
        page: int = 1,
        page_size: int = 30,
        orderby: str = "create_time",
        desc: bool = True,
        id: str = None,
        name: str = None,
    ) -> List[DataSet]:
        """列出数据集"""
        params = {
            "page": page,
            "page_size": page_size,
            "orderby": orderby,
            "desc": str(desc).lower(),
            "id": id,
            "name": name,
        }

        try:
            response = requests.get(
                f"{self.base_url}/api/v1/datasets",
                headers=self.headers,
                params={k: v for k, v in params.items() if v is not None},
            )
            response.raise_for_status()
            return [
                DataSet(
                    **{
                        **item,
                        "create_time": datetime.fromtimestamp(
                            item["create_time"] / 1000
                        ).isoformat(),  # 转换时间戳
                        "avatar": item.get("avatar", ""),
                        "description": item.get("description", ""),
                    }
                )
                for item in response.json().get("data", [])
            ]
        except requests.exceptions.RequestException as e:
            logger.error(f"List datasets error: {str(e)}")
            raise

    def _refresh_datasets(self):
        """拉取数据集并返回展示结构"""
        try:
            datasets = self.list_datasets()
            table_data = []
            for ds in datasets:
                table_data.append(
                    [
                        False,  # 默认未选中
                        ds.id,
                        ds.name,
                        ds.description,
                        ds.embedding_model,
                        ds.create_time,
                    ]
                )
            return table_data
        except Exception as e:
            logger.error(f"刷新数据集失败: {e}")
            raise gr.Error(f"刷新失败: {e}")

    def update_dataset(self, dataset_id: str, update_data: Dict) -> DataSet:
        """更新数据集（修复create_time处理）"""
        try:
            response = requests.put(
                f"{self.base_url}/api/v1/datasets/{dataset_id}",
                headers=self.headers,
                json=update_data,
            )
            response.raise_for_status()
            response_json = response.json()

            # 验证响应结构
            if "data" not in response_json:
                logger.error("响应缺少data字段")
                raise ValueError("无效的API响应")

            data = response_json["data"]

            # 处理可能缺失的字段
            create_time = data.get("create_time", datetime.now().timestamp() * 1000)
            data["create_time"] = datetime.fromtimestamp(create_time / 1000).isoformat()

            return DataSet(**data)
        except Exception as e:
            logger.error(f"更新数据集失败: {str(e)}")
            raise

    def upload_documents(
        self, dataset_id: str, files: List[Tuple[str, bytes]]
    ) -> List[dict]:
        try:
            # 添加调试日志
            logger.debug(f"▼▼▼ 开始上传文档 ▼▼▼")
            logger.debug(f"目标数据集: {dataset_id}")
            logger.debug(f"文件数量: {len(files)}")

            for i, (filename, content) in enumerate(files):
                logger.debug(
                    f"文件#{i + 1}: {filename} ({len(content)} bytes) | 前16字节: {content[:16].hex()}"
                )

            # 构建请求
            file_fields = [
                ("files", (filename, content, "application/octet-stream"))
                for filename, content in files
            ]

            # 记录原始请求信息
            logger.debug(
                f"请求URL: {self.base_url}/api/v1/datasets/{dataset_id}/documents"
            )
            logger.debug(
                f"请求头: { {'Authorization': f'Bearer [REDACTED]'} }"
            )  # 隐藏真实API Key

            response = requests.post(
                f"{self.base_url}/api/v1/datasets/{dataset_id}/documents",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=file_fields,
                timeout=30,
            )

            # 记录响应详情
            logger.debug(f"▲▲▲ 上传响应 ▲▲▲")
            logger.debug(f"状态码: {response.status_code}")
            logger.debug(f"响应头: {dict(response.headers)}")
            logger.debug(f"响应内容: {response.text[:500]}...")  # 截断长内容

            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            logger.exception("上传发生致命错误")
            raise gr.Error(f"上传失败: {str(e)}")

    def list_documents(
        self,
        dataset_id: str,
        page: int = 1,
        page_size: int = 30,
        keywords: str = None,
        document_id: str = None,
    ) -> dict:
        """查询文档（根据API 2.4）"""
        try:
            params = {
                "page": page,
                "page_size": page_size,
                "keywords": keywords,
                "id": document_id,
            }

            response = requests.get(
                f"{self.base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=self.headers,
                params={k: v for k, v in params.items() if v is not None},
            )
            response.raise_for_status()

            # 转换时间格式和状态
            data = response.json().get("data", {})

            for doc in data.get("docs", []):

                doc["create_time"] = datetime.fromtimestamp(
                    doc["create_time"] / 1000
                ).strftime("%Y-%m-%d %H:%M:%S")
                doc["status"] = self._map_status(doc.get("run", ""))

            return {"docs": data.get("docs", []), "total": data.get("total", 0)}
        except Exception as e:
            logger.error(f"查询文档失败: {str(e)}")
            return {"docs": [], "total": 0}

    def _map_status(self, run_status: str) -> str:
        """映射文档状态"""
        status_map = {
            "UNSTART": "🟡 未开始",
            "RUNNING": "🔄 解析中",
            "DONE": "✅ 已完成",
            "FAILED": "❌ 失败",
        }
        return status_map.get(run_status, "❓ 未知状态")

    def delete_documents(self, dataset_id: str, document_ids: List[str]) -> bool:
        """删除文档（根据API 2.5）"""
        try:
            response = requests.delete(
                f"{self.base_url}/api/v1/datasets/{dataset_id}/documents",
                headers=self.headers,
                json={"ids": document_ids},
            )
            return response.json().get("code", 1) == 0
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False

    def parse_documents(self, dataset_id: str, document_ids: List[str]) -> bool:
        """解析文档（根据API 2.6）"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/datasets/{dataset_id}/chunks",
                headers=self.headers,
                json={"document_ids": document_ids},
            )
            # DYC 新加入调试信息查看错误
            logger.info(f"解析接口返回: {response.status_code}, {response.text}")
            if response.status_code == 200:
                logger.info("解析请求成功")
                return True
            return response.json().get("code", 1) == 0
        except Exception as e:
            logger.error(f"解析文档失败: {str(e)}")
            return False

    # 文档管理功能

    def cancel_parse(self, dataset_id: str, document_ids: List[str]) -> None:
        """取消文档解析"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/datasets/{dataset_id}/chunks/cancel",
                headers=self.headers,
                json={"document_ids": document_ids},
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Cancel parsing error: {str(e)}")
            raise

    def _update_doc_list(self, dataset_id):
        """更新文档列表"""
        if not dataset_id:
            return pd.DataFrame(columns=["ID", "文件名", "状态", "上传时间"])

        documents = self.list_documents(dataset_id)

        return pd.DataFrame(
            [
                {
                    "ID": doc.id,
                    "文件名": doc.display_name,
                    "状态": doc.parser_config.get("run", "未知"),
                    "上传时间": self._format_timestamp(
                        doc.parser_config.get("create_time")
                    ),
                }
                for doc in documents
            ]
        )


custom_css = f"""
/* 主色调 */
:root {{
  --primary: #F8F9FA;         /* 主背景 */
  --secondary: #FFFFFF;      /* 卡片背景 */
  --accent: #6C63FF;         /* 强调色-紫罗兰 */
  --text: #2D3436;           /* 正文颜色 */
  --gradient: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
}}

* {{
    font-family: "Cambria Math", "Times New Roman", serif;
}}

body {{
    font-family: "Cambria Math", "Times New Roman", serif;
}}

/* 中文字体设置 */
@font-face {{
    font-family: "STSong";
    src: local("华文中宋");
}}

html, body {{
    font-family: "Cambria Math", "Times New Roman", "STSong", serif;
}}

/* 中文文本使用华文中宋 */
* {{
    font-family: "STSong", "Cambria Math", "Times New Roman", serif;
}}

/* 数学公式字体设置为 Cambria Math */
.mathjax {{
    font-family: "Cambria Math", "Times New Roman", serif;
}}

.main-container {{
  background: var(--gradient);
  min-height: 100vh;
  padding: 2rem;
}}

.contact-panel {{
  background: var(--secondary);
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
  padding: 1.5rem;
  margin-right: 1rem;
}}

.chat-panel {{
  background: var(--secondary);
  border-radius: 12px;
  box-shadow: 0 4px 30px rgba(108, 99, 255, 0.05);
  padding: 2rem;
}}

.message-flow {{
  min-height: 70vh;
  max-height: 70vh;
  overflow-y: auto;
  padding: 1rem;
  background: var(--primary);
  border-radius: 8px;
}}

.input-area {{
  margin-top: 1.5rem;
  position: relative;
}}

.message-bubble.user {{
  background: var(--accent);
  color: white;
  border-radius: 18px 18px 4px 18px;
}}

.message-bubble.bot {{
  background: var(--secondary);
  border: 1px solid #E9ECEF;
  border-radius: 18px 18px 18px 4px;
}}

button:hover {{
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(108, 99, 255, 0.2);
}}

.input-area::before {{
  content: "";
  position: absolute;
  top: -15px;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent) 0%, rgba(108,99,255,0) 100%);
}}

#problem-id-input {{
    background: #f8f9fa;
    border: 2px solid #6C63FF;
    border-radius: 8px;
    padding: 12px;
}}

button#problem-id-search {{
    transition: all 0.3s ease;
}}
button#problem-id-search:hover {{
    transform: scale(1.05);
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
}}

.result-table {{
    margin-top: 20px;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}}
.result-table th {{
    background: #6C63FF !important;
    color: white !important;
}}
.sidebar {{
    background-image: url("https://raw.githubusercontent.com/Yu-chen-Deng/yu-chen-deng.github.io/refs/heads/main/Du/img/sidebar-bg.png");
}}
"""


class ChatInterface:
    def __init__(self, rag_client: RAGFlow):
        self.client = rag_client
        self.custom_css = custom_css
        self.login_interface = None
        self.main_interface = None
        self.guest_interface = None
        self.register_interface = None
        self.user_state = None
        self.login_status = None

    def create_interface(self):
        with gr.Blocks(
            theme=gr.themes.Monochrome(), css=self.custom_css, title="智能问答系统"
        ) as demo:
            self.user_state = gr.State(value={"logged_in": False, "role": None})

            # === 登录界面 ===
            with gr.Column(visible=True) as self.login_interface:
                gr.Markdown("# 🤖 KAG - 登录")

                username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                password = gr.Textbox(
                    label="密码", type="password", placeholder="请输入密码"
                )

                with gr.Row():
                    login_btn = gr.Button("登录", variant="primary")
                    guest_btn = gr.Button("游客登录", variant="secondary")
                    register_btn = gr.Button("注册", variant="secondary")

                self.login_status = gr.Markdown("")

            # === 注册界面 ===
            with gr.Column(visible=False) as self.register_interface:
                gr.Markdown("# ✨ 注册新账号")

                new_username = gr.Textbox(label="用户名", placeholder="请输入新用户名")
                new_password = gr.Textbox(
                    label="密码", type="password", placeholder="请输入新密码"
                )
                confirm_password = gr.Textbox(
                    label="确认密码", type="password", placeholder="请确认密码"
                )

                register_status = gr.Markdown("")

                # 定义注册用户函数
                def register_user(username, password, confirm_password):
                    if password != confirm_password:
                        return gr.update(), "❌ 两次密码不一致。"

                    conn = sqlite3.connect("users.db")
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT * FROM users WHERE username = ?", (username,)
                    )
                    if cursor.fetchone():
                        conn.close()
                        return gr.update(), "❌ 用户名已存在，请选择其他用户名。"

                    cursor.execute(
                        "INSERT INTO users (username, password) VALUES (?, ?)",
                        (username, password),
                    )
                    conn.commit()
                    conn.close()
                    return gr.update(visible=True), "✅ 注册成功，您可以登录了！"

                register_btn.click(
                    fn=register_user,
                    inputs=[new_username, new_password, confirm_password],
                    outputs=[self.login_interface, register_status],
                )

                back_to_login_btn = gr.Button("返回登录", variant="secondary")
                back_to_login_btn.click(
                    fn=lambda: (
                        gr.update(visible=True),  # 显示登录界面
                        gr.update(visible=False),  # 隐藏注册界面
                    ),
                    inputs=[],
                    outputs=[self.login_interface, self.register_interface],
                )

            # === 主界面 ===
            with gr.Column(visible=False) as self.main_interface:
                gr.Markdown("# 🤖 KAG")

                with gr.Sidebar(position="left", elem_classes=["sidebar"]) as sidebar:
                    gr.Markdown("# 🧭 功能导航")
                    tab_select = gr.Radio(
                        choices=["智能问答", "参考文件查询", "数据集管理", "文档管理"],
                        label="请选择模块",
                        value="智能问答",
                    )

                # 子页面容器
                chat_tab = gr.Column(visible=True)
                reference_tab = gr.Column(visible=False)
                dataset_tab = gr.Column(visible=False)
                document_tab = gr.Column(visible=False)

                with chat_tab:
                    self._build_chat_interface()
                with reference_tab:
                    self._build_reference_interface()
                with dataset_tab:
                    self._build_dataset_interface()
                with document_tab:
                    self._build_document_interface()

                # Tab切换
                def switch_tab(selected):
                    return (
                        gr.update(visible=selected == "智能问答"),
                        gr.update(visible=selected == "参考文件查询"),
                        gr.update(visible=selected == "数据集管理"),
                        gr.update(visible=selected == "文档管理"),
                    )

                tab_select.change(
                    fn=switch_tab,
                    inputs=tab_select,
                    outputs=[chat_tab, reference_tab, dataset_tab, document_tab],
                )

            # === 简化版主界面（游客） ===
            with gr.Column(visible=False) as self.guest_interface:
                gr.Markdown("# 🙋 游客智能问答")
                self._build_chat_interface(minimal=True)  # 用不同参数区分精简版

            # === 登录回调逻辑 ===
            def admin_login(username, password):
                if username == "admin" and password == "123456":
                    return (
                        gr.update(visible=False),  # 隐藏登录
                        gr.update(visible=True),  # 显示主界面
                        gr.update(visible=False),  # 隐藏游客界面
                        {"logged_in": True, "role": "admin"},
                        "✅ 登录成功，欢迎管理员！",
                    )

                if not username or not password:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "❌ 用户名和密码不能为空。",
                    )

                # 如果是数据库中的用户，验证用户名和密码
                conn = sqlite3.connect("users.db")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM users WHERE username = ? AND password = ?",
                    (username, password),
                )
                user = cursor.fetchone()

                conn.close()

                if user:
                    # 如果用户名和密码匹配，登录成功
                    return (
                        gr.update(visible=False),  # 隐藏登录界面
                        gr.update(visible=True),  # 显示主界面
                        gr.update(visible=False),  # 隐藏游客界面
                        {"logged_in": True, "role": "user"},  # 设置角色为普通用户
                        f"✅ 登录成功，欢迎 {username}！",
                    )
                else:
                    # 如果用户名或密码错误，返回错误信息
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "❌ 登录失败，用户名或密码错误。",
                    )

            def guest_login():
                return (
                    gr.update(visible=False),  # 隐藏登录
                    gr.update(visible=False),  # 隐藏主界面
                    gr.update(visible=True),  # 显示游客界面
                    {"logged_in": True, "role": "guest"},
                    "🎉 已以游客身份登录",
                )

            login_btn.click(
                fn=admin_login,
                inputs=[username, password],
                outputs=[
                    self.login_interface,
                    self.main_interface,
                    self.guest_interface,
                    self.user_state,
                    self.login_status,
                ],
            )

            guest_btn.click(
                fn=guest_login,
                inputs=[],
                outputs=[
                    self.login_interface,
                    self.main_interface,
                    self.guest_interface,
                    self.user_state,
                    self.login_status,
                ],
            )

            register_btn.click(
                fn=lambda: (
                    gr.update(visible=False),  # 隐藏登录界面
                    gr.update(visible=True),  # 显示注册界面
                ),
                inputs=[],
                outputs=[self.login_interface, self.register_interface],
            )

        return demo

    def _handle_return_to_login(self):
        """处理返回登录界面的逻辑"""
        return (
            gr.update(visible=True),  # 显示登录界面
            gr.update(visible=False),  # 隐藏主界面
            gr.update(visible=False),  # 隐藏游客界面
            {"logged_in": False, "role": None},  # 重置用户状态
            "",
        )

    def _build_reference_interface(self):
        """重构后的参考文件查询界面"""
        with gr.Row():
            with gr.Column(scale=4):
                # 新增问题ID输入区域
                with gr.Group():
                    problem_id_input = gr.Textbox(
                        label="请输入问题ID",
                        placeholder="例如：1",
                        max_lines=1,
                        elem_id="problem-id-input",
                    )
                    search_btn = gr.Button("开始检索", variant="primary")

                # 结果展示表格
                result_table = gr.DataFrame(
                    headers=["文档名称"],
                    datatype=["str"],
                    interactive=False,
                    elem_classes=["result-table"],
                    label="相关文档列表",
                )

            # 右侧信息展示（可选）
            with gr.Column(scale=1):
                gr.Markdown("### 使用说明")
                gr.Markdown(
                    """
                1. 在输入框输入完整的问题ID
                2. 点击「开始检索」按钮
                3. 系统将展示与该问题相关的所有文档
                """
                )

        # 事件绑定
        search_btn.click(
            fn=self.client.get_reference_files,
            inputs=problem_id_input,
            outputs=result_table,
        )
        problem_id_input.submit(
            fn=self.client.get_reference_files,
            inputs=problem_id_input,
            outputs=result_table,
        )

    def _build_chat_interface(self, minimal=False):
        """调整后的聊天界面（移除原右侧问题ID输入）"""
        if minimal:
            with gr.Row():

                # 左侧聊天区域（移除问题ID相关组件）
                with gr.Column(scale=3):
                    with gr.Row():
                        return_btn = gr.Button(
                            "返回登录", variant="secondary", size="sm"
                        )
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        elem_classes=["chat-history"],
                        height=600,
                        type="messages",
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                        ],
                        render_markdown=True,
                    )
                    msg = gr.Textbox(placeholder="请输入您的问题...", lines=3)

                    with gr.Row():
                        submit_btn = gr.Button("发送", variant="primary")
                        clear_btn = gr.Button("清空历史")

                    with gr.Accordion("高级设置", open=False):
                        model_select = gr.Dropdown(
                            choices=["qwen2.5:3b"], value="qwen2.5:3b"
                        )
                        max_tokens = gr.Slider(100, 1000, value=300, label="最大长度")
                        temperature = gr.Slider(0, 2, value=0.7, label="随机性")

                    submit_btn.click(
                        self._respond,
                        inputs=[msg, chatbot, model_select, max_tokens, temperature],
                        outputs=[msg, chatbot],  # 移除reference_files输出
                    )
                    clear_btn.click(lambda: ([], [], [], []), outputs=[chatbot, msg])
                    return_btn.click(
                        fn=self._handle_return_to_login,  # 返回登录界面
                        inputs=[],
                        outputs=[
                            self.login_interface,
                            self.main_interface,
                            self.guest_interface,
                            self.user_state,
                            self.login_status,
                        ],
                    )

        else:
            with gr.Row():
                # 左侧聊天区域（移除问题ID相关组件）
                with gr.Column(scale=3):
                    with gr.Row():
                        return_btn = gr.Button(
                            "返回登录", variant="secondary", size="sm"
                        )
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        elem_classes=["chat-history"],
                        height=600,
                        type="messages",
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                        ],
                        render_markdown=True,
                    )
                    msg = gr.Textbox(placeholder="请输入您的问题...", lines=3)

                    with gr.Row():
                        submit_btn = gr.Button("发送", variant="primary")
                        clear_btn = gr.Button("清空历史")

                    with gr.Accordion("高级设置", open=False):
                        model_select = gr.Dropdown(
                            choices=["qwen2.5:14b"], value="qwen2.5:14b"
                        )
                        max_tokens = gr.Slider(100, 1000, value=300, label="最大长度")
                        temperature = gr.Slider(0, 2, value=0.7, label="随机性")

                # 右侧面板现在只显示相关问题
                with gr.Column(scale=1, elem_classes=["info-panel"]):
                    related_questions = gr.JSON(label="相关问题推荐")

            # 调整事件绑定
            submit_btn.click(
                self._respond,
                inputs=[msg, chatbot, model_select, max_tokens, temperature],
                outputs=[msg, chatbot, related_questions],  # 移除reference_files输出
            )
            clear_btn.click(
                lambda: ([], [], [], []), outputs=[chatbot, related_questions, msg]
            )

    def _respond(self, message, history, model, max_tokens, temperature):
        try:
            # 生成回答
            answer = self._get_answer(message, history, model, max_tokens, temperature)

            # 生成相关问题
            related_questions = self.client.get_related_questions(message, history)

            # 转换为Gradio要求的消息格式（统一字典格式）
            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer},
            ]

            return "", new_history, related_questions
        except Exception as e:  # 修复冒号缺失
            error_msg = f"系统错误：{str(e)}"
            logger.exception(error_msg)
            return "", history, []

    def _get_answer(self, message, history, model, max_tokens, temperature):
        """获取聊天回复"""
        try:
            response = self.client.chat_completions(
                messages=self._format_messages(message, history),
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            logger.info(f"LLM 返回: {response}")  # 加这一行查看真实结构
            #  检查结构
            if not response or "choices" not in response:
                raise ValueError(f"响应中无 'choices': {response}")

            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"请求异常：{str(e)}")
            return "⚠️ 抱歉，模型响应失败。"

    # 旧格式适配方式（如果 history 是 [(问题, 答案)]）
    def _format_messages(self, message, history):
        """兼容旧版元组格式和字典格式"""
        messages = []
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append(item)  # 直接使用字典格式
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                # 将旧版元组转为字典
                messages.extend(
                    [
                        {"role": "user", "content": item[0]},
                        {"role": "assistant", "content": item[1]},
                    ]
                )
        messages.append({"role": "user", "content": message})
        return messages

    # def _delete_selected_datasets(self, dataset_table):
    #     """从表格中提取选中的数据集并调用后端接口删除"""
    #     print(dataset_table)
    #     try:
    #         # 表头是 ["选中", "ID", "名称", "描述", "嵌入模型", "创建时间"]
    #         selected_ids = [
    #             row[1]  # ID在第2列（索引1）
    #             for row in dataset_table
    #             if str(row[0]).strip().lower() in ["true", "1", "yes"]  # 选中列
    #         ]
    #         if not selected_ids:
    #             raise gr.Error("请至少选择一个要删除的数据集。")

    #         self.delete_datasets(selected_ids)
    #         return self._refresh_datasets()
    #     except Exception as e:
    #         logger.error(f"删除数据集失败: {e}")
    #         raise gr.Error(f"删除失败：{e}")

    def wait_for_document_ready(
        self, dataset_id: str, document_id: str, timeout: int = 30
    ) -> bool:
        """等待文档状态变成 UNSTART，可以开始解析"""
        for i in range(timeout):
            docs = self.client.list_documents(dataset_id, document_id=document_id)[
                "docs"
            ]
            if not docs:
                logger.debug(f"文档{document_id}还没查到，继续等待...")
                time.sleep(1)
                continue
            run_status = docs[0].get("run", "")
            logger.debug(f"当前文档{document_id} run状态: {run_status}")
            print(f"Timeout left: {timeout - i} second(s).")
            if run_status in ("UNSTART", "DONE"):
                return True
            time.sleep(1)
        logger.error(f"文档{document_id}在{timeout}秒内未准备好(run状态={run_status})")
        return False

    def _delete_selected_datasets(self, selected_names):
        # 通过名称映射到完整数据集对象
        all_datasets = self.client.list_datasets()
        selected_datasets = [ds for ds in all_datasets if ds.name in selected_names]

        # 原删除逻辑（假设需要数据集ID）
        dataset_ids = [ds.id for ds in selected_datasets]
        print(dataset_ids)
        self.client.delete_datasets(dataset_ids)

        # 返回更新后的列表
        return self._refresh_datasets()

    def _refresh_datasets(self):
        """拉取数据集并返回展示结构"""
        try:
            datasets = self.client.list_datasets()
            table_data = []
            for ds in datasets:
                table_data.append(
                    [
                        True,  # 默认未选中
                        ds.id,
                        ds.name,
                        ds.description,
                        ds.embedding_model,
                        ds.create_time,
                    ]
                )
            return table_data
        except Exception as e:
            logger.error(f"刷新数据集失败: {e}")
            raise gr.Error(f"刷新失败: {e}")

    def _refresh_dataset_options(self):
        """返回所有数据集，用于刷新下拉列表或复选框"""
        try:
            datasets = self.client.list_datasets()
            return [f"{ds.name}（{ds.id}）" for ds in datasets if ds.id]
        except Exception as e:
            logger.error(f"刷新数据集列表失败: {str(e)}")
            return self._get_dataset_choices()

    def _get_dataset_choices(self):
        datasets = self.vector_db.list_datasets()
        return [f"{ds['name']}（{ds['id']}）" for ds in datasets]

    def _delete_selected_datasets_by_ids(self, selected_list):
        """通过复选框选择的显示文本删除对应数据集"""
        try:
            ids = [
                item.split("（")[-1].strip("）") for item in selected_list
            ]  # 提取 ID
            if ids:
                self.client.delete_datasets(ids)
            return self._refresh_dataset_options()
        except Exception as e:
            logger.error(f"删除数据集失败: {str(e)}")
            return []

    def _create_dataset_and_refresh(
        self, name, description, chunk_method, embedding_model
    ):
        """创建数据集并刷新列表"""
        try:
            self.client.create_dataset(
                name=name,
                description=description,
                chunk_method=chunk_method,
                embedding_model=embedding_model,
            )
        except Exception as e:
            logger.error(f"创建失败: {str(e)}")
        return self._refresh_dataset_options()

    def _build_dataset_interface(self):
        """构建数据集管理界面"""
        with gr.Row():
            # 创建数据集表单
            with gr.Column(scale=2):
                with gr.Column(variant="panel"):
                    gr.Markdown("## 创建新数据集")
                    ds_name = gr.Textbox(label="数据集名称")
                    ds_description = gr.Textbox(label="数据集描述")
                    chunk_method = gr.Dropdown(
                        choices=["naive", "knowledge_graph"],
                        label="分块方法",
                        value="naive",
                    )
                    embedding_model = gr.Dropdown(
                        choices=["nomic-embed-text:latest"],
                        label="嵌入模型",
                        value="nomic-embed-text:latest",
                    )
                    create_btn = gr.Button("创建数据集", variant="primary")

                # 数据集列表及操作
                with gr.Column(scale=3):

                    dataset_list = gr.DataFrame(  # 数据集列表
                        label="数据集列表",
                        headers=["选中", "ID", "名称", "描述", "嵌入模型", "创建时间"],
                        datatype=["bool", "str", "str", "str", "str", "str"],
                        interactive=True,
                        row_count=5,
                        col_count=6,
                    )

                    # 操作按钮组
                    with gr.Column(scale=0.5):
                        refresh_btn = gr.Button("刷新数据集列表")
                        # 原 refresh_btn 后添加：
                        dataset_options = [
                            ds.name for ds in self.client.list_datasets()
                        ]
                        dataset_selector = gr.CheckboxGroup(
                            choices=dataset_options, label="选择要删除的数据集"
                        )
                        delete_btn = gr.Button("删除选中的数据集", variant="stop")

            # 数据集修改表单
            with gr.Column(scale=2):
                gr.Markdown("## 修改数据集")
                dataset_id_input = gr.Textbox(label="数据集ID")
                with gr.Row():

                    update_btn = gr.Button("提交修改", variant="primary")
                update_name = gr.Textbox(label="新名称")
                update_description = gr.Textbox(label="新描述")
                update_embedding_model = gr.Dropdown(
                    label="嵌入模型",
                    choices=["nomic-embed-text:latest"],
                    value="nomic-embed-text:latest",
                )
                update_chunk_method = gr.Dropdown(
                    label="分块方法",
                    choices=["naive", "knowledge_graph"],
                    value="naive",
                )

        # 事件绑定
        create_btn.click(
            self._create_dataset_and_refresh,
            inputs=[ds_name, ds_description, chunk_method, embedding_model],
            outputs=[dataset_list],
        )

        refresh_btn.click(
            self._refresh_datasets, inputs=[], outputs=[dataset_list]  # 刷新数据集列表
        )

        delete_btn.click(
            self._delete_selected_datasets,
            inputs=[dataset_selector],
            outputs=[dataset_list],
        )

        update_btn.click(
            self._update_dataset_info,
            inputs=[
                dataset_id_input,
                update_name,
                update_description,
                update_embedding_model,
                update_chunk_method,
            ],
            outputs=[dataset_list],
        )

    def _create_dataset_and_refresh(
        self, name, description, chunk_method, embedding_model
    ):
        """创建数据集并刷新列表"""
        self.client.create_dataset(
            name=name,
            description=description,
            chunk_method=chunk_method,
            embedding_model=embedding_model,
        )
        return self._format_datasets()

    def _refresh_datasets(self, page=1, page_size=30):
        return self._format_datasets()

    def _format_datasets(self):
        datasets = self.client.list_datasets()
        data = []
        for ds in datasets:
            data.append(
                [
                    False,  # 选中
                    ds.id,
                    ds.name,
                    ds.description or "",
                    ds.embedding_model,
                    ds.create_time,
                ]
            )
        return pd.DataFrame(
            data, columns=["选中", "ID", "名称", "描述", "嵌入模型", "创建时间"]
        )

    # 刷新数据集列表

    def _load_dataset_info(self, dataset_id):
        """加载数据集信息到表单"""
        info = self.client.get_dataset_info(dataset_id)
        return [
            info.get("name", ""),
            info.get("description", ""),
            info.get("embedding_model", ""),
            info.get("chunk_method", ""),
        ]

    def _docs_to_table(self, docs: List[dict]):
        return pd.DataFrame(
            [
                {
                    "ID": doc.get("id"),
                    "文件名": doc.get("name") or doc.get("display_name"),
                    "状态": doc.get("status", "未知"),
                    "创建时间": doc.get("create_time"),
                    "大小": round(doc.get("size", 0) / 1024, 2),
                }
                for doc in docs
            ]
        )

    def _build_document_interface(self):
        """重构后的文档管理界面"""
        with gr.Column():
            with gr.Row():
                # 数据集选择
                dataset_selector = gr.Dropdown(
                    label="📁 选择知识库",
                    choices=[],  # 初始空，或者给一批默认数据
                    interactive=True,
                    scale=2,
                )

                # 操作按钮组
                with gr.Row(scale=1):
                    refresh_btn = gr.Button("🔄 刷新文档", variant="secondary")
                    refresh_btn2 = gr.Button("🔄 刷新知识库", variant="secondary")
                    parse_btn = gr.Button("🔧 解析文档", variant="primary")
                    delete_btn = gr.Button("🗑️ 删除文档", variant="stop")

            # 文档上传区域
            with gr.Column(variant="panel"):
                gr.Markdown("## 📤 上传文档")
                file_upload = gr.File(
                    label="选择文件（支持多选）",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".xlsx", ".txt", ".md"],
                    height=150,
                )

                upload_btn = gr.Button("🚀 开始上传", variant="primary")

            # 文档列表
            doc_list = gr.DataFrame(
                headers=["ID", "文件名", "状态", "创建时间", "大小"],
                datatype=["str", "str", "str", "str", "number"],
                interactive=False,
                wrap=True,
            )

            # ========== 封装异常处理逻辑 ==========
            def safe_upload(ds_id, files):
                try:
                    return self._handle_upload(ds_id, files)
                except Exception as e:
                    raise gr.Error(f"上传失败：{str(e)}")

            def safe_delete(ds_id, df):
                try:
                    return self._handle_delete(ds_id, df)
                except Exception as e:
                    raise gr.Error(f"删除失败：{str(e)}")

            # ========== 事件绑定 ==========
            dataset_selector.change(
                fn=lambda ds_id: self._docs_to_table(
                    self.client.list_documents(ds_id)["docs"]
                ),
                inputs=dataset_selector,
                outputs=doc_list,
            ).then(lambda: gr.Info("已切换知识库"), None, None)

            upload_btn.click(
                fn=safe_upload, inputs=[dataset_selector, file_upload], outputs=doc_list
            ).then(lambda: gr.Info("上传成功！"), None, None)

            def refresh_doc_list(ds_id):
                docs = self.client.list_documents(ds_id)["docs"]
                return self._docs_to_table(docs)

            def refresh_dataset_options():
                options = self._get_dataset_options()
                return gr.update(choices=options)

            refresh_btn.click(
                fn=refresh_doc_list,
                inputs=dataset_selector,
                outputs=doc_list,
            ).then(lambda: gr.Info("已刷新文档列表"), None, None)
            refresh_btn2.click(
                fn=refresh_dataset_options,
                inputs=[],
                outputs=[dataset_selector],
            )
            delete_btn.click(
                fn=safe_delete, inputs=[dataset_selector, doc_list], outputs=doc_list
            ).then(lambda: gr.Info("删除成功！"), None, None)

            parse_btn.click(
                fn=lambda ds_id, df: self._handle_parse(ds_id, df),
                inputs=[dataset_selector, doc_list],
                outputs=doc_list,
            ).then(lambda: gr.Info("解析已开始"), None, None)

    def _handle_delete(self, dataset_id, doc_df):
        """处理删除逻辑"""
        if doc_df.empty:
            raise gr.Error("请先选择要删除的文档")

        doc_ids = doc_df["ID"].tolist()
        if self.client.delete_documents(dataset_id, doc_ids):
            return self.client.list_documents(dataset_id)["docs"]
        raise gr.Error("删除操作失败")

    def _handle_parse(self, dataset_id, doc_df):
        """处理解析逻辑"""
        if doc_df.empty:
            raise gr.Error("请先选择要解析的文档")

        doc_ids = doc_df["ID"].tolist()

        # DYC新增： 文档状态检查，等待所有文档变为 UNSTART
        for doc_id in doc_ids:
            ready = self.client.wait_for_document_ready(dataset_id, doc_id, timeout=30)
            if not ready:
                raise gr.Error(f"文档 {doc_id} 未准备好，无法解析")

        if self.client.parse_documents(dataset_id, doc_ids):
            return self.client.list_documents(dataset_id)["docs"]
        raise gr.Error("解析操作失败")

    def _load_dataset_info(self, dataset_id):
        """加载数据集信息到表单"""
        info = self.client.get_dataset_info(dataset_id)
        return [
            info.get("name", ""),
            info.get("description", ""),
            info.get("embedding_model", ""),
            info.get("chunk_method", ""),
        ]

    def _update_dataset_info(
        self, dataset_id, name, description, embedding_model, chunk_method
    ):
        """提交数据集修改"""
        self.client.update_dataset(
            dataset_id=dataset_id,
            update_data={
                "name": name,
                "description": description,
                "embedding_model": embedding_model,
                "chunk_method": chunk_method,
            },
        )

    def _get_dataset_options(self):
        """获取数据集下拉选项"""
        datasets = self.client.list_datasets()
        # 添加空值过滤
        return [(ds.name or "未命名数据集", ds.id) for ds in datasets if ds.id]

    def _update_doc_list(self, dataset_id):
        """更新文档列表"""
        if not dataset_id:
            return []
        return self.client.list_documents(dataset_id)

    # def _handle_upload(self, dataset_id, files):
    #     """处理上传逻辑"""
    #     if not dataset_id:
    #         raise gr.Error("请先选择知识库")

    #     try:
    #         # 转换Gradio文件对象为FileStorage
    #         uploaded_files = [(file.name, file) for file in files]

    #         # 调用API
    #         result = self.client.upload_documents(dataset_id, uploaded_files)
    #         return self.client.list_documents(dataset_id)["docs"]
    #     except Exception as e:
    #         logger.error(f"上传异常：{str(e)}")
    #         raise gr.Error("文件上传失败，请检查文件格式或网络连接")

    # DYC修复
    def _handle_upload(self, dataset_id, files):
        if not dataset_id:
            raise gr.Error("请先选择知识库")

        try:
            multipart_files = []
            open_files = []  # 保存打开的文件对象，上传后再关闭
            for file_path in files:
                f = open(file_path, "rb")
                open_files.append(f)  # 保存引用以防止提前关闭
                filename = os.path.basename(file_path)
                multipart_files.append(
                    ("file", (filename, f, self._detect_mime(filename)))
                )

            response = requests.post(
                url=f"{self.client.base_url}/api/v1/datasets/{dataset_id}/documents",
                headers={
                    k: v
                    for k, v in self.client.headers.items()
                    if k.lower() != "content-type"
                },
                files=multipart_files,
            )
            response.raise_for_status()

            return self.client.list_documents(dataset_id)["docs"]

        except Exception as e:
            logger.exception("上传处理异常：")
            raise gr.Error(f"上传失败：{str(e)}")

        finally:
            # 不论成功失败都关闭文件
            for f in open_files:
                f.close()

    def _detect_mime(self, filename):
        """自动检测文件类型"""
        from mimetypes import guess_type

        return guess_type(filename)[0] or "application/octet-stream"

    # 数据集事件处理函数


if __name__ == "__main__":
    # 在程序启动时创建数据库（如果没有）
    create_db()

    # 初始化客户端
    rag_client = RAGFlow(api_key=API_KEY, base_url=BASE_URL, chat_id=CHAT_ID)

    # 创建界面
    interface = ChatInterface(rag_client)
    demo = interface.create_interface()
    demo.launch(server_port=7860, pwa=True, allowed_paths=["./"])
