import re
import pandas as pd

def normalize_math(text: str) -> str:
    """
    将 Markdown/LaTeX 公式转换为纯文本可读格式
    """
    # 数学环境
    text = re.sub(r'\$\$(.+?)\$\$', r'\1', text)
    text = re.sub(r'\$(.+?)\$', r'\1', text)

    # 分数
    text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', text)

    # \text{...} \mathrm{...} 等
    text = re.sub(r'\\(text|mathrm)\{([^{}]+)\}', r'\2', text)

    # 下标/上标
    text = re.sub(r'([_^])\{([^{}]+)\}', r'\1\2', text)

    # 简单替换
    text = text.replace('\\cdot', '*').replace('\\times', '*')
    text = text.replace('\\leqslant', '<=').replace('\\geqslant', '>=')
    text = text.replace('\\left', '').replace('\\right', '')

    # 删除残余命令
    text = re.sub(r'\\[a-zA-Z]+\s*', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def parse_markdown(md_path: str) -> pd.DataFrame:
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    current_concept = None
    current_qtype = None
    problem = None

    in_answer = False
    in_steps  = False

    for line in lines:
        s = line.strip()

        # 标题处理
        if s.startswith('#'):
            s = s.lstrip('#').strip()

        # 考向处理
        if s.startswith('考向'):
            raw = re.sub(r'^[一二三四五六七八九十]+', '', s.replace('考向', '')).strip()
            current_concept = raw
            continue

        # 概念标题（题型01、题型02等）
        if re.match(r'^题型\s*\d+', s):
            current_concept = re.sub(r'^题型\s*\d+\s*', '', s)
            continue

        # 题型（question type）处理，如一、选择题
        m_qt = re.match(r'^([一二三四五六七八九十]+)、\s*(.+?题)$', s)
        if m_qt:
            current_qtype = m_qt.group(2).strip()
            continue

        # 新题号
        m_q = re.match(r'^(\d+)[\.\．]\s*(.+)', line)
        if m_q:
            # 保存上一题
            if problem:
                records.append(problem)
            # 重置状态
            in_answer = in_steps = False
            # 新题
            raw_q = re.sub(r'^\d+[．.]\s*（.*?）\s*', '', m_q.group(2))
            raw_q = re.sub(r'^（.*?）\s*', '', raw_q)  # 删除题干最前面的（）内容
            problem = {
                'problem_id':  m_q.group(1),
                'difficulty':  '',
                'concepts':    '一元一次不等式（组）::' + (current_concept or ''),
                'question':    normalize_math(raw_q),
                'options':     [],
                'answer_line': '',
                'steps':       '',
                'answer_code': None,
                'question_type': current_qtype or ''
            }
            continue

        # 选项
        if problem and re.match(r'^[A-D]\.', s):
            problem['options'].append(normalize_math(s))
            continue

        # 答案开始
        if problem and '答案' in s and not in_answer and not in_steps:
            raw_ans = re.sub(r'【答案】\s*', '', s)
            # 选项型答案
            codes = re.findall(r'[A-D]', raw_ans)
            if codes:
                problem['answer_code'] = codes[0]
            problem['answer_line'] = normalize_math(raw_ans)
            in_answer = True
            continue

        # 收集多行答案：直到遇到解析标签或下一个题号
        if problem and in_answer and not in_steps:
            if re.match(r'^\d+[\.\．]', s):
                # 碰到新题号，回退并保存
                in_answer = False
            elif re.search(r'【(?:分析|点睛|详解|提示)】', s):
                in_answer = False
                in_steps  = True
                # 也将这一行作为第一行解析
            else:
                problem['answer_line'] += '\n' + normalize_math(s)
                continue

        # 收集解析（steps）
        if problem and (in_steps or re.search(r'【(?:分析|点睛|详解|提示)】', s)):
            in_steps = True
            # 去掉任意标签
            raw_step = re.sub(r'【(?:分析|点睛|详解|提示)】\s*', '', s)
            problem['steps'] += normalize_math(raw_step) + '\n'
            continue

        # 题干续行
        if problem and not in_answer and not in_steps:
            if s and not re.match(r'^[A-D]\.', s) and '答案' not in s:
                problem['question'] += ' ' + normalize_math(s)
            continue

    # 最后一题
    if problem:
        records.append(problem)



    # 组装 DataFrame
    rows = []
    for p in records:
        q = p['question']
        if p['options']:
            q += '\n' + '\n'.join(p['options'])
        if p['answer_code']:
            for opt in p['options']:
                if opt.startswith(p['answer_code'] + '.'):
                    ans = p['answer_code'] + '.' + opt.split('.', 1)[1].strip()
                    break
        else:
            ans = p['answer_line']

        p['question_type'] = '选择题' if p['options'] else '填空题'

        rows.append({
            'problem_id': p['problem_id'],
            'difficulty': p['difficulty'],
            'concepts':   p['concepts'],
            'question':   q,
            'steps': p['steps'].strip(),
            'answer':     ans.strip(),
            'question_type': p['question_type']
        })

    return pd.DataFrame(rows, columns=[
        'problem_id','difficulty','concepts',
        'question','steps','answer','question_type'
    ])


if __name__ == '__main__':
    df = parse_markdown('md/第08讲 一元一次不等式（组）及其应用（练习）（解析版）.md')
    df.to_excel('new/一元一次不等式（组）及其应用.xlsx', index=False)
    print(f'已生成 ercigenshi.xlsx，共 {len(df)} 道题目。')
