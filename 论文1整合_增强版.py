# -*- coding: utf-8 -*-
"""
论文题目分析增强版。

保留原有描述性分析、BGE 相似度和 BERTopic 聚类（batch 模式），并新增：
1. SQLite 历史题库的增量导入、自动去重和 CSV 导出；
2. 新题目与历史题目的描述性分析、Top-K 相似度对比和联合聚类；
3. 千问 OpenAI 兼容接口智能解释，以及无密钥/调用失败时的本地规则解释。

常用命令：
    python 论文1整合.py ingest --history 历史题目.csv
    python 论文1整合.py compare --new-file 新题目.csv --store-new
    python 论文1整合.py compare --title "一个新题目" --major "专业名称"
    python 论文1整合.py batch --csv 历史题目.csv --output 完整分析结果

千问密钥仅从 DASHSCOPE_API_KEY（或 --api-key-env 指定的环境变量）读取，
不会写入源代码、数据库或分析报告。可通过 QWEN_BASE_URL/--qwen-base-url
设置与 API Key 地域匹配的业务空间专属地址。
"""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.request
import warnings
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

# 容器/虚拟机中避免 joblib 物理核探测输出无关堆栈。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.stats import norm

try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False

    class _JiebaFallback:
        """未安装 jieba 时提供最小兼容分词，保证核心流程可运行。"""

        @staticmethod
        def lcut(text):
            return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z][A-Za-z0-9_+-]*", str(text))

        @staticmethod
        def add_word(_word):
            return None

    jieba = _JiebaFallback()

# 绘图库
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import seaborn as sns

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

try:
    import joypy
except ImportError:
    joypy = None

# 机器学习与NLP库
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from bertopic import BERTopic
    from bertopic.backend import BaseEmbedder
    from umap import UMAP
    from hdbscan import HDBSCAN
    HAS_BERTOPIC = True
except ImportError:
    BERTopic = UMAP = HDBSCAN = None
    HAS_BERTOPIC = False

    class BaseEmbedder:
        """BERTopic 未安装时供 BGEBackend 继承的兼容基类。"""

        pass

# 模型加载尝试 (优先 FlagEmbedding)
try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_BGE = True
except ImportError:
    BGEM3FlagModel = None
    HAS_BGE = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

warnings.filterwarnings("ignore")

# ==============================================================================
#  全局配置区域 (请根据需要修改此处)
# ==============================================================================
# 输入文件路径
CSV_FILE_PATH = r"C:\Users\12920\Desktop\论文同质化\输入\LXY2023_2025.csv"
# 停用词与字典路径 (描述性分析用)
STOPWORDS_PATH = r"C:\Users\12920\Downloads\TYCD.txt"
TECH_DICT_PATH = r"C:\Users\12920\Downloads\ZYCD.txt"

# 输出总目录
BASE_OUTPUT_DIR = r"C:\Users\12920\Desktop\论文同质化\输出"

# ==============================================================================
#  0. 基础工具函数与环境初始化
# ==============================================================================

def get_font_path():
    """获取用于词云的中文字体绝对路径 (解决 OSError)"""
    windows_fonts = [
        r"C:\Windows\Fonts\simhei.ttf",      # 黑体
        r"C:\Windows\Fonts\msyh.ttf",        # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttf",      # 微软雅黑加粗
        r"C:\Windows\Fonts\simsun.ttc",      # 宋体
        r"C:\Windows\Fonts\STKAITI.TTF",     # 楷体
    ]
    for f in windows_fonts:
        if os.path.exists(f): return f
    try:
        for font in fm.fontManager.ttflist:
            if any(x in font.name.lower() for x in ['simhei', 'microsoft yahei', 'cjk', 'chinese']):
                return font.fname
    except:
        pass
    return "SimHei.ttf" # Fallback

def init_environment():
    """初始化字体和输出目录"""
    # 1. 创建目录结构 (完全对应原脚本结构)
    dirs = {
        "root": BASE_OUTPUT_DIR,
        "wordcloud": os.path.join(BASE_OUTPUT_DIR, "1_描述性_词云"),
        "methods": os.path.join(BASE_OUTPUT_DIR, "2_描述性_研究方法"),
        "heatmap": os.path.join(BASE_OUTPUT_DIR, "3_描述性_热力图"),
        "stats": os.path.join(BASE_OUTPUT_DIR, "4_描述性_基础统计"),
        "similarity": os.path.join(BASE_OUTPUT_DIR, "5_相似度分析"),
        "cluster": os.path.join(BASE_OUTPUT_DIR, "6_聚类分析"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # 2. 设置 Matplotlib 字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'SimSun', 'Arial Unicode MS']
    found = False
    for font in chinese_fonts:
        if any(f.name == font for f in fm.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [font]
            found = True
            break
    if not found:
        plt.rcParams['font.sans-serif'] = ['SimHei']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 3. 加载 Jieba 字典
    if os.path.exists(TECH_DICT_PATH):
        with open(TECH_DICT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().replace(" ", "").replace("\t", "")
                if w and not w.startswith("#"):
                    jieba.add_word(w)
                    
    return dirs

def load_stopwords():
    """加载停用词"""
    stopwords = set()
    if os.path.exists(STOPWORDS_PATH):
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w: stopwords.add(w)
    # 添加内置停用词
    stopwords.update({
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
        '基于', '研究', '分析', '设计', '应用', '系统', '开发', '实现', '算法', '模型', '现状', '对策', '影响', '探讨',
        '天津商业大学', '天津', '天津市', '为例', '大学生', '某', 'xx', 'XX', '数据','期间','之间'
    })
    return stopwords

def load_data(filepath):
    """读取并清洗数据"""
    print(f"正在读取数据: {filepath}")
    if not os.path.exists(filepath):
        print(f"错误：文件不存在 - {filepath}")
        return None

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filepath, encoding='gbk')
        except:
            print("错误：无法读取CSV文件，请检查编码。")
            return None

    col_map = {'年份': 'year', '题目': 'title', '标题': 'title', '专业': 'major', '导师': 'teacher'}
    df.columns = df.columns.str.strip()
    df.rename(columns=col_map, inplace=True)
    
    if 'major' not in df.columns: df['major'] = '未分类'
    if 'teacher' not in df.columns: df['teacher'] = '未知'
    
    df = df.dropna(subset=['title', 'year'])
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df['title'] = df['title'].astype(str)
    
    # 重置索引，确保索引连续，这对后续处理很重要
    df.reset_index(drop=True, inplace=True)
    
    print(f"数据加载完成，有效记录数: {len(df)}")
    return df

# ==============================================================================
#  1. 模型后端类 (共享模型实例，但不共享向量数据)
# ==============================================================================

class BGEBackend(BaseEmbedder):
    """自定义 Embedder 后端"""
    def __init__(self, model_path="BAAI/bge-m3"):
        self.is_flag = False
        self.model = None
        if HAS_BGE:
            print(f"Loading SOTA Model (FlagEmbedding): {model_path} ...")
            try:
                self.model = BGEM3FlagModel(model_path, use_fp16=True)
                self.is_flag = True
            except Exception as e:
                print(f"FlagEmbedding 加载出错: {e}，将尝试 SentenceTransformer...")
        
        if not self.is_flag and HAS_SENTENCE_TRANSFORMERS:
            fallback_model = "paraphrase-multilingual-MiniLM-L12-v2"
            print(f"正在加载降级模型: {fallback_model} ...")
            self.model = SentenceTransformer(fallback_model)

        if self.model is None:
            raise RuntimeError(
                "未安装 FlagEmbedding 或 sentence-transformers。"
                "新题对比可改用 --embedding tfidf；旧版 batch 模式需先安装语义模型依赖。"
            )
    
    def embed(self, documents, verbose=False):
        if self.is_flag:
            # batch_size=12 for safe memory usage
            embeddings = self.model.encode(documents, batch_size=12, max_length=512)['dense_vecs']
        else:
            embeddings = self.model.encode(documents, show_progress_bar=verbose)
        return embeddings

# ==============================================================================
#  2. 模块一：描述性分析 (完整还原 `最终版描述性分析.py`)
# ==============================================================================

def run_descriptive_analysis(df, dirs, stopwords_set):
    print("\n>>> 开始执行：描述性分析模块")
    FONT_PATH = get_font_path()
    
    # 辅助函数
    def tokenize(text):
        ws = jieba.lcut(str(text))
        clean_words = []
        for w in ws:
            w = w.strip()
            if not w or re.fullmatch(r"[\s\d\W]+", w): continue
            if len(w) == 1 and re.match(r"[\u4e00-\u9fa5]", w): continue
            if w in stopwords_set: continue
            clean_words.append(w)
        return clean_words
    
    # 生成 token_clean 列用于热力图
    df["token_clean"] = df["title"].apply(tokenize)
    
    # ------------------------------------------------------
    # 1. 题目字数统计 (还原所有图表)
    # ------------------------------------------------------
    print("  [1/5] 进行字数统计...")
    def count_mixed_len(text):
        text = str(text)
        english_words = re.findall(r'[a-zA-Z]+', text)
        n_english = len(english_words)
        text_no_english = re.sub(r'[a-zA-Z]+', '', text)
        text_clean = re.sub(r'\s+', '', text_no_english)
        return n_english + len(text_clean)

    df['title_len'] = df['title'].apply(count_mixed_len)
    
    # 计算完列后再生成 valid_year 副本
    df_valid_year = df[df['year'] > 1900].copy()

    # 导出字数异常
    abnormal_df = df[(df['title_len'] < 12) | (df['title_len'] > 30)]
    abnormal_path = os.path.join(dirs["stats"], "论文题目字数异常(非12-30).csv")
    cols_to_save = [c for c in ['title', 'title_len', 'major', 'year'] if c in df.columns]
    abnormal_df[cols_to_save].to_csv(abnormal_path, index=False, encoding='utf-8-sig')

    # 字数异常比率统计表
    ratio_stats = []
    for major in df['major'].unique():
        sub = df[df['major'] == major]
        cnt = len(sub)
        abn = len(sub[(sub['title_len'] < 12) | (sub['title_len'] > 30)])
        ratio_stats.append({
            "专业": major, "总题目数": cnt, "字数异常题目数": abn, "异常比率": f"{abn/cnt:.2%}" if cnt else "0%"
        })
    total_abn = len(abnormal_df)
    ratio_stats.append({"专业": "总体", "总题目数": len(df), "字数异常题目数": total_abn, "异常比率": f"{total_abn/len(df):.2%}"})
    pd.DataFrame(ratio_stats).to_csv(os.path.join(dirs["stats"], "题目字数异常比率统计表.csv"), index=False, encoding='utf-8-sig')

    # 【核心还原】字数分布直方图 (带正态拟合)
    def draw_title_dist(data_series, title, filename):
        n = len(data_series)
        if n < 5: return 
        max_val = data_series.max()
        max_len = int(np.ceil(max_val / 5.0) * 5)
        if max_len < 5: max_len = 5
        bins = np.arange(0, max_len + 5, 5)
        
        counts, bin_edges = np.histogram(data_series, bins=bins)
        xs = (bin_edges[:-1] + bin_edges[1:]) / 2

        plt.figure(figsize=(8,5))
        ax = plt.gca()
        ax.bar(xs, counts, width=4.5, color='steelblue', edgecolor='black')

        y_max_plot = max(counts) 
        mu, sigma = np.mean(data_series), np.std(data_series, ddof=1)
        if sigma > 0:
            xs_dense = np.linspace(min(data_series), max(data_series), 200)
            pdf = norm.pdf(xs_dense, mu, sigma)
            if pdf.max() > 0:
                scale = max(counts) / pdf.max()
                pdf_scaled = pdf * scale
                ax.plot(xs_dense, pdf_scaled, linewidth=2, color='red', zorder=5, label="正态拟合")
                ax.legend()
                y_max_plot = max(y_max_plot, pdf_scaled.max())

        offset = y_max_plot * 0.015
        for x, h in zip(xs, counts):
            if h > 0: ax.text(x, h + offset, str(int(h)), ha="center", va="bottom", fontsize=9)

        ax.set_ylim(0, y_max_plot * 1.15)
        plt.title(f"{title} (N={n})", fontsize=14)
        plt.xlabel("题目字数区间", fontsize=12)
        plt.ylabel("论文数量", fontsize=12)
        plt.xticks(np.arange(0, max_len + 5, 5))
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["stats"], filename))
        plt.close()

    # 批量生成字数分布图
    draw_title_dist(df['title_len'], "总体-论文题目字数分布", "总体_题目字数分布.png")
    for major in df['major'].unique():
        safe_major = re.sub(r'[\\/:*?"<>|]', "_", str(major))
        draw_title_dist(df[df['major'] == major]['title_len'], f"{major} 专业论文题目字数分布", f"{safe_major}_题目字数分布.png")
    for year in sorted(df_valid_year['year'].unique()):
        draw_title_dist(df_valid_year[df_valid_year['year'] == year]['title_len'], f"总体-{year}年 题目字数分布", f"总体_{year}年_题目字数分布.png")
    for major in df['major'].unique():
        sub_major = df_valid_year[df_valid_year['major'] == major]
        safe_major = re.sub(r'[\\/:*?"<>|]', "_", str(major))
        for year in sorted(sub_major['year'].unique()):
            draw_title_dist(sub_major[sub_major['year'] == year]['title_len'], f"{major}-{year}年 题目字数分布", f"{safe_major}_{year}年_题目字数分布.png")

    # ------------------------------------------------------
    # 2. 历年字数趋势 (还原所有折线图)
    # ------------------------------------------------------
    print("  [2/5] 绘制字数趋势图...")
    if not df_valid_year.empty:
        # (1) 总体
        plt.figure(figsize=(10, 6))
        annual_mean = df_valid_year.groupby('year')['title_len'].mean().reset_index()
        sns.lineplot(data=annual_mean, x='year', y='title_len', marker='o', linewidth=2.5, color='purple')
        for x, y in zip(annual_mean['year'], annual_mean['title_len']):
            plt.text(x, y + 0.1, f"{y:.1f}", ha='center', va="bottom", fontsize=9)
        plt.title("历年论文题目平均字数趋势 (总体)")
        plt.xlabel("年份")
        plt.ylabel("平均字数")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(sorted(annual_mean['year'].unique().astype(int)))
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["stats"], "理学院总体_历年字数趋势.png"))
        plt.close()

        # (2) 各专业对比
        plt.figure(figsize=(12, 7))
        major_means = df_valid_year.groupby(['year', 'major'])['title_len'].mean().reset_index()
        sns.lineplot(data=major_means, x='year', y='title_len', hue='major', marker='o', linewidth=2, palette="tab10")
        plt.title("各专业历年论文题目平均字数趋势对比")
        plt.xticks(sorted(major_means['year'].unique()))
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["stats"], "各专业对比_历年字数趋势.png"))
        plt.close()

        # (3) 各专业单独
        for major in df['major'].unique():
            sub_df = df_valid_year[df_valid_year['major'] == major]
            if len(sub_df) < 5: continue
            safe_major = re.sub(r'[\\/:*?"<>|]', "_", str(major))
            trend = sub_df.groupby('year')['title_len'].mean().reset_index()
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=trend, x='year', y='title_len', marker='o', color='teal')
            for idx, row in trend.iterrows():
                plt.text(row['year'], row['title_len'] + 0.1, f"{row['title_len']:.1f}", ha='center', va="bottom")
            plt.title(f"{major}-历年题目平均字数趋势")
            plt.xticks(sorted(trend['year'].unique()))
            plt.tight_layout()
            plt.savefig(os.path.join(dirs["stats"], f"{safe_major}_历年字数趋势.png"))
            plt.close()

    # ------------------------------------------------------
    # 3. 结尾词统计 (还原水平柱状图)
    # ------------------------------------------------------
    print("  [3/5] 统计结尾词...")
    def get_last_word(title):
        ws = jieba.lcut(str(title).strip())
        for w in reversed(ws):
            if w.strip() and not re.fullmatch(r"[\W\d\s]+", w): return w
        return ""
    
    df['ending'] = df['title'].apply(get_last_word)
    # 再次更新 df_valid_year 以包含 ending
    df_valid_year = df[df['year'] > 1900].copy()

    def draw_horizontal_bar(data_series, title, filename, top_k=15):
        if data_series.empty: return
        counts = data_series.value_counts().drop("", errors='ignore')
        if counts.empty: return
        vals = counts.values[:top_k]
        labels = counts.index[:top_k]
        
        plt.figure(figsize=(8, max(5, len(vals)*0.5)))
        ax = plt.gca()
        ys = np.arange(len(vals))
        ax.barh(ys[::-1], vals, height=0.6, color='steelblue', edgecolor='black')
        for y_pos, x_val in zip(ys[::-1], vals):
            ax.text(x_val + 0.1, y_pos, str(int(x_val)), va="center", ha="left")
        ax.set_yticks(ys[::-1])
        ax.set_yticklabels(labels)
        plt.title(f"{title} (N={vals.sum()})")
        plt.xlabel("出现次数")
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["stats"], filename))
        plt.close()

    draw_horizontal_bar(df['ending'], "总体-论文结尾词频率统计", "总体_结尾词统计.png")
    for year in sorted(df_valid_year['year'].unique()):
        sub = df_valid_year[df_valid_year['year'] == year]
        if len(sub) >= 5: draw_horizontal_bar(sub['ending'], f"{year}年-结尾词统计", f"理学院总体_{year}年_结尾词统计.png", top_k=10)
    for major in df['major'].unique():
        sub = df[df['major'] == major]
        safe_major = re.sub(r'[\\/:*?"<>|]', "_", str(major))
        draw_horizontal_bar(sub['ending'], f"{major}-总体结尾词统计", f"{safe_major}_总体结尾词统计.png")
        # 专业+年份
        sub_valid = df_valid_year[df_valid_year['major'] == major]
        for year in sorted(sub_valid['year'].unique()):
            sub_y = sub_valid[sub_valid['year'] == year]
            if len(sub_y) >= 5: draw_horizontal_bar(sub_y['ending'], f"{major}-{year}年 结尾词统计", f"{safe_major}_{year}年_结尾词.png", top_k=10)

    # ------------------------------------------------------
    # 4. 词云生成 (还原逻辑)
    # ------------------------------------------------------
    print("  [4/5] 生成词云...")
    wc_stopwords = set(stopwords_set)
    wc_stopwords.update({'基于', '研究', '分析', '设计', '应用', '系统', '开发', '实现', '算法', '模型', '现状', '对策', '影响', '探讨', '天津商业大学', '天津', '天津市', '为例', '大学生', '某', 'xx', 'XX', '数据','期间','之间'})
    
    def clean_and_tokenize_wc(text):
        if not isinstance(text, str): return []
        text = re.sub(r'[—\-]?以[^，。,;]*为例', '', text) # 还原去“以...为例”逻辑
        words = jieba.lcut(text)
        return [w for w in words if len(w.strip()) > 1 and not re.fullmatch(r"[\s\d\W]+", w) and w not in wc_stopwords]

    def draw_wordcloud(text_series, title_text, filename, min_samples=5):
        if WordCloud is None:
            return
        n = len(text_series)
        if n < min_samples: return
        all_tokens = []
        for t in text_series: all_tokens.extend(clean_and_tokenize_wc(t))
        if not all_tokens: return
        freq = Counter(all_tokens)
        
        wc = WordCloud(font_path=FONT_PATH, background_color="white", width=1000, height=600, 
                       max_words=200, collocations=False, random_state=42).generate_from_frequencies(freq)
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{title_text} (n={n})")
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["wordcloud"], filename), dpi=300)
        plt.close()

    draw_wordcloud(df['title'], "总体研究方向词云", "总体_词云.png")
    for major in df['major'].unique():
        safe_major = re.sub(r'[\\/:*?"<>|]', "_", str(major))
        draw_wordcloud(df[df['major'] == major]['title'], f"{major} 研究方向词云", f"{safe_major}_词云.png")
    for year in sorted(df_valid_year['year'].unique()):
        draw_wordcloud(df_valid_year[df_valid_year['year'] == year]['title'], f"总体_{year}年 研究方向词云", f"总体_{year}年_词云.png")
    for major in df['major'].unique():
        for year in sorted(df_valid_year['year'].unique()):
            sub = df_valid_year[(df_valid_year['major'] == major) & (df_valid_year['year'] == year)]
            safe_major = re.sub(r'[\\/:*?"<>|]', "_", str(major))
            if len(sub) >= 5:
                draw_wordcloud(sub['title'], f"{major}_{year}年 研究方向词云", f"{safe_major}_{year}年_词云.png")

    # ------------------------------------------------------
    # 5. 专业相关性热力图 (还原)
    # ------------------------------------------------------
    print("  [5/5] 生成相关性热力图...")
    all_words_heatmap = []
    for t in df['token_clean']: all_words_heatmap.extend(t)
    top_words = [w for w, _ in Counter(all_words_heatmap).most_common(50)]
    
    def get_word_vector(texts):
        words = []
        for t in texts: words.extend(tokenize(t))
        freq = Counter(words)
        return [freq.get(w,0) for w in top_words]

    # (1) 总体热力图
    major_vecs = {m: get_word_vector(df[df['major']==m]['title']) for m in df['major'].unique()}
    if major_vecs:
        corr_df = pd.DataFrame(major_vecs, index=top_words).corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr_df, annot=True, cmap="YlGnBu")
        plt.title(f"总体-专业研究主题相关系数热力图 (N={len(df)})")
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["heatmap"], "总体_相关性热力图.png"))
        plt.close()

    # (2) 各年热力图
    for year in sorted(df_valid_year['year'].unique()):
        sub = df_valid_year[df_valid_year['year'] == year]
        if len(sub) < 10: continue
        major_vecs = {}
        valid_majors = []
        for m in sub['major'].unique():
            sub_m = sub[sub['major'] == m]
            if len(sub_m) < 3: continue 
            major_vecs[m] = get_word_vector(sub_m['title'])
            valid_majors.append(m)
        if len(valid_majors) >= 2:
            corr = pd.DataFrame(major_vecs, index=top_words).corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title(f"{year}年-专业相关性热力图")
            plt.tight_layout()
            plt.savefig(os.path.join(dirs["heatmap"], f"{year}年_相关性热力图.png"))
            plt.close()

# ==============================================================================
#  3. 模块二：相似度计算 (完整还原 `相似度计算BGE3.py`)
# ==============================================================================

def run_similarity_analysis(df, embedder_instance, dirs):
    print("\n>>> 开始执行：相似度分析模块")
    
    # 核心修正：使用“原始数据 (A)”进行向量化
    # 相似度计算是基于 df['title'] 的
    print("  正在计算原始标题的向量 (For Similarity Analysis)...")
    embeddings = embedder_instance.embed(df['title'].tolist(), verbose=True)
    
    output_list_yearly = []
    output_list_all = []
    
    # 1. 年度分析
    print("  [1/2] 进行年度相似度分析...")
    years = sorted(df['year'].unique())
    for year in years:
        indices = df[df['year'] == year].index.tolist()
        if len(indices) < 2: continue
        
        year_vectors = embeddings[indices]
        sim_matrix = cosine_similarity(year_vectors)
        np.fill_diagonal(sim_matrix, -1)
        
        pairs = np.argwhere(sim_matrix > 0.85)
        pairs = pairs[pairs[:, 1] > pairs[:, 0]]
        
        found_count = 0
        for idx_i, idx_j in pairs:
            real_i = indices[idx_i]
            real_j = indices[idx_j]
            output_list_yearly.append({
                'year': year,
                'major1': df.loc[real_i, 'major'],
                'title1': df.loc[real_i, 'title'],
                'teacher1': df.loc[real_i, 'teacher'],
                'major2': df.loc[real_j, 'major'],
                'title2': df.loc[real_j, 'title'],
                'teacher2': df.loc[real_j, 'teacher'],
                'similarity': sim_matrix[idx_i, idx_j]
            })
            found_count += 1
        print(f"    年份 {year}: 发现 {found_count} 对相似论文")

    res_yearly = pd.DataFrame(output_list_yearly)
    res_yearly.to_csv(os.path.join(dirs["similarity"], '年度论文相似度超过0.85.csv'), index=False, encoding='utf-8-sig')
    
    if not res_yearly.empty:
        stats = res_yearly.groupby(['year', 'major1', 'major2']).size().reset_index(name='相似对数')
        stats.to_csv(os.path.join(dirs["similarity"], '每年专业论文相似度统计表.csv'), index=False, encoding='utf-8-sig')

    # 2. 全局汇总分析
    print("  [2/2] 进行跨年份汇总相似度分析...")
    if len(embeddings) < 5000:
        sim_matrix_all = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix_all, -1)
        pairs_all = np.argwhere(sim_matrix_all > 0.85)
        pairs_all = pairs_all[pairs_all[:, 1] > pairs_all[:, 0]]
        
        for i, j in pairs_all:
             output_list_all.append({
                'year1': df.loc[i, 'year'],
                'major1': df.loc[i, 'major'],
                'title1': df.loc[i, 'title'],
                'teacher1': df.loc[i, 'teacher'],
                'year2': df.loc[j, 'year'],
                'major2': df.loc[j, 'major'],
                'title2': df.loc[j, 'title'],
                'teacher2': df.loc[j, 'teacher'],
                'similarity': sim_matrix_all[i, j]
            })
            
        res_all = pd.DataFrame(output_list_all)
        res_all.to_csv(os.path.join(dirs["similarity"], '所有年份论文相似度超过0.85.csv'), index=False, encoding='utf-8-sig')
        
        # 统计与热力图
        if not res_all.empty:
            all_counts = res_all.groupby(['major1', 'major2']).size().reset_index(name='count')
            all_counts.to_csv(os.path.join(dirs["similarity"], '历届专业论文相似度统计.csv'), index=False, encoding='utf-8-sig')

            print("    正在绘制相似度热力图...")
            all_majors = sorted(list(set(df['major'].unique())))
            inter_major = res_all[res_all['major1'] != res_all['major2']]
            intra_major = res_all[res_all['major1'] == res_all['major2']]
            
            inter_sym = pd.concat([
                inter_major[['major1', 'major2']],
                inter_major.rename(columns={'major1': 'major2', 'major2': 'major1'})[['major1', 'major2']]
            ])
            df_sym = pd.concat([intra_major[['major1', 'major2']], inter_sym])
            matrix_data = pd.crosstab(df_sym['major1'], df_sym['major2'])
            matrix_data = matrix_data.reindex(index=all_majors, columns=all_majors, fill_value=0)

            plt.figure(figsize=(14, 11))
            sns.heatmap(matrix_data, annot=True, fmt="d", cmap="YlOrRd", linewidths=.5, cbar_kws={'label': '相似论文对数'})
            plt.title('各专业论文题目相似度分布热力图 (阈值 > 0.85)', fontsize=16, pad=25)
            plt.xticks(rotation=0); plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(dirs["similarity"], '专业相似度热力图.png'), dpi=300, bbox_inches='tight')
    else:
        print("    数据量过大，跳过全局全矩阵计算。")

# ==============================================================================
#  4. 模块三：聚类分析 (完整还原 `聚类BGE3.py`)
# ==============================================================================

def clean_title_for_cluster(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'（.*?）', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'以.*?为例', '', text)
    text = re.sub(r'以.*?为对象', '', text)
    text = re.sub(r'以.*?为视.*?角', '', text)
    useless_starts = ['基于', '关于', '浅析', '试论', '探究', '面向', '通过', '简析']
    for word in useless_starts: text = text.replace(word, '')
    separators = [r'——', r'：', r':', r'--']
    for sep in separators:
        parts = re.split(sep, text)
        if len(parts) > 1 and len(parts[0].strip()) > 2: text = parts[0]
    return text.strip()

def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    if x.size < 2 or y.size < 2: return 
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ellipse = patches.Ellipse(
        (np.mean(x), np.mean(y)), width=lambda_[0] * n_std * 2, height=lambda_[1] * n_std * 2,
        angle=np.rad2deg(np.arccos(v[0, 0])), facecolor=facecolor, **kwargs
    )
    return ax.add_patch(ellipse)

def run_clustering_analysis(df_original, embedder_instance, dirs):
    print("\n>>> 开始执行：聚类分析模块")
    if not HAS_BERTOPIC:
        raise RuntimeError("batch 聚类需要安装 bertopic、umap-learn 和 hdbscan。")
    
    # 核心修正：数据预处理流程还原 (A -> B)
    # 1. 复制数据
    df_cluster = df_original.copy()
    
    # 2. 清洗标题
    df_cluster['title_clean'] = df_cluster['title'].apply(clean_title_for_cluster)
    
    # 3. 筛选数据 (Data A 变为 Data B)
    # 只有长度 > 1 的清洗后标题才会被用于聚类
    df_cluster = df_cluster[df_cluster['title_clean'].str.len() > 1]
    
    docs = df_cluster['title_clean'].tolist()
    print(f"  清洗后有效聚类样本数: {len(docs)}")
    
    # 4. 生成聚类专用向量 (For Clustering Analysis)
    # 必须使用 Data B (清洗后的标题) 来生成向量，不能复用 Data A 的向量
    print("  正在计算清洗后标题的向量 (For Clustering Analysis)...")
    embeddings_subset = embedder_instance.embed(docs, verbose=True)

    # 5. 训练模型
    print("  配置 UMAP 和 HDBSCAN...")
    umap_model = UMAP(n_neighbors=30, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=20, min_samples=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    stopwords_extended = ["研究", "应用", "分析", "设计", "的", "中", "技术", "系统", "一种", "方法", "探讨", "实现", "策略", "与", "及", "下", "问题", "对策", "影响", "模式", "路径", "视角", "视阈", "背景", "现状", "评价", "体系", "构建", "优化", "思考", "综述", "实证", "模型", "机制", "协同", "动因", "效应", "改革", "创新", "发展", "我国", "中国", "比较", "管理", "能力", "因素", "作用", "对策", "建议", "关联", "特征", "基于", "视角下"]
    vectorizer_model = CountVectorizer(stop_words=stopwords_extended, ngram_range=(1, 3))

    print("  训练 BERTopic...")
    topic_model = BERTopic(
        embedding_model=embedder_instance, 
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=16,
        language="multilingual",
        calculate_probabilities=True,
        verbose=True
    )
    
    # 使用刚计算的 embeddings_subset
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings_subset)
    
    print("  处理噪音...")
    try:
        new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings_subset)
        topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)
        df_cluster['Topic'] = new_topics
    except:
        df_cluster['Topic'] = topics

    # 生成 2D 坐标
    print("  生成 2D 可视化坐标...")
    umap_vis = UMAP(n_neighbors=30, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    coords = umap_vis.fit_transform(embeddings_subset)
    df_cluster['x'] = coords[:, 0]
    df_cluster['y'] = coords[:, 1]
    
    plot_df = df_cluster[df_cluster['Topic'] != -1].copy()
    unique_topics = sorted(plot_df['Topic'].unique())
    
    palette = sns.color_palette("tab20", len(unique_topics)) if len(unique_topics) <= 20 else sns.color_palette("husl", len(unique_topics))
    color_map = dict(zip(unique_topics, palette))
    years = sorted(plot_df['year'].unique())
    full_years = np.arange(min(years), max(years) + 1)

    # 计算斜率用于排序
    topic_stats = []
    full_years_df = pd.DataFrame({'year': full_years})
    for t in unique_topics:
        sub = plot_df[plot_df['Topic'] == t]
        counts_per_year = sub.groupby('year').size().reset_index(name='count')
        merged = pd.merge(full_years_df, counts_per_year, on='year', how='left').fillna(0)
        slope = np.polyfit(np.arange(len(merged)), merged['count'].values, 1)[0] if len(merged) > 1 else 0
        topic_stats.append({'Topic': t, 'Total': len(sub), 'Slope': slope})
    stats_df = pd.DataFrame(topic_stats)
    
    # 排序逻辑
    sorted_by_vol = stats_df.sort_values('Total', ascending=False)['Topic'].tolist()
    stream_order_deque = deque()
    for i, t in enumerate(sorted_by_vol):
        if i % 2 == 0: stream_order_deque.append(t)
        else: stream_order_deque.appendleft(t)
    stream_sort_order = list(stream_order_deque)
    
    bubble_sort_stats = stats_df.sort_values('Slope', ascending=True).reset_index(drop=True)
    topic_to_rank_map = dict(zip(bubble_sort_stats['Topic'], bubble_sort_stats.index))
    
    def get_trend_label(row):
        t_id = int(row['Topic'])
        s = row['Slope']
        marker = "↑" if s > 0.05 else ("↓" if s < -0.05 else "~")
        return f"Topic {t_id} ({marker}{s:.2f})"

    bubble_sort_stats['Label'] = bubble_sort_stats.apply(get_trend_label, axis=1)
    rank_to_label_map = dict(zip(bubble_sort_stats.index, bubble_sort_stats['Label']))
    
    # --- 绘图开始 ---
    
    # 生成 Legend Handles
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[t], markersize=10) for t in unique_topics]
    labels = [f"Topic {t}" for t in unique_topics]

    # [图A] 全局散点图
    print("  绘制 [图A: 全局散点图]...")
    plt.figure(figsize=(14, 12)) 
    ax_global = plt.gca()
    for t in unique_topics:
        subset = plot_df[plot_df['Topic'] == t]
        if len(subset) == 0: continue
        draw_confidence_ellipse(subset['x'], subset['y'], ax_global, n_std=2.0, edgecolor=color_map[t], linestyle='--', linewidth=2.5, alpha=0.8)
        sns.scatterplot(data=subset, x='x', y='y', color=color_map[t], ax=ax_global, s=50, alpha=0.7, edgecolor='white')
        ax_global.text(subset['x'].mean(), subset['y'].mean(), str(t), fontsize=14, fontweight='bold', color='black', ha='center', va='center')
    
    plt.title("主题聚类散点图 (BGE-M3 Powered)", fontsize=18, fontweight='bold')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    # 【恢复】全局散点图图例
    plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', title="Topic ID")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["cluster"], "主题聚类散点图.png"), dpi=300)
    plt.close()

    # [图B] 分年散点图
    print("  绘制 [图B: 分年散点图]...")
    n_years = len(years)
    cols = math.ceil(math.sqrt(n_years)) if n_years > 4 else 2
    rows = math.ceil(n_years / cols)
    fig1, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), sharex=True, sharey=True)
    if n_years > 1: axes = axes.flatten()
    else: axes = [axes]
    
    for i, year in enumerate(years):
        ax = axes[i]
        # 背景轮廓
        for t in unique_topics:
            sub_g = plot_df[plot_df['Topic'] == t]
            if len(sub_g) < 2: continue
            draw_confidence_ellipse(sub_g['x'], sub_g['y'], ax, n_std=2.0, edgecolor=color_map[t], linestyle='--', linewidth=2, alpha=0.5)
            ax.text(sub_g['x'].mean(), sub_g['y'].mean(), str(t), fontsize=10, fontweight='bold', color=color_map[t], ha='center', va='center', alpha=0.6)
        # 当年数据
        sub_y = plot_df[plot_df['year'] == year]
        if not sub_y.empty:
            sns.scatterplot(data=sub_y, x='x', y='y', hue='Topic', palette=color_map, ax=ax, s=60, legend=False, alpha=0.95, edgecolor='white')
        ax.set_title(f"{year}年", fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    for j in range(i + 1, len(axes)): fig1.delaxes(axes[j])
    # 【恢复】分年散点图统一图例
    fig1.legend(handles, labels, loc='center right', title="Topic ID", bbox_to_anchor=(1.08, 0.5), fontsize=10)
    plt.tight_layout()
    fig1.savefig(os.path.join(dirs["cluster"], "主题时间聚类散点图.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # [图C] 曲线平滑河流图
    print("  绘制 [图C: 曲线平滑河流图]...")
    counts = plot_df.groupby(['year', 'Topic']).size().unstack(fill_value=0)
    counts = counts.reindex(full_years, fill_value=0)
    x_smooth = np.linspace(full_years.min(), full_years.max(), 300) 
    y_smooth_list = []
    labels_list = []
    colors_list = []
    
    for topic in stream_sort_order:
        if topic not in counts.columns: continue
        y_vals = counts[topic].values
        try:
            spl = make_interp_spline(full_years, y_vals, k=3 if len(full_years)>3 else 1)
            y_smooth = np.maximum(spl(x_smooth), 0)
        except:
            y_smooth = np.interp(x_smooth, full_years, y_vals)
        y_smooth_list.append(y_smooth)
        labels_list.append(f"Topic {topic}")
        colors_list.append(color_map[topic])
    
    plt.figure(figsize=(14, 8))
    plt.stackplot(x_smooth, y_smooth_list, labels=labels_list, colors=colors_list, baseline='sym', alpha=0.9)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # 【恢复】河流图顶部图例
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title="Topic ID", ncol=2)
    plt.title('主题演变趋势 (Streamgraph)', fontsize=16, fontweight='bold')
    
    # 【恢复】精确的 X 轴设置 (整数年份)
    plt.xlim(full_years.min(), full_years.max())
    plt.xticks(ticks=full_years, labels=[str(int(y)) for y in full_years], fontsize=12)
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["cluster"], "主题趋势河流图.png"), dpi=300)
    plt.close()

    # [图D] 山峦图 (Joypy)
    print("  绘制 [图D: 山峦图]...")
    try:
        joy_df = plot_df.copy()
        joy_df['year_smooth'] = joy_df['year'] + np.random.uniform(-0.4, 0.4, size=len(joy_df))
        plt.figure(figsize=(10, 16)) 
        fig3, axes3 = joypy.joyplot(
            data=joy_df, column="year_smooth", by="Topic",
            labels=[f"Topic {t}" for t in unique_topics], color=[color_map[t] for t in unique_topics],
            range_style='own', grid="y", linewidth=1.2, legend=False, overlap=0.3, 
            title="各主题时间分布趋势 (Ridgeline)", alpha=0.85, x_range=[min(years)-0.8, max(years)+0.8], figsize=(10, 16)
        )
        if isinstance(axes3, list) or isinstance(axes3, np.ndarray): axes3[-1].set_xticks(years)
        plt.savefig(os.path.join(dirs["cluster"], "主题趋势山峦图.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"   警告: 山峦图绘制失败: {e}")

    # [图E] 气泡图
    print("  绘制 [图E: 气泡图]...")
    plt.figure(figsize=(16, 14)) 
    ax = plt.gca()
    
    bubble_data = plot_df.groupby(['year', 'Topic']).size().reset_index(name='Count')
    bubble_data['Rank_Y'] = bubble_data['Topic'].map(topic_to_rank_map)
    
    sns.scatterplot(data=bubble_data, x='year', y='Rank_Y', size='Count', hue='Topic', palette=color_map, sizes=(50, 1200), alpha=0.75, edgecolor='gray', legend=False)
    
    # 增长/衰退区划分线
    try:
        zero_cross = bubble_sort_stats[bubble_sort_stats['Slope'] > 0].index
        if len(zero_cross) > 0:
            split_y = zero_cross[0] - 0.5
            plt.axhline(y=split_y, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
            plt.text(min(years), split_y + 0.3, ' Growth Zone (Slope > 0) ↑ ', va='bottom', ha='left', fontsize=12, color='darkred', fontweight='bold')
            plt.text(min(years), split_y - 0.3, ' Decline Zone (Slope < 0) ↓ ', va='top', ha='left', fontsize=12, color='darkgreen', fontweight='bold')
    except:
        pass

    plt.title("主题年度热度趋势气泡图 (BGE-M3 Reduced)", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("年份", fontsize=14)
    plt.ylabel("Topic (Sorted by Growth Slope)", fontsize=14)
    plt.xticks(full_years, fontsize=11)
    
    y_ticks_locs = sorted(rank_to_label_map.keys())
    y_ticks_labels = [rank_to_label_map[i] for i in y_ticks_locs]
    plt.yticks(y_ticks_locs, y_ticks_labels, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 【恢复】气泡图图例与布局调整
    legend_handles_color = [plt.Line2D([0], [0], marker='o', color='w', label=f"Topic {tid}", markerfacecolor=color_map[tid], markersize=8) for tid in sorted(unique_topics)]
    ax.add_artist(plt.legend(handles=legend_handles_color, bbox_to_anchor=(1.02, 1), loc='upper left', title="Topic ID"))
    plt.subplots_adjust(right=0.75, left=0.15) 
    
    plt.savefig(os.path.join(dirs["cluster"], "主题趋势气泡图.png"), dpi=300)
    plt.close()

    # Excel 导出
    print("  导出 Excel 主题内容详情表...")
    topic_info_list = []
    for tid in unique_topics:
        words_data = topic_model.get_topic(tid) 
        if not words_data: continue
        keywords_str = "、".join([w[0] for w in words_data[:4]])
        count = len(plot_df[plot_df['Topic'] == tid])
        slope = stats_df[stats_df['Topic'] == tid]['Slope'].values[0]
        trend_str = "显著增加" if slope > 1.0 else ("增加" if slope > 0.2 else ("显著减少" if slope < -1.0 else ("减少" if slope < -0.2 else "稳定")))
        
        topic_info_list.append({
            "Topic ID": tid, "代表性核心词 (KeyWords)": keywords_str, "论文数量": count,
            "趋势斜率(Slope)": round(slope, 3), "趋势判定": trend_str
        })
    
    df_excel = pd.DataFrame(topic_info_list).sort_values("趋势斜率(Slope)", ascending=False)
    df_excel.to_excel(os.path.join(dirs["cluster"], "主题内容概括表.xlsx"), index=False)

# ============================================================================== 
#  5. 新题目对比、历史存储与智能解释
# ============================================================================== 

DEFAULT_DB_PATH = str(Path.cwd() / "论文题目历史库.sqlite3")
DEFAULT_COMPARE_OUTPUT = str(Path.cwd() / "新题目对比结果")
DEFAULT_QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

COLUMN_ALIASES = {
    "题目": "title",
    "标题": "title",
    "论文题目": "title",
    "title": "title",
    "年份": "year",
    "年度": "year",
    "year": "year",
    "专业": "major",
    "major": "major",
    "导师": "teacher",
    "指导教师": "teacher",
    "teacher": "teacher",
}


def normalize_title(text):
    """用于查重和存储去重的稳定标题规范化。"""
    value = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE)


def count_title_length(text):
    """中文字符按字计数，连续英文按一个词计数。"""
    value = str(text or "")
    english_words = re.findall(r"[A-Za-z]+", value)
    without_english = re.sub(r"[A-Za-z]+", "", value)
    return len(english_words) + len(re.sub(r"\s+", "", without_english))


def _clean_optional_text(value, default=""):
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def _clean_year(value):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def standardize_question_dataframe(df, require_year=False):
    """统一中英文列名；新题目允许缺少年份、专业或导师。"""
    if df is None:
        raise ValueError("题目数据不能为空。")
    result = df.copy()
    result.columns = [str(c).strip() for c in result.columns]
    rename_map = {c: COLUMN_ALIASES[c] for c in result.columns if c in COLUMN_ALIASES}
    result.rename(columns=rename_map, inplace=True)
    if "title" not in result.columns:
        raise ValueError("数据中必须包含“题目”“标题”或“title”列。")

    for col, default in (("year", pd.NA), ("major", "未分类"), ("teacher", "未知")):
        if col not in result.columns:
            result[col] = default

    result["title"] = result["title"].map(lambda x: _clean_optional_text(x))
    result = result[result["title"].str.len() > 0].copy()
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    if require_year:
        result = result.dropna(subset=["year"])
    result["major"] = result["major"].map(lambda x: _clean_optional_text(x, "未分类") or "未分类")
    result["teacher"] = result["teacher"].map(lambda x: _clean_optional_text(x, "未知") or "未知")
    return result.reset_index(drop=True)


def read_question_file(filepath, require_year=False):
    """读取 CSV/Excel 题目文件并标准化。"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    elif suffix in {".csv", ".txt"}:
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                raw = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            raise ValueError(f"无法识别文件编码：{path}") from last_error
    else:
        raise ValueError("仅支持 .csv、.txt、.xlsx、.xls 文件。")
    return standardize_question_dataframe(raw, require_year=require_year)


class HistoryStore:
    """基于 SQLite 的历史题目仓库，支持增量导入和稳定去重。"""

    def __init__(self, db_path):
        self.db_path = str(Path(db_path).expanduser().resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    title_normalized TEXT NOT NULL,
                    year INTEGER,
                    major TEXT NOT NULL DEFAULT '未分类',
                    teacher TEXT NOT NULL DEFAULT '未知',
                    source TEXT NOT NULL DEFAULT '',
                    fingerprint TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_questions_year ON questions(year);
                CREATE INDEX IF NOT EXISTS idx_questions_major ON questions(major);
                CREATE INDEX IF NOT EXISTS idx_questions_title_norm ON questions(title_normalized);
                """
            )

    @staticmethod
    def _fingerprint(title, year, major, teacher):
        parts = [normalize_title(title), str(year or ""), major.strip(), teacher.strip()]
        return hashlib.sha256("\u241f".join(parts).encode("utf-8")).hexdigest()

    def add_dataframe(self, df, source=""):
        data = standardize_question_dataframe(df)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for row in data.itertuples(index=False):
            title = _clean_optional_text(row.title)
            year = _clean_year(row.year)
            major = _clean_optional_text(row.major, "未分类") or "未分类"
            teacher = _clean_optional_text(row.teacher, "未知") or "未知"
            rows.append(
                (
                    title,
                    normalize_title(title),
                    year,
                    major,
                    teacher,
                    str(source or ""),
                    self._fingerprint(title, year, major, teacher),
                    now,
                )
            )
        with self._connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO questions
                    (title, title_normalized, year, major, teacher, source, fingerprint, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted = connection.total_changes - before
        return {"submitted": len(rows), "inserted": inserted, "duplicates": len(rows) - inserted}

    def load_dataframe(self):
        with self._connect() as connection:
            result = pd.read_sql_query(
                """
                SELECT id, title, year, major, teacher, source, created_at
                FROM questions
                ORDER BY COALESCE(year, 0), id
                """,
                connection,
            )
        if not result.empty:
            result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
        return result

    def count(self):
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM questions").fetchone()[0])

    def export_csv(self, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.load_dataframe().to_csv(output, index=False, encoding="utf-8-sig")
        return output


def _character_ngrams(text, sizes=(2, 3)):
    value = normalize_title(text)
    grams = set()
    for size in sizes:
        grams.update(value[i:i + size] for i in range(max(0, len(value) - size + 1)))
    return grams or ({value} if value else set())


def character_jaccard(left, right):
    left_set = _character_ngrams(left)
    right_set = _character_ngrams(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def build_text_matrix(texts, backend="auto", model_path="BAAI/bge-m3"):
    """在同一空间编码历史题目与新题目，自动降级到中文字符 TF-IDF。"""
    cleaned = [str(text).strip() for text in texts]
    if not cleaned:
        raise ValueError("没有可用于向量化的题目。")

    semantic_available = HAS_BGE or HAS_SENTENCE_TRANSFORMERS
    if backend in {"auto", "bge"} and semantic_available:
        try:
            encoder = BGEBackend(model_path)
            matrix = np.asarray(encoder.embed(cleaned, verbose=True), dtype=np.float32)
            return matrix, "bge-m3" if encoder.is_flag else "sentence-transformers", None
        except Exception:
            if backend == "bge":
                raise
            print("【提示】语义模型不可用，自动改用中文字符 TF-IDF。")
    elif backend == "bge":
        raise RuntimeError("--embedding bge 需要 FlagEmbedding 或 sentence-transformers。")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(cleaned)
    return matrix, "char-tfidf", vectorizer


def _risk_label(score, exact_duplicate, threshold):
    if exact_duplicate:
        return "重复题目"
    if score >= threshold:
        return "高相似"
    if score >= max(0.60, threshold - 0.20):
        return "中等相似"
    return "低相似"


def compare_new_to_history(history_df, new_df, matrix, backend_used, top_k=5, threshold=0.85):
    history_count = len(history_df)
    history_matrix = matrix[:history_count]
    new_matrix = matrix[history_count:]
    scores = cosine_similarity(new_matrix, history_matrix)
    details = []
    summaries = []

    for new_index, new_row in new_df.reset_index(drop=True).iterrows():
        new_code = f"N{new_index + 1:04d}"
        order = np.argsort(scores[new_index])[::-1][:min(top_k, history_count)]
        top_score = float(scores[new_index, order[0]]) if len(order) else 0.0
        exact_any = False
        top_title = ""
        top_jaccard = 0.0
        for rank, history_index in enumerate(order, start=1):
            history_row = history_df.iloc[int(history_index)]
            semantic_score = float(scores[new_index, history_index])
            exact = normalize_title(new_row["title"]) == normalize_title(history_row["title"])
            lexical_score = character_jaccard(new_row["title"], history_row["title"])
            exact_any = exact_any or exact
            if rank == 1:
                top_title = history_row["title"]
                top_jaccard = lexical_score
            details.append(
                {
                    "新题目编号": new_code,
                    "新题目": new_row["title"],
                    "新题目专业": new_row.get("major", "未分类"),
                    "排名": rank,
                    "历史记录ID": history_row.get("id", ""),
                    "历史题目": history_row["title"],
                    "历史年份": history_row.get("year", pd.NA),
                    "历史专业": history_row.get("major", "未分类"),
                    "历史导师": history_row.get("teacher", "未知"),
                    "相似度": round(semantic_score, 6),
                    "字符Jaccard": round(lexical_score, 6),
                    "完全重复": "是" if exact else "否",
                    "风险等级": _risk_label(semantic_score, exact, threshold),
                    "向量方法": backend_used,
                }
            )
        summaries.append(
            {
                "新题目编号": new_code,
                "新题目": new_row["title"],
                "年份": new_row.get("year", pd.NA),
                "专业": new_row.get("major", "未分类"),
                "导师": new_row.get("teacher", "未知"),
                "题目字数": count_title_length(new_row["title"]),
                "最相似历史题目": top_title,
                "最高相似度": round(top_score, 6),
                "最高字符Jaccard": round(top_jaccard, 6),
                "风险等级": _risk_label(top_score, exact_any, threshold),
                "向量方法": backend_used,
            }
        )
    return pd.DataFrame(details), pd.DataFrame(summaries)


def cluster_new_and_history(history_df, new_df, matrix, max_clusters=8):
    """对历史题目和新题目进行联合聚类，并返回簇概览与二维坐标。"""
    total = len(history_df) + len(new_df)
    if total == 0:
        raise ValueError("聚类数据不能为空。")

    labels = np.zeros(total, dtype=int)
    best_score = None
    best_model = None
    if total >= 3:
        upper = min(max(2, int(max_clusters)), total - 1, max(2, int(math.sqrt(total)) + 2))
        for cluster_count in range(2, upper + 1):
            try:
                model = KMeans(n_clusters=cluster_count, random_state=42, n_init=20)
                candidate_labels = model.fit_predict(matrix)
                if len(np.unique(candidate_labels)) < 2:
                    continue
                kwargs = {"metric": "cosine"}
                if total > 2000:
                    kwargs.update({"sample_size": 2000, "random_state": 42})
                score = float(silhouette_score(matrix, candidate_labels, **kwargs))
                if best_score is None or score > best_score:
                    best_score = score
                    best_model = model
                    labels = candidate_labels
            except (ValueError, TypeError):
                continue

    if total >= 2 and getattr(matrix, "shape", (0, 0))[1] >= 1:
        try:
            components = min(2, matrix.shape[1])
            coordinates = TruncatedSVD(n_components=components, random_state=42).fit_transform(matrix)
            if coordinates.shape[1] == 1:
                coordinates = np.column_stack([coordinates[:, 0], np.zeros(total)])
        except ValueError:
            coordinates = np.column_stack([np.arange(total, dtype=float), np.zeros(total)])
    else:
        coordinates = np.zeros((total, 2), dtype=float)

    history_view = history_df.copy()
    history_view["数据类型"] = "历史题目"
    history_view["题目编号"] = history_view.get("id", pd.Series(range(1, len(history_view) + 1))).map(
        lambda value: f"H{int(value):06d}" if not pd.isna(value) else ""
    )
    new_view = new_df.copy().reset_index(drop=True)
    new_view["数据类型"] = "新题目"
    new_view["题目编号"] = [f"N{i + 1:04d}" for i in range(len(new_view))]
    all_questions = pd.concat([history_view, new_view], ignore_index=True, sort=False)
    all_questions["聚类编号"] = labels.astype(int)
    all_questions["二维坐标X"] = coordinates[:, 0]
    all_questions["二维坐标Y"] = coordinates[:, 1]

    titles = all_questions["title"].astype(str).tolist()
    keyword_matrix = None
    keyword_vectorizer = None
    try:
        keyword_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), max_features=8000, sublinear_tf=True
        )
        keyword_matrix = keyword_vectorizer.fit_transform(titles)
    except ValueError:
        pass

    cluster_rows = []
    for cluster_id in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == cluster_id)
        subset = all_questions.iloc[indices]
        keywords = []
        if keyword_matrix is not None:
            mean_weights = np.asarray(keyword_matrix[indices].mean(axis=0)).ravel()
            features = keyword_vectorizer.get_feature_names_out()
            top_indices = mean_weights.argsort()[::-1]
            keywords = [features[i] for i in top_indices if mean_weights[i] > 0][:8]

        representative_titles = subset["title"].astype(str).head(3).tolist()
        try:
            centroid = matrix[indices].mean(axis=0)
            center_scores = cosine_similarity(matrix[indices], centroid).ravel()
            representative_titles = [
                all_questions.iloc[indices[i]]["title"]
                for i in center_scores.argsort()[::-1][:3]
            ]
        except (ValueError, TypeError):
            pass

        cluster_rows.append(
            {
                "聚类编号": int(cluster_id),
                "题目总数": int(len(indices)),
                "历史题目数": int((subset["数据类型"] == "历史题目").sum()),
                "新题目数": int((subset["数据类型"] == "新题目").sum()),
                "聚类关键词": "、".join(keywords),
                "代表题目": "；".join(representative_titles),
                "本簇新题目": "；".join(subset.loc[subset["数据类型"] == "新题目", "title"].astype(str)),
            }
        )

    cluster_summary = pd.DataFrame(cluster_rows)
    metrics = {
        "cluster_count": int(len(np.unique(labels))),
        "silhouette_score": None if best_score is None else round(best_score, 6),
        "selection": "silhouette" if best_model is not None else "single_cluster_small_or_degenerate_data",
    }
    return all_questions, cluster_summary, metrics


def _top_terms(texts, stopwords_set, top_n=15):
    terms = []
    for text in texts:
        if HAS_JIEBA:
            candidates = jieba.lcut(str(text))
        else:
            normalized = normalize_title(text)
            candidates = [normalized[i:i + 2] for i in range(max(0, len(normalized) - 1))]
        for word in candidates:
            word = str(word).strip().lower()
            if len(word) < 2 or word in stopwords_set or re.fullmatch(r"[\W\d_]+", word):
                continue
            terms.append(word)
    return [word for word, _ in Counter(terms).most_common(top_n)]


def build_descriptive_profiles(history_df, new_df, stopwords_set):
    """生成新旧题目的规模、长度、年份、专业和高频词描述。"""
    rows = []
    for label, data in (("历史题目", history_df), ("新题目", new_df)):
        lengths = data["title"].map(count_title_length)
        valid_years = pd.to_numeric(data["year"], errors="coerce").dropna().astype(int)
        rows.append(
            {
                "数据集": label,
                "题目数": int(len(data)),
                "平均字数": round(float(lengths.mean()), 2) if len(lengths) else 0.0,
                "中位数字数": round(float(lengths.median()), 2) if len(lengths) else 0.0,
                "最短字数": int(lengths.min()) if len(lengths) else 0,
                "最长字数": int(lengths.max()) if len(lengths) else 0,
                "标题唯一率": round(data["title"].map(normalize_title).nunique() / len(data), 4) if len(data) else 0.0,
                "年份范围": (
                    f"{valid_years.min()}-{valid_years.max()}" if len(valid_years) else "未知"
                ),
                "专业数": int(data["major"].nunique(dropna=True)),
                "导师数": int(data["teacher"].nunique(dropna=True)),
                "高频词": "、".join(_top_terms(data["title"], stopwords_set)),
            }
        )
    return pd.DataFrame(rows)


def plot_cluster_result(cluster_df, output_path):
    """保存一张突出显示新题目的二维聚类图。"""
    if cluster_df.empty:
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 9))
    cluster_ids = sorted(cluster_df["聚类编号"].unique())
    palette = sns.color_palette("tab20", max(1, len(cluster_ids)))
    color_map = dict(zip(cluster_ids, palette))
    for cluster_id in cluster_ids:
        cluster_part = cluster_df[cluster_df["聚类编号"] == cluster_id]
        for data_type, marker, size in (("历史题目", "o", 38), ("新题目", "*", 180)):
            part = cluster_part[cluster_part["数据类型"] == data_type]
            if part.empty:
                continue
            plt.scatter(
                part["二维坐标X"], part["二维坐标Y"],
                marker=marker, s=size, color=color_map[cluster_id],
                alpha=0.65 if data_type == "历史题目" else 1.0,
                edgecolors="white", linewidths=0.6,
                label=f"簇{cluster_id}-{data_type}",
            )
    for _, row in cluster_df[cluster_df["数据类型"] == "新题目"].iterrows():
        label = str(row["title"])
        label = label if len(label) <= 18 else label[:18] + "…"
        plt.annotate(label, (row["二维坐标X"], row["二维坐标Y"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    plt.title("新题目与历史题目联合聚类")
    plt.xlabel("降维坐标 1")
    plt.ylabel("降维坐标 2")
    plt.grid(linestyle="--", alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()
    return output


def _dataframe_records(df):
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False))


def build_local_explanation(new_summary, similarity_details, cluster_summary, profiles, threshold):
    """无网络或无 API Key 时的可审计规则解释。"""
    high_count = int(new_summary["风险等级"].isin(["重复题目", "高相似"]).sum())
    lines = [
        "# 新题目对比智能解释",
        "",
        "## 总体判断",
        "",
        f"本次分析 {len(new_summary)} 个新题目，其中 {high_count} 个达到重复/高相似等级。"
        f"高相似阈值设为 {threshold:.2f}；该结果用于选题初筛，不等同于学术不端认定。",
        "",
        "## 单题解释与建议",
        "",
    ]
    cluster_lookup = {}
    for _, row in cluster_summary.iterrows():
        cluster_lookup[int(row["聚类编号"])] = row.get("聚类关键词", "")

    for _, row in new_summary.iterrows():
        code = row["新题目编号"]
        risk = row["风险等级"]
        score = float(row["最高相似度"])
        if risk == "重复题目":
            advice = "建议更换研究对象、核心变量或方法，并避免仅调整虚词。"
        elif risk == "高相似":
            advice = "建议核查研究问题的增量，至少从样本、场景、变量关系或方法上形成明确差异。"
        elif risk == "中等相似":
            advice = "方向可继续，但应在摘要或开题说明中突出与相近选题的差异。"
        else:
            advice = "与历史题库的标题重合较低，可继续进行内容层面的人工审查。"
        cluster_id = row.get("所属聚类", pd.NA)
        cluster_text = ""
        if not pd.isna(cluster_id):
            keywords = cluster_lookup.get(int(cluster_id), "")
            cluster_text = f"，归入簇 {int(cluster_id)}" + (f"（{keywords}）" if keywords else "")
        lines.extend(
            [
                f"### {code}：{row['新题目']}",
                "",
                f"风险等级为“{risk}”，最高相似度 {score:.3f}{cluster_text}。"
                f"最接近的历史题目是“{row['最相似历史题目']}”。{advice}",
                "",
            ]
        )

    lines.extend(
        [
            "## 数据说明",
            "",
            f"描述性统计覆盖历史题目 {int(profiles.iloc[0]['题目数'])} 条、新题目 {int(profiles.iloc[1]['题目数'])} 条。"
            "相似度、字符重合度和聚类结果应结合专业判断共同使用。",
        ]
    )
    return "\n".join(lines)


def call_qwen_chat(system_prompt, user_prompt, api_key, model, base_url, timeout=60, retries=2):
    """使用千问 OpenAI 兼容 Chat Completions HTTP 接口。"""
    if not api_key:
        raise ValueError("未提供千问 API Key。")
    if "{WorkspaceId}" in base_url:
        raise ValueError("QWEN_BASE_URL 中的 {WorkspaceId} 尚未替换。")
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1800,
        "stream": False,
    }
    request_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=request_data,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            if not str(content).strip():
                raise ValueError("千问返回了空内容。")
            return str(content).strip()
        except urllib.error.HTTPError as exc:
            last_error = RuntimeError(f"千问 API 返回 HTTP {exc.code}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                break
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                break
        time.sleep(2 ** attempt)
    raise RuntimeError(f"千问调用失败：{type(last_error).__name__}") from last_error


def generate_smart_explanation(
    new_summary,
    similarity_details,
    cluster_summary,
    profiles,
    threshold,
    llm_mode="auto",
    api_key_env="DASHSCOPE_API_KEY",
    qwen_model=DEFAULT_QWEN_MODEL,
    qwen_base_url=DEFAULT_QWEN_BASE_URL,
):
    local_report = build_local_explanation(
        new_summary, similarity_details, cluster_summary, profiles, threshold
    )
    if llm_mode == "off":
        return local_report, "local_rules"

    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        note = "\n\n> 未检测到千问环境变量，本报告由本地规则生成。"
        return local_report + note, "local_rules_no_api_key"

    top_matches = (
        similarity_details.sort_values(["新题目编号", "排名"])
        .groupby("新题目编号", group_keys=False)
        .head(3)
    )
    analysis_payload = {
        "threshold": threshold,
        "descriptive_profiles": _dataframe_records(profiles),
        "new_question_summary": _dataframe_records(new_summary.head(30)),
        "top_history_matches": _dataframe_records(top_matches.head(90)),
        "cluster_summary": _dataframe_records(cluster_summary),
    }
    system_prompt = (
        "你是严谨的高校论文选题审查助手。只依据给定结构化分析结果解释，不虚构论文内容。"
        "请用中文 Markdown 输出：总体结论、逐题风险与证据、聚类含义、可操作的差异化修改建议、方法局限。"
        "明确说明标题相似不等同于抄袭，所有数值保留三位小数。"
    )
    user_prompt = "请解释以下新题目与历史题库的分析结果：\n" + json.dumps(
        analysis_payload, ensure_ascii=False
    )
    try:
        explanation = call_qwen_chat(
            system_prompt, user_prompt, api_key, qwen_model, qwen_base_url
        )
        return "# 千问智能解释\n\n" + explanation, f"qwen:{qwen_model}"
    except Exception as exc:
        fallback_note = f"\n\n> 千问调用未成功（{type(exc).__name__}），已自动回退到本地规则解释。"
        return local_report + fallback_note, "local_rules_api_fallback"


def save_compare_results(
    output_dir,
    profiles,
    new_summary,
    similarity_details,
    cluster_df,
    cluster_summary,
    explanation,
    metadata,
):
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "profiles": output / "01_新旧题目描述性统计.csv",
        "new_summary": output / "02_新题目分析汇总.csv",
        "similarities": output / "03_新题目与历史题目相似度明细.csv",
        "cluster_summary": output / "04_聚类概览.csv",
        "cluster_details": output / "05_聚类明细.csv",
        "explanation": output / "06_智能解释.md",
        "metadata": output / "07_分析元数据.json",
        "cluster_plot": output / "08_新旧题目聚类图.png",
    }
    profiles.to_csv(paths["profiles"], index=False, encoding="utf-8-sig")
    new_summary.to_csv(paths["new_summary"], index=False, encoding="utf-8-sig")
    similarity_details.to_csv(paths["similarities"], index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(paths["cluster_summary"], index=False, encoding="utf-8-sig")
    cluster_df.to_csv(paths["cluster_details"], index=False, encoding="utf-8-sig")
    paths["explanation"].write_text(explanation, encoding="utf-8")
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    plot_cluster_result(cluster_df, paths["cluster_plot"])
    return paths


def run_new_question_analysis(
    history_df,
    new_df,
    output_dir,
    embedding="auto",
    model_path="BAAI/bge-m3",
    top_k=5,
    threshold=0.85,
    max_clusters=8,
    llm_mode="auto",
    api_key_env="DASHSCOPE_API_KEY",
    qwen_model=DEFAULT_QWEN_MODEL,
    qwen_base_url=DEFAULT_QWEN_BASE_URL,
):
    history = standardize_question_dataframe(history_df)
    new_questions = standardize_question_dataframe(new_df)
    if history.empty:
        raise ValueError("历史题库为空，请先使用 ingest 命令或 --history 导入历史题目。")
    if new_questions.empty:
        raise ValueError("没有有效的新题目。")
    if not 0 < threshold <= 1:
        raise ValueError("相似度阈值必须在 (0, 1] 范围内。")
    if top_k < 1:
        raise ValueError("top-k 必须大于或等于 1。")

    combined_titles = history["title"].tolist() + new_questions["title"].tolist()
    matrix, backend_used, _ = build_text_matrix(
        combined_titles, backend=embedding, model_path=model_path
    )
    similarity_details, new_summary = compare_new_to_history(
        history, new_questions, matrix, backend_used, top_k=top_k, threshold=threshold
    )
    cluster_df, cluster_summary, cluster_metrics = cluster_new_and_history(
        history, new_questions, matrix, max_clusters=max_clusters
    )

    new_cluster_map = (
        cluster_df[cluster_df["数据类型"] == "新题目"]
        .set_index("题目编号")["聚类编号"]
        .to_dict()
    )
    new_summary["所属聚类"] = new_summary["新题目编号"].map(new_cluster_map).astype("Int64")
    similarity_details["所属聚类"] = similarity_details["新题目编号"].map(new_cluster_map).astype("Int64")
    profiles = build_descriptive_profiles(history, new_questions, load_stopwords())
    explanation, explanation_source = generate_smart_explanation(
        new_summary,
        similarity_details,
        cluster_summary,
        profiles,
        threshold,
        llm_mode=llm_mode,
        api_key_env=api_key_env,
        qwen_model=qwen_model,
        qwen_base_url=qwen_base_url,
    )
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history_count": len(history),
        "new_count": len(new_questions),
        "embedding_backend": backend_used,
        "similarity_threshold": threshold,
        "top_k": top_k,
        "clustering": cluster_metrics,
        "explanation_source": explanation_source,
        "qwen_model": qwen_model if explanation_source.startswith("qwen:") else None,
    }
    paths = save_compare_results(
        output_dir,
        profiles,
        new_summary,
        similarity_details,
        cluster_df,
        cluster_summary,
        explanation,
        metadata,
    )
    return {
        "profiles": profiles,
        "new_summary": new_summary,
        "similarity_details": similarity_details,
        "cluster_df": cluster_df,
        "cluster_summary": cluster_summary,
        "metadata": metadata,
        "paths": paths,
    }


def _collect_new_questions(args):
    frames = []
    if args.new_file:
        frames.append(read_question_file(args.new_file))
    if args.title:
        frames.append(
            standardize_question_dataframe(
                pd.DataFrame(
                    {
                        "title": args.title,
                        "year": [args.year] * len(args.title),
                        "major": [args.major] * len(args.title),
                        "teacher": [args.teacher] * len(args.title),
                    }
                )
            )
        )
    if not frames:
        raise ValueError("compare 命令至少需要 --new-file 或一个 --title。")
    return pd.concat(frames, ignore_index=True)


def command_ingest(args):
    store = HistoryStore(args.db)
    data = read_question_file(args.history)
    stats = store.add_dataframe(data, source=Path(args.history).name)
    print(
        f"历史题库导入完成：提交 {stats['submitted']} 条，新增 {stats['inserted']} 条，"
        f"重复跳过 {stats['duplicates']} 条；当前共 {store.count()} 条。"
    )
    print(f"数据库：{store.db_path}")
    return 0


def command_export(args):
    store = HistoryStore(args.db)
    path = store.export_csv(args.output_file)
    print(f"已导出 {store.count()} 条历史题目：{path}")
    return 0


def command_compare(args):
    store = HistoryStore(args.db)
    if args.history:
        imported = read_question_file(args.history)
        stats = store.add_dataframe(imported, source=Path(args.history).name)
        print(f"历史文件预导入：新增 {stats['inserted']} 条，重复跳过 {stats['duplicates']} 条。")

    history = store.load_dataframe()
    new_questions = _collect_new_questions(args)
    result = run_new_question_analysis(
        history,
        new_questions,
        args.output,
        embedding=args.embedding,
        model_path=args.model_path,
        top_k=args.top_k,
        threshold=args.threshold,
        max_clusters=args.max_clusters,
        llm_mode=args.llm,
        api_key_env=args.api_key_env,
        qwen_model=args.qwen_model,
        qwen_base_url=args.qwen_base_url,
    )

    store_stats = None
    if args.store_new:
        source = Path(args.new_file).name if args.new_file else "命令行新题目"
        store_stats = store.add_dataframe(new_questions, source=f"分析后入库:{source}")
    export_path = store.export_csv(Path(args.output) / "09_历史题库导出.csv")

    summary = result["new_summary"]
    risk_counts = summary["风险等级"].value_counts().to_dict()
    print("\n新题目分析完成。")
    print(f"历史题目：{len(history)} 条；新题目：{len(new_questions)} 条。")
    print(f"相似度方法：{result['metadata']['embedding_backend']}；风险分布：{risk_counts}")
    print(f"解释来源：{result['metadata']['explanation_source']}")
    if store_stats is not None:
        print(f"新题目入库：新增 {store_stats['inserted']} 条，重复跳过 {store_stats['duplicates']} 条。")
    print(f"结果目录：{Path(args.output).expanduser().resolve()}")
    print(f"历史题库导出：{export_path}")
    return 0


def command_batch(args):
    """保留原脚本的完整批处理流程。"""
    global CSV_FILE_PATH, STOPWORDS_PATH, TECH_DICT_PATH, BASE_OUTPUT_DIR
    CSV_FILE_PATH = args.csv
    STOPWORDS_PATH = args.stopwords
    TECH_DICT_PATH = args.tech_dict
    BASE_OUTPUT_DIR = args.output

    dirs = init_environment()
    stopwords = load_stopwords()
    data = load_data(CSV_FILE_PATH)
    if data is None or data.empty:
        raise ValueError("数据读取失败或为空。")
    embedder = BGEBackend(args.model_path)
    run_descriptive_analysis(data, dirs, stopwords)
    run_similarity_analysis(data, embedder, dirs)
    run_clustering_analysis(data, embedder, dirs)
    print(f"完整批量分析已完成：{Path(BASE_OUTPUT_DIR).expanduser().resolve()}")
    return 0


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="论文题目历史库、新题查重、描述分析、相似度、聚类与千问解释工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command")

    ingest = commands.add_parser("ingest", help="将 CSV/Excel 历史题目增量写入 SQLite")
    ingest.add_argument("--history", required=True, help="历史题目 CSV/Excel 文件")
    ingest.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite 历史题库路径")
    ingest.set_defaults(handler=command_ingest)

    compare = commands.add_parser("compare", help="将一个或多个新题目与历史题库比较")
    compare.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite 历史题库路径")
    compare.add_argument("--history", help="比较前先增量导入此历史 CSV/Excel")
    compare.add_argument("--new-file", help="包含新题目的 CSV/Excel")
    compare.add_argument("--title", action="append", help="直接提供一个新题目；可重复使用")
    compare.add_argument("--year", type=int, help="通过 --title 输入题目的年份")
    compare.add_argument("--major", default="未分类", help="通过 --title 输入题目的专业")
    compare.add_argument("--teacher", default="未知", help="通过 --title 输入题目的导师")
    compare.add_argument("--output", default=DEFAULT_COMPARE_OUTPUT, help="分析结果目录")
    compare.add_argument("--top-k", type=int, default=5, help="每个新题返回的相近历史题数")
    compare.add_argument("--threshold", type=float, default=0.85, help="高相似阈值")
    compare.add_argument("--embedding", choices=["auto", "bge", "tfidf"], default="auto")
    compare.add_argument("--model-path", default="BAAI/bge-m3", help="BGE 模型名称或本地路径")
    compare.add_argument("--max-clusters", type=int, default=8, help="自动聚类时允许的最大簇数")
    compare.add_argument("--store-new", action="store_true", help="分析完成后把新题目写入历史题库")
    compare.add_argument("--llm", choices=["auto", "on", "off"], default="auto", help="千问解释策略")
    compare.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="保存千问密钥的环境变量名")
    compare.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL, help="千问模型名")
    compare.add_argument(
        "--qwen-base-url",
        default=DEFAULT_QWEN_BASE_URL,
        help="千问 OpenAI 兼容 base URL；可改为业务空间专属地址",
    )
    compare.set_defaults(handler=command_compare)

    export = commands.add_parser("export", help="将 SQLite 历史题库导出为 CSV")
    export.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite 历史题库路径")
    export.add_argument("--output-file", required=True, help="导出 CSV 路径")
    export.set_defaults(handler=command_export)

    batch = commands.add_parser("batch", help="运行原脚本的完整描述/相似度/BERTopic 流程")
    batch.add_argument("--csv", default=CSV_FILE_PATH, help="历史数据 CSV")
    batch.add_argument("--stopwords", default=STOPWORDS_PATH, help="停用词文件")
    batch.add_argument("--tech-dict", default=TECH_DICT_PATH, help="jieba 专业词典")
    batch.add_argument("--output", default=BASE_OUTPUT_DIR, help="批处理输出目录")
    batch.add_argument("--model-path", default="BAAI/bge-m3", help="BGE 模型名称或本地路径")
    batch.set_defaults(handler=command_batch)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        return int(args.handler(args) or 0)
    except KeyboardInterrupt:
        print("\n操作已取消。")
        return 130
    except Exception as exc:
        print(f"【错误】{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
