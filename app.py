import streamlit as st
import subprocess
import pandas as pd
from pathlib import Path

# 页面设置
st.set_page_config(page_title="论文题目分析工具", layout="wide")
st.title("📚 论文题目相似度与聚类分析")

# ----- 侧边栏：参数调整 (对应你的需求 B) -----
st.sidebar.header("⚙️ 参数调整")
threshold = st.sidebar.slider(
    "相似度阈值 (越高越严格)",
    min_value=0.60,
    max_value=0.95,
    value=0.85,
    step=0.01,
    help="高于此值的题目会被标记为'高相似'或'重复'"
)

# ----- 主界面：文件上传 -----
uploaded_file = st.file_uploader("请上传 CSV 文件（必须包含“题目”和“年份”列）", type=["csv"])

# 定义结果文件夹路径
RESULT_DIR = Path("网页运行结果")

# ----- 核心逻辑：分析按钮 -----
if uploaded_file is not None:
    # 保存上传的文件（加个判断，避免重复写入）
    temp_path = Path("temp_upload.csv")
    if not temp_path.exists() or temp_path.read_bytes() != uploaded_file.getvalue():
        temp_path.write_bytes(uploaded_file.getvalue())
    
    st.success(f"文件 {uploaded_file.name} 上传成功！")
    
    # 预览数据
    try:
        df_preview = pd.read_csv(temp_path, encoding='gbk')  # 修复编码
        st.subheader("数据预览")
        st.dataframe(df_preview.head())
    except Exception as e:
        st.warning(f"预览失败（不影响分析）: {e}")

    # 开始分析按钮
    if st.button("🚀 开始分析（BGE-M3 模型）"):
        with st.spinner("正在分析，首次加载模型需要几秒钟，请耐心等待..."):
            # 构建命令，动态传入阈值
            cmd = [
                "python", "论文1整合_增强版.py",
                "compare",
                "--history", "temp_upload.csv",
                "--new-file", "temp_upload.csv",
                "--model-path", "./bge-m3",
                "--threshold", str(threshold),  # 应用侧边栏的阈值
                "--output", str(RESULT_DIR)
            ]
            
            # 修复日志编码报错 (errors='replace' 解决解码问题)
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='gbk', errors='replace')
            
            # 显示日志（收拢在折叠框里）
            with st.expander("查看详细运行日志"):
                st.text(result.stdout)
                if result.stderr:
                    st.text(result.stderr)
            
            if result.returncode == 0:
                st.success("✅ 分析完成！")
                # 刷新页面后，下面的“结果文件区域”会自动检测到文件夹并显示下载按钮
            else:
                st.error("❌ 分析出错，请检查日志。")

# ----- 关键修复：结果文件持久化显示 (解决你提到的下载问题) -----
# 只要结果文件夹存在，不管页面刷新多少次，这里都会显示下载按钮！
st.markdown("---")  # 分割线
st.subheader("📁 历史分析结果下载")

if RESULT_DIR.exists():
    files = list(RESULT_DIR.glob("*"))
    if files:
        st.info(f"发现 {len(files)} 个结果文件（无需重新分析，直接下载）")
        for file in files:
            if file.is_file():
                with open(file, "rb") as f:
                    st.download_button(
                        label=f"⬇️ 下载 {file.name}",
                        data=f,
                        file_name=file.name,
                        mime="text/csv" if file.suffix == ".csv" else "text/plain",
                        key=file.name  # 给每个按钮一个唯一key，防止冲突
                    )
    else:
        st.caption("暂无结果文件，请先运行一次分析。")
else:
    st.caption("还没有分析结果，请上传文件并点击分析按钮。")