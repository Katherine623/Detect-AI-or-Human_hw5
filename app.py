"""
AI 文章檢測器 Streamlit 應用程式
"""

import streamlit as st
import time
from model import AIDetector, SimpleAIDetector

# 頁面設定
st.set_page_config(
    page_title="AI 文章檢測器",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .ai-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .human-result {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """載入 AI 檢測模型"""
    try:
        detector = AIDetector()
        return detector, True
    except Exception as e:
        st.warning(f"⚠️ 無法載入深度學習模型，使用簡化版檢測器。錯誤: {str(e)}")
        detector = SimpleAIDetector()
        return detector, False

# 主標題
st.markdown('<div class="main-header">🤖 AI 文章檢測器</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">檢測文章是由 AI 還是人類撰寫</div>', unsafe_allow_html=True)

# 側邊欄
with st.sidebar:
    st.header("ℹ️ 關於")
    st.info("""
    這個工具使用機器學習模型來分析文章內容，
    判斷文章是由 AI 還是人類撰寫。
    
    **使用方法：**
    1. 在文字框中輸入或貼上文章
    2. 點擊「開始檢測」按鈕
    3. 查看檢測結果
    """)
    
    st.header("📊 模型資訊")
    if st.button("載入模型"):
        with st.spinner("正在載入模型..."):
            st.session_state.detector, st.session_state.model_loaded = load_model()
        if st.session_state.model_loaded:
            st.success("✅ 深度學習模型載入成功！")
        else:
            st.info("ℹ️ 使用簡化版檢測器")
    
    if st.session_state.detector is not None:
        st.success("✅ 模型已就緒")
    else:
        st.warning("⚠️ 請先載入模型")
    
    st.header("📝 範例文章")
    if st.button("載入 AI 文章範例"):
        st.session_state.example_text = """Artificial intelligence has revolutionized numerous industries in recent years. Machine learning algorithms can now process vast amounts of data with unprecedented efficiency. These technological advancements have enabled computers to perform tasks that were once exclusively human domains. From natural language processing to image recognition, AI systems continue to demonstrate remarkable capabilities. The integration of deep learning techniques has particularly enhanced the performance of these systems."""
    
    if st.button("載入人類文章範例"):
        st.session_state.example_text = """I remember the first time I tried to write an essay. It was tough! My thoughts were all over the place, and I couldn't figure out how to organize them. But you know what? That's totally normal. Writing is messy. Sometimes I'd write a sentence, hate it, delete it, then write it again almost the same way. That's just how it goes, right?"""

# 主要內容區域
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 輸入文章")
    
    # 文字輸入區
    default_text = st.session_state.get('example_text', '')
    text_input = st.text_area(
        "請輸入或貼上要檢測的文章內容：",
        value=default_text,
        height=300,
        placeholder="在這裡輸入文章內容...",
        key="text_input"
    )
    
    # 文字統計
    if text_input:
        word_count = len(text_input.split())
        char_count = len(text_input)
        st.caption(f"📊 字數統計：{word_count} 個詞 | {char_count} 個字元")
    
    # 檢測按鈕
    detect_button = st.button("🔍 開始檢測", type="primary", use_container_width=True)

with col2:
    st.header("⚙️ 設定")
    
    show_details = st.checkbox("顯示詳細分析", value=True)
    show_probabilities = st.checkbox("顯示機率圖表", value=True)

# 處理檢測
if detect_button:
    if st.session_state.detector is None:
        st.error("❌ 請先在側邊欄載入模型！")
    elif not text_input or len(text_input.strip()) < 10:
        st.warning("⚠️ 請輸入至少 10 個字元的文章內容")
    else:
        # 顯示進度
        with st.spinner("🔍 正在分析文章..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # 進行預測
            result = st.session_state.detector.predict(text_input)
        
        st.success("✅ 分析完成！")
        
        # 顯示結果
        st.header("📊 檢測結果")
        
        # 主要結果
        result_class = "ai-result" if result['prediction'] == "AI" else "human-result"
        result_icon = "🤖" if result['prediction'] == "AI" else "👤"
        
        st.markdown(f"""
        <div class="result-box {result_class}">
            <h2 style="margin:0;">{result_icon} 檢測結果：{result['prediction']}</h2>
            <h3 style="margin-top:1rem;">信心分數：{result['confidence']:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 機率圖表
        if show_probabilities:
            st.subheader("📈 機率分佈")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="🤖 AI 撰寫機率",
                    value=f"{result['ai_probability']*100:.2f}%"
                )
                st.progress(result['ai_probability'])
            
            with col2:
                st.metric(
                    label="👤 人類撰寫機率",
                    value=f"{result['human_probability']*100:.2f}%"
                )
                st.progress(result['human_probability'])
        
        # 詳細分析
        if show_details and hasattr(st.session_state.detector, 'analyze_text_features'):
            st.subheader("📝 文字特徵分析")
            features = st.session_state.detector.analyze_text_features(text_input)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>字數</h4>
                    <h2>{features['word_count']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>句子數</h4>
                    <h2>{features['sentence_count']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>平均詞長</h4>
                    <h2>{features['avg_word_length']:.1f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>平均句長</h4>
                    <h2>{features['avg_sentence_length']:.1f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # 解釋說明
        with st.expander("❓ 如何解讀結果"):
            st.markdown("""
            - **信心分數**：表示模型對預測結果的信心程度（0-100%）
            - **機率分佈**：顯示文章屬於 AI 或人類撰寫的機率
            - **文字特徵**：分析文章的基本統計資訊
            
            **注意事項：**
            - 此工具僅供參考，不保證 100% 準確
            - 較短的文章可能影響檢測準確度
            - 建議結合多種方法進行判斷
            """)

# 頁尾
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎓 5114056002_HW5 | Built with Streamlit & Transformers</p>
</div>
""", unsafe_allow_html=True)
