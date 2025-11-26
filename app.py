"""
AI 文章檢測器 Streamlit 應用程式
"""

import streamlit as st
import time
import numpy as np
from typing import Dict
import re
from collections import Counter

# ==================== AI 檢測模型 ====================

class SimpleAIDetector:
    """
    簡化版的 AI 檢測器，使用基本的文字特徵分析
    結合多種啟發式規則來提高準確度
    """
    
    def __init__(self):
        """初始化檢測器"""
        # AI 常用的連接詞和轉折詞
        self.ai_markers = {
            'however', 'moreover', 'furthermore', 'additionally', 'consequently',
            'therefore', 'thus', 'hence', 'nevertheless', 'nonetheless'
        }
        
    def calculate_perplexity_score(self, text: str) -> float:
        """
        計算文字的複雜度分數
        AI 文章通常有較低的複雜度（更流暢）
        """
        words = text.lower().split()
        if len(words) < 2:
            return 0.5
        
        # 計算詞彙多樣性
        unique_ratio = len(set(words)) / len(words)
        
        # 檢查重複的 bigrams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0.5
        
        return (unique_ratio + bigram_diversity) / 2
    
    def check_sentence_uniformity(self, sentences: list) -> float:
        """檢查句子長度的均勻性 - AI 通常更均勻"""
        if len(sentences) < 2:
            return 0.5
        
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5
        
        # 計算變異係數
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        cv = std_len / mean_len if mean_len > 0 else 0
        
        # CV 越小表示越均勻（更像 AI）
        uniformity_score = max(0, min(1, 1 - cv))
        return uniformity_score
    
    def count_ai_markers(self, text: str) -> float:
        """計算 AI 常用詞的出現頻率"""
        words = set(text.lower().split())
        marker_count = len(words.intersection(self.ai_markers))
        return min(1.0, marker_count / 3)  # 正規化到 0-1
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        使用多種啟發式規則預測
        """
        if not text or len(text.strip()) < 10:
            return {
                "prediction": "Unknown",
                "ai_probability": 0.5,
                "human_probability": 0.5,
                "confidence": 0
            }
        
        # 計算基本特徵
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # 多維度評分
        ai_score = 0.0
        weights = []
        
        # 1. 句子長度均勻性 (權重: 25%)
        uniformity = self.check_sentence_uniformity(sentences)
        ai_score += uniformity * 0.25
        weights.append(0.25)
        
        # 2. 文字複雜度 (權重: 20%)
        perplexity = self.calculate_perplexity_score(text)
        # AI 文章通常有較高的複雜度分數（更流暢）
        ai_score += perplexity * 0.20
        weights.append(0.20)
        
        # 3. AI 常用詞標記 (權重: 15%)
        marker_score = self.count_ai_markers(text)
        ai_score += marker_score * 0.15
        weights.append(0.15)
        
        # 4. 平均句子長度 (權重: 20%)
        # AI 通常保持在 15-25 個詞之間
        if 15 <= avg_sentence_length <= 25:
            sentence_score = 1.0
        elif 10 <= avg_sentence_length < 15 or 25 < avg_sentence_length <= 30:
            sentence_score = 0.6
        else:
            sentence_score = 0.3
        ai_score += sentence_score * 0.20
        weights.append(0.20)
        
        # 5. 用詞正式度 (權重: 10%)
        # AI 通常用較長的詞
        formality_score = min(1.0, (avg_word_length - 3) / 4) if avg_word_length > 3 else 0
        ai_score += formality_score * 0.10
        weights.append(0.10)
        
        # 6. 文章完整度 (權重: 10%)
        # AI 通常產生較完整的文章
        completeness_score = min(1.0, word_count / 100) if word_count > 50 else 0.3
        ai_score += completeness_score * 0.10
        weights.append(0.10)
        
        # 正規化 AI 機率
        ai_prob = min(0.95, max(0.05, ai_score))
        human_prob = 1 - ai_prob
        
        prediction = "AI" if ai_prob > 0.5 else "Human"
        confidence = abs(ai_prob - 0.5) * 200  # 轉換為 0-100
        
        return {
            "prediction": prediction,
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "confidence": confidence
        }
    
    def analyze_text_features(self, text: str) -> Dict[str, any]:
        """分析文字特徵"""
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "vocabulary_diversity": len(set(words)) / len(words) if words else 0
        }

# ==================== Streamlit 應用程式 ====================

@st.cache_resource
def load_model():
    """載入 AI 檢測模型"""
    detector = SimpleAIDetector()
    return detector, True

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

st.markdown('<div class="main-header">🤖 AI 文章檢測器</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">檢測文章是由 AI 還是人類撰寫</div>', unsafe_allow_html=True)

# 側邊欄
with st.sidebar:
    st.header("ℹ️ 關於")
    st.info("""
    這個工具使用機器學習演算法來分析文章內容，
    判斷文章是由 AI 還是人類撰寫。
    
    **使用方法：**
    1. 在文字框中輸入或貼上文章
    2. 點擊「開始檢測」按鈕
    3. 查看檢測結果
    
    **檢測特徵：**
    - 句子長度均勻性
    - 詞彙多樣性
    - 用詞正式度
    - AI 常用詞標記
    """)
    
    st.header("📊 模型資訊")
    if st.button("載入模型"):
        with st.spinner("正在載入模型..."):
            st.session_state.detector, st.session_state.model_loaded = load_model()
        st.success("✅ 檢測器已就緒！")
    
    if st.session_state.detector is not None:
        st.success("✅ 模型已就緒")
    else:
        st.warning("⚠️ 請先載入模型")
    
    st.header("📝 範例文章")
    if st.button("載入 AI 文章範例"):
        st.session_state.text_input = """Artificial intelligence has revolutionized numerous industries in recent years. Machine learning algorithms can now process vast amounts of data with unprecedented efficiency. These technological advancements have enabled computers to perform tasks that were once exclusively human domains. From natural language processing to image recognition, AI systems continue to demonstrate remarkable capabilities. The integration of deep learning techniques has particularly enhanced the performance of these systems."""
        st.rerun()
    
    if st.button("載入人類文章範例"):
        st.session_state.text_input = """I remember the first time I tried to write an essay. It was tough! My thoughts were all over the place, and I couldn't figure out how to organize them. But you know what? That's totally normal. Writing is messy. Sometimes I'd write a sentence, hate it, delete it, then write it again almost the same way. That's just how it goes, right?"""
        st.rerun()

# 主要內容區域
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 輸入文章")
    
    # 文字輸入區
    text_input = st.text_area(
        "請輸入或貼上要檢測的文章內容：",
        value=st.session_state.get('text_input', ''),
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
        if show_details:
            st.subheader("📝 文字特徵分析")
            features = st.session_state.detector.analyze_text_features(text_input)
            
            cols = st.columns(5)
            
            with cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>字數</h4>
                    <h2>{features['word_count']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>句子數</h4>
                    <h2>{features['sentence_count']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>平均詞長</h4>
                    <h2>{features['avg_word_length']:.1f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>平均句長</h4>
                    <h2>{features['avg_sentence_length']:.1f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[4]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>詞彙多樣性</h4>
                    <h2>{features['vocabulary_diversity']:.2f}</h2>
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
