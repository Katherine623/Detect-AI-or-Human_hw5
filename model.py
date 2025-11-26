"""
AI Text Detector Model
使用預訓練的 RoBERTa 模型來檢測文章是由 AI 還是人類撰寫
"""

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
from typing import Dict

class AIDetector:
    def __init__(self):
        """
        初始化 AI 檢測器
        使用 roberta-base-openai-detector 模型
        """
        self.model_name = "roberta-base-openai-detector"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # 載入預訓練模型和 tokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except:
            # 如果無法載入該模型，使用備用方案
            print("無法載入 roberta-base-openai-detector，使用備用模型...")
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        預測文章是由 AI 還是人類撰寫
        
        Args:
            text: 要檢測的文章內容
            
        Returns:
            包含預測結果的字典：
            - prediction: "AI" 或 "Human"
            - ai_probability: AI 撰寫的機率 (0-1)
            - human_probability: 人類撰寫的機率 (0-1)
            - confidence: 信心分數 (0-100)
        """
        if not text or len(text.strip()) == 0:
            return {
                "prediction": "Unknown",
                "ai_probability": 0.5,
                "human_probability": 0.5,
                "confidence": 0
            }
        
        # Tokenize 輸入文字
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # 移動到相同的裝置
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 進行預測
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # 取得機率值
        probs = probabilities.cpu().numpy()[0]
        
        # 假設 label 0 = Human, label 1 = AI
        human_prob = float(probs[0])
        ai_prob = float(probs[1])
        
        # 判斷預測結果
        prediction = "AI" if ai_prob > human_prob else "Human"
        confidence = max(ai_prob, human_prob) * 100
        
        return {
            "prediction": prediction,
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "confidence": confidence
        }
    
    def analyze_text_features(self, text: str) -> Dict[str, any]:
        """
        分析文字特徵
        
        Args:
            text: 要分析的文字
            
        Returns:
            文字特徵的字典
        """
        words = text.split()
        sentences = text.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "avg_sentence_length": np.mean([len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0
        }

# 簡化版本的檢測器（基於啟發式規則）
class SimpleAIDetector:
    """
    簡化版的 AI 檢測器，使用基本的文字特徵分析
    當無法載入深度學習模型時使用
    """
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        使用簡單的啟發式規則預測
        """
        if not text or len(text.strip()) == 0:
            return {
                "prediction": "Unknown",
                "ai_probability": 0.5,
                "human_probability": 0.5,
                "confidence": 0
            }
        
        # 計算基本特徵
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # 簡單的評分系統
        ai_score = 0
        
        # AI 文章通常有更統一的句子長度
        if 15 <= avg_sentence_length <= 25:
            ai_score += 0.2
        
        # AI 文章通常用詞較為正式
        if avg_word_length > 5:
            ai_score += 0.15
        
        # AI 文章通常結構較為完整
        if word_count > 100:
            ai_score += 0.15
        
        # 標準化分數
        ai_prob = min(max(ai_score + 0.5, 0), 1)
        human_prob = 1 - ai_prob
        
        prediction = "AI" if ai_prob > 0.5 else "Human"
        confidence = abs(ai_prob - 0.5) * 200  # 轉換為 0-100
        
        return {
            "prediction": prediction,
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "confidence": confidence
        }
