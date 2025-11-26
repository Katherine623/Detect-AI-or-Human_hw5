# AI 文章檢測器 (AI Text Detector)

這是一個使用機器學習模型來檢測文章是由 AI 還是人類撰寫的應用程式。

> 💡 **靈感來源：** [JustDone AI Detector](https://justdone.com/ai-detector)

## 功能特色

- 🤖 檢測文章是由 AI 或人類撰寫
- 📊 顯示預測信心分數
- 🎯 使用預訓練的深度學習模型
- 🚀 簡單易用的網頁介面

## 安裝步驟

1. 克隆此專案
```bash
git clone <your-repo-url>
cd 5114056002_HW5
```

2. 安裝相依套件
```bash
pip install -r requirements.txt
```

3. 執行應用程式
```bash
streamlit run app.py
```

## 使用方法

1. 在網頁介面的文字框中輸入或貼上要檢測的文章
2. 點擊「檢測」按鈕
3. 查看檢測結果和信心分數

## 技術棧

- **Frontend**: Streamlit
- **Model**: Transformer-based model (RoBERTa)
- **Backend**: Python, PyTorch, Hugging Face Transformers

## 部署到 Streamlit Cloud

1. 將程式碼推送到 GitHub
2. 前往 [Streamlit Cloud](https://streamlit.io/cloud)
3. 連接你的 GitHub repository
4. 選擇主檔案 `app.py`
5. 點擊部署

## 授權

MIT License
