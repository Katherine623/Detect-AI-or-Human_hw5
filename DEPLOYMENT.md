# Git 設定和部署指南

## 步驟 1：初始化 Git Repository

在專案目錄中執行以下命令：

```powershell
# 進入專案目錄
cd "c:\Users\wecan\OneDrive\桌面\在職專班\資工\資訊處理w3\5114056002_HW5"

# 初始化 Git
git init

# 設定你的 Git 使用者資訊（如果還沒設定）
git config user.name "你的名字"
git config user.email "你的email@example.com"

# 添加所有檔案
git add .

# 提交
git commit -m "Initial commit: AI Text Detector"
```

## 步驟 2：在 GitHub 創建 Repository

1. 前往 https://github.com
2. 登入你的帳號
3. 點擊右上角的 "+" 按鈕，選擇 "New repository"
4. 輸入 Repository 名稱（例如：ai-text-detector）
5. 選擇 Public 或 Private
6. 不要勾選 "Initialize this repository with a README"
7. 點擊 "Create repository"

## 步驟 3：連接並推送到 GitHub

```powershell
# 添加遠端 repository（替換成你的 GitHub username 和 repo 名稱）
git remote add origin https://github.com/你的username/ai-text-detector.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

## 步驟 4：部署到 Streamlit Cloud

1. 前往 https://streamlit.io/cloud
2. 使用 GitHub 帳號登入
3. 點擊 "New app"
4. 選擇：
   - Repository: 你剛創建的 repository
   - Branch: main
   - Main file path: app.py
5. 點擊 "Deploy!"
6. 等待部署完成（約 2-5 分鐘）

## 步驟 5：分享你的應用

部署完成後，你會獲得一個 URL，例如：
`https://你的username-ai-text-detector-app-xxxxx.streamlit.app`

你可以分享這個 URL 給任何人使用！

## 更新應用程式

當你修改程式碼後：

```powershell
git add .
git commit -m "描述你的修改"
git push
```

Streamlit Cloud 會自動檢測到變更並重新部署。

## 常見問題

**Q: 如何在本地測試？**
```powershell
# 安裝相依套件
pip install -r requirements.txt

# 執行應用程式
streamlit run app.py
```

**Q: 模型載入太慢怎麼辦？**
- 模型會在首次使用時下載，之後會被快取
- 可以考慮使用更輕量的模型

**Q: Streamlit Cloud 部署失敗？**
- 檢查 requirements.txt 是否正確
- 確保沒有超過資源限制
- 查看部署日誌了解錯誤訊息
