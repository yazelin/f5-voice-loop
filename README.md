# f5-voice-loop

基於 **F5-TTS**（Flow Matching 流匹配非自回歸語音克隆模型）的台灣國語對話迴圈。

```
麥克風 ──► whisper.cpp（本機 GPU）──► llmshare / Groq（雲端）──► F5-TTS（本機 GPU）──► paplay
  arecord        語音轉文字                 生成簡答               流匹配克隆（~2GB 顯存）
```

---

## 🎯 核心優勢與實測數據

### 1. 顯存極省（~2.1 GB VRAM），8GB 筆電多工首選
* **CosyVoice**：需 **4.5 ~ 5.3 GB** 顯存，在 8GB 筆電上極易與背景程式或地端 LLM 衝突導致 CUDA OOM。
* **F5-TTS**：僅需 **~2.1 GB** 顯存，可輕鬆與 `llama-server`（1.2 GB）及 `whisper-server`（0.8 GB）同時常駐！

### 2. 非自回歸架構：零吞字、零跳針
傳統自回歸模型（如 GPT-SoVITS 或早期語音模型）在長句或生僻字容易出現漏字、吞字尾或重複跳針；F5-TTS 採用 Flow Matching 擴散流匹配架構，**語音長度與文字完全線性對齊，發音精準穩定**。

### 3. 原生內建「雙層台灣化系統（`taiwanize.py`）」
* **詞彙層台灣化**：自動將 `視頻`➔`影片`、`內存`➔`記憶體`、`默認`➔`預設`、`項目`➔`專案`、`服務器`➔`伺服器`、`這兒`➔`這裡`。
* **兩岸聲調與破音字修正**：
  * 「倒**垃圾**」➔ 正確發音 **`勒瑟`**（避免大陸普通話 `lājī`）
  * 「**我和你**」➔ 正確發音 **`我汗你`**
  * 「**品質** / 質量」➔ 正確發音 **`品直`**
  * 「**企業**」➔ 正確發音 **`氣業`**
  * 「**星期一**」➔ 正確發音 **`星旗一`**（二聲）
  * 「**微糖微冰**」➔ 正確發音 **`為糖為冰`**（去齒唇化）

### 4. 預設 Jinn 知性音色
預設自動載入 `assets/jinn-tiffy-10s.wav` 與同名逐字稿，啟動即擁有極致自然的台灣口音與說話節奏。

---

## 🚀 快速開始

### 1. 環境安裝
複用既有 PyTorch CUDA 環境，無需重複下載龐大權重：
```bash
cd ~/f5-voice-loop
bash setup.sh
```

### 2. 啟動對話迴圈
```bash
# 預設直接使用 Jinn 聲音（assets/jinn-tiffy-10s.wav）
~/CosyVoice/.venv/bin/python voice_loop.py

# 切換 Groq 超低延遲 LLM 後端
~/CosyVoice/.venv/bin/python voice_loop.py --backend groq

# 指定自訂參考聲音
~/CosyVoice/.venv/bin/python voice_loop.py --voice assets/我的聲音.wav
```

---

## ⌨️ 互動指令清單

在提示字元 `請按 Enter 錄音，或輸入指令` 下：

| 指令 | 說明 | 備註 |
| :--- | :--- | :--- |
| **`:say <問題>`** | **打字向 LLM 提問**（不開麥克風） | 會呼叫 LLM 並記錄到對話記憶（`history`） |
| **`:tts <文字>`** | **純文字發音測試** | 跳過 LLM、不進記憶，純聽 TTS 發音與校音 |
| **`:voice <檔案.wav>`** | 即時切換參考聲音 | 支援同名 `.txt` 自動讀取逐字稿 |
| **`:record`** | 當場錄一段 10 秒聲音當參考音 | 錄完直接切換 |
| **`:speed <倍率>`** | 調整語速 | 例：`:speed 1.1` |
| **`:backend <後端>`** | 切換 LLM 後端 | `llmshare` / `groq` / `local` |
| **`:len <字數>`** | 設定回答字數上限 | 預設 40 字 |
| **`:history`** | 查看當前對話歷史 | 顯示前幾輪問答 |
| **`:clear`** | 清空對話記憶 | 開始新話題 |
| **`:help`** | 顯示指令說明 | |
| **`:q`** 或 `Ctrl+C` | 離開程式 | 立即釋放顯存與記憶體 |

---

## 🛠️ 技術細節

* **聲學模型**：F5-TTS Base (`model_1250000.safetensors` ~1.3 GB)
* **聲碼器 (Vocoder)**：Vocos Mel-24kHz
* **文字前處理**：`taiwanize.py` ➔ `rjieba` ➔ `pypinyin`
* **STT 引擎**：Whisper small（掛載 `-nf` 關閉溫度回退、`-sns` 抑制非語音、`-nth 0.6` 靜音過濾，徹底解決幻聽卡死問題）

---

## 📜 授權

MIT © 林亞澤 (yazelin)
