# f5-voice-loop

基於 **F5-TTS**（Flow Matching 流匹配非自回歸語音克隆模型）的對話語音迴圈。

```
麥克風 ──► whisper.cpp（本機 GPU）──► llmshare / Groq（雲端）──► F5-TTS（本機 GPU）──► paplay
  arecord        語音轉文字                 生成簡答               流匹配克隆（~2GB 顯存）
```

---

## 核心優勢

1. **顯存極省（~2.0 GB VRAM）**：比 CosyVoice 省下約 1.5 GB 顯存，讓 8GB 顯存機器輕鬆兼顧地端 LLM。
2. **非自回歸架構，零吞字、零跳針**：徹底解決傳統 TTS / GPT-SoVITS 容易漏字或重複唸同一句的問題。
3. **台灣腔情緒克隆**：只需 5~10 秒的參考音檔（例如 `assets/我的聲音.wav`），完整還原台灣人語調、尾音與情感起伏。
4. **`:say` 快速測試**：直接打字測試克隆發音，無需每次都透過麥克風錄音。

---

## 快速開始

### 1. 安裝與依賴檢查
```bash
cd ~/f5-voice-loop
bash setup.sh
```

### 2. 啟動對話迴圈
```bash
# 預設自動載入 Jinn 的聲音（assets/jinn-tiffy-10s.wav）
~/CosyVoice/.venv/bin/python voice_loop.py

# 或指定其他聲音樣本（例如 assets/我的聲音.wav）
~/CosyVoice/.venv/bin/python voice_loop.py --voice assets/我的聲音.wav
```

### 3. 指令說明
在互動提示下輸入：
* `:say <文字>` — 直接測試發音（不錄音、不呼叫 LLM）
* `:voice <檔案.wav>` — 動態切換參考聲音與音色
* `:record` — 當場錄製一段新的 10 秒聲音當參考音
* `:speed <倍率>` — 調整語速（例：`:speed 1.1`）
* `:backend <llmshare/groq/local>` — 切換 LLM 後端
* `:clear` — 清空對話記憶
* `:help` — 顯示說明
* `:q` — 離開

---

## 授權

MIT © 林亞澤
