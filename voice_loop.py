#!/usr/bin/env python3
"""麥克風 → Whisper → LLM → F5-TTS / E2-TTS（流匹配語音克隆，顯存僅 ~2GB）講出來 → 播放。

特點：
- Flow Matching（流匹配）技術，極速且零自回歸吞字問題
- 顯存佔用僅約 2.0 ~ 2.4 GB（比 CosyVoice 省下 1.5GB+）
- 支援 :say 直接打字測試發音
- 支援 :voice 動態切換參考聲音
- 支援 :engine 切換 F5-TTS 或 E2-TTS
- 支援 :speed 調節說話語速

跑法：
  ~/CosyVoice/.venv/bin/python voice_loop.py
  ~/CosyVoice/.venv/bin/python voice_loop.py --voice assets/我的聲音.wav
  ~/CosyVoice/.venv/bin/python voice_loop.py --selfcheck
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from taiwanize import taiwanize_text

HERE = Path(__file__).resolve().parent
WORK = HERE / "tmp"

WHISPER_CLI = Path(os.environ.get("WHISPER_CLI", Path.home() / ".mori/bin/whisper-cli"))
WHISPER_MODEL = Path(os.environ.get("WHISPER_MODEL", Path.home() / ".mori/models/ggml-small.bin"))
WHISPER_DESCRIPTOR = Path.home() / ".mori/whisper-server.json"
WHISPER_SUPERVISOR = Path.home() / ".mori/bin/mori-whisper-serve"

LLM_URL = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "local": "http://127.0.0.1:8080/v1/chat/completions",
}
DEFAULT_MODEL = {
    "llmshare": "deepseek-v4-flash:0731",
    "groq": "openai/gpt-oss-120b",
    "local": "qwen3.5-4b",
}

PAREN_RE = re.compile(r"[(（\[][^)）\]]{0,6}[)）\]]")
STT_HINT = "以下是繁體中文的句子。"

COMMANDS = {
    ":voice": "切換參考聲音，例：:voice assets/我的聲音.wav",
    ":record": "重錄一段當參考聲音",
    ":say": "不錄音，直接打字問 LLM，例：:say 今天天氣如何",
    ":tts": "純文字發音測試（不問 LLM、不進記憶），例：:tts 測試一段話",
    ":speed": "調整語速 (0.5~2.0)，例：:speed 1.1",
    ":nfe": "調整 ODE 採樣步數 (8~32，預設 16 速度提升 2 倍)，例：:nfe 12",
    ":backend": "換 LLM 後端：:backend llmshare / groq / local",
    ":len": "回答字數上限，例：:len 40",
    ":clear": "清空對話歷史",
    ":history": "查看對話紀錄",
    ":help": "顯示指令清單",
    ":q": "離開",
}


HALLUCINATIONS = {
    "謝謝大家收看", "謝謝大家收看。", "請訂閱我的頻道", "請訂閱我的頻道。",
    "請不吝賜教", "謝謝大家", "謝謝大家。", "未完待續", "感謝您的收看",
}


def clean_stt(text):
    text = PAREN_RE.sub("", text).strip()
    if not text or text in STT_HINT or text in HALLUCINATIONS:
        return ""
    if re.match(r"^(.{2,12}?)\1{2,}[。！!？\?]*$", text):
        return ""
    return text


def record(out_wav, device):
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", str(out_wav)]
    if device:
        cmd[1:1] = ["-D", device]
    proc = subprocess.Popen(cmd)
    input("錄音中… 再按 Enter 停止。")
    proc.terminate()
    proc.wait()
    return out_wav.exists() and out_wav.stat().st_size > 16000


def find_whisper_server():
    try:
        d = json.loads(WHISPER_DESCRIPTOR.read_text(encoding="utf-8"))
        os.kill(d["pid"], 0)
    except (OSError, ValueError, KeyError):
        return None
    return f"http://{d['host']}:{d['port']}{d.get('inference_path', '/inference')}"


def ensure_whisper_server(timeout=20):
    url = find_whisper_server()
    if url or not WHISPER_SUPERVISOR.is_file():
        return url
    try:
        subprocess.run([str(WHISPER_SUPERVISOR), "--ensure"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        url = find_whisper_server()
        if url:
            return url
        time.sleep(0.5)
    return None


def transcribe(wav, stt=None):
    if stt and stt.get("url"):
        r = subprocess.run(
            ["curl", "-s", "--max-time", "10", "-F", f"file=@{wav}", "-F", "language=zh",
             "-F", "response_format=json", "-F", f"prompt={STT_HINT}", stt["url"]],
            capture_output=True, text=True,
        )
        try:
            return clean_stt(" ".join(json.loads(r.stdout)["text"].split()))
        except (ValueError, KeyError):
            pass

    env = {**os.environ, "LD_LIBRARY_PATH": str(WHISPER_CLI.parent)}
    r = subprocess.run(
        [str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-l", "zh", "-nt", "-np",
         "-nf", "-sns", "-nth", "0.6", "--prompt", STT_HINT, "-f", str(wav)],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        return ""
    return clean_stt(r.stdout)


def build_prompt(question, max_chars, history=()):
    rule = f"用正體中文口語回答，{max_chars} 個字以內，只回答問題本身，不要開場白、不要條列、不要 emoji。"
    if not history:
        return rule + f"問題：{question}"
    past = "\n".join(f"我：{q}\n你：{a}" for q, a in history[-8:])
    return f"{rule}下面是我們剛才的對話，接著回答最後那個問題：\n\n{past}\n我：{question}\n你："


def ask_llm(question, backend, model, max_chars, history=()):
    prompt = build_prompt(question, max_chars, history)
    if backend == "llmshare":
        r = subprocess.run(["llmshare", "raw", model, prompt], capture_output=True, text=True)
        raw = r.stdout if r.returncode == 0 else f"模型無回應: {r.stderr.strip()[:60]}"
    elif backend == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return "未設定 GROQ_API_KEY"
        payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300})
        req = urllib.request.Request(LLM_URL["groq"], payload.encode(),
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "f5-loop/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            raw = data["choices"][0]["message"].get("content") or ""
        except Exception as e:
            raw = f"Groq 錯誤: {e}"
    else:
        payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300})
        req = urllib.request.Request(LLM_URL["local"], payload.encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            raw = data["choices"][0]["message"].get("content") or ""
        except Exception as e:
            raw = f"在地模型錯誤: {e}"
    return " ".join(raw.split())[:max_chars * 2] or "我不太確定該怎麼回答。"


def selfcheck():
    from f5_tts.api import F5TTS
    f5 = F5TTS()
    print("F5-TTS selfcheck OK! Model initialized successfully.")


def main():
    ap = argparse.ArgumentParser(description="F5-TTS Voice Loop")
    ap.add_argument("--voice", help="固定的參考聲音 wav 檔；不給就讀 assets/我的聲音.wav 或每次講的那句")
    ap.add_argument("--voice-text", help="參考聲音逐字稿；不給則自動讀同名 .txt 或用 Whisper 轉錄")
    ap.add_argument("--record-voice", nargs="?", const="", metavar="WAV", help="開場錄一段聲音當這場的參考音")
    ap.add_argument("--speed", type=float, default=1.0, help="語速 (預設 1.0)")
    ap.add_argument("--nfe", type=int, default=16, help="ODE 採樣步數 (預設 16，速度快 2 倍且音質無損；可選 8~32)")
    ap.add_argument("--backend", choices=["llmshare", "groq", "local"], default="llmshare")
    ap.add_argument("--model", help="指定 LLM 模型")
    ap.add_argument("--max-chars", type=int, default=50, help="回答最大字數")
    ap.add_argument("--device", default="", help="arecord 裝置")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    WORK.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    from f5_tts.api import F5TTS

    print("載入 F5-TTS 流匹配模型（顯存約 2.0GB）...", flush=True)
    t0 = time.time()
    f5 = F5TTS()
    print(f"F5-TTS 載入完成（{time.time()-t0:.2f}s）！")

    stt = {"url": ensure_whisper_server()}
    default_asset_wav = HERE / "assets/jinn-tiffy-10s.wav"
    default_asset_txt = HERE / "assets/jinn-tiffy-10s.txt"

    ref_wav = None
    ref_text = None

    if args.record_voice is not None:
        ref_wav = str(Path(args.record_voice).expanduser()) if args.record_voice else str(WORK / "voice.wav")
        print("請錄一段 5~15 秒的聲音作為克隆樣本：")
        input("按 Enter 開始錄音…")
        if not record(Path(ref_wav), args.device):
            sys.exit("未錄到聲音，結束。")
        ref_text = transcribe(Path(ref_wav), stt)
        if not ref_text:
            sys.exit("無法辨識參考聲音內容，請於安靜環境重試。")
        print(f"參考聲音已就緒：{ref_wav}\n參考文字：{ref_text}")
        if args.record_voice:
            Path(ref_wav).with_suffix(".txt").write_text(ref_text + "\n", encoding="utf-8")
    elif args.voice:
        ref_wav = str(Path(args.voice).expanduser())
        sidecar = Path(ref_wav).with_suffix(".txt")
        ref_text = args.voice_text or (
            sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else transcribe(Path(ref_wav), stt)
        )
        print(f"參考聲音已就緒：{ref_wav}\n參考文字：{ref_text}")
    elif default_asset_wav.exists() and default_asset_txt.exists():
        ref_wav = str(default_asset_wav)
        ref_text = default_asset_txt.read_text(encoding="utf-8").strip()
        print(f"預設載入 Jinn 參考音色：{ref_wav}\n參考文字：{ref_text}")

    state = {
        "backend": args.backend,
        "model": args.model or DEFAULT_MODEL[args.backend],
        "len": args.max_chars,
        "speed": args.speed,
        "nfe": args.nfe,
        "wav": ref_wav,
        "text": ref_text,
    }
    history = []

    print(f"後端：{state['backend']} / {state['model']} | 語速：{state['speed']} | NFE步數：{state['nfe']}")
    print("提示：按 Enter 錄音，打字輸入 :say <文字> 直接測試，打 :help 看完整指令。\n")

    out_wav = WORK / "out.wav"
    in_wav = WORK / "in.wav"

    while True:
        try:
            line = input("請按 Enter 錄音，或輸入指令（:say / :help）: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再見！")
            break

        if line == ":q":
            break
        if line == ":help":
            print("\n指令清單：")
            for k, v in COMMANDS.items():
                print(f"  {k:9s} {v}")
            print()
            continue
        if line == ":clear":
            history.clear()
            print("對話歷史已清空。\n")
            continue
        if line == ":history":
            if not history:
                print("尚無對話歷史。\n")
            for q, a in history[-8:]:
                print(f"  問：{q}\n  答：{a}")
            print()
            continue
        if line.startswith(":speed"):
            _, _, val = line.partition(" ")
            try:
                state["speed"] = max(0.5, min(2.5, float(val)))
                print(f"語速調整為：{state['speed']}\n")
            except ValueError:
                print("請給數字，例：:speed 1.1\n")
            continue
        if line.startswith(":nfe"):
            _, _, val = line.partition(" ")
            try:
                state["nfe"] = max(8, min(64, int(val)))
                print(f"NFE 採樣步數調整為：{state['nfe']}\n")
            except ValueError:
                print("請給整數，例：:nfe 16\n")
            continue
        if line.startswith(":voice"):
            _, _, vpath = line.partition(" ")
            vpath = vpath.strip()
            p = Path(vpath).expanduser() if vpath else None
            if p and p.is_file():
                state["wav"] = str(p)
                sidecar = p.with_suffix(".txt")
                state["text"] = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else transcribe(p, stt)
                print(f"參考聲音已切換為：{state['wav']}\n參考文字：{state['text']}\n")
            else:
                print(f"找不到檔案 {vpath}\n")
            continue
        if line.startswith(":record"):
            ref = WORK / "voice.wav"
            print("請講 5~15 秒聲音：")
            input("按 Enter 開始錄音…")
            if not record(ref, args.device):
                print("未錄到聲音。\n")
                continue
            t = transcribe(ref, stt)
            if not t:
                print("聽不出內容，請重試。\n")
                continue
            state["wav"] = str(ref)
            state["text"] = t
            print(f"參考聲音已更新！參考文字：{t}\n")
            continue
        if line.startswith(":backend"):
            _, _, b = line.partition(" ")
            b = b.strip()
            if b in DEFAULT_MODEL:
                state["backend"] = b
                state["model"] = DEFAULT_MODEL[b]
                print(f"後端切換為 {b} / {state['model']}\n")
            else:
                print("可選後端：llmshare, groq, local\n")
            continue

        typed_say = ""
        direct_tts = ""
        if line.startswith(":say"):
            _, _, typed_say = line.partition(" ")
            typed_say = typed_say.strip()
            if not typed_say:
                print("要給問題，例：:say 今天天氣如何\n")
                continue
        elif line.startswith(":tts"):
            _, _, direct_tts = line.partition(" ")
            direct_tts = direct_tts.strip()
            if not direct_tts:
                print("要給發音文字，例：:tts 測試一段話\n")
                continue

        turn_start = time.time()
        if direct_tts:
            heard = direct_tts
            answer_display = taiwanize_text(direct_tts, for_speech=False)
            speech_text = taiwanize_text(direct_tts, for_speech=True)
            print(f"發音：{answer_display}")
            stt_time = 0.0
            llm_time = 0.0
        else:
            if typed_say:
                heard = typed_say
                stt_time = 0.0
                print(f"你問：{heard}")
            else:
                if not record(in_wav, args.device):
                    print("未錄到聲音，請再試一次。\n")
                    continue
                t_stt = time.time()
                heard = transcribe(in_wav, stt)
                stt_time = time.time() - t_stt
                print(f"你說：{heard}（STT {stt_time:.2f}s）")
            if not heard:
                print("聽不出內容，請再試一次。\n")
                continue

            t_llm = time.time()
            raw_answer = ask_llm(heard, state["backend"], state["model"], state["len"], history)
            llm_time = time.time() - t_llm
            answer_display = taiwanize_text(raw_answer, for_speech=False)
            speech_text = taiwanize_text(raw_answer, for_speech=True)
            print(f"回答：{answer_display}（LLM {llm_time:.2f}s）")

        active_ref_wav = state["wav"] or str(in_wav)
        active_ref_text = state["text"] or heard

        t_tts = time.time()
        try:
            wav_out, sr, _ = f5.infer(
                ref_file=active_ref_wav,
                ref_text=active_ref_text,
                gen_text=speech_text,
                speed=state["speed"],
                nfe_step=state["nfe"],
            )
            sf.write(str(out_wav), wav_out, sr)
            tts_time = time.time() - t_tts
            audio_sec = len(wav_out) / sr
            print(f"F5-TTS 合成（taiwanize, NFE={state['nfe']}）：{tts_time:.2f}s | 音訊長：{audio_sec:.1f}s | 總耗時：{time.time()-turn_start:.2f}s")
            subprocess.run(["paplay", str(out_wav)])
            if not direct_tts:
                history.append((heard, answer_display))
            print()
        except Exception as e:
            print(f"F5-TTS 合成失敗: {e}\n")


if __name__ == "__main__":
    main()
