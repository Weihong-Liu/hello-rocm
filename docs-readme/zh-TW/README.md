<div align=center>
  <img src="../../images/head.png" alt="hello-rocm">
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*開源 · 社群驅動 · 讓 AMD AI 生態更易用*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="./README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>

## hello-rocm 是什麼？

&emsp;&emsp;**hello-rocm** 是一套面向 **AMD GPU + ROCm** 的大語言模型實作教學：涵蓋環境基線、部署、微調與基礎設施／算子主題，協助開發者在 ROCm 上完成推理與訓練流程。

&emsp;&emsp;自 **ROCm 7.10.0** 起，ROCm 已可像 CUDA 一樣在 Python 虛擬環境中安裝，並支援 **Linux 與 Windows**。本倉庫以簡體中文 **根目錄 [README.md](../../README.md)** 為主文件；英文摘要請見 **[README_en.md](../../README_en.md)**。

## 建議學習順序

1. **[00-Environment](../../00-Environment/README.md)** — ROCm、PyTorch、**uv** 等統一環境  
2. **01-Deploy** — LM Studio、vLLM、Ollama、llama.cpp 等部署  
3. **02-Fine-tune** — LoRA 等微調實例  
4. **03-Infra** — HIP／軟體棧與算子相關章節  

## 目錄結構（摘要）

| 路徑 | 說明 |
|:---|:---|
| [00-Environment](../../00-Environment/) | ROCm 基線安裝（Windows／Ubuntu） |
| [01-Deploy](../../01-Deploy/) | 大模型部署教學 |
| [02-Fine-tune](../../02-Fine-tune/) | 微調教學 |
| [03-Infra](../../03-Infra/) | 基礎設施與算子實踐 |
| [04-References](../../04-References/) | 參考資料 |
| [05-AMD-YES](../../05-AMD-YES/) | 社群案例 |

## 參與貢獻

&emsp;&emsp;請先閱讀 **[規範指南](../../规范指南.md)** 與 **[CONTRIBUTING.md](../../CONTRIBUTING.md)**，歡迎提交 Issue 與 Pull Request。
