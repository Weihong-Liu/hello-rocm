<div align=center>
  <img src="../../images/head.png" alt="hello-rocm">
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*オープンソース · コミュニティ主導 · AMD AI をより使いやすく*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="./README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="../ko-KR/README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>

## hello-rocm について

&emsp;&emsp;**hello-rocm** は、**AMD GPU と ROCm** を使った大規模言語モデル（LLM）の学習用チュートリアル集です。環境構築、デプロイ、ファインチューニング、インフラ／演算子レベルの話題までをカバーします。

&emsp;&emsp;**ROCm 7.10.0** 以降、ROCm は CUDA と同様に Python 仮想環境へのインストールが可能になり、**Linux と Windows** の両方が公式にサポートされています。

## 推奨ラーニングパス

1. **[00-Environment](../../00-Environment/README.md)** — ROCm、PyTorch、**uv** による統一ベースライン  
2. **01-Deploy** — LM Studio、vLLM、Ollama、llama.cpp など  
3. **02-Fine-tune** — LoRA などのファインチューニング例  
4. **03-Infra** — HIP やスタック／カーネル関連の章  

## ドキュメントの所在

| 言語 | ファイル |
|:---|:---|
| 簡体字中国語（メイン） | ルートの **[README.md](../../README.md)** |
| 英語 | **[README_en.md](../../README_en.md)** |

## リポジトリ構成（抜粋）

| パス | 内容 |
|:---|:---|
| [00-Environment](../../00-Environment/) | ROCm ベースライン（Windows / Ubuntu） |
| [01-Deploy](../../01-Deploy/) | LLM デプロイ |
| [02-Fine-tune](../../02-Fine-tune/) | ファインチューニング |
| [03-Infra](../../03-Infra/) | インフラ・演算子 |
| [04-References](../../04-References/) | 参考資料 |
| [05-AMD-YES](../../05-AMD-YES/) | コミュニティ事例 |

## コントリビューション

&emsp;&emsp;**[规范指南](../../规范指南.md)** と **[CONTRIBUTING.md](../../CONTRIBUTING.md)** をご覧のうえ、Issue / Pull Request を歓迎します。
