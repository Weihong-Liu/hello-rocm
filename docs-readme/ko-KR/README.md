<div align=center>
  <img src="../../images/head.png" alt="hello-rocm">
  <strong>AMD YES! 🚀</strong>
</div>

<div align="center">

*오픈소스 · 커뮤니티 주도 · AMD AI 생태계를 더 쉽게*

<p align="center">
  <a href="../../README_en.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../../README.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
  <a href="../es-ES/README.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9"></a>
  <a href="../fr-FR/README.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9"></a>
  <a href="./README.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9"></a>
  <a href="../ar-SA/README.md"><img alt="العربية" src="https://img.shields.io/badge/العربية-d9d9d9"></a>
  <a href="../vi-VN/README.md"><img alt="Tiếng_Việt" src="https://img.shields.io/badge/Tiếng_Việt-d9d9d9"></a>
  <a href="../de-DE/README.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9"></a>
</p>

</div>

## hello-rocm 소개

&emsp;&emsp;**hello-rocm**은 **AMD GPU와 ROCm**으로 대규모 언어 모델(LLM)을 배포·미세 조정하기 위한 튜토리얼 모음입니다. 통합 환경, 배포, 파인튜닝, 인프라/연산자 수준의 내용을 다룹니다.

&emsp;&emsp;**ROCm 7.10.0**부터 ROCm은 CUDA와 유사하게 Python 가상 환경에 설치할 수 있으며 **Linux와 Windows**를 공식 지원합니다.

## 권장 학습 순서

1. **[00-Environment](../../00-Environment/README.md)** — ROCm + PyTorch + **uv** 기준 환경  
2. **01-Deploy** — LM Studio, vLLM, Ollama, llama.cpp 등  
3. **02-Fine-tune** — LoRA 등 파인튜닝 예제  
4. **03-Infra** — HIP 및 스택/커널 실습  

## 전체 문서 위치

| 언어 | 파일 |
|:---|:---|
| 간체中文(본문) | 루트 **[README.md](../../README.md)** |
| English | **[README_en.md](../../README_en.md)** |

## 저장소 구조(요약)

| 경로 | 내용 |
|:---|:---|
| [00-Environment](../../00-Environment/) | ROCm 기본 설치(Windows / Ubuntu) |
| [01-Deploy](../../01-Deploy/) | LLM 배포 |
| [02-Fine-tune](../../02-Fine-tune/) | 파인튜닝 |
| [03-Infra](../../03-Infra/) | 인프라·연산자 |
| [04-References](../../04-References/) | 참고 자료 |
| [05-AMD-YES](../../05-AMD-YES/) | 커뮤니티 사례 |

## 기여

&emsp;&emsp;**[规范指南](../../规范指南.md)** 및 **[CONTRIBUTING.md](../../CONTRIBUTING.md)**를 읽은 뒤 Issue와 Pull Request를 환영합니다.
