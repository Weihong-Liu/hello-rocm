# docs-readme

本目录下的 **`*/README.md` 与仓库根目录 [`README.md`](../README.md) 结构一致**，并已修正指向教程、图片与配置文件的**相对路径**（从 `docs-readme/<locale>/` 出发使用 `../../` 等），便于在 GitHub 上按语言浏览。

| 语言 / Locale | 说明 | 入口 |
|:---|:---|:---|
| 简体中文 `zh-CN` | 与根目录 `README.md` **同文**（路径已改写） | [zh-CN/README.md](./zh-CN/README.md) |
| 繁體中文 `zh-TW` | 由简体全文经 **zhconv** 转为繁体，结构与根目录一致 | [zh-TW/README.md](./zh-TW/README.md) |
| 日本語、Español、Français、한국어、العربية、Tiếng Việt、Deutsch | 默认由脚本生成英文正文 + 路径；可用 **DeepSeek** 逐段翻译为目标语言（见下文） | 见下表 |

| Locale | Path |
|:---|:---|
| Japanese | [ja-JP/README.md](./ja-JP/README.md) |
| Spanish | [es-ES/README.md](./es-ES/README.md) |
| French | [fr-FR/README.md](./fr-FR/README.md) |
| Korean | [ko-KR/README.md](./ko-KR/README.md) |
| Arabic | [ar-SA/README.md](./ar-SA/README.md) |
| Vietnamese | [vi-VN/README.md](./vi-VN/README.md) |
| German | [de-DE/README.md](./de-DE/README.md) |

## 重新生成

根目录 `README.md` 或 `README_en.md` 更新后，可在仓库根执行：

```bash
python3 -m venv .venv-docs-readme
.venv-docs-readme/bin/pip install zhconv
.venv-docs-readme/bin/python docs-readme/tools/generate_locale_readmes.py
```

（`zh-TW` 依赖 **zhconv** 做简繁转换。）

## DeepSeek 逐段翻译（日语 / 西语等）

1. 在 [DeepSeek 开放平台](https://platform.deepseek.com/) 创建 **API Key**，**不要**把密钥写入仓库或提交到 Git。
2. 复制 [`docs-readme/.env.example`](./.env.example) 为 `docs-readme/.env`，填入 `DEEPSEEK_API_KEY`（该文件已加入 `.gitignore`）。
3. 在仓库根执行（需联网）：

```bash
set -a && source docs-readme/.env && set +a   # 或: export DEEPSEEK_API_KEY=...
python3 docs-readme/tools/generate_locale_readmes.py
python3 docs-readme/tools/translate_readmes_deepseek.py
# 仅部分语言：
# python3 docs-readme/tools/translate_readmes_deepseek.py --locales ja-JP fr-FR
```

可选环境变量：`DEEPSEEK_MODEL`（默认 `deepseek-chat`）、`DEEPSEEK_API_BASE`（默认 `https://api.deepseek.com`）。若官方调整端点，请参阅其最新文档。

主文档仍以仓库根目录 **[README.md](../README.md)**（简中）与 **[README_en.md](../README_en.md)**（英文）为权威来源；`zh-CN` / `zh-TW` 由 `generate_locale_readmes.py` 与根目录对齐；其余语言在运行翻译脚本后以 DeepSeek 输出为准。
