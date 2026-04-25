# hello-rocm-skill

帮助你在 [hello-rocm](https://github.com/datawhalechina/hello-rocm) 教程项目中快速找到所需内容的 Claude Code Skill —— 在 AMD GPU 上部署、微调、优化大语言模型，一站导航。

## 安装方式

将 `hello-rocm-skill/` 目录复制到你 IDE 的 skills 文件夹：

| IDE | 目标路径 |
|-----|---------|
| **Claude Code** | `.claude/skills/hello-rocm-skill/` |
| **Cursor** | `.cursor/skills/hello-rocm-skill/` |
| **Windsurf** | `.windsurf/skills/hello-rocm-skill/` |
| **Trae** | `.trae/skills/hello-rocm-skill/` |
| **Codex / Qoder** | `.agents/skills/hello-rocm-skill/` |
| **Kiro** | `.kiro/skills/hello-rocm-skill/` |
| **OpenCode** | `.opencode/skills/hello-rocm-skill/` |
| **VS Code Copilot** | `.github/skills/hello-rocm-skill/` |

### 一行安装（Claude Code）

```bash
git clone https://github.com/datawhalechina/hello-rocm.git
cp -r hello-rocm/hello-rocm-skill/ .claude/skills/hello-rocm-skill/
```

### 验证

开启新对话说：**"我想在 AMD GPU 上部署模型"**

## 三种用户画像快速入口

| 我是... | 试试说... |
|---------|----------|
| **刚买了 AMD 设备** | "我的 GPU 支持 ROCm 吗？" / "帮我装 ROCm 跑第一个模型" |
| **想用 AMD 做 AI 学习** | "我想用 LoRA 微调模型" / "有什么实战项目？" |
| **有编程基础要用 ROCm** | "帮我写一个 HIP kernel" / "如何在 ROCm 上部署 vLLM 生产环境？" |

## 这个 Skill 做什么

- 从你的对话中自动判断经验水平（无需填问卷）
- 精准导航到 hello-rocm 项目中你需要的教程文件
- 总结关键内容并给出下一步建议
- 超出项目覆盖范围时，回退到 AMD 官方文档

## 这个 Skill 不做什么

- 不在你的机器上执行安装操作
- 不诊断硬件故障
- 不替代 AMD 官方技术支持

## 文件结构

| 文件 | 用途 |
|------|------|
| `SKILL.md` | 核心 agent 指令（LLM 消费） |
| `skill.json` | 机器可读的技能清单 |
| `README.md` | 本文件 —— 人类阅读（中文） |
| `README_EN.md` | 本文件 —— 人类阅读（英文） |
| `references/quick-deploy/SKILL.md` | 新用户 5 步闪电部署检查表 |
| `references/troubleshooting/SKILL.md` | 常见错误排查 |

## 链接

- [hello-rocm 项目](https://github.com/datawhalechina/hello-rocm)
- [ROCm 官方文档](https://rocm.docs.amd.com/)

## 版本

0.1.0 —— 初版框架，触发表随项目推进逐步扩充。
