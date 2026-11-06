# 许可证与素材记录

## 1. 项目代码

RVCre 自有整合代码使用 **GPL-3.0-only**。用户的目标是全开源，本项目不把“必须是 MIT/Apache”误当成额外要求。GPL 允许商业使用；它与仅限非商业使用的模型条款不是同一件事。

VoxCPM 的代码/权重、Seed-VC 的代码/权重及其他依赖仍保留各自许可证。主仓库采用 GPL 不会抹去 Apache、MIT 等上游的版权、NOTICE 和归属要求。精确版本和主要资产来源见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

发行整合包时应保留适用的许可证文件和版权声明，并提供 GPL 覆盖部分要求的对应源码和构建/安装信息。不能只因为调用了子进程或 HTTP，就概括为“所有其他代码自动免受 GPL 影响”；结合方式与通信内容也有关。GNU 官方说明：[组合与聚合](https://www.gnu.org/licenses/gpl-faq.html.en#MereAggregation)。

个人本地修改与内部使用不要求自动公开仓库。程序输出也不会仅因为运行了 GPL 代码就一概变成 GPL；输出中包含的受保护素材和权重条款仍要按实际内容判断。来源：[本地修改](https://www.gnu.org/licenses/gpl-faq.html#GPLRequireSourcePostedPublic)、[程序输出](https://www.gnu.org/licenses/gpl-faq.html#WhatCaseIsOutputGPL)。

## 2. 默认后端与备选

| 项目 | 代码 | 官方权重/所选权重 | 本项目角色 |
|---|---|---|---|
| VoxCPM2 | Apache-2.0 | Apache-2.0 | 主 TTS 与声音设计 |
| Seed-VC v2 | GPL-3.0 | GPL-3.0 | 主 VC |
| ASTRAL quantization | GPL-3.0 | GPL-3.0 | Seed-VC v2 内容表示依赖 |
| OpenVoice V2 | MIT | MIT | 后续宽松许可备选，尚未接入 |
| RVC | MIT | 官方基础模型仓库标 MIT；第三方角色模型另看其许可 | 后续复用旧音色资产，尚未接入 |
| FreeVC | MIT | Coqui 官方 `freevc24` 发布条目标 MIT | 已评估的旧方案，未采用 |
| Amphion / Vevo / Vevo2 | Amphion 框架 MIT | Vevo CC-BY-NC-4.0；Vevo2 CC-BY-NC-ND-4.0 | 不纳入默认可自由修改和发布的模型链路 |

许可证可以随新版本或资产改变，因此下载按 revision 固定。安装器不声称“某个项目根目录是 MIT，所以所有下载的模型、说话人音色和训练数据也是 MIT”。

这里的本地开源路线指代码与所选模型权重的可获取、可运行和相应许可；不承诺上游全部基础训练语料和每一步训练资源都已公开。

## 3. 用户音源

项目按用户提供的素材使用授权推进本地开发与训练，不额外加入审批打勾页面。音源来源只做简明记录：来源类型、名称/链接或本地描述、备注、原文件 SHA-256。缺少来源说明不被伪装为模型错误。

当前仓库没有用户提到的原神音源。后续提供的原神或其他作品音频保存在本地数据目录，不随源码、示例或测试包提交。**用户允许本项目使用这些材料，不等于作品的音频、角色或相关第三方权利自动改成 GPL。** 发布代码、发布原音、发布训练权重、发布生成音频是不同的行为，记录其实际范围即可，不把它们混成一个“全开源素材”开关。

示例文件只有人工编写的文字、JSON 结构与路径占位符；测试音频由测试过程产生的简单合成信号临时创建，不包含角色台词。

## 4. 来源索引

- [VoxCPM LICENSE](https://github.com/OpenBMB/VoxCPM/blob/4b8dd4440d33396b3f429201f8af0529b4d2c601/LICENSE)；[VoxCPM2 模型卡](https://huggingface.co/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915)。
- [Seed-VC LICENSE](https://github.com/Plachtaa/seed-vc/blob/51383efd921027683c89e5348211d93ff12ac2a8/LICENSE)；[所选权重](https://huggingface.co/Plachta/Seed-VC/tree/257283f9f41585055e8f858fba4fd044e5caed6e)。
- [OpenVoice V2 模型卡](https://huggingface.co/myshell-ai/OpenVoiceV2)。
- [RVC LICENSE](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)；[基础模型仓库](https://huggingface.co/lj1995/VoiceConversionWebUI)。
- [Coqui FreeVC 发布清单](https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json)。
- [Vevo 权重](https://huggingface.co/amphion/Vevo)；[Vevo2 权重](https://huggingface.co/RMSnow/Vevo2)。
