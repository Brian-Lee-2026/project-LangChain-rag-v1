from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from app.models.schemas import MessageTurn

# 提示词版本号会跟随问答日志一起记录，方便后续回溯不同版本的效果差异。
PROMPT_VERSION = "v1.0.0"

SYSTEM_PROMPT = """
你是一名具身机器人行业知识问答助手，负责回答“使用场景、商业价值、部署约束、未来需求和能力缺口”等问题。

请严格遵守以下规则：
1. 优先基于检索资料回答，不要凭空编造事实。
2. 如果问题涉及趋势判断，要区分“资料中的事实”与“基于资料的推断”。
3. 如果资料不足，请明确说明“现有知识库不足以确认”，并给出建议补充的资料方向。
4. 回答使用中文，先给简洁结论，再给分点分析。
5. 如果用户问题偏离具身机器人主题，也请礼貌提醒并尽量收敛到当前知识域。
"""

USER_TEMPLATE = """
【Prompt版本】
{prompt_version}

【最近对话】
{history}

【检索资料】
{context}

【用户问题】
{question}

请按以下结构回答：
1. 结论
2. 场景/需求分析
3. 关键约束或风险
4. 依据摘要
"""


def build_chat_messages(
    *,
    question: str,
    history: Sequence[MessageTurn],
    context_docs: Sequence[dict[str, Any]],
) -> list[BaseMessage]:
    # `system` 消息负责约束回答原则，`human` 消息负责提供问题、历史对话和检索上下文。
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT.strip()),
            ("human", USER_TEMPLATE.strip()),
        ]
    )
    return prompt.format_messages(
        prompt_version=PROMPT_VERSION,
        question=question.strip(),
        history=render_history(history),
        context=render_context(context_docs),
    )


def render_history(history: Sequence[MessageTurn]) -> str:
    if not history:
        return "无"
    # 历史对话在提示词中压平成纯文本，避免把前端数据结构直接暴露给模型。
    return "\n".join(
        f"{'用户' if turn.role == 'user' else '助手'}: {turn.content.strip()}" for turn in history
    )


def render_context(context_docs: Sequence[dict[str, Any]]) -> str:
    if not context_docs:
        return "当前没有检索到有效资料。"
    blocks: list[str] = []
    for index, item in enumerate(context_docs, start=1):
        # 为每个检索片段编号，方便模型在回答中引用，也便于后续人工核对来源。
        blocks.append(
            "\n".join(
                [
                    f"[资料{index}] 标题: {item['title']}",
                    f"来源: {item['source']}",
                    f"相关度: {item['score']:.3f}",
                    "内容:",
                    item["content"].strip(),
                ]
            )
        )
    return "\n\n".join(blocks)
