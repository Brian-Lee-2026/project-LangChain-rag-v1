from app.models.schemas import MessageTurn
from app.services.prompting import PROMPT_VERSION, render_context, render_history


def test_render_history_formats_roles():
    history = [
        MessageTurn(role="user", content="仓储场景需求是什么？"),
        MessageTurn(role="assistant", content="重点是稳定抓取和系统对接。"),
    ]

    text = render_history(history)

    assert "用户: 仓储场景需求是什么？" in text
    assert "助手: 重点是稳定抓取和系统对接。" in text


def test_render_context_includes_prompt_version_related_fields():
    context = render_context(
        [
            {
                "title": "仓储与物流场景",
                "source": "data/knowledge/02_logistics_warehouse.md",
                "score": 0.87,
                "content": "仓储物流适合优先自动化。",
            }
        ]
    )

    assert "相关度: 0.870" in context
    assert "仓储与物流场景" in context
    assert PROMPT_VERSION == "v1.0.0"
