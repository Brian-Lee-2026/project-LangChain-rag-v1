from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MessageTurn(BaseModel):
    # 目前只保留 `user` / `assistant` 两类历史消息，足够支撑前端追问场景。
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=4000)
    # `conversation_id` 由前端保持，用来串联多轮问答与反馈记录。
    conversation_id: str | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=12)
    # `history` 不直接存全部聊天记录，只传最近几轮，由后端再按配置截断。
    history: list[MessageTurn] = Field(default_factory=list)


class SourceDocument(BaseModel):
    title: str
    source: str
    excerpt: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    # `request_id` 主要用于日志追踪和后续人工反馈回写。
    request_id: str
    conversation_id: str
    answer: str
    sources: list[SourceDocument]
    model: str
    prompt_version: str
    latency_ms: int
    created_at: datetime


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=6, max_length=100)
    conversation_id: str | None = Field(default=None, max_length=100)
    # `rating` 和 `is_accurate` 同时保留，兼容主观评分与客观准确率标注。
    rating: int | None = Field(default=None, ge=1, le=5)
    is_accurate: bool | None = None
    notes: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    status: str = "recorded"
    feedback_id: str
    recorded_at: datetime


class FeedbackSummary(BaseModel):
    # 这个模型直接服务前端与运维查看，不额外暴露原始反馈明细。
    total_feedback: int
    accurate_votes: int
    inaccurate_votes: int
    accuracy_rate: float | None
    average_rating: float | None
    latest_feedback_at: datetime | None


class HealthResponse(BaseModel):
    # 健康检查同时返回索引规模，方便判断知识库是否真的加载成功。
    status: str
    app_name: str
    environment: str
    document_count: int
    chunk_count: int
    model: str
    prompt_version: str
