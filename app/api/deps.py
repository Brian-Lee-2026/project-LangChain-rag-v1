from fastapi import Request

from app.services.feedback_service import FeedbackService
from app.services.rag_service import RAGService


def get_rag_service(request: Request) -> RAGService:
    # 从应用生命周期里取单例服务，避免每次请求重新构建索引和模型。
    return request.app.state.rag_service


def get_feedback_service(request: Request) -> FeedbackService:
    return request.app.state.feedback_service
