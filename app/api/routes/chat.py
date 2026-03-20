from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.deps import get_rag_service
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService, RAGServiceError

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="知识问答")
def chat(
    request: Request,
    payload: ChatRequest,
    service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        return service.ask(payload, request_id=request.state.request_id)
    except ValueError as exc:
        # 这里保留参数校验类错误，直接返回 400 给前端，便于用户修正输入。
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RAGServiceError as exc:
        # 将底层模型、检索或外部依赖错误统一收敛为 503，避免泄露过多实现细节。
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
