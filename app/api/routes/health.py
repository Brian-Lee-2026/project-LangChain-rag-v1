from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.models.schemas import HealthResponse
from app.services.rag_service import RAGService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="服务健康检查")
def health(service: RAGService = Depends(get_rag_service)) -> HealthResponse:
    # 健康检查顺带暴露当前索引规模和模型信息，方便联调时快速确认加载状态。
    return service.health()
