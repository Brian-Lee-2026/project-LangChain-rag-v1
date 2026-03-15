from fastapi import APIRouter, Depends

from app.api.deps import get_feedback_service
from app.models.schemas import FeedbackRequest, FeedbackResponse, FeedbackSummary
from app.services.feedback_service import FeedbackService

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse, summary="记录人工反馈")
def submit_feedback(
    payload: FeedbackRequest,
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackResponse:
    # 路由层只负责接收反馈，聚合统计逻辑全部下沉到服务层。
    return service.record_feedback(payload)


@router.get("/summary", response_model=FeedbackSummary, summary="查看准确率反馈摘要")
def feedback_summary(
    service: FeedbackService = Depends(get_feedback_service),
) -> FeedbackSummary:
    # 摘要接口用于快速查看当前人工标注的准确率趋势。
    return service.get_summary()
