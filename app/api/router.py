from fastapi import APIRouter

from app.api.routes import chat, feedback, health

api_router = APIRouter()
# 统一在这里挂载子路由，方便后续继续扩展版本化接口。
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(feedback.router)
