import importlib.util
import json
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("aimanga_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"无法加载 bootstrap: {_BOOTSTRAP_PATH}")
_bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_bootstrap)
_bootstrap.ensure_project_root()

from agent.commander import Commander
from memory.chat_history_store import ChatHistoryStore
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

SESSION_COOKIE_NAME = "aimanga_session"
INDEX_PATH = get_abs_path("web/index.html")
IMAGE_DIR = Path(get_abs_path("images")).resolve()
memory_conf = agent_conf.get("memory", {})


class ChatRequest(BaseModel):
    message: str


def build_sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def ensure_session_id(request: Request) -> tuple[str, bool]:
    existing = request.cookies.get(SESSION_COOKIE_NAME)
    if existing:
        return existing, False
    return uuid.uuid4().hex, True


def set_session_cookie(response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
    )


def image_path_to_url(image_path: str) -> str | None:
    if not image_path:
        return None
    candidate = Path(image_path)
    if not candidate.name:
        return None
    return f"/images/{candidate.name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    commander = Commander(mode="mcp", mcp_server_command=sys.executable)
    chat_store = ChatHistoryStore(
        store_path=memory_conf.get("store_path", "memory/portrait_memory.sqlite3"),
        enabled=memory_conf.get("enabled", True),
    )
    await commander.setup()
    app.state.commander = commander
    app.state.chat_store = chat_store
    try:
        yield
    finally:
        await commander.close()


app = FastAPI(title="AIManga Web", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=get_abs_path("web")), name="static")


@app.get("/")
async def index(request: Request):
    session_id, created = ensure_session_id(request)
    response = FileResponse(INDEX_PATH)
    if created:
        set_session_cookie(response, session_id)
    return response


@app.get("/api/chat/history")
async def get_chat_history(request: Request):
    session_id, created = ensure_session_id(request)
    chat_store: ChatHistoryStore = request.app.state.chat_store
    response = JSONResponse({"session_id": session_id, "messages": chat_store.list_messages(session_id)})
    if created:
        set_session_cookie(response, session_id)
    return response


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    session_id, created = ensure_session_id(request)
    commander: Commander = request.app.state.commander
    chat_store: ChatHistoryStore = request.app.state.chat_store
    user_message = payload.message.strip()

    async def event_generator():
        assistant_parts: list[str] = []
        image_url: str | None = None

        yield build_sse_event("session", {"session_id": session_id})

        if not user_message:
            error_message = "消息不能为空。"
            chat_store.append_message(session_id=session_id, role="assistant", content=error_message)
            yield build_sse_event("error", {"message": error_message})
            yield build_sse_event("done", {"ok": False})
            return

        chat_store.append_message(session_id=session_id, role="user", content=user_message)

        try:
            async for event in commander.execute_stream(user_query=user_message, session_id=session_id):
                event_type = event.get("event")
                if event_type == "delta":
                    text = event.get("text", "")
                    assistant_parts.append(text)
                    yield build_sse_event("delta", {"text": text})
                elif event_type == "image":
                    image_url = image_path_to_url(event.get("image_path", ""))
                    if image_url:
                        yield build_sse_event("image", {"url": image_url})

            final_text = "".join(assistant_parts).strip() or "未能生成有效回复"
            chat_store.append_message(
                session_id=session_id,
                role="assistant",
                content=final_text,
                image_url=image_url,
            )
            yield build_sse_event("done", {"ok": True})
        except Exception as exc:
            logger.error(f"[web] 处理聊天请求失败: {exc}", exc_info=True)
            error_message = f"请求处理失败：{exc}"
            chat_store.append_message(session_id=session_id, role="assistant", content=error_message)
            yield build_sse_event("error", {"message": error_message})
            yield build_sse_event("done", {"ok": False})

    response = StreamingResponse(event_generator(), media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    if created:
        set_session_cookie(response, session_id)
    return response


@app.get("/images/{filename}")
async def get_image(filename: str):
    image_path = (IMAGE_DIR / filename).resolve()
    if IMAGE_DIR not in image_path.parents and image_path != IMAGE_DIR / filename:
        return JSONResponse({"detail": "invalid image path"}, status_code=400)
    if not image_path.exists():
        return JSONResponse({"detail": "image not found"}, status_code=404)
    return FileResponse(image_path)
