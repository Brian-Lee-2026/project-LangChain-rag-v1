const apiBase = "/api/v1";
const state = {
  // 用本地持久化的 conversationId 串联多轮提问和反馈标注。
  conversationId: window.localStorage.getItem("conversationId") || crypto.randomUUID(),
  history: [],
};

window.localStorage.setItem("conversationId", state.conversationId);

const form = document.querySelector("#chat-form");
const textarea = document.querySelector("#question");
const sendButton = document.querySelector("#send-btn");
const statusNode = document.querySelector("#status");
const messagesNode = document.querySelector("#messages");
const template = document.querySelector("#message-template");

document.querySelectorAll(".preset").forEach((button) => {
  button.addEventListener("click", () => {
    textarea.value = button.dataset.question || "";
    textarea.focus();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = textarea.value.trim();
  if (!question) {
    return;
  }

  appendMessage({
    role: "用户",
    body: escapeHtml(question),
    meta: new Date().toLocaleTimeString(),
  });
  state.history.push({ role: "user", content: question });

  setLoading(true, "正在检索并生成回答...");
  textarea.value = "";

  try {
    const response = await fetch(`${apiBase}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        conversation_id: state.conversationId,
        history: state.history,
      }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "请求失败");
    }

    // 前端保留简化版历史记录，供下一轮提问继续带上上下文。
    state.history.push({ role: "assistant", content: payload.answer });
    appendMessage({
      role: "助手",
      body: formatParagraphs(payload.answer),
      meta: `${payload.model} · ${payload.latency_ms} ms`,
      sources: payload.sources,
      feedback: {
        requestId: payload.request_id,
        conversationId: payload.conversation_id,
      },
    });
    setLoading(false, "回答完成，可继续追问");
  } catch (error) {
    appendMessage({
      role: "系统",
      body: `<p>${escapeHtml(error.message || "请求失败")}</p>`,
      meta: new Date().toLocaleTimeString(),
    });
    setLoading(false, "发生错误，请查看日志");
  }
});

function appendMessage({ role, body, meta, sources = [], feedback = null }) {
  const fragment = template.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  // 通过角色切换样式，保持测试页里用户、助手、系统消息的视觉区分。
  article.classList.add(role === "用户" ? "is-user" : role === "助手" ? "is-assistant" : "is-system");
  fragment.querySelector(".message-role").textContent = role;
  fragment.querySelector(".message-meta").textContent = meta;
  fragment.querySelector(".message-body").innerHTML = body;

  if (sources.length) {
    const sourceWrap = fragment.querySelector(".sources");
    sources.forEach((item) => {
      const card = document.createElement("section");
      card.className = "source-card";
      card.innerHTML = `
        <strong>${escapeHtml(item.title)}</strong>
        <span>${escapeHtml(item.source)} · 相关度 ${(item.relevance_score * 100).toFixed(1)}%</span>
        <p>${escapeHtml(item.excerpt)}</p>
      `;
      sourceWrap.appendChild(card);
    });
  }

  if (feedback) {
    // 只有助手回答才渲染反馈按钮，避免对系统提示或用户消息误打标。
    const wrap = fragment.querySelector(".feedback-actions");
    const goodButton = buildFeedbackButton("回答准确", true, 5, feedback);
    const badButton = buildFeedbackButton("回答不准确", false, 2, feedback);
    wrap.appendChild(goodButton);
    wrap.appendChild(badButton);
  }

  messagesNode.appendChild(fragment);
  messagesNode.scrollTop = messagesNode.scrollHeight;
}

function buildFeedbackButton(label, isAccurate, rating, feedback) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "feedback-btn";
  button.textContent = label;
  button.addEventListener("click", async () => {
    // 点击后立即禁用，防止同一条回答被重复提交多次反馈。
    button.disabled = true;
    try {
      await fetch(`${apiBase}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: feedback.requestId,
          conversation_id: feedback.conversationId,
          is_accurate: isAccurate,
          rating,
        }),
      });
      button.textContent = "已记录";
    } catch (error) {
      button.disabled = false;
      button.textContent = "记录失败";
    }
  });
  return button;
}

function setLoading(loading, text) {
  sendButton.disabled = loading;
  textarea.disabled = loading;
  statusNode.textContent = text;
}

function formatParagraphs(text) {
  // 后端回答以段落文本为主，这里做最小格式化，便于浏览器中阅读。
  return text
    .split(/\n{2,}/)
    .map((block) => `<p>${escapeHtml(block).replace(/\n/g, "<br />")}</p>`)
    .join("");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
