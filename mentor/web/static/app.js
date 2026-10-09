"use strict";

const $ = (sel) => document.querySelector(sel);

const QUICK = {
  teach: "Teach me this lesson step by step, starting from the beginning.",
  hint: "I'm stuck. Look at my code and latest test run and give me the smallest hint for the first exercise I haven't solved. Don't give me the solution.",
  review: "Review my exercises.py like a senior engineer.",
  quiz: "Quiz me on this lesson: one question at a time, getting harder as we go.",
};

const state = {
  modules: [],
  current: null, // module id
  file: "exercises.py", // file open in the editor
  dirty: false,
  busy: false,
  lastTestOutput: "",
};

const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } },
};

// ---------- markdown ----------

function renderMarkdown(el, text) {
  el.innerHTML = DOMPurify.sanitize(marked.parse(text));
  el.querySelectorAll("pre code").forEach((block) => hljs.highlightElement(block));
  el.querySelectorAll("a[href^='http']").forEach((a) => { a.target = "_blank"; a.rel = "noopener"; });
}

// ---------- api ----------

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// ---------- modules ----------

async function loadModules() {
  const data = await api("/api/modules");
  state.modules = data.modules;
  const select = $("#module-select");
  select.innerHTML = "";
  let group = null;
  for (const m of state.modules) {
    if (!group || group.dataset.phase !== m.phase) {
      group = document.createElement("optgroup");
      group.dataset.phase = m.phase;
      group.label = m.phase.replace(/^phase(\d+)_/, (_, n) => `Phase ${Number(n)} · `).replaceAll("_", " ");
      select.append(group);
    }
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.completed ? "✓" : "  "} ${m.id} · ${m.title}`;
    group.append(opt);
  }
  const done = state.modules.filter((m) => m.completed).length;
  $("#progress").textContent = `${done} / ${state.modules.length} done`;
}

function firstUnfinished() {
  return (state.modules.find((m) => !m.completed) || state.modules[0])?.id;
}

async function openModule(id) {
  if (state.dirty && !confirm("You have unsaved code. Leave this module anyway?")) {
    $("#module-select").value = state.current;
    return;
  }
  const mod = await api(`/api/module/${id}`);
  state.current = id;
  store.set("module", id);
  $("#module-select").value = id;
  renderMarkdown($("#lesson"), mod.lesson);
  $("#lesson").scrollTop = 0;
  state.file = mod.files[0] || "exercises.py";
  const picker = $("#file-select");
  picker.innerHTML = "";
  for (const f of mod.files) picker.append(new Option(f, f));
  picker.hidden = mod.files.length < 2;
  loadIntoEditor(mod.code);
  $("#test-panel").hidden = true;
  $("#chat-scope").textContent = id;
  await loadChat();
}

// ---------- tabs & editor ----------

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $("#lesson").hidden = name !== "lesson";
  $("#exercises").hidden = name !== "exercises";
  if (name === "exercises") editor.refresh();
  $("#explain-btn").hidden = true;
}

const dark = matchMedia("(prefers-color-scheme: dark)").matches;
const editor = CodeMirror.fromTextArea($("#editor"), {
  mode: "python",
  theme: dark ? "material-darker" : "default",
  lineNumbers: true,
  indentUnit: 4,
  matchBrackets: true,
  autoCloseBrackets: true,
  extraKeys: {
    Tab: (cm) => cm.somethingSelected() ? cm.indentSelection("add") : cm.replaceSelection("    "),
    "Shift-Tab": (cm) => cm.indentSelection("subtract"),
    "Ctrl-S": () => save(),
    "Cmd-S": () => save(),
    "Ctrl-Enter": () => runTests(),
    "Cmd-Enter": () => runTests(),
  },
});
editor.on("change", () => setDirty(true));

function setDirty(dirty) {
  state.dirty = dirty;
  $("#save-state").textContent = dirty ? "Unsaved changes" : "Saved";
}

function loadIntoEditor(code) {
  editor.setOption("mode", state.file.endsWith(".py") ? "python" : null);
  editor.setValue(code);
  editor.clearHistory();
  setDirty(false);
}

async function openFile(path) {
  if (state.dirty) await save();
  const { code } = await api(`/api/module/${state.current}/file?path=${encodeURIComponent(path)}`);
  state.file = path;
  $("#file-select").value = path;
  loadIntoEditor(code);
}

async function save() {
  await api(`/api/module/${state.current}/code`, { path: state.file, code: editor.getValue() });
  setDirty(false);
}

async function runTests() {
  const btn = $("#run");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const result = await api(`/api/module/${state.current}/test`, { path: state.file, code: editor.getValue() });
    setDirty(false);
    state.lastTestOutput = result.output;
    const verdict = $("#test-verdict");
    verdict.textContent = result.passed ? "All tests passed 🎉" : "Some tests failed";
    verdict.className = result.passed ? "pass" : "fail";
    $("#ask-failure").textContent = result.passed ? "Get a code review" : "Ask mentor about this";
    $("#ask-failure").dataset.passed = result.passed;
    $("#test-output").textContent = result.output.trim();
    $("#test-panel").hidden = false;
    if (result.passed) await loadModules();
    $("#module-select").value = state.current;
  } catch (err) {
    alert(`Couldn't run the tests: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run tests";
  }
}

// ---------- mentor chat ----------

const WELCOME = `<p><strong>Hi! I'm your mentor for this module.</strong></p>
<p>Read the lesson on the left and ask me anything as you go, or press <strong>Teach me</strong> and I'll walk you through it.</p>
<p>I can see your code and test results, so when you're stuck just press <strong>Hint</strong>.</p>`;

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = role === "assistant" ? "msg assistant markdown" : `msg ${role}`;
  if (role === "assistant") renderMarkdown(el, text);
  else el.textContent = text;
  $("#messages").append(el);
  return el;
}

function scrollChat() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

async function loadChat() {
  const { messages } = await api(`/api/chat/${state.current}`);
  const box = $("#messages");
  box.innerHTML = "";
  if (!messages.length) box.innerHTML = `<div class="welcome">${WELCOME}</div>`;
  for (const m of messages) addMessage(m.role, m.text);
  scrollChat();
}

function setBusy(busy) {
  state.busy = busy;
  $("#send").disabled = busy;
  document.querySelectorAll(".quick button").forEach((b) => { b.disabled = busy; });
}

async function send(text) {
  text = text.trim();
  if (!text || state.busy) return;
  if (state.dirty) await save(); // the mentor always sees your latest code
  setBusy(true);
  $(".welcome")?.remove();
  addMessage("user", text);
  const reply = addMessage("assistant", "");
  reply.classList.add("pending");
  scrollChat();

  let full = "";
  let frame = 0;
  try {
    const res = await fetch(`/api/chat/${state.current}/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) throw new Error((await res.json()).error);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      full += decoder.decode(value, { stream: true });
      if (!frame) {
        frame = requestAnimationFrame(() => {
          frame = 0;
          const nearBottom = $("#messages").scrollHeight - $("#messages").scrollTop - $("#messages").clientHeight < 80;
          renderMarkdown(reply, full);
          if (nearBottom) scrollChat();
        });
      }
    }
  } catch (err) {
    full += `\n\n[error] ${err.message}`;
  }
  cancelAnimationFrame(frame);
  const error = full.indexOf("\n\n[error] ");
  if (error !== -1) {
    renderMarkdown(reply, full.slice(0, error));
    addMessage("error", full.slice(error + 2));
  } else {
    renderMarkdown(reply, full);
  }
  if (!reply.textContent.trim()) reply.remove();
  reply.classList.remove("pending");
  setBusy(false);
  scrollChat();
}

// ---------- explain selection ----------

function selectedLessonText() {
  const sel = getSelection();
  if (!sel || sel.isCollapsed || !$("#lesson").contains(sel.anchorNode)) return "";
  return sel.toString().trim();
}

document.addEventListener("selectionchange", () => {
  const btn = $("#explain-btn");
  const text = selectedLessonText();
  if (!text) { btn.hidden = true; return; }
  const rect = getSelection().getRangeAt(0).getBoundingClientRect();
  btn.style.top = `${Math.max(8, rect.top - 40)}px`;
  btn.style.left = `${Math.min(innerWidth - 200, Math.max(8, rect.left))}px`;
  btn.hidden = false;
});

$("#explain-btn").addEventListener("mousedown", (e) => {
  e.preventDefault(); // keep the selection
  const text = selectedLessonText();
  $("#explain-btn").hidden = true;
  if (text) send(`Explain this part of the lesson in a different way, with a simple example:\n\n> ${text.replaceAll("\n", "\n> ")}`);
});

// ---------- resizable divider ----------

function initDivider() {
  const split = $("#split");
  const saved = store.get("leftWidth");
  if (saved) split.style.setProperty("--left-width", saved);
  const divider = $("#divider");
  divider.addEventListener("pointerdown", (e) => {
    if (matchMedia("(max-width: 800px)").matches) return;
    divider.setPointerCapture(e.pointerId);
    divider.classList.add("dragging");
    const move = (ev) => {
      const pct = Math.min(80, Math.max(30, (ev.clientX / split.clientWidth) * 100));
      split.style.setProperty("--left-width", `${pct}%`);
    };
    const up = () => {
      divider.classList.remove("dragging");
      divider.removeEventListener("pointermove", move);
      store.set("leftWidth", split.style.getPropertyValue("--left-width"));
      editor.refresh();
    };
    divider.addEventListener("pointermove", move);
    divider.addEventListener("pointerup", up, { once: true });
  });
}

// ---------- wiring ----------

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
document.querySelectorAll("[data-quick]").forEach((b) => b.addEventListener("click", () => send(QUICK[b.dataset.quick])));
$("#module-select").addEventListener("change", (e) => openModule(e.target.value));
$("#next-module").addEventListener("click", () => {
  const i = state.modules.findIndex((m) => m.id === state.current);
  if (i + 1 < state.modules.length) openModule(state.modules[i + 1].id);
});
$("#save").addEventListener("click", save);
$("#file-select").addEventListener("change", (e) => openFile(e.target.value));
$("#run").addEventListener("click", runTests);
$("#ask-failure").addEventListener("click", (e) => {
  send(e.target.dataset.passed === "true"
    ? QUICK.review
    : "My tests failed. Explain what the first failure means and give me one small hint. Don't write the fix for me.");
});
$("#new-chat").addEventListener("click", async () => {
  if (!confirm("Start a fresh conversation for this module? The current one will be deleted.")) return;
  await api(`/api/chat/${state.current}/reset`, {});
  await loadChat();
});

const input = $("#input");
$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value;
  input.value = "";
  input.style.height = "";
  send(text);
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("#composer").requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "";
  input.style.height = `${input.scrollHeight}px`;
});
window.addEventListener("beforeunload", (e) => { if (state.dirty) e.preventDefault(); });

(async () => {
  initDivider();
  await loadModules();
  const saved = store.get("module");
  const start = state.modules.some((m) => m.id === saved) ? saved : firstUnfinished();
  if (start) await openModule(start);
})();
