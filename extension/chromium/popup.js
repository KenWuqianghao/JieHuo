const enabled = document.querySelector("#enabled");
const lastRoute = document.querySelector("#last-route");
const modelStatus = document.querySelector("#model-status");
const form = document.querySelector("#preview-form");
const queryInput = document.querySelector("#query");
const preview = document.querySelector("#preview");
const warm = document.querySelector("#warm");

function renderRoute(route) {
  if (!route) return "No routes yet";
  return `${route.query} -> ${route.label} (${Math.round(route.confidence * 100)}%, neural)`;
}

function renderStatus(status) {
  if (!status?.ok) return `Error: ${status?.error || "model unavailable"}`;
  if (status.status === "ready") return "Ready";
  if (status.status === "loading") return "Loading";
  return "Idle";
}

async function loadState() {
  const settings = await chrome.storage.sync.get({ enabled: true });
  enabled.checked = settings.enabled;

  const session = await chrome.storage.session.get("lastRoute");
  lastRoute.textContent = renderRoute(session.lastRoute);

  const status = await chrome.runtime.sendMessage({ target: "service_worker", type: "status" });
  modelStatus.textContent = renderStatus(status);
}

enabled.addEventListener("change", async () => {
  await chrome.runtime.sendMessage({
    target: "service_worker",
    type: "setEnabled",
    enabled: enabled.checked,
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  preview.textContent = "Classifying";
  const response = await chrome.runtime.sendMessage({
    target: "service_worker",
    type: "classify",
    query: queryInput.value,
  });
  if (!response?.ok) {
    preview.textContent = `Error: ${response?.error || "model unavailable"}`;
    return;
  }
  preview.textContent = renderRoute({ query: queryInput.value.trim(), ...response.result });
  modelStatus.textContent = "Ready";
});

warm.addEventListener("click", async () => {
  modelStatus.textContent = "Loading";
  const response = await chrome.runtime.sendMessage({ target: "service_worker", type: "status" });
  modelStatus.textContent = renderStatus(response);
});

loadState();
