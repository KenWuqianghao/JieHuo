const query = new URLSearchParams(location.search).get("q") || "";
const normalized = query.trim().slice(0, 500);
const queryNode = document.querySelector("#query");
const statusNode = document.querySelector("#status");

queryNode.textContent = normalized || "No query";

async function route() {
  if (!normalized) {
    statusNode.textContent = "No query to route";
    return;
  }

  const response = await chrome.runtime.sendMessage({
    target: "service_worker",
    type: "route",
    query: normalized,
  });

  if (!response?.ok) {
    const error = response?.error || "Local JieHuo model unavailable";
    location.href = `status.html?query=${encodeURIComponent(normalized)}&error=${encodeURIComponent(error)}`;
    return;
  }

  statusNode.textContent = `${response.route.label} (${Math.round(
    response.route.confidence * 100
  )}%)`;
  location.replace(response.route.targetUrl);
}

route().catch((err) => {
  location.href = `status.html?query=${encodeURIComponent(normalized)}&error=${encodeURIComponent(
    err instanceof Error ? err.message : String(err)
  )}`;
});
