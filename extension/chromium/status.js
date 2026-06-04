const params = new URLSearchParams(location.search);
const query = params.get("query") || "";
const error = params.get("error") || "Local JieHuo model unavailable";

document.querySelector("#query").textContent = query || "No query";
document.querySelector("#error").textContent = error;

document.querySelector("#retry").addEventListener("click", async () => {
  if (!query) return;

  const response = await chrome.runtime.sendMessage({
    target: "service_worker",
    type: "classify",
    query,
  });

  if (!response?.ok) {
    document.querySelector("#error").textContent = response?.error || "Local JieHuo model unavailable";
    return;
  }

  const encoded = encodeURIComponent(query.trim().slice(0, 500));
  const targetUrl =
    response.result.label === "perplexity"
      ? `https://www.perplexity.ai/search?q=${encoded}`
      : `https://www.google.com/search?q=${encoded}`;

  location.href = targetUrl;
});
