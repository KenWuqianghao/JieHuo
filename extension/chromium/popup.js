const enabled = document.querySelector("#enabled");
const lastRoute = document.querySelector("#last-route");
const form = document.querySelector("#preview-form");
const queryInput = document.querySelector("#query");
const preview = document.querySelector("#preview");

function renderRoute(route) {
  if (!route) return "No routes yet";
  return `${route.query} -> ${route.label} (${Math.round(route.confidence * 100)}%)`;
}

async function loadState() {
  const settings = await chrome.storage.sync.get({ enabled: true });
  enabled.checked = settings.enabled;

  const session = await chrome.storage.session.get("lastRoute");
  lastRoute.textContent = renderRoute(session.lastRoute);
}

enabled.addEventListener("change", async () => {
  await chrome.storage.sync.set({ enabled: enabled.checked });
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const route = JieHuoRouter.route(queryInput.value);
  preview.textContent = renderRoute(route);
});

loadState();
