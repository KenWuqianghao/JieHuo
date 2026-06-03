importScripts("router.js");

const DEFAULT_SETTINGS = {
  enabled: true,
  interceptHost: "jiehuo.vercel.app",
  interceptPath: "/search",
};

async function getSettings() {
  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS, ...stored };
}

function queryFromUrl(url) {
  const parsed = new URL(url);
  return parsed.searchParams.get("q") || parsed.searchParams.get("query") || "";
}

async function routeTab(tabId, query) {
  const normalized = JieHuoRouter.normalizeQuery(query);
  if (!normalized) {
    await chrome.tabs.update(tabId, { url: chrome.runtime.getURL("popup.html") });
    return;
  }

  const route = JieHuoRouter.route(normalized);
  await chrome.storage.session.set({ lastRoute: { ...route, at: Date.now() } });
  await chrome.tabs.update(tabId, { url: route.targetUrl });
}

chrome.runtime.onInstalled.addListener(async () => {
  const existing = await chrome.storage.sync.get(Object.keys(DEFAULT_SETTINGS));
  await chrome.storage.sync.set({ ...DEFAULT_SETTINGS, ...existing });
});

chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.frameId !== 0 || details.tabId < 0) return;

  const settings = await getSettings();
  if (!settings.enabled) return;

  let parsed;
  try {
    parsed = new URL(details.url);
  } catch {
    return;
  }

  if (parsed.hostname !== settings.interceptHost || parsed.pathname !== settings.interceptPath) {
    return;
  }

  await routeTab(details.tabId, queryFromUrl(details.url));
});

chrome.omnibox.onInputEntered.addListener(async (text, disposition) => {
  const route = JieHuoRouter.route(text);
  await chrome.storage.session.set({ lastRoute: { ...route, at: Date.now() } });

  if (disposition === "currentTab") {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      await chrome.tabs.update(tab.id, { url: route.targetUrl });
      return;
    }
  }

  await chrome.tabs.create({ url: route.targetUrl });
});

chrome.omnibox.onInputChanged.addListener((text, suggest) => {
  if (!JieHuoRouter.normalizeQuery(text)) {
    suggest([]);
    return;
  }

  const route = JieHuoRouter.route(text);
  suggest([
    {
      content: text,
      description: `JieHuo routes this to ${route.label} (${Math.round(
        route.confidence * 100
      )}% confidence)`,
    },
  ]);
});
