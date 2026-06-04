const DEFAULT_SETTINGS = {
  enabled: true,
  interceptHost: "jiehuo.vercel.app",
  interceptPath: "/search",
};

const OFFSCREEN_DOCUMENT = "offscreen.html";
const MODEL_TIMEOUT_MS = 45000;
const LOCAL_ROUTE_PAGE = "route.html";
const JIEHUO_RULE_ID = 1;

let creatingOffscreenDocument;

function normalizeQuery(query) {
  return String(query || "").trim().slice(0, 500);
}

function buildTargetUrl(query, label) {
  const encoded = encodeURIComponent(normalizeQuery(query));
  if (label === "perplexity") {
    return `https://www.perplexity.ai/search?q=${encoded}`;
  }
  return `https://www.google.com/search?q=${encoded}`;
}

async function installSearchRedirectRule() {
  const settings = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  if (!settings.enabled) {
    await removeSearchRedirectRule();
    return;
  }

  await chrome.declarativeNetRequest.updateDynamicRules({
    removeRuleIds: [JIEHUO_RULE_ID],
    addRules: [
      {
        id: JIEHUO_RULE_ID,
        priority: 1,
        action: {
          type: "redirect",
          redirect: {
            regexSubstitution: `${chrome.runtime.getURL(LOCAL_ROUTE_PAGE)}?q=\\1`,
          },
        },
        condition: {
          regexFilter: "^https://jiehuo\\.vercel\\.app/search\\?(?:.*&)?(?:q|query)=([^&#]+).*$",
          resourceTypes: ["main_frame"],
        },
      },
    ],
  });
}

async function removeSearchRedirectRule() {
  await chrome.declarativeNetRequest.updateDynamicRules({
    removeRuleIds: [JIEHUO_RULE_ID],
  });
}

async function ensureOffscreenDocument() {
  const offscreenUrl = chrome.runtime.getURL(OFFSCREEN_DOCUMENT);

  if ("getContexts" in chrome.runtime) {
    const existingContexts = await chrome.runtime.getContexts({
      contextTypes: ["OFFSCREEN_DOCUMENT"],
      documentUrls: [offscreenUrl],
    });
    if (existingContexts.length > 0) return;
  } else {
    const clients = await self.clients.matchAll();
    if (clients.some((client) => client.url === offscreenUrl)) return;
  }

  if (!creatingOffscreenDocument) {
    creatingOffscreenDocument = chrome.offscreen.createDocument({
      url: OFFSCREEN_DOCUMENT,
      reasons: ["WORKERS"],
      justification: "Run the local JieHuo ONNX model for browser search routing.",
    });
  }

  await creatingOffscreenDocument;
  creatingOffscreenDocument = undefined;
}

function withTimeout(promise, timeoutMs) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error("Local JieHuo model timed out")), timeoutMs);
    }),
  ]);
}

async function classifyWithLocalModel(query) {
  await ensureOffscreenDocument();
  return withTimeout(
    chrome.runtime.sendMessage({
      target: "offscreen",
      type: "classify",
      query,
    }),
    MODEL_TIMEOUT_MS
  );
}

async function modelStatus() {
  await ensureOffscreenDocument();
  return chrome.runtime.sendMessage({
    target: "offscreen",
    type: "status",
  });
}

chrome.runtime.onInstalled.addListener(async () => {
  const existing = await chrome.storage.sync.get(Object.keys(DEFAULT_SETTINGS));
  await chrome.storage.sync.set({ ...DEFAULT_SETTINGS, ...existing });
  await installSearchRedirectRule();
});

chrome.runtime.onStartup.addListener(() => {
  installSearchRedirectRule();
});

chrome.omnibox.onInputEntered.addListener(async (text, disposition) => {
  const normalized = normalizeQuery(text);
  if (!normalized) return;

  const result = await classifyWithLocalModel(normalized);
  if (!result?.ok) {
    await chrome.tabs.create({
      url: chrome.runtime.getURL(
        `status.html?query=${encodeURIComponent(normalized)}&error=${encodeURIComponent(
          result?.error || "Local JieHuo model unavailable"
        )}`
      ),
    });
    return;
  }

  const route = {
    query: normalized,
    ...result.result,
    targetUrl: buildTargetUrl(normalized, result.result.label),
  };
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
  if (!normalizeQuery(text)) {
    suggest([]);
    return;
  }

  suggest([
    {
      content: text,
      description: "Route with the local JieHuo neural model",
    },
  ]);
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.target !== "service_worker") return false;

  if (message.type === "classify") {
    classifyWithLocalModel(message.query)
      .then(sendResponse)
      .catch((err) =>
        sendResponse({
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        })
      );
    return true;
  }

  if (message.type === "setEnabled") {
    chrome.storage.sync
      .set({ enabled: Boolean(message.enabled) })
      .then(() => (message.enabled ? installSearchRedirectRule() : removeSearchRedirectRule()))
      .then(() => sendResponse({ ok: true, enabled: Boolean(message.enabled) }))
      .catch((err) =>
        sendResponse({
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        })
      );
    return true;
  }

  if (message.type === "route") {
    classifyWithLocalModel(message.query)
      .then((result) => {
        if (!result?.ok) {
          sendResponse(result);
          return;
        }
        const normalized = normalizeQuery(message.query);
        const route = {
          query: normalized,
          ...result.result,
          targetUrl: buildTargetUrl(normalized, result.result.label),
        };
        chrome.storage.session
          .set({ lastRoute: { ...route, at: Date.now() } })
          .then(() => sendResponse({ ok: true, route }))
          .catch((err) =>
            sendResponse({
              ok: false,
              error: err instanceof Error ? err.message : String(err),
            })
          );
      })
      .catch((err) =>
        sendResponse({
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        })
      );
    return true;
  }

  if (message.type === "status") {
    modelStatus()
      .then(sendResponse)
      .catch((err) =>
        sendResponse({
          ok: false,
          status: "error",
          error: err instanceof Error ? err.message : String(err),
        })
      );
    return true;
  }

  return false;
});
