const DEFAULT_SETTINGS = {
  enabled: true,
  interceptHost: "jiehuo.vercel.app",
  interceptPath: "/search",
};

const form = document.querySelector("#settings-form");
const enabled = document.querySelector("#enabled");
const status = document.querySelector("#status");

async function loadSettings() {
  const settings = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  enabled.checked = settings.enabled;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await chrome.runtime.sendMessage({
    target: "service_worker",
    type: "setEnabled",
    enabled: enabled.checked,
  });
  status.textContent = "Saved";
});

loadSettings();
