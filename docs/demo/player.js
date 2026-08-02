const mediaElements = [...document.querySelectorAll("audio, video")];

for (const media of mediaElements) {
  media.addEventListener("play", () => {
    for (const other of mediaElements) {
      if (other !== media && !other.paused) {
        other.pause();
      }
    }
  });
}

const copyButton = document.querySelector("[data-copy]");
const copyStatus = document.querySelector(".copy-status");

copyButton?.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(copyButton.dataset.copy ?? "");
    copyStatus.textContent = "Copied.";
    copyButton.textContent = "Copied";
  } catch {
    copyStatus.textContent = "Select and copy the command manually.";
  }

  window.setTimeout(() => {
    copyStatus.textContent = "";
    copyButton.textContent = "Copy command";
  }, 1800);
});
