let typingInterval = null;
let isStreaming = false;
let stopRequested = false;
let chatLanguage = "english";

// Toggle chat popup
function toggleChat() {
  const chatPopup = document.getElementById("chatPopup");
  chatPopup.style.display = chatPopup.style.display === "flex" ? "none" : "flex";

  if (chatPopup.style.display === "flex") {
    setTimeout(() => document.getElementById("userInput").focus(), 100);
  }
}

// Update header & messages on language change
function setChatLanguage() {
  const select = document.getElementById("langSelect");
  chatLanguage = select.value;

  const title = document.getElementById("chatTitle");
  if (title) title.textContent = chatLanguage === "english" ? "Help " : "ਸਹਾਇਤਾ ";

  const input = document.getElementById("userInput");
  input.placeholder =
    chatLanguage === "english"
      ? "Type your message here..."
      : "ਇੱਥੇ ਆਪਣਾ ਸੁਨੇਹਾ ਲਿਖੋ...";

  const initialMsg = document.getElementById("initialBotMsg");
  if (initialMsg) {
    initialMsg.textContent =
      chatLanguage === "english"
        ? "Hey! How can I help you today?"
        : "ਹੈਲੋ! ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਸਹਾਇਤਾ ਕਰ ਸਕਦਾ ਹਾਂ?";
  }
}

async function sendMessage() {
  const input = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");

  if (isStreaming) {
    stopRequested = true;
    return;
  }

  const message = input.value.trim();
  if (!message) return;

  const chatBody = document.getElementById("chatBody");

  const userMsg = document.createElement("div");
  userMsg.className = "chat-message user";
  userMsg.textContent = message;
  chatBody.appendChild(userMsg);
  chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });

  input.value = "";
  input.disabled = true;

  stopRequested = false;
  isStreaming = true;
  sendBtn.textContent = "■";

  const botMsg = document.createElement("div");
  botMsg.className = "chat-message bot";
  chatBody.appendChild(botMsg);
  chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });

  let dotCount = 0;
  typingInterval = setInterval(() => {
    if (!isStreaming) return;
    dotCount = (dotCount + 1) % 4;
    botMsg.textContent = "typing" + ".".repeat(dotCount);
    chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
  }, 400);

  try {
    const res = await fetch("/get", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ msg: message, lang: chatLanguage }),
    });

    const data = await res.json();
    const fullText =
      data.response || "Sorry, I don’t have an answer for that. 🙏";

    clearInterval(typingInterval);
    typingInterval = null;

    botMsg.textContent = "";

    await revealTextSimulatingStream(botMsg, fullText, () => {
      isStreaming = false;
      stopRequested = false;
      input.disabled = false;
      sendBtn.textContent = "➤";
      input.focus();
    });
  } catch (err) {
    clearInterval(typingInterval);
    typingInterval = null;

    botMsg.textContent = "Error fetching response. 😢";
    console.error(err);

    isStreaming = false;
    stopRequested = false;
    input.disabled = false;
    sendBtn.textContent = "➤";
  }
}

function revealTextSimulatingStream(element, text, onFinish) {
  return new Promise((resolve) => {
    const chatBody = element.parentElement;
    let i = 0;
    const chunkSize = 1;
    const tickMs = 20;

    function step() {
      if (stopRequested) {
        isStreaming = false;
        stopRequested = false;
        if (onFinish) onFinish();
        resolve();
        return;
      }

      if (i < text.length) {
        const nextChunk = text.slice(i, i + chunkSize);
        element.innerHTML += nextChunk.replace(/\n/g, "<br>");
        i += chunkSize;

        chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });

        setTimeout(step, tickMs);
      } else {
        if (onFinish) onFinish();
        resolve();
      }
    }

    step();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("userInput");

  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      event.preventDefault();
      if (isStreaming) stopRequested = true;
      else sendMessage();
    }
  });
});
