const consentBox = document.getElementById("consentBox");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");
const video = document.getElementById("camera");

let stream = null;

// Initial state
startBtn.disabled = true;
stopBtn.disabled = true;

// Enable Start only after consent
consentBox.addEventListener("change", () => {
  startBtn.disabled = !consentBox.checked;
});

// START analysis
startBtn.addEventListener("click", async () => {
  try {
    const res = await fetch("/start", { method: "POST" });
    const data = await res.json();

    // Treat both cases as success
    if (data.status === "started" || data.status === "already_running") {
      statusText.innerText = "Status: Analysis running";
      startBtn.disabled = true;
      stopBtn.disabled = false;
      return;
    }

    throw new Error("Unexpected response");

  } catch (err) {
    alert("Could not start analysis.");
    console.error(err);
  }
});

// STOP analysis
stopBtn.addEventListener("click", async () => {
  try {
    // Stop browser camera
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      video.srcObject = null;
      video.style.display = "none";
    }

    const res = await fetch("/stop", {
      method: "POST"
    });

    if (!res.ok) {
      throw new Error("Failed to stop backend");
    }

    statusText.innerText = "Status: Stopped";
    startBtn.disabled = false;
    stopBtn.disabled = true;

  } catch (err) {
    alert("Could not stop analysis.");
    console.error(err);
  }
});