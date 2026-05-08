const typingTarget = document.getElementById("typingText");
const textInput = document.getElementById("textInput");
const charCounter = document.getElementById("charCounter");
const form = document.getElementById("emotionForm");
const analyzeBtn = document.getElementById("analyzeBtn");
const buttonText = analyzeBtn.querySelector(".button-text");
const buttonLoader = analyzeBtn.querySelector(".button-loader");
const clearBtn = document.getElementById("clearBtn");
const copyBtn = document.getElementById("copyBtn");
const resultCard = document.getElementById("resultCard");
const resultSection = document.getElementById("resultSection");
const historyList = document.getElementById("historyList");
const errorBox = document.getElementById("errorBox");

const typingMessage = "AI Emotion Detector";
const historyStore = [];

function typeHeadline() {
    let index = 0;

    function tick() {
        if (index <= typingMessage.length) {
            typingTarget.textContent = typingMessage.slice(0, index);
            index += 1;
            setTimeout(tick, 95);
        }
    }

    tick();
}

function updateCounter() {
    charCounter.textContent = textInput.value.length.toString();
}

function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    clearBtn.disabled = isLoading;
    buttonText.textContent = isLoading ? "Analyzing..." : "Analyze Emotion";
    buttonLoader.classList.toggle("hidden", !isLoading);
    analyzeBtn.classList.toggle("is-loading", isLoading);
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hideError() {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
}

function createScoreMarkup(scores, fallbackEmotion) {
    if (!scores || Object.keys(scores).length === 0) {
        return `
            <div class="score-chip">
                <strong>${capitalize(fallbackEmotion)}</strong>
                <span>Confidence scores are unavailable for this model.</span>
            </div>
        `;
    }

    return Object.entries(scores)
        .sort((first, second) => second[1] - first[1])
        .map(([emotion, score]) => {
            const meta = window.EMOTION_META[emotion] || {};
            return `
                <div class="score-chip">
                    <strong>${meta.emoji || "🤖"} ${capitalize(emotion)}</strong>
                    <span>${score.toFixed(2)}%</span>
                </div>
            `;
        })
        .join("");
}

function renderResult(result) {
    const confidence = typeof result.confidence === "number" ? result.confidence : null;
    const emotionColor = result.color || "#00d4ff";
    const confidenceText = confidence !== null ? `${confidence.toFixed(2)}%` : "Not available";
    const cleanedTextMarkup = result.cleaned_text
        ? `<p class="cleaned-preview"><strong>Processed Text:</strong> ${result.cleaned_text}</p>`
        : "";

    resultCard.innerHTML = `
        <div class="result-live" style="--emotion-color: ${emotionColor}">
            <div class="result-badge">
                <span>${result.emoji}</span>
                <span>Emotion Detected</span>
            </div>

            <div class="result-main">
                <div>
                    <h3 class="result-emotion">${capitalize(result.emotion)} ${result.emoji}</h3>
                    <p class="result-description">${result.description}</p>
                </div>
                <div class="result-emoji">${result.emoji}</div>
            </div>

            <div class="confidence-block">
                <div class="confidence-label">
                    <span>Confidence</span>
                    <strong>${confidenceText}</strong>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: ${confidence !== null ? confidence : 18}%"></div>
                </div>
            </div>

            ${cleanedTextMarkup}

            <div class="score-grid">
                ${createScoreMarkup(result.scores, result.emotion)}
            </div>
        </div>
    `;

    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function truncate(text, maxLength = 110) {
    return text.length > maxLength ? `${text.slice(0, maxLength).trim()}...` : text;
}

function renderHistory() {
    if (historyStore.length === 0) {
        historyList.innerHTML = `<div class="history-empty">No predictions yet. Your recent analyses will appear here.</div>`;
        return;
    }

    historyList.innerHTML = historyStore
        .map((item) => `
            <article class="history-item">
                <div class="history-topline">
                    <span class="history-emotion" style="color: ${item.color}">
                        ${item.emoji} ${capitalize(item.emotion)}
                    </span>
                    <span class="history-confidence">${item.confidence}</span>
                </div>
                <p class="history-snippet">${truncate(item.text)}</p>
            </article>
        `)
        .join("");
}

function pushHistory(text, result) {
    historyStore.unshift({
        text,
        emotion: result.emotion,
        emoji: result.emoji,
        color: result.color,
        confidence: typeof result.confidence === "number" ? `${result.confidence.toFixed(2)}%` : "N/A",
    });

    if (historyStore.length > 8) {
        historyStore.pop();
    }

    renderHistory();
}

function capitalize(text) {
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : "";
}

async function copyResult() {
    const emotion = resultCard.querySelector(".result-emotion");
    if (!emotion) {
        showError("Analyze some text first so there is a result to copy.");
        return;
    }

    const confidenceText = resultCard.querySelector(".confidence-label strong")?.textContent || "Not available";
    const description = resultCard.querySelector(".result-description")?.textContent || "";
    const payload = `Prediction: ${emotion.textContent}\nConfidence: ${confidenceText}\nDescription: ${description}`;

    try {
        await navigator.clipboard.writeText(payload);
        copyBtn.textContent = "Copied";
        setTimeout(() => {
            copyBtn.textContent = "Copy Result";
        }, 1400);
    } catch (error) {
        showError("Copy failed in this browser. Please try again.");
    }
}

async function handleSubmit(event) {
    event.preventDefault();
    hideError();

    const text = textInput.value.trim();
    if (!text) {
        showError("Please enter text before clicking Analyze Emotion.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text }),
        });

        const payload = await response.json();

        if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Prediction failed. Please try again.");
        }

        renderResult(payload.result);
        pushHistory(text, payload.result);
    } catch (error) {
        showError(error.message || "Something went wrong while contacting the server.");
    } finally {
        setLoading(false);
    }
}

function resetWorkspace() {
    textInput.value = "";
    updateCounter();
    hideError();
    resultCard.innerHTML = `
        <div class="result-placeholder">
            <div class="placeholder-orb"></div>
            <p>Your prediction card will appear here after analysis.</p>
        </div>
    `;
}

function createParticles() {
    const particleRoot = document.getElementById("particles");
    const totalParticles = window.innerWidth < 768 ? 14 : 24;

    for (let index = 0; index < totalParticles; index += 1) {
        const particle = document.createElement("span");
        particle.className = "particle";
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.bottom = `${-10 - Math.random() * 70}px`;
        particle.style.animationDuration = `${10 + Math.random() * 12}s`;
        particle.style.animationDelay = `${Math.random() * 5}s`;
        particle.style.opacity = `${0.2 + Math.random() * 0.6}`;
        particleRoot.appendChild(particle);
    }
}

typeHeadline();
createParticles();
updateCounter();
renderHistory();

textInput.addEventListener("input", updateCounter);
form.addEventListener("submit", handleSubmit);
clearBtn.addEventListener("click", resetWorkspace);
copyBtn.addEventListener("click", copyResult);
