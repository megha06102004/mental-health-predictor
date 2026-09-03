// DASS-21 Clinical Questionnaire items and frontend logic
const DASS21_QUESTIONS = [
  { id: "Q1", subscale: "Stress", text: "I found it hard to wind down." },
  { id: "Q2", subscale: "Anxiety", text: "I was aware of dryness of my mouth." },
  { id: "Q3", subscale: "Depression", text: "I couldn't seem to experience any positive feeling at all." },
  { id: "Q4", subscale: "Anxiety", text: "I experienced breathing difficulty (e.g. excessively rapid breathing)." },
  { id: "Q5", subscale: "Depression", text: "I found it difficult to work up the initiative to do things." },
  { id: "Q6", subscale: "Stress", text: "I tended to over-react to situations." },
  { id: "Q7", subscale: "Anxiety", text: "I experienced trembling (e.g. in the hands)." },
  { id: "Q8", subscale: "Stress", text: "I felt that I was using a lot of nervous energy." },
  { id: "Q9", subscale: "Anxiety", text: "I was worried about situations in which I might panic and make a fool of myself." },
  { id: "Q10", subscale: "Depression", text: "I felt that I had nothing to look forward to." },
  { id: "Q11", subscale: "Stress", text: "I found myself getting agitated." },
  { id: "Q12", subscale: "Stress", text: "I found it difficult to relax." },
  { id: "Q13", subscale: "Depression", text: "I felt down-hearted and blue." },
  { id: "Q14", subscale: "Stress", text: "I was intolerant of anything that kept me from getting on with what I was doing." },
  { id: "Q15", subscale: "Anxiety", text: "I felt I was close to panic." },
  { id: "Q16", subscale: "Depression", text: "I was unable to become enthusiastic about anything." },
  { id: "Q17", subscale: "Depression", text: "I felt I wasn't worth much as a person." },
  { id: "Q18", subscale: "Stress", text: "I felt that I was rather touchy." },
  { id: "Q19", subscale: "Anxiety", text: "I was aware of the action of my heart in the absence of physical exertion." },
  { id: "Q20", subscale: "Anxiety", text: "I felt scared without any good reason." },
  { id: "Q21", subscale: "Depression", text: "I felt that life was meaningless." }
];

const LIKERT_OPTIONS = [
  { val: 0, label: "0 - Did not apply to me at all" },
  { val: 1, label: "1 - Applied to some degree" },
  { val: 2, label: "2 - Applied considerably" },
  { val: 3, label: "3 - Applied very much" }
];

const userAnswers = new Array(21).fill(null);

function initQuestions() {
  const container = document.getElementById("questionsList");
  container.innerHTML = "";

  DASS21_QUESTIONS.forEach((q, idx) => {
    const card = document.createElement("div");
    card.className = "question-card";
    card.id = `qCard_${idx}`;

    card.innerHTML = `
      <div class="q-header">
        <span class="q-id">Item ${idx + 1} of 21</span>
        <span class="q-subscale">${q.subscale}</span>
      </div>
      <div class="q-text">${q.text}</div>
      <div class="options-grid">
        ${LIKERT_OPTIONS.map(opt => `
          <button type="button" class="option-btn" data-idx="${idx}" data-val="${opt.val}" onclick="selectOption(${idx}, ${opt.val})">
            ${opt.label}
          </button>
        `).join("")}
      </div>
    `;

    container.appendChild(card);
  });
}

function selectOption(qIdx, val) {
  userAnswers[qIdx] = val;

  const card = document.getElementById(`qCard_${qIdx}`);
  card.classList.add("answered");

  const buttons = card.querySelectorAll(".option-btn");
  buttons.forEach(btn => {
    if (parseInt(btn.getAttribute("data-val")) === val) {
      btn.classList.add("selected");
    } else {
      btn.classList.remove("selected");
    }
  });

  updateProgress();
}

function updateProgress() {
  const answeredCount = userAnswers.filter(a => a !== null).length;
  const pct = (answeredCount / 21) * 100;

  document.getElementById("progressBar").style.width = `${pct}%`;
  document.getElementById("progressText").textContent = `${answeredCount} of 21 Answered`;

  const submitBtn = document.getElementById("submitBtn");
  submitBtn.disabled = answeredCount < 21;
}

async function submitAssessment() {
  const submitBtn = document.getElementById("submitBtn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Running ML & Deep Learning Pipelines...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ responses: userAnswers })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Analyze Psychological Profile ⚡";
  }
}

function renderResults(data) {
  document.getElementById("quizContainer").classList.add("hidden");
  const resSec = document.getElementById("resultsSection");
  resSec.classList.remove("hidden");
  resSec.scrollIntoView({ behavior: "smooth" });

  // Update scores and badges
  const updateMetric = (type, score, severity, dlInfo) => {
    document.getElementById(`score${type}`).textContent = score;
    const badge = document.getElementById(`badge${type}`);
    badge.textContent = `Scikit-Learn: ${severity}`;
    badge.className = `badge badge-${severity.toLowerCase().replace(" ", "-")}`;

    const dlTag = document.getElementById(`dl${type}`);
    if (dlInfo) {
      const confPct = Math.round(dlInfo.confidence * 100);
      dlTag.textContent = `Keras MTL-DNN: ${dlInfo.predicted_severity} (${confPct}%)`;
    } else {
      dlTag.style.display = "none";
    }
  };

  const dl = data.deep_learning_predictions || {};
  updateMetric("Depression", data.scores.depression, data.clinical_severity.depression, dl.Depression);
  updateMetric("Anxiety", data.scores.anxiety, data.clinical_severity.anxiety, dl.Anxiety);
  updateMetric("Stress", data.scores.stress, data.clinical_severity.stress, dl.Stress);

  // Radar Chart
  document.getElementById("radarChart").src = data.radar_chart_base64;

  // Contributing Symptoms
  const sympList = document.getElementById("symptomsList");
  sympList.innerHTML = "";
  if (data.top_symptoms && data.top_symptoms.length > 0) {
    data.top_symptoms.forEach(s => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${s.subscale} (${s.id}):</strong> "${s.statement}" <span style="float:right; opacity:0.8;">Severity: ${s.score}/3</span>`;
      sympList.appendChild(li);
    });
  } else {
    sympList.innerHTML = "<li>No elevated symptom clusters detected in responses.</li>";
  }

  // Recommendations
  const recList = document.getElementById("recommendationsList");
  recList.innerHTML = "";
  data.coping_strategies.forEach(tip => {
    const li = document.createElement("li");
    li.textContent = tip;
    recList.appendChild(li);
  });
}

function resetQuiz() {
  userAnswers.fill(null);
  document.querySelectorAll(".option-btn").forEach(b => b.classList.remove("selected"));
  document.querySelectorAll(".question-card").forEach(c => c.classList.remove("answered"));
  updateProgress();

  document.getElementById("resultsSection").classList.add("hidden");
  document.getElementById("quizContainer").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.addEventListener("DOMContentLoaded", () => {
  initQuestions();
  document.getElementById("submitBtn").addEventListener("click", submitAssessment);
  document.getElementById("resetBtn").addEventListener("click", resetQuiz);
});
