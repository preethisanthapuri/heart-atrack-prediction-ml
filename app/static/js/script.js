(() => {
  const form = document.getElementById("prediction-form");
  const button = document.getElementById("submit-button");
  const error = document.getElementById("form-error");
  const empty = document.getElementById("result-empty");
  const content = document.getElementById("result-content");
  const panel = document.getElementById("result-panel");
  const percent = value => `${(value * 100).toFixed(1)}%`;

  form.addEventListener("submit", async event => {
    event.preventDefault();
    error.hidden = true;
    if (!form.reportValidity()) return;
    const payload = {};
    for (const field of form.elements) {
      if (field.name) payload[field.name] = field.value;
    }
    button.disabled = true;
    button.querySelector("span:first-child").textContent = "Generating…";
    panel.setAttribute("aria-busy", "true");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "The prediction request failed.");

      document.getElementById("predicted-class").textContent = `Class ${result.predicted_dataset_class}`;
      document.getElementById("p-class-1").textContent = percent(result.probability_class_1);
      document.getElementById("p-class-0").textContent = percent(result.probability_class_0);
      document.getElementById("bar-class-1").style.width = percent(result.probability_class_1);
      document.getElementById("bar-class-0").style.width = percent(result.probability_class_0);

      const explanation = document.getElementById("explanation-list");
      explanation.replaceChildren();
      result.explanation.forEach(item => {
        const li = document.createElement("li");
        const feature = document.createElement("span");
        const contribution = document.createElement("span");
        feature.textContent = item.feature;
        contribution.textContent = `${item.model_contribution_log_odds >= 0 ? "+" : ""}${item.model_contribution_log_odds.toFixed(3)} · ${item.direction}`;
        contribution.className = item.model_contribution_log_odds >= 0 ? "contribution-positive" : "contribution-negative";
        li.append(feature, contribution);
        explanation.append(li);
      });

      const warnings = document.getElementById("warning-list");
      warnings.replaceChildren();
      warnings.hidden = result.warnings.length === 0;
      result.warnings.forEach(message => {
        const p = document.createElement("p");
        p.textContent = message;
        warnings.append(p);
      });
      empty.hidden = true;
      content.hidden = false;
    } catch (exception) {
      error.textContent = exception.message;
      error.hidden = false;
    } finally {
      button.disabled = false;
      button.querySelector("span:first-child").textContent = "Generate estimate";
      panel.setAttribute("aria-busy", "false");
    }
  });
})();
