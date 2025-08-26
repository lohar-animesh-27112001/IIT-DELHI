// const form = document.getElementById("predictionForm");
// const resultDiv = document.getElementById("result");

// form.addEventListener("submit", async (e) => {
//   e.preventDefault();
//   resultDiv.textContent = "Predicting...";

//   const formData = new FormData(form);
//   const data = {};
//   formData.forEach((value, key) => {
//     data[key] = key === "smoker"
//       ? value === "true"
//       : isNaN(value) ? value : parseFloat(value);
//   });

//   try {
//     const response = await fetch("http://127.0.0.1:8000/predict", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify(data),
//     });

//     const result = await response.json();
//     if (response.ok) {
//       resultDiv.textContent = `Predicted Insurance Premium Category: ${result.insurance_premium_category}`;
//     } else {
//       resultDiv.textContent = `Error: ${result.detail || "Unable to predict"}`;
//     }
//   } catch (error) {
//     resultDiv.textContent = `Error: ${error.message}`;
//   }
// });

// script.js: Another way to handle form submission and prediction
async function makePrediction() {
    const input = document.getElementById("userInput").value;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ data: input })
        });

        const result = await response.json();
        document.getElementById("result").innerText = "Prediction: " + result.prediction;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error occurred.";
    }
}
