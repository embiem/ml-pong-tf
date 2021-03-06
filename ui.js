import {
  linearRegressionModel,
  multiLayerPerceptronRegressionModel1Hidden,
  multiLayerPerceptronRegressionModel2Hidden,
  run
} from ".";

const statusElement = document.getElementById("status");
export function updateStatus(message) {
  statusElement.innerText = message;
}

const baselineStatusElement = document.getElementById("baselineStatus");
export function updateBaselineStatus(message) {
  baselineStatusElement.innerText = message;
}

export function updateModelStatus(message, modelName) {
  const statElement = document.querySelector(`#${modelName} .status`);
  statElement.innerText = message;
}

const NUM_TOP_WEIGHTS_TO_DISPLAY = 5;
/**
 * Updates the weights output area to include information about the weights
 * learned in a simple linear model.
 * @param {List} weightsList list of objects with 'value':number and
 *     'description':string
 */
export function updateWeightDescription(weightsList) {
  const inspectionHeadlineElement = document.getElementById(
    "inspectionHeadline"
  );
  inspectionHeadlineElement.innerText = `Top ${NUM_TOP_WEIGHTS_TO_DISPLAY} weights by magnitude`;
  // Sort weights objects by descending absolute value.
  weightsList.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  var table = document.getElementById("myTable");
  // Clear out table contents
  table.innerHTML = "";
  // Add new rows to table.
  weightsList.forEach((weight, i) => {
    if (i < NUM_TOP_WEIGHTS_TO_DISPLAY) {
      let row = table.insertRow(-1);
      let cell1 = row.insertCell(0);
      let cell2 = row.insertCell(1);
      if (weight.value < 0) {
        cell2.setAttribute("class", "negativeWeight");
      } else {
        cell2.setAttribute("class", "positiveWeight");
      }
      cell1.innerHTML = weight.description;
      cell2.innerHTML = weight.value.toFixed(4);
    }
  });
}

function downloadLocalstorage() {
  const text = JSON.stringify(localStorage);

  var element = document.createElement("a");
  element.setAttribute(
    "href",
    "data:text/plain;charset=utf-8," + encodeURIComponent(text)
  );
  element.setAttribute("download", `localstorage.json`);

  element.style.display = "none";
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

export async function setup() {
  const trainSimpleLinearRegression = document.getElementById("simple-mlr");
  const trainNeuralNetworkLinearRegression1Hidden = document.getElementById(
    "nn-mlr-1hidden"
  );
  const trainNeuralNetworkLinearRegression2Hidden = document.getElementById(
    "nn-mlr-2hidden"
  );

  trainSimpleLinearRegression.addEventListener(
    "click",
    async e => {
      const model = linearRegressionModel();
      await run(model, "linear", true, 200);
      await model.save("localstorage://simpleNN-lr-model");
      //downloadLocalstorage();
    },
    false
  );

  trainNeuralNetworkLinearRegression1Hidden.addEventListener(
    "click",
    async () => {
      const model = multiLayerPerceptronRegressionModel1Hidden();
      await run(model, "oneHidden", false, 300);
      await model.save("localstorage://1hiddenNN-lr-model");
      //downloadLocalstorage();
    },
    false
  );

  trainNeuralNetworkLinearRegression2Hidden.addEventListener(
    "click",
    async () => {
      const model = multiLayerPerceptronRegressionModel2Hidden();
      await run(model, "twoHidden", false, 400);
      await model.save("localstorage://2hiddenNN-lr-model");
      downloadLocalstorage();
    },
    false
  );
}
