import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

import { PongDataset, featureDescriptions } from "./data";
import * as ui from "./ui";

// Some hyperparameters for model training.
const NUM_EPOCHS = 200;
const BATCH_SIZE = 20;
const LEARNING_RATE = 0.01;

const pongData = new PongDataset();
const tensors = {};

// Convert loaded data into tensors and creates normalized versions of the
// features.
export function arraysToTensors() {
  tensors.trainFeatures = tf.tensor2d(pongData.trainFeatures);
  tensors.trainTarget = tf.tensor2d(pongData.trainTarget);
  tensors.testFeatures = tf.tensor2d(pongData.testFeatures);
  tensors.testTarget = tf.tensor2d(pongData.testTarget);
}

/**
 * Builds and returns Linear Regression Model.
 *
 * @returns {tf.Sequential} The linear regression model.
 */
export function linearRegressionModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [pongData.numFeatures], units: 1 }));

  model.summary();
  return model;
}

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 1 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression model.
 */
export function multiLayerPerceptronRegressionModel1Hidden() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [pongData.numFeatures],
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal"
    })
  );
  model.add(tf.layers.dense({ units: 1 }));

  model.summary();
  return model;
}

/**
 * Builds and returns Multi Layer Perceptron Regression Model
 * with 2 hidden layers, each with 10 units activated by sigmoid.
 *
 * @returns {tf.Sequential} The multi layer perceptron regression mode  l.
 */
export function multiLayerPerceptronRegressionModel2Hidden() {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [pongData.numFeatures],
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal"
    })
  );
  model.add(
    tf.layers.dense({
      units: 50,
      activation: "sigmoid",
      kernelInitializer: "leCunNormal"
    })
  );
  model.add(tf.layers.dense({ units: 1 }));

  model.summary();
  return model;
}

/**
 * Describe the current linear weights for a human to read.
 *
 * @param {Array} kernel Array of floats of length 4.  One value per feature.
 * @returns {List} List of objects, each with a string feature name, and value
 *     feature weight.
 */
export function describeKernelElements(kernel) {
  tf.util.assert(
    kernel.length == 4,
    `kernel must be a array of length 4, got ${kernel.length}`
  );
  const outList = [];
  for (let idx = 0; idx < kernel.length; idx++) {
    outList.push({ description: featureDescriptions[idx], value: kernel[idx] });
  }
  return outList;
}

/**
 * Compiles `model` and trains it using the train data and runs model against
 * test data. Issues a callback to update the UI after each epcoh.
 *
 * @param {tf.Sequential} model Model to be trained.
 * @param {boolean} weightsIllustration Whether to print info about the learned
 *  weights.
 */
export async function run(
  model,
  modelName,
  weightsIllustration,
  nEpochs = NUM_EPOCHS
) {
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError"
  });

  let trainLogs = [];
  const container = document.querySelector(`#${modelName} .chart`);

  ui.updateStatus("Starting training process...");
  await model.fit(tensors.trainFeatures, tensors.trainTarget, {
    batchSize: BATCH_SIZE,
    epochs: nEpochs,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await ui.updateModelStatus(
          `Epoch ${epoch + 1} of ${nEpochs} completed.`,
          modelName
        );
        trainLogs.push(logs);
        tfvis.show.history(container, trainLogs, ["loss", "val_loss"]);

        if (weightsIllustration) {
          model.layers[0]
            .getWeights()[0]
            .data()
            .then(kernelAsArr => {
              const weightsList = describeKernelElements(kernelAsArr);
              ui.updateWeightDescription(weightsList);
            });
        }
      }
    }
  });

  ui.updateStatus("Running on test data...");
  const result = model.evaluate(tensors.testFeatures, tensors.testTarget, {
    batchSize: BATCH_SIZE
  });
  const testLoss = result.dataSync()[0];

  const trainLoss = trainLogs[trainLogs.length - 1].loss;
  const valLoss = trainLogs[trainLogs.length - 1].val_loss;
  await ui.updateModelStatus(
    `Final train-set loss: ${trainLoss.toFixed(4)}\n` +
      `Final validation-set loss: ${valLoss.toFixed(4)}\n` +
      `Test-set loss: ${testLoss.toFixed(4)}`,
    modelName
  );
}

export function computeBaseline() {
  const avgPrice = tensors.trainTarget.mean();
  console.log(`Average price: ${avgPrice.dataSync()}`);
  const baseline = tensors.testTarget
    .sub(avgPrice)
    .square()
    .mean();
  console.log(`Baseline loss: ${baseline.dataSync()}`);
  const baselineMsg = `Baseline loss (meanSquaredError) is ${baseline
    .dataSync()[0]
    .toFixed(2)}`;
  ui.updateBaselineStatus(baselineMsg);
}
