/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const Papa = require("papaparse");
import { arraysToTensors, computeBaseline } from ".";
import * as ui from "./ui";

const TRAIN_FEATURES_FN = "train-features.csv";
const TRAIN_TARGET_FN = "train-target.csv";
const TEST_FEATURES_FN = "test-features.csv";
const TEST_TARGET_FN = "test-target.csv";

/**
 * Given CSV data returns an array of arrays of numbers.
 *
 * @param {Array<Object>} data Downloaded data.
 *
 * @returns {Promise.Array<number[]>} Resolves to data with values parsed as floats.
 */
const parseCsv = async data => {
  return new Promise(resolve => {
    data = data.map(row => {
      return Object.keys(row).map(key => parseFloat(row[key]));
    });
    resolve(data);
  });
};

/**
 * Downloads and returns the csv.
 *
 * @param {File} file file to be loaded.
 *
 * @returns {Promise.Array<number[]>} Resolves to parsed csv data.
 */
export const loadCsv = async file => {
  return new Promise(resolve => {
    console.log(`  * Downloading data from: ${file.name}`);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: results => {
        resolve(parseCsv(results["data"]));
      }
    });
  });
};

/** Helper class to handle loading training and test data. */
export class BostonHousingDataset {
  constructor() {
    // Arrays to hold the data.
    this.trainFeatures = null;
    this.trainTarget = null;
    this.testFeatures = null;
    this.testTarget = null;

    const uploadDataInput = document.getElementById("upload-data");
    uploadDataInput.addEventListener("change", e => {
      const files = {};
      for (let i = 0; i < e.target.files.length; i++) {
        const file = e.target.files[i];
        files[file.name] = file;
      }

      this.loadData(files);
    });
  }

  get numFeatures() {
    // If numFetures is accessed before the data is loaded, raise an error.
    if (this.trainFeatures == null) {
      throw new Error("'loadData()' must be called before numFeatures");
    }
    return this.trainFeatures[0].length;
  }

  /** Loads training and test data. */
  async loadData(files) {
    [
      this.trainFeatures,
      this.trainTarget,
      this.testFeatures,
      this.testTarget
    ] = await Promise.all([
      loadCsv(files[TRAIN_FEATURES_FN]),
      loadCsv(files[TRAIN_TARGET_FN]),
      loadCsv(files[TEST_FEATURES_FN]),
      loadCsv(files[TEST_TARGET_FN])
    ]);

    shuffle(this.trainFeatures, this.trainTarget);
    shuffle(this.testFeatures, this.testTarget);

    ui.updateStatus("Data loaded, converting to tensors");
    arraysToTensors();
    ui.updateStatus(
      "Data is now available as tensors.\n" + "Click a train button to begin."
    );
    // TODO Explain what baseline loss is. How it is being computed in this
    // Instance
    ui.updateBaselineStatus("Estimating baseline loss");
    computeBaseline();
    await ui.setup();
  }
}

export const featureDescriptions = [
  "Ball X Position",
  "Ball Y Position",
  "Ball X Velocity",
  "Ball Y Velocity"
];

/**
 * Shuffles data and target (maintaining alignment) using Fisher-Yates
 * algorithm.flab
 */
function shuffle(data, target) {
  let counter = data.length;
  let temp = 0;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    // data:
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    // target:
    temp = target[counter];
    target[counter] = target[index];
    target[index] = temp;
  }
}
