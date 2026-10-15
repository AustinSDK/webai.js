// Sample usage: learn the x-intercept of y = m*x + b
// run with:  node example.js
import webai from "./webai.js";

const ai = new webai();
ai.logging = function(){}

// ---- ground truth ----
function xIntercept(m, b) {
  return -b / m; // only called with |m| >= M_MIN
}

// ---- data ranges (keep |m| away from 0 so the output stays bounded) ----
const M_MIN = 0.5, M_MAX = 5;
const B_MAX = 10;
const X_MAX = B_MAX / M_MIN; // largest possible |x| = 20

// ---- normalization: network sees small numbers, we convert back after ----
const encodeIn = (m, b) => [m / M_MAX, b / B_MAX];          // each in [-1, 1]
const encodeOut = (x) => (x / X_MAX + 1) / 2;               // [-20, 20] -> [0, 1]
const decodeOut = (y) => (y * 2 - 1) * X_MAX;               // back to real x

function randomSample() {
  const sign = Math.random() < 0.5 ? -1 : 1;
  const m = sign * (M_MIN + Math.random() * (M_MAX - M_MIN));
  const b = (Math.random() * 2 - 1) * B_MAX;
  return { m, b };
}

// 2 inputs -> 2 hidden layers of 16 -> 1 output
ai.passSchema({
  input: 2,
  hidden: [16, 16],
  out: 1,
  learningRate: 0.05,
  epochs: 500,
  seed: 1,
});

const trainingData = [];
for (let j = 0; j < 5000; j++) {
  const { m, b } = randomSample();
  trainingData.push({ input: encodeIn(m, b), output: [encodeOut(xIntercept(m, b))] });
}
ai.passTrainingData(trainingData);

const result = ai.tryTraining();
console.log("Training done:", result);

// in-range test: slope 6/7, x-intercept 6  ->  b = -6 * (6/7)
const m = 6 / 7;
const b = -6 * m; // -5.142857...
const predicted = decodeOut(ai.testCacheOnProblem(encodeIn(m, b))[0]);
console.log([m, b], "->", predicted.toFixed(3), "(expected", xIntercept(m, b).toFixed(3) + ")");

// full report
const report = ai.automateTesting();
console.log("Accuracy:", report.accuracy * 100 + "%");

// peek inside
console.log("One weight (layer 0, neuron 1, input 0):", ai.fetch("w0_1_0"));
console.log("Loss at epoch 1 vs last:", ai.fetch("loss")[0].toFixed(4), "->", ai.fetch("loss").slice(-1)[0].toFixed(6));