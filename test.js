// Sample usage: teach the network XOR (the classic "needs a hidden layer" test)
// run with:  node example.js
import webai from "./webai.js";

const ai = new webai()

// 2 inputs -> 4 hidden neurons -> 1 output
ai.passSchema({input: 2, hidden: [4], out: 1, learningRate: 0.5, epochs: 5000, seed: 1})

ai.passTrainingData([
    {input: [0, 0], output: [0]},
    {input: [0, 1], output: [1]},
    {input: [1, 0], output: [1]},
    {input: [1, 1], output: [0]},
])

const result = ai.tryTraining()
console.log("Training done:", result)

// try it on the inputs
for(const input of [[0,0],[0,1],[1,0],[1,1]]){
    console.log(input, "->", ai.testCacheOnProblem(input)[0].toFixed(3))
}

// full report
const report = ai.automateTesting()
console.log("Accuracy:", report.accuracy * 100 + "%")

// peek inside
console.log("One weight (layer 0, neuron 1, input 0):", ai.fetch("w0_1_0"))
console.log("Loss at epoch 1 vs last:", ai.fetch("loss")[0].toFixed(4), "->", ai.fetch("loss").slice(-1)[0].toFixed(6))