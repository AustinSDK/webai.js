# webai.js
> Just as a disclaimer, this whole project was designed and architected by human. Although implemented mostly through AI code.

A tiny neural network library written from scratch in plain JavaScript. No dependencies, no build step, one file.

It's built to be **readable**: the whole training loop (forward pass, error, backward pass, update) is in one place, so you can learn how neural networks actually work by reading the source.

```js
const ai = new webai()
ai.passSchema({input: 2, hidden: [4], out: 1})
ai.passTrainingData([
    {input: [0, 0], output: [0]},
    {input: [0, 1], output: [1]},
    {input: [1, 0], output: [1]},
    {input: [1, 1], output: [0]},
])
ai.tryTraining()
ai.testCacheOnProblem([0, 1]) // -> [0.979...]
```

## Features

- Any number of inputs, outputs, and hidden layers (e.g. `2-4-2`, `3-8-8-1`)
- Backpropagation with sigmoid activation and mean squared error
- Shuffled training each epoch
- Seedable random numbers, so runs are repeatable
- Built-in schema validation with readable error messages
- Inspect weights, biases, and the loss history after training
- Works in the browser and in Node

## Install

Copy `webai.js` into your project. That's it.

**Browser**

```html
<script src="webai.js"></script>
<script>
    const ai = new webai()
</script>
```

**Node**

```js
const webai = require("./webai.js")
const ai = new webai()
```

## Quick start: learning XOR

XOR is the classic test because a network with no hidden layer can't learn it.

```js
const webai = require("./webai.js")
const ai = new webai()

// 2 inputs -> 4 hidden neurons -> 1 output
ai.passSchema({input: 2, hidden: [4], out: 1, learningRate: 0.5, epochs: 5000, seed: 1})

ai.passTrainingData([
    {input: [0, 0], output: [0]},
    {input: [0, 1], output: [1]},
    {input: [1, 0], output: [1]},
    {input: [1, 1], output: [0]},
])

console.log(ai.tryTraining())
// { epochs: 5000, finalLoss: 0.0004... }

for (const input of [[0,0],[0,1],[1,0],[1,1]]) {
    console.log(input, "->", ai.testCacheOnProblem(input)[0].toFixed(3))
}
// [0,0] -> 0.016   [0,1] -> 0.979   [1,0] -> 0.979   [1,1] -> 0.024
```

A runnable version is in [`example.js`](example.js): `node example.js`.

## Schema options

Passed to `passSchema()`.

| Option         | Required | Default | Description                                                        |
| -------------- | -------- | ------- | ------------------------------------------------------------------ |
| `input`        | yes      |         | Number of input values per sample                                  |
| `out`          | yes      |         | Number of output values per sample                                 |
| `hidden`       | no       | `[]`    | Hidden layer sizes, e.g. `[4]` or `[8, 8]`                         |
| `learningRate` | no       | `0.5`   | How big each weight nudge is. Lower is slower but steadier         |
| `epochs`       | no       | `5000`  | How many full passes over the training data                        |
| `seed`         | no       | random  | Any number. The same seed gives the same starting weights and order |

## API

| Method                           | What it does                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `passSchema(schema)`             | Builds the network from a schema (see above). Resets any previous training                             |
| `testSchema(sample)`             | Checks a sample matches the schema. Returns `true` or throws a readable error                          |
| `passTrainingDataSample(sample)` | Adds one sample: `{input: [...], output: [...]}`                                                       |
| `passTrainingData(data)`         | Adds a list of samples                                                                                 |
| `tryTraining()`                  | Trains for `epochs` passes. Returns `{epochs, finalLoss}`                                              |
| `testCacheOnProblem(input)`      | Runs an input array through the network and returns the output array                                   |
| `automateTesting()`              | Runs every stored training sample and returns `{rows, accuracy}` with expected vs. got for each        |
| `fetch(what)`                    | Reads data back out (see below)                                                                        |

### `fetch(what)`

| Value                | Returns                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| `"all"` (default)    | Schema, all layers (weights and biases), and loss history               |
| `"weights"`          | Weights for every layer                                                 |
| `"biases"`           | Biases for every layer                                                  |
| `"loss"`             | Average error per epoch, useful for seeing if training worked           |
| `"schema"`           | The schema you passed in, with defaults filled in                       |
| `"w<layer>_<neuron>_<input>"` | A single weight, e.g. `"w0_1_0"` is layer 0, neuron 1, input 0 |

## How it works

Each layer is a list of neurons. A neuron takes every value from the previous layer, multiplies each by its own weight, adds them up, adds a bias, and squashes the result with sigmoid:

```
out = sigmoid( (v1 * w1) + (v2 * w2) + ... + bias )
```

A `2-4-1` network therefore has two sets of weights: 4 neurons x 2 inputs, then 1 neuron x 4 hidden values. The hidden layer is made of computed *values*, not stored weights.

Training repeats these four steps for each sample, `epochs` times:

1. **Forward pass:** run the input through every layer to get an output
2. **Error:** compare the output to the target (mean squared error)
3. **Backward pass:** work out how much each neuron is to blame, starting at the output and passing blame back through the weights
4. **Update:** nudge every weight and bias a little in the direction that reduces the error

## Tips

- **Scale your data to 0-1.** Sigmoid outputs are always between 0 and 1, so targets outside that range can't be learned.
- **Check the loss.** If `fetch("loss")` is still dropping at the end, add more epochs. If it's flat and high, try more hidden neurons or a different learning rate.
- **Don't start from zero weights.** The library already uses random starting values, because identical weights make every neuron learn the same thing.

## Limitations

- Sigmoid is the only activation, used on every layer
- Trains one sample at a time (no mini-batches)
- No save/load for trained models yet
- No train/test split helper, so `automateTesting()` checks against the training data

## Roadmap

- [ ] Save and load weights as JSON
- [ ] More activation functions (ReLU, tanh)
- [ ] Mini-batch training
- [ ] Gradient checking to verify the backward pass
- [ ] Input normalization helper
- [ ] Train/test split

## Project layout

```
webai.js     the library
example.js   XOR demo
README.md    this file
```
