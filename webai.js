// simple ai
let version = [0,1,2].join(".")
console.log(`[WebAI] Simple AI (v${version}) initialized.`)

/*
Steps to ai:
    1. Training data
    2. Training
        2.1 Forward Pass
        2.2 Error
        2.3 Backward Pass
        2.4 Update
    3. Test
*/

class webai{
    // options: {logging: fn(level, message, data) | false, logEvery: N epochs}
    // `logging` can be reassigned any time: ai.logging = (level, msg, data) => {...}
    constructor(options = {}){
        this.logging = options.logging === undefined
            ? (level, msg) => console.log(`[WebAI:${level}] ${msg}`)
            : options.logging
        this.logEvery = options.logEvery || 500
        this.scary = {
            trained: false,
            cache: {
                samples: [],     // training data: [{input:[..], output:[..]}, ...]
                lossHistory: []  // average error per epoch
            },
            schema: {},
            layers: []           // [{weights: [[..],[..]], biases: [..]}, ...]
        }
        this._rng = Math.random
    }

    // ---------- helpers ----------
    // level: "info" | "debug" | "warn"
    _log(level, message, data){
        if(typeof this.logging === "function") this.logging(level, message, data)
    }

    _sigmoid(x){ return 1 / (1 + Math.exp(-x)) }

    // small seedable random generator so runs can be repeated (mulberry32)
    _seedRandom(seed){
        let a = seed >>> 0
        this._rng = () => {
            a = (a + 0x6D2B79F5) >>> 0
            let t = a
            t = Math.imul(t ^ (t >>> 15), t | 1)
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296
        }
    }

    // ---------- 1. schema ----------
    // {input: 2, hidden: [4], out: 1, learningRate: 0.5, epochs: 5000, seed: 1}
    // Yes, more than one output works fine: out: 2, out: 10, etc.
    passSchema(schema){
        const s = Object.assign({hidden: [], learningRate: 0.5, epochs: 5000}, schema)
        if(!(s.input > 0) || !(s.out > 0)) throw new Error("[WebAI] schema needs input and out > 0")
        this.scary.schema = s
        if(s.seed !== undefined) this._seedRandom(s.seed)

        // sizes like [2, 4, 1]. Each neighbouring pair becomes one layer of weights.
        const sizes = [s.input, ...s.hidden, s.out]
        this.scary.layers = []
        for(let i = 1; i < sizes.length; i++){
            const prev = sizes[i-1], size = sizes[i]
            const weights = []
            for(let n = 0; n < size; n++){
                const row = []
                for(let p = 0; p < prev; p++) row.push(this._rng() * 2 - 1) // random -1..1, NOT zeros
                weights.push(row)
            }
            this.scary.layers.push({weights, biases: new Array(size).fill(0)})
        }
        this.scary.trained = false
        this.scary.cache.lossHistory = []
        this._log("info", `Schema set: ${sizes.join(" -> ")} (${this.scary.layers.length} layers, lr=${s.learningRate}, epochs=${s.epochs}${s.seed !== undefined ? ", seed=" + s.seed : ""})`, s)
    }

    // check a sample matches the schema. Returns true or throws a readable error.
    testSchema(example){
        const s = this.scary.schema
        if(!s.input) throw new Error("[WebAI] call passSchema first")
        if(!example || !Array.isArray(example.input) || !Array.isArray(example.output))
            throw new Error("[WebAI] sample must look like {input: [..], output: [..]}")
        if(example.input.length !== s.input)
            throw new Error(`[WebAI] expected ${s.input} inputs, got ${example.input.length}`)
        if(example.output.length !== s.out)
            throw new Error(`[WebAI] expected ${s.out} outputs, got ${example.output.length}`)
        return true
    }

    // ---------- 2. training data ----------
    passTrainingDataSample(sample){
        this.testSchema(sample)
        this.scary.cache.samples.push({input: sample.input.slice(), output: sample.output.slice()})
        this._log("debug", `Added sample #${this.scary.cache.samples.length}: [${sample.input}] -> [${sample.output}]`, sample)
    }
    passTrainingData(data){
        for(const sample of data) this.passTrainingDataSample(sample)
    }

    // ---------- 2.1 forward pass ----------
    // returns the activations of every layer (index 0 = the raw inputs)
    _forward(input){
        const acts = [input]
        for(const layer of this.scary.layers){
            const prev = acts[acts.length - 1]
            const out = layer.weights.map((row, n) => {
                let sum = layer.biases[n]
                for(let p = 0; p < prev.length; p++) sum += prev[p] * row[p]
                return this._sigmoid(sum)
            })
            acts.push(out)
        }
        return acts
    }

    // ---------- 2. training ----------
    tryTraining(){
        const {samples} = this.scary.cache
        const {epochs, learningRate} = this.scary.schema
        if(samples.length === 0) throw new Error("[WebAI] no training data")
        const layers = this.scary.layers
        this.scary.cache.lossHistory = []
        this._log("info", `Training started: ${samples.length} samples, ${epochs} epochs`)

        for(let epoch = 0; epoch < epochs; epoch++){
            // shuffle the order each epoch so it doesn't memorise the sequence
            const order = samples.map((_, i) => i)
            for(let i = order.length - 1; i > 0; i--){
                const j = Math.floor(this._rng() * (i + 1))
                ;[order[i], order[j]] = [order[j], order[i]]
            }

            let epochLoss = 0
            for(const idx of order){
                const {input, output: target} = samples[idx]

                // 2.1 forward pass
                const acts = this._forward(input)
                const result = acts[acts.length - 1]

                // 2.2 error (mean squared error, for display)
                for(let k = 0; k < target.length; k++) epochLoss += (result[k] - target[k]) ** 2

                // 2.3 backward pass: work out a "delta" (blame) for every neuron, last layer first
                const deltas = new Array(layers.length)
                // output layer: (how wrong) * (slope of sigmoid)
                deltas[layers.length - 1] = result.map((o, k) => (o - target[k]) * o * (1 - o))
                // hidden layers: blame flows back through the weights
                for(let l = layers.length - 2; l >= 0; l--){
                    const next = layers[l + 1], act = acts[l + 1]
                    deltas[l] = act.map((a, n) => {
                        let blame = 0
                        for(let k = 0; k < next.weights.length; k++) blame += deltas[l+1][k] * next.weights[k][n]
                        return blame * a * (1 - a)
                    })
                }

                // 2.4 update: nudge each weight against its blame
                for(let l = 0; l < layers.length; l++){
                    const prev = acts[l]
                    for(let n = 0; n < layers[l].weights.length; n++){
                        for(let p = 0; p < prev.length; p++)
                            layers[l].weights[n][p] -= learningRate * deltas[l][n] * prev[p]
                        layers[l].biases[n] -= learningRate * deltas[l][n]
                    }
                }
            }
            this.scary.cache.lossHistory.push(epochLoss / samples.length)
            if((epoch + 1) % this.logEvery === 0 || epoch === 0 || epoch === epochs - 1)
                this._log("debug", `Epoch ${epoch + 1}/${epochs} - loss: ${(epochLoss / samples.length).toFixed(6)}`, {epoch: epoch + 1, loss: epochLoss / samples.length})
        }

        this.scary.trained = true
        const h = this.scary.cache.lossHistory
        this._log("info", `Training finished. Final loss: ${h[h.length - 1].toFixed(6)}`)
        return {epochs, finalLoss: h[h.length - 1]}
    }

    // ---------- get stuff back out ----------
    // "all" | "weights" | "biases" | "loss" | "schema" | "w<layer>_<neuron>_<input>" like "w0_1_0"
    fetch(output="all"){
        const s = this.scary
        if(output === "all") return JSON.parse(JSON.stringify({schema: s.schema, layers: s.layers, loss: s.cache.lossHistory}))
        if(output === "weights") return s.layers.map(l => l.weights)
        if(output === "biases") return s.layers.map(l => l.biases)
        if(output === "loss") return s.cache.lossHistory.slice()
        if(output === "schema") return s.schema
        const m = /^w(\d+)_(\d+)_(\d+)$/.exec(output)
        if(m) return s.layers[m[1]]?.weights[m[2]]?.[m[3]]
        return undefined
    }

    // ---------- 3. test ----------
    // give it inputs, get the outputs back
    testCacheOnProblem(input){
        if(!this.scary.trained) this._log("warn", "model has not been trained yet")
        if(input.length !== this.scary.schema.input)
            throw new Error(`[WebAI] expected ${this.scary.schema.input} inputs, got ${input.length}`)
        const acts = this._forward(input)
        this._log("debug", `Test: [${input}] -> [${acts[acts.length - 1].map(v => v.toFixed(4))}]`)
        return acts[acts.length - 1]
    }

    // run every cached training sample through the model and report how it did
    automateTesting(){
        const rows = this.scary.cache.samples.map(({input, output}) => {
            const got = this.testCacheOnProblem(input)
            const error = got.reduce((sum, g, k) => sum + Math.abs(g - output[k]), 0) / got.length
            const correct = got.every((g, k) => Math.round(g) === Math.round(output[k]))
            return {input, expected: output, got, error, correct}
        })
        const accuracy = rows.filter(r => r.correct).length / rows.length
        this._log("info", `Testing: ${rows.filter(r => r.correct).length}/${rows.length} correct (${(accuracy * 100).toFixed(1)}%)`, rows)
        return {rows, accuracy}
    }
}

// works as a plain <script> in the browser (global `webai`) and with require() in node
export default webai