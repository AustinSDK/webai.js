// simple ai (v0.1.1)

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
    constructor(){}
    passSchema(schema){}                // such as {input: 2, out: 1}... does ai even work with more then one output?
    testSchema(example){}               // debugging i suppose
    passTrainingDataSample(sample){}    // A single sample
    passTrainingData(data){}            // data in a list matching schema
    tryTraining(){}                     // try to train the model
    fetch(output="all"){}               // fetch either a specific output ("w1") or all "null"
    testCacheOnProblem(schema){}        // test inputs to get export.
    automateTesting(){}                 // automates testing the model based on previously cached information
}