require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

// extract the data from the loadCSV module
let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
	shuffle: true,
	splitTest: 50,
	dataColumns: ['horsepower', 'weight', 'displacement'],
	labelColumns: ['mpg'],
});

// create a new regression, pass in feature, labels, and options
const regression = new LinearRegression(features, labels, {
	learningRate: 0.1,
	iterations: 100,
});

// train the regression
regression.train();

// test the regression's success
// you're looking for r2 to be a low positive number.
const r2 = regression.test(testFeatures, testLabels);

// create a graph
plot({
	x: regression.mseHistory.reverse(),
	xLabel: 'Iteration #',
	yLabel: 'Mean Squared Error',
});

console.log('R2 is ', r2);
