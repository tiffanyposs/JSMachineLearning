require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

// extract the data from the loadCSV module
let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
	shuffle: true,
	splitTest: 50,
	dataColumns: ['horsepower', 'weight', 'displacement'],
	labelColumns: ['mpg'],
});

// create a new regression, pass in feature, labels, and options
const regression = new LinearRegression(features, labels, {
	learningRate: 0.1,
	iterations: 3, // implementing Batch or Stochastic Gradient Decent lowers this number
	batchSize: 10, // if you switch this to 1, then you're using Stochastic
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

regression.predict([
	[130, 1.752, 307], // 18 ---> guess: 18.68
	[100, 1.641, 250], // 19 ---> guess: 21.74
	[69, 0.8065, 72] // 35 ---> guess: 32.67
]).print();


// passedemissions,mpg,cylinders,displacement,horsepower,weight,acceleration,modelyear,carname
// FALSE,18,8,307,130,1.752,12,70,chevrolet chevelle malibu
// TRUE,19,6,250,100,1.641,15,71,pontiac firebird
// TRUE,35,4,72,69,0.8065,18,71,datsun 1200
