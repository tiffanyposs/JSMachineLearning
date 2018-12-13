require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
	shuffle: true,
	splitTest: 50,
	dataColumns: ['horsepower', 'weight', 'displacement'],
	labelColumns: ['mpg'],
});

const regression = new LinearRegression(features, labels, {
	learningRate: 10,
	iterations: 100,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);

console.log('MSE histor: ', regression.mseHistory);
// your looking for r2 to be a low positive number. If it is negative, that means
// just taking the average is better than the answer you're getting. So you want a positive
// number
console.log('R2 is ', r2);
// console.log('Updated m is: ', regression.weights.get(1, 0));
// console.log('Updated b is: ', regression.weights.get(0, 0));
