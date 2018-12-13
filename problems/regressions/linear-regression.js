const tf = require('@tensorflow/tfjs');
const _ = require('lodash');


class LinearRegression {

	// setup defaults
	constructor(features, labels, options) {
		this.features = tf.tensor(features);
		this.labels = tf.tensor(labels);

		// create a tensor of 1s
		this.features = tf.ones([this.features.shape[0], 1]).concat(this.features, 1);

		this.options = Object.assign({
			learningRate: 0.1,
			iterations: 1000
		}, options);

		this.weights = tf.zeros([2, 1]);
	}

	gradientDescent() {
		// matrix multiplication
		const currentGuesses = this.features.matMul(this.weights)
		const differences = currentGuesses.sub(this.labels);
		const slopes = this.features
		  .transpose()
			.matMul(differences)
			.div(this.features.shape[0]);

		this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
	}

	// gradientDescent() {
	// 	// (m * current_feature + b)
	// 	const currentGuessesForMPG = this.features.map(row => {
	// 		return this.m * row[0] + this.b;
	// 	});
	//
	// 	// (guess - current_label) ---> add them all together, * 2, divide by number of features
	// 	const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
	// 		return guess - this.labels[i][0];
	// 	})) * 2 / this.features.length;
	//
	// 	// (-1 * original_feature * (original_label - guess)) --- add them all togeter, * 2, divide by number of features
	// 	const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
	// 		return -1 * this.features[i][0] * (this.labels[i][0] - guess)
	// 	})) * 2 / this.features.length;
	//
	// 	this.m = this.m - mSlope * this.options.learningRate;
	// 	this.b = this.b - bSlope * this.options.learningRate;
	// }

	// loop through each allowed iteration
	train() {
		for (let i = 0; i < this.options.iterations; i++) {
			this.gradientDescent();
		}
	}

	test(testFeatures, testLabels) {
		testFeatures = tf.tensor(testFeatures);
		testLabels = tf.tensor(testLabels);
		testFeatures = tf.ones([testFeatures.shape[0], 1]).concat(testFeatures, 1);
		const predictions = testFeatures.matMul(this.weights);
		// predictions.print()
		// subtract the predictions from the real data, square all of them, add them together, get the value
		const res = testLabels.sub(predictions).pow(2).sum().get();

		// subtract the mean of the real data from the real data, square it, sum it, get the value
		const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

		return 1 - res / tot;
	}

}

module.exports = LinearRegression;
