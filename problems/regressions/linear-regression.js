const tf = require('@tensorflow/tfjs');
const _ = require('lodash');


class LinearRegression {

	// setup defaults
	constructor(features, labels, options) {
		this.features = this.processFeatures(features);
		this.labels = tf.tensor(labels);
		this.mseHistory = [];

		this.options = Object.assign({
			learningRate: 0.1,
			iterations: 1000
		}, options);

		this.weights = tf.zeros([this.features.shape[1], 1]);
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

	// loop through each allowed iteration
	train() {
		for (let i = 0; i < this.options.iterations; i++) {
			this.gradientDescent();
			this.recordMSE();
			this.updateLearningRate();
		}
	}

	test(testFeatures, testLabels) {
		testFeatures = this.processFeatures(testFeatures);
		testLabels = tf.tensor(testLabels);
		const predictions = testFeatures.matMul(this.weights);

		// subtract the predictions from the real data, square all of them, add them together, get the value
		const res = testLabels.sub(predictions).pow(2).sum().get();

		// subtract the mean of the real data from the real data, square it, sum it, get the value
		const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

		return 1 - res / tot;
	}

	processFeatures(features) {
		features = tf.tensor(features);

		if (this.mean && this.variance) {
			features = features.sub(this.mean).div(this.variance.pow(0.5))
		} else {
			features = this.standardize(features);
		}

		features = tf.ones([features.shape[0], 1]).concat(features, 1);

		return features;
	}

	standardize(features) {
		const { mean, variance } = tf.moments(features, 0);
		this.mean = mean;
		this.variance = variance;
		return features.sub(mean).div(variance.pow(0.5));
	}

	// records mean square error
	recordMSE() {
		const mse = this.features
			.matMul(this.weights)
			.sub(this.labels)
			.pow(2)
			.sum()
			.div(this.features.shape[0])
			.get();

		this.mseHistory.unshift(mse);
	}

	updateLearningRate() {
		if (this.mseHistory.length < 2) return;
		if (this.mseHistory[0] > this.mseHistory[1]) {
			this.options.learningRate /= 2;
		} else {
			this.options.learningRate *= 1.05;
		}
	}

}




module.exports = LinearRegression;
