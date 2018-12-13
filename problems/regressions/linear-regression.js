const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {

	/**
	 * Sets up all default properties
	 * @name constructor
	 * @param features - matrix of features
	 * @param labels - matrix of features
	 * @param options - override default options
	*/
	constructor(features, labels, options) {
		this.features = this.processFeatures(features); // all features
		this.labels = tf.tensor(labels); // all labels
		this.mseHistory = []; // history of Mean Square Error

		this.options = Object.assign({
			learningRate: 0.1,
			iterations: 1000
		}, options);

		this.weights = tf.zeros([this.features.shape[1], 1]); // m and b values
	}

	/**
	 * multiply the features with the weights (m & b guesses)
	 * get the differences between the current guesses and the labels
	 * find the average slope between the features and the differences
	 * update the weights (m & b) to be the weights minus the slopes * learning rate
	 * @name gradientDescent
	*/
	gradientDescent() {
		// matrix multiplication
		const currentGuesses = this.features.matMul(this.weights);
		const differences = currentGuesses.sub(this.labels);
		const slopes = this.features
		  .transpose()
			.matMul(differences)
			.div(this.features.shape[0]);

		this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
	}

	/**
	 * Initializes the training
	 * loop through each allowed iteration and update the gradient Decent
	 * recored the Mean Square Error
	 * update the learning rate
	 * @name train
	*/
	train() {
		for (let i = 0; i < this.options.iterations; i++) {
			this.gradientDescent();
			this.recordMSE();
			this.updateLearningRate();
		}
	}

	/**
	 * Test the results
	 * process the testFeatures and testLabels the same way as for training data
	 * make predictions based on the final (m & b) (this.weights) by multiplying testFeatures matrix by the weights
	 * get the res (sum of squares of residuals) by subtracting the predictions from the actual results, square all of them, add them together, get the value
	 * get the tot (total sum of squares) by subtracting the mean of the real data from the real data, square it, sum it, get the value
	 * return 1 minus the res / tot
	 * @name test
	 * @param testFeatures
	 * @param testLabels
	 * @return {Number}
	*/
	test(testFeatures, testLabels) {
		testFeatures = this.processFeatures(testFeatures);
		testLabels = tf.tensor(testLabels);

		const predictions = testFeatures.matMul(this.weights);
		const res = testLabels.sub(predictions).pow(2).sum().get();
		const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

		return 1 - res / tot;
	}

	/**
	  * Process a feature set of data
		* convert into a tensor
		* standardize the features
		* lastly, create a ones matrix with the number of rows of the feature set and concat with with features
		* @name processFeatures
		* @param features {Array}
		* @return {Tensor}
	*/
	processFeatures(features) {
		features = tf.tensor(features);
		features = this.standardize(features);
		features = tf.ones([features.shape[0], 1]).concat(features, 1);

		return features;
	}

	/**
	 * If the mean and variance already exist, run the standardization on the existing mean and variance
	 * If not, get the mean and variance by using tf.moments on features
	 * set the mean and variance props
	 * then run the standardization on the updated mean and variance
	 * @name standardize
	 * @param features
	*/
	standardize(features) {
		if(this.mean && this.variance) {
			return features.sub(this.mean).div(this.variance.pow(0.5));
		}

		const { mean, variance } = tf.moments(features, 0);
		this.mean = mean;
		this.variance = variance;

		return features.sub(mean).div(variance.pow(0.5));
	}

	/**
	  * Tracks the Mean Square Error history
		* Matrix Multiply the features by the weights (m & b)
		* subtract the labels
		* square it
		* add them all together
		* divide by the number of features
		* add it to the mseHistory prop
		* @name recordMSE
	*/
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

	/**
	 * Dynamically updates the learning rate
	 * if the mean square error history doesn't have enough history, skip
	 * if the last MSE is greater than the second to last MSE ---> divide the learning rate by half (it's way too big)
	 * else increase the learning rate by 5% ( make it slightly more in the right direction)
	 * @name updateLearningRate
	*/
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
