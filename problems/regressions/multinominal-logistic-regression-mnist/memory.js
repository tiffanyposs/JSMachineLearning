const _ = require('lodash');

const loadData = () => {
	const randoms = _.range(0, 999999);
	const firstHalf = randoms.slice(0, randoms.length / 2);
	const secondHalf = randoms.slice(randoms.length / 2, randoms.length - 1);
	return { firstHalf, secondHalf };
};

const { firstHalf, secondHalf } = loadData();





// const _ = require('lodash');
//
// const loadData = () => {
// 	const randoms = _.range(0, 999999);
// 	return randoms;
// };
//
// const data = loadData();
// const firstHalf = data.slice(0, data.length / 2);
// const secondHalf = data.slice(data.length / 2, data.length - 1);
//
// debugger;
//
// console.log(firstHalf.length)
// console.log(secondHalf.length)
// console.log(data.length);
