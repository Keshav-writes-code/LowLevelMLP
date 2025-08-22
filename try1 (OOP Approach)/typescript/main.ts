import { MLP } from "./NeuralNet";
import { verticalBarGraph } from "./utils";

let model = new MLP(10, [23, 443, 12], 3);

let old_output = model.forward_propogation([
  23, 43, 23, 12, 4, 54, 23, 43, 12, 3,
]);

const history: number[] = [model.loss([0, 1, 0])];
for (let index = 0; index < 1000; index++) {
  model.backpropogate([23, 43, 23, 12, 4, 54, 23, 43, 12, 3], [0, 1, 0], 0.03);
  history.push(model.loss([0, 1, 0]));
}
let new_output = model.forward_propogation([
  23, 43, 23, 12, 4, 54, 23, 43, 12, 3,
]);
let new_loss = model.loss([0, 1, 0]);

console.log(old_output);
console.log(new_output);
console.log(history);

verticalBarGraph(history);
