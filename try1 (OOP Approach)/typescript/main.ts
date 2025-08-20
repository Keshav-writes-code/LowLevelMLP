import { MLP } from "./NeuralNet";

let model = new MLP(10, [23, 443, 12], 3);

let output = model.forward_propogation([23, 43, 23, 12, 4, 54, 23, 43, 12, 3]);
console.log(output);
