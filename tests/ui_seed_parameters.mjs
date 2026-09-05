import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

// Execute the production parameter collector without starting the wallet/UI.
// Its unrelated controls and size/schedule helpers are supplied as fixtures.
const source = fs.readFileSync(new URL("../src/create_page.js", import.meta.url), "utf8");
const functions = ["clampInt", "clampFloat", "readBaseTrainParams"].map((name) => {
  const match = source.match(new RegExp(`^  function ${name}\\b[\\s\\S]*?^  }`, "m"));
  assert.ok(match, `production function ${name} must be present`);
  return match[0];
}).join("\n");

const control = (value) => ({ value: String(value) });
const context = vm.createContext({
  treesNum: control(10),
  depthNum: control(3),
  lrNum: control(0.1),
  minLeafNum: control(2),
  binsNum: control(32),
  binningMode: control("linear"),
  seedNum: control(0),
  earlyStopOn: { checked: false },
  patienceNum: control(10),
  trainSplitNum: control(70),
  valSplitNum: control(20),
  selectedTask: "regression",
  buildLrScheduleFromUI: () => null,
  clampForSize: (trees, depth) => ({ trees, depth }),
  readImbalanceConfigForParams: () => null,
});
vm.runInContext(functions, context);

for (const [input, expected] of [
  ["0", 0],
  ["1", 1],
  ["42", 42],
  ["2147483647", 2147483647],
  ["", 42],
]) {
  context.seedNum.value = input;
  const params = vm.runInContext("readBaseTrainParams({ nClasses: 1 })", context);
  assert.equal(params.seed, expected, `seed input ${JSON.stringify(input)}`);
}

console.log("UI training parameters preserve seed 0 and the supported integer range.");
