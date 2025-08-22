export function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}
/**
 * Display a vertical bar graph in the terminal for large datasets
 * @param data Array of numbers to display
 * @param options Configuration options for the graph
 */
export function verticalBarGraph(
  data: number[],
  options: {
    title?: string;
    barChar?: string;
    maxHeight?: number;
    maxBars?: number;
    startIndex?: number;
    showLabels?: boolean;
    showValues?: boolean;
    sampling?: "none" | "every-n" | "average";
    samplingFactor?: number;
  } = {},
): void {
  const {
    title = "Data Visualization",
    barChar = "█",
    maxHeight = 20,
    maxBars = getTerminalWidth() - 10,
    startIndex = 0,
    showLabels = true,
    showValues = true,
    sampling = "none",
    samplingFactor = 1,
  } = options;

  function getTerminalWidth(): number {
    return process.stdout.columns || 80;
  }

  console.log(`\n${title}`);

  if (data.length === 0) {
    console.log("No data to display");
    return;
  }

  // Apply sampling if needed
  let displayData: number[] = [];
  let originalIndices: number[] = [];

  if (sampling === "none") {
    const endIndex = Math.min(startIndex + maxBars, data.length);
    displayData = data.slice(startIndex, endIndex);
    originalIndices = Array.from(
      { length: displayData.length },
      (_, i) => i + startIndex,
    );
  } else if (sampling === "every-n") {
    for (let i = 0; i < data.length; i += samplingFactor) {
      if (displayData.length >= maxBars) break;
      displayData.push(data[i]);
      originalIndices.push(i);
    }
  } else if (sampling === "average") {
    for (let i = 0; i < data.length; i += samplingFactor) {
      if (displayData.length >= maxBars) break;
      const chunk = data.slice(i, i + samplingFactor);
      const avg = chunk.reduce((sum, val) => sum + val, 0) / chunk.length;
      displayData.push(avg);
      originalIndices.push(i);
    }
  }

  // Calculate the maximum value for scaling
  const maxValue = Math.max(...displayData);

  // Create the vertical bars
  const scaledData = displayData.map((value) =>
    Math.max(1, Math.round((value / maxValue) * maxHeight)),
  );

  // Build the chart from top to bottom
  const chart: string[][] = [];

  // Add values at the top if requested
  if (showValues) {
    const valueRow: string[] = [];
    for (let i = 0; i < displayData.length; i++) {
      // Truncate value to fit in column
      const value = displayData[i].toString().slice(0, 3);
      valueRow.push(value.padStart(1).padEnd(1));
    }
    chart.push(valueRow);
  }

  // Build the bars
  for (let row = 0; row < maxHeight; row++) {
    const chartRow: string[] = [];
    for (let col = 0; col < scaledData.length; col++) {
      // If current height is within bar height, add bar character
      chartRow.push(maxHeight - row <= scaledData[col] ? barChar : " ");
    }
    chart.push(chartRow);
  }

  // Add index labels at the bottom if requested
  if (showLabels) {
    const labelRow: string[] = [];
    for (let i = 0; i < originalIndices.length; i++) {
      // Add label every 5 indices to avoid crowding
      const index = originalIndices[i];
      const label = i % 5 === 0 ? index.toString().slice(-1) : "";
      labelRow.push(label.padStart(1));
    }
    chart.push(labelRow);
  }

  // Render the chart
  for (const row of chart) {
    console.log(row.join(""));
  }

  // Show summary of what's displayed
  if (data.length > displayData.length) {
    console.log(`\nShowing ${displayData.length} of ${data.length} items`);
    if (sampling !== "none") {
      console.log(`Sampling: ${sampling} with factor ${samplingFactor}`);
    } else {
      console.log(
        `Range: ${startIndex + 1}-${startIndex + displayData.length}`,
      );
    }
  }

  const sum = displayData.reduce((acc, val) => acc + val, 0);
  const avg = sum / displayData.length;
  console.log(`\nAvg: ${avg.toFixed(2)} | Max: ${maxValue.toFixed(2)}`);
}
