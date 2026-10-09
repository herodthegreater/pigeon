import { describe, expect, it } from "vitest";
import { Engine, Unsupported } from "./client";
import { DetectorEngine, NoWorker } from "../detector/client";

// Node has no Web Workers, which is exactly the browser the cages must turn away without a Reload button.
describe("a browser without Web Workers", () => {
  it("gets Pigeon's Unsupported error, which the cage answers with 'open it in a current browser'", async () => {
    await expect(new Engine().ready).rejects.toBeInstanceOf(Unsupported);
  });
  it("gets the detector's NoWorker error, which the YOLOv8n cage answers the same way", async () => {
    await expect(new DetectorEngine(() => {}).ready).rejects.toBeInstanceOf(NoWorker);
  });
});
