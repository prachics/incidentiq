import { render, screen } from "@testing-library/react";
import { Timeline } from "./Timeline";

describe("Timeline", () => {
  it("lists steps in order with durations, and marks failures", () => {
    render(
      <Timeline
        running={false}
        events={[
          { node: "intake", duration_ms: 812, at: 1, ok: true, service: "checkout-service" },
          { node: "execute_tool", duration_ms: 64, at: 2, ok: false },
          { node: "act", duration_ms: 1500, at: 3, ok: true, decision: "approved", attempts: 2 },
        ]}
      />,
    );
    const steps = screen.getAllByRole("listitem");
    expect(steps).toHaveLength(3);
    expect(steps[0]).toHaveTextContent("Read the report");
    expect(steps[0]).toHaveTextContent("checkout-service");
    expect(steps[0]).toHaveTextContent("812 ms");
    expect(steps[1]).toHaveTextContent("Ran the diagnostic");
    expect(steps[1]?.querySelector("p")).toHaveClass("text-red-800");
    expect(steps[2]).toHaveTextContent("approved · 2 attempts");
    expect(steps[2]).toHaveTextContent("1.5 s");
  });

  it("shows that work is in progress", () => {
    render(<Timeline running events={[]} />);
    expect(screen.getByText("Starting…")).toBeInTheDocument();
  });
});
