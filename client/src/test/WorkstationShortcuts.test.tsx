import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const runAnalysis = vi.fn();

vi.mock("../services/api", () => ({
  runAnalysis: (...args: unknown[]) => runAnalysis(...args),
  fetchHealth: () => new Promise(() => undefined),
}));

// Every audio call is a no-op; the engine touches WebAudio, which jsdom lacks.
vi.mock("../audio/engine", () => ({
  audio: new Proxy({}, { get: () => () => undefined }),
}));

// The boot overlay holds the keyboard until it finishes; skip straight past it.
vi.mock("../components/cinematic/BootSequence", async () => {
  const { useEffect } = await import("react");
  return {
    BootSequence: ({ onComplete }: { onComplete: () => void }) => {
      useEffect(onComplete, [onComplete]);
      return null;
    },
  };
});

vi.mock("../components/hud/AmbientField", () => ({ AmbientField: () => null }));

/**
 * A stand-in for the real panel: the browser's metadata probe never resolves in
 * jsdom, so loading a clip through the actual dropzone is impossible. What is
 * under test is how the workstation wires a selection to the shortcut, not how
 * the panel reads a file.
 */
vi.mock("../components/video/SubjectPanel", () => ({
  SubjectPanel: ({
    label,
    onSelect,
  }: {
    label: string;
    onSelect: (file: File, url: string, meta: unknown) => void;
  }) => (
    <div>
      {["first", "second"].map((which) => (
        <button
          key={which}
          type="button"
          onClick={() =>
            onSelect(new File(["x"], `${label}-${which}.mp4`, { type: "video/mp4" }), "blob:x", {
              source: `${label}-${which}.mp4`,
              duration: 6,
              width: 1280,
              height: 720,
              format: "MP4",
            })
          }
        >
          {`load ${label} ${which}`}
        </button>
      ))}
    </div>
  ),
}));

const { Workstation } = await import("../components/workstation/Workstation");

const loadBoth = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("button", { name: "load Video A first" }));
  await user.click(screen.getByRole("button", { name: "load Video B first" }));
};

describe("Workstation shortcuts", () => {
  beforeEach(() => {
    runAnalysis.mockReset();
    runAnalysis.mockReturnValue(new Promise(() => undefined));
    sessionStorage.clear();
  });

  it("sends the clip currently in the slot when Enter is pressed after a replace", async () => {
    const user = userEvent.setup();
    render(<Workstation />);

    await loadBoth(user);
    await user.click(screen.getByRole("button", { name: "load Video A second" }));

    await user.keyboard("{Enter}");

    await waitFor(() => expect(runAnalysis).toHaveBeenCalled());
    const [sentA, sentB] = runAnalysis.mock.calls[0] as [File, File];
    expect(sentA.name).toBe("Video A-second.mp4");
    expect(sentB.name).toBe("Video B-first.mp4");
  });

  it("matches what the Run comparison button sends", async () => {
    const user = userEvent.setup();
    render(<Workstation />);

    await loadBoth(user);
    await user.click(screen.getByRole("button", { name: "load Video A second" }));
    await user.click(screen.getByRole("button", { name: /run comparison/i }));

    await waitFor(() => expect(runAnalysis).toHaveBeenCalled());
    expect((runAnalysis.mock.calls[0] as [File, File])[0].name).toBe("Video A-second.mp4");
  });

  it("does not start an analysis before both slots are filled", async () => {
    const user = userEvent.setup();
    render(<Workstation />);

    await user.click(screen.getByRole("button", { name: "load Video A first" }));
    await user.keyboard("{Enter}");

    expect(runAnalysis).not.toHaveBeenCalled();
  });
});
