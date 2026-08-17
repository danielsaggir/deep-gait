import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Testing Library only auto-registers cleanup when Vitest globals are enabled,
// so unmount explicitly to stop renders leaking between tests.
afterEach(cleanup);

// jsdom implements neither, and the reducer revokes object URLs on replace.
URL.createObjectURL ??= () => "blob:test";
URL.revokeObjectURL ??= () => undefined;
