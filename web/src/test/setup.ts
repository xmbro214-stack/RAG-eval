import "@testing-library/jest-dom/vitest";

Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: {
    clear: () => undefined,
    getItem: () => null,
    removeItem: () => undefined,
    setItem: () => undefined
  }
});
