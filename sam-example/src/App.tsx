import { SamVersion, SamVersions, useIsActive, useStore } from "./store";

function App() {
  const isActive = useIsActive();
  const start = useStore((state) => state.start);
  const stop = useStore((state) => state.stop);
  const loadImage = useStore((state) => state.loadImage);
  const model = useStore((state) => state.model);
  const setModel = useStore((state) => state.setModel);
  const setVersion = useStore((state) => state.setVersion);
  const version = useStore((state) => state.version);
  return (
    <div className="h-screen w-screen bg-zinc-300 flex flex-col items-center space-y-3 p-3">
      <p>Select the model that you want to use.</p>
      <div className="flex space-x-2">
        <button className="bg-blue-400 p-2 rounded-lg max-w-[140px] overflow-hidden whitespace-nowrap" onClick={setModel}>
          {model?.split("/").pop() || "Select model"}
        </button>
        <select value={version} onChange={(e) => setVersion(e.target.value as SamVersion)}>
          {Object.entries(SamVersions).map(([key, value]) => (
            <option key={key} value={key}>
              {value}
            </option>
          ))}
        </select>
      </div>
      Status: {isActive ? "Active" : "Inactive"}
      {model && (
        <div className="flex space-x-2">
          <button className="bg-red-500 p-2 py-1 rounded-lg" onClick={stop}>
            Stop
          </button>
          <button className="bg-green-500 p-2 py-1 rounded-lg" onClick={start}>
            Start
          </button>
        </div>
      )}
      {isActive && <button onClick={loadImage}>Load image</button>}
    </div>
  );
}

export default App;
