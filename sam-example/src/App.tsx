import { SamVersion, SamVersions, getRandomId, useIsActive, useStore } from "./store";

function App() {
  const isActive = useIsActive();
  const start = useStore((state) => state.start);
  const stop = useStore((state) => state.stop);
  const loadImage = useStore((state) => state.loadImage);
  const model = useStore((state) => state.model);
  const setModel = useStore((state) => state.setModel);
  const setVersion = useStore((state) => state.setVersion);
  const version = useStore((state) => state.version);
  const points = useStore((state) => state.points);
  const addPoint = useStore((state) => state.addPoint);
  const editPoint = useStore((state) => state.editPoint);
  const removePoint = useStore((state) => state.removePoint);
  const predictPoint = useStore((state) => state.predictPoint);

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
      <div className="flex flex-col space-y-2">
        {points.map((p) => (
          <div key={p.id} className="flex space-x-3">
            <input type="number" value={p.x} onChange={(e) => editPoint(p.id, { x: Number(e.target.value) })} />
            <input type="number" value={p.y} onChange={(e) => editPoint(p.id, { y: Number(e.target.value) })} />
            <select value={p.label} onChange={(e) => editPoint(p.id, { label: Number(e.target.value) })}>
              <option value={0}>Back</option>
              <option value={1}>Fore</option>
            </select>
            <button onClick={() => removePoint(p.id)}>X</button>
          </div>
        ))}
        <button onClick={() => addPoint({ id: getRandomId(), label: 1, x: 0, y: 0 })}>Add point</button>
        <button onClick={predictPoint}>Predict</button>
      </div>
    </div>
  );
}

export default App;
