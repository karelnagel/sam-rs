import { SamVersion, SamVersions, useEvents, useIsActive, useStore } from "./store";
import { Image } from "./Image";

export default function App() {
  useEvents();
  const isActive = useIsActive();
  const start = useStore((state) => state.start);
  const stop = useStore((state) => state.stop);
  const loadImage = useStore((state) => state.loadImage);
  const model = useStore((state) => state.model);
  const setModel = useStore((state) => state.setModel);
  const setVersion = useStore((state) => state.setVersion);
  const version = useStore((state) => state.version);
  const points = useStore((state) => state.points);
  const editPoint = useStore((state) => state.editPoint);
  const removePoint = useStore((state) => state.removePoint);
  const predictPoint = useStore((state) => state.predictPoint);

  return (
    <div className=" flex flex-col items-center space-y-3 p-3 overflow-hidden">
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
      <div className="grid grid-cols-3 gap-2">
        <Image />
        <div className="flex flex-col space-y-2">
          <p className="text-lg font-bold">Points:</p>
          {points.map((p) => (
            <div key={p.id} className="grid grid-cols-4 space-x-3">
              <input type="number" value={p.x} onChange={(e) => editPoint(p.id, { x: Number(e.target.value) })} />
              <input type="number" value={p.y} onChange={(e) => editPoint(p.id, { y: Number(e.target.value) })} />
              <select value={p.label} onChange={(e) => editPoint(p.id, { label: Number(e.target.value) })}>
                <option value={0}>Back</option>
                <option value={1}>Fore</option>
              </select>
              <button onClick={() => removePoint(p.id)}>X</button>
            </div>
          ))}
          <button disabled={!points.length} className="bg-blue-400 p-2 rounded-lg text-white disabled:bg-slate-400" onClick={predictPoint}>
            Predict
          </button>
        </div>
      </div>
    </div>
  );
}
