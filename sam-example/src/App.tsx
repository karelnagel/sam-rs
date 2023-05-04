import { useEffect, useRef } from "react";
import { SamVersion, SamVersions, getRandomId, useEvents, useIsActive, useStore } from "./store";
import { convertFileSrc } from "@tauri-apps/api/tauri";

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
      <Image />
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

const Image = () => {
  const masks = useStore((state) => state.masks);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  let image = useStore((state) => state.image);
  if (image) image = convertFileSrc(image);
  if (!image) return null;

  useEffect(() => {
    const mask = masks?.[0];
    const img = imageRef.current;
    const canvas = canvasRef.current;
    console.log(mask);
    if (!img) return;
    if (!mask) return;
    if (!canvas) return;

    const ctx = canvas.getContext("2d")!;

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      for (let row = 0; row < mask.length; row++) {
        for (let col = 0; col < mask[row].length; col++) {
          if (!mask[row][col]) {
            ctx.fillStyle = "rgba(100, 100, 0, 0.5)";
            ctx.fillRect(col, row, 1, 1);
          }
        }
      }
    };
  }, [masks]);
  return (
    <div className="">
      <img ref={imageRef} src={image} className="object-none hidden" style={{}} />
      <canvas ref={canvasRef} />
    </div>
  );
};
