import { invoke } from "@tauri-apps/api/tauri";
import { create } from "zustand";
import { open } from "@tauri-apps/api/dialog";

const getRandomId = () => Math.random().toString(36).substr(2, 9);
type Point = {
  id: string;
  x: number;
  y: number;
  label: number;
};
type StoreType = {
  isActive: boolean;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  check: () => Promise<void>;
  image?: string;
  loadImage: () => Promise<void>;
  points: Point[];
  addPoint: (point: Point) => void;
  removePoint: (id: string) => void;
};
export const useStore = create<StoreType>((set, get) => ({
  isActive: false,
  start: async () => {
    const model = (await open({
      multiple: true,
      title: "Select model",
      directory: false,
      filters: [
        {
          name: "Model",
          extensions: ["bin", "bin.gz", "gz"],
        },
      ],
    })) as string;
    console.log(model);
    await invoke("start_model", { model: model[0] });
    set({ isActive: true });
  },

  stop: async () => {
    await invoke("stop_model");
    set({ isActive: false });
  },
  check: async () => {
    const active: boolean = await invoke("is_model_active");
    set({ isActive: active });
  },
  loadImage: async () => {
    const image = (await open({
      multiple: true,
      title: "Select image",
      directory: false,
      filters: [
        {
          name: "Image",
          extensions: ["png", "jpg", "jpeg"],
        },
      ],
    })) as string;
    await invoke("load_image", { image });
    set({ image });
  },
  points: [],
  addPoint: (point) => set((state) => ({ points: [...state.points, point] })),
  removePoint: (id) => set((state) => ({ points: state.points.filter((p) => p.id !== id) })),
}));

function App() {
  const isActive = useStore((state) => state.isActive);
  const start = useStore((state) => state.start);
  const stop = useStore((state) => state.stop);
  const loadImage = useStore((state) => state.loadImage);
  return (
    <div className="h-screen w-screen bg-red-400">
      <button className="bg-red-500" onClick={stop}>
        Stop
      </button>
      <button className="bg-green-500" onClick={start}>
        Start
      </button>
      {isActive && <button onClick={loadImage}>Load image</button>}
    </div>
  );
}

export default App;
