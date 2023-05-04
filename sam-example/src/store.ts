import { invoke } from "@tauri-apps/api/tauri";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { open } from "@tauri-apps/api/dialog";

export const SamVersions = {
  SamVitH: "Vit H",
  SamVitB: "Vit B",
  SamVitL: "Vit L",
  SamTest: "Test version",
};
export type SamVersion = keyof typeof SamVersions;
export const getRandomId = () => Math.random().toString(36).substr(2, 9);
export type Point = {
  id: string;
  x: number;
  y: number;
  label: number;
};
export type StoreType = {
  isActive: boolean;
  model?: string;
  setModel: () => Promise<void>;
  setVersion: (version: SamVersion) => void;
  version: SamVersion;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  check: () => Promise<void>;
  image?: string;
  loadImage: () => Promise<void>;
  points: Point[];
  addPoint: (point: Point) => void;
  removePoint: (id: string) => void;
};
export const useStore = create(
  persist<StoreType>(
    (set, get) => ({
      version: "SamVitH",
      setVersion: (version) => set({ version }),
      setModel: async () => {
        let model = await open({
          multiple: true,
          title: "Select model",
          directory: false,
          filters: [
            {
              name: "Model",
              extensions: ["bin", "bin.gz", "gz"],
            },
          ],
        });
        if (!model) return;
        if (Array.isArray(model)) model = model[0];

        set({ model });
      },
      isActive: false,
      start: async () => {
        let model = get().model;
        let version = get().version;
        await invoke("start_model", { model: model?.replace(".bin.gz", ""), version });
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
        let path = (await open({
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
        if (!path) return;
        if (Array.isArray(path)) path = path[0];

        await invoke("load_image", { path });
        set({ image: path });
      },
      points: [],
      addPoint: (point) => set((state) => ({ points: [...state.points, point] })),
      removePoint: (id) => set((state) => ({ points: state.points.filter((p) => p.id !== id) })),
    }),
    { name: "sam" }
  )
);
