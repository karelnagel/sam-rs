import { useEffect, useRef, useState } from "react";
import { getRandomId, useStore } from "./store";
import { convertFileSrc } from "@tauri-apps/api/tauri";

export const Image = () => {
  const masks = useStore((state) => state.masks);
  const imageRef = useRef<HTMLImageElement>(null);
  const masksRef = useRef<HTMLCanvasElement>(null);
  const pointsRef = useRef<HTMLCanvasElement>(null);
  const addPoint = useStore((state) => state.addPoint);
  const points = useStore((state) => state.points);

  let image = useStore((state) => state.image);
  if (image) image = convertFileSrc(image);
  if (!image) return null;

  const [scale, setScale] = useState<number>();

  const handleClick = (event: any, label = 1) => {
    event.preventDefault();
    const canvas = pointsRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / scale!;
    const y = (event.clientY - rect.top) / scale!;

    addPoint({ id: getRandomId(), x, y, label });
  };
  useEffect(() => {
    const img = imageRef.current;
    const masks = masksRef.current;
    const points = pointsRef.current;
    if (!img || !masks || !points) return;
    img.onload = () => {
      setScale(img.width / img.naturalWidth);
      masks.height = img.height;
      masks.width = img.width;
      points.height = img.height;
      points.width = img.width;
    };
  }, [image]);

  useEffect(() => {
    const canvas = masksRef.current;
    if (!canvas) return;
    if (!scale) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const mask = masks?.[0];
    if (!mask) return;
    for (let row = 0; row < mask.length; row++) {
      for (let col = 0; col < mask[row].length; col++) {
        if (!mask[row][col]) {
          ctx.fillStyle = "rgba(100, 100, 0, 0.5)";
          ctx.fillRect(col * scale, row * scale, scale, scale);
        }
      }
    }
  }, [masks]);

  useEffect(() => {
    const canvas = pointsRef.current;
    if (!canvas) return;
    if (!scale) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x * scale, p.y * scale, 5, 0, 2 * Math.PI);
      ctx.fillStyle = p.label ? "red" : "blue";
      ctx.fill();
      ctx.closePath();
    });
  }, [points]);
  return (
    <div className=" shrink-0 col-span-2 w-full relative">
      <div className="w-full ">
        <img ref={imageRef} src={image} className="object-contain h-full w-full" />
        <canvas onContextMenu={(e) => handleClick(e, 0)} className="absolute top-0 left-0" ref={masksRef} onClick={handleClick} />
        <canvas onContextMenu={(e) => handleClick(e, 0)} className="absolute top-0 left-0" ref={pointsRef} onClick={handleClick} />
      </div>
    </div>
  );
};
