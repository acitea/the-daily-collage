import { useState, useRef } from 'react';
import type { Hitbox } from '../types/vibe';

const CATEGORY_COLORS: Record<string, string> = {
  emergencies: '#fca5a5',
  crime: '#fbb6ce',
  festivals: '#a7f3d0',
  transportation: '#93c5fd',
  weather_temp: '#fed7aa',
  weather_wet: '#bae6fd',
  sports: '#c4b5fd',
  economics: '#fde68a',
  politics: '#fbcfe8',
};

interface VibeCanvasProps {
  imageUrl: string;
  hitboxes: Hitbox[];
  alt?: string;
  selectedCategories: Set<string>;
  onSelectCategory: (category: string) => void;
}

export const VibeCanvas = ({ imageUrl, hitboxes, alt = 'City Vibe', selectedCategories, onSelectCategory }: VibeCanvasProps) => {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [hoveredCategory, setHoveredCategory] = useState<string | null>(null);
  const headlinesRef = useRef<HTMLDivElement>(null);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImageSize({ width: img.offsetWidth, height: img.offsetHeight });
  };

  const getScaledHitbox = (hitbox: Hitbox) => {
    if (!imageSize.width || !imageSize.height) return hitbox;
    
    // Assuming hitboxes are in absolute pixel coordinates from the original image
    // If they're normalized (0-1), remove the scaling
    return {
      ...hitbox,
      x: hitbox.x,
      y: hitbox.y,
      w: hitbox.w,
      h: hitbox.h,
    };
  };

  return (
    <div className="relative w-full">
      <div className="relative w-full inline-block">
        {/* Base image with grayscale when hovering/selecting */}
        <img
          src={imageUrl}
          alt={alt}
          className="w-full h-auto block"
          onLoad={handleImageLoad}
          style={{
            filter: hoveredCategory || selectedCategories.size > 0 ? 'grayscale(100%)' : 'grayscale(0%)',
            transition: 'filter 200ms ease-out',
          }}
        />

        {/* Colored overlays for selected/hovered areas - show full color */}
        {(hoveredCategory || selectedCategories.size > 0) && (
          <img
            src={imageUrl}
            alt={alt}
            className="absolute top-0 left-0 w-full h-auto pointer-events-none"
            style={{
              clipPath: getClipPath(hoveredCategory, selectedCategories, hitboxes, imageSize),
              transition: 'clip-path 200ms ease-out',
            }}
          />
        )}
        
        {imageSize.width > 0 && hitboxes.map((hitbox, idx) => {
          const scaled = getScaledHitbox(hitbox);
          const isHovered = hoveredCategory === hitbox.category;
          const isSelected = selectedCategories.has(hitbox.category);
          const isRelevant = isHovered || isSelected;
          const categoryColor = CATEGORY_COLORS[hitbox.category] || '#e5e7eb';
          
          return (
            <div
              key={idx}
              className="absolute group"
              style={{
                left: `${scaled.x}px`,
                top: `${scaled.y}px`,
                width: `${scaled.w}px`,
                height: `${scaled.h}px`,
              }}
              onMouseEnter={() => setHoveredCategory(hitbox.category)}
              onMouseLeave={() => setHoveredCategory(null)}
              onClick={() => {
                onSelectCategory(hitbox.category);
                // Smooth scroll to headlines
                setTimeout(() => {
                  headlinesRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
              }}
            >
              {/* Highlight overlay with category color */}
              <div
                className="absolute inset-0 cursor-pointer border-2 transition-all duration-200"
                style={{
                  borderColor: isRelevant ? categoryColor : 'transparent',
                  backgroundColor: isRelevant ? categoryColor + '33' : 'transparent',
                  borderStyle: 'solid',
                }}
              />
              
              {/* Category label on hover */}
              {isHovered && (
                <div
                  className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 text-white px-3 py-1 rounded text-sm font-semibold whitespace-nowrap z-10 pointer-events-none"
                  style={{ backgroundColor: categoryColor }}
                >
                  {hitbox.category}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Ref for scrolling to headlines */}
      <div ref={headlinesRef} style={{ pointerEvents: 'none' }} />
    </div>
  );
}

// Helper function to generate clip-path for multiple selected categories
function getClipPath(hoveredCategory: string | null, selectedCategories: Set<string>, hitboxes: Hitbox[], imageSize: { width: number; height: number }): string {
  if (!imageSize.width || !imageSize.height) return 'none';
  
  const categoriesToShow = hoveredCategory ? new Set([hoveredCategory]) : selectedCategories;
  if (categoriesToShow.size === 0) return 'none';
  
  const relevantHitboxes = hitboxes.filter((h) => categoriesToShow.has(h.category));
  if (relevantHitboxes.length === 0) return 'none';
  
  // Create polygon points for all relevant hitboxes
  const points = relevantHitboxes
    .map((h) => {
      const xPercent = (h.x / imageSize.width) * 100;
      const yPercent = (h.y / imageSize.height) * 100;
      const wPercent = (h.w / imageSize.width) * 100;
      const hPercent = (h.h / imageSize.height) * 100;
      
      return `${xPercent.toFixed(2)}% ${yPercent.toFixed(2)}%, ${(xPercent + wPercent).toFixed(2)}% ${yPercent.toFixed(2)}%, ${(xPercent + wPercent).toFixed(2)}% ${(yPercent + hPercent).toFixed(2)}%, ${xPercent.toFixed(2)}% ${(yPercent + hPercent).toFixed(2)}%`;
    })
    .join(', ');
  
  return `polygon(${points})`;
}
