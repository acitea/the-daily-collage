import { useState } from 'react';
import { Modal, Text, Anchor, Stack, Badge, Group } from '@mantine/core';
import { IconExternalLink } from '@tabler/icons-react';
import type { Hitbox } from '../types/vibe';

interface VibeCanvasProps {
  imageUrl: string;
  hitboxes: Hitbox[];
  alt?: string;
}

export const VibeCanvas = ({ imageUrl, hitboxes, alt = 'City Vibe' }: VibeCanvasProps) => {
  const [selectedHitbox, setSelectedHitbox] = useState<Hitbox | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

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
      <img
        src={imageUrl}
        alt={alt}
        className="w-full h-auto block"
        onLoad={handleImageLoad}
      />
      
      {imageSize.width > 0 && hitboxes.map((hitbox, idx) => {
        const scaled = getScaledHitbox(hitbox);
        return (
          <div
            key={idx}
            className="absolute cursor-pointer hover:bg-blue-500/20 transition-colors border-2 border-transparent hover:border-blue-500"
            style={{
              left: `${scaled.x}px`,
              top: `${scaled.y}px`,
              width: `${scaled.w}px`,
              height: `${scaled.h}px`,
            }}
            onClick={() => setSelectedHitbox(hitbox)}
            title={`${hitbox.category}: ${hitbox.tag}`}
          />
        );
      })}

      <Modal
        opened={selectedHitbox !== null}
        onClose={() => setSelectedHitbox(null)}
        title={
          <Group>
            <Badge color="blue" size="lg">
              {selectedHitbox?.category}
            </Badge>
            <Text fw={600}>{selectedHitbox?.tag}</Text>
          </Group>
        }
        size="lg"
      >
        {selectedHitbox && (
          <Stack gap="md">
            {selectedHitbox.articles.length > 0 ? (
              selectedHitbox.articles.map((article, idx) => (
                <div key={idx} className="p-3 border border-gray-200 rounded">
                  <Anchor
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-2"
                  >
                    <Text size="sm" fw={500} className="flex-1">
                      {article.title}
                    </Text>
                    <IconExternalLink size={16} className="mt-1 flex-shrink-0" />
                  </Anchor>
                  <Text size="xs" c="dimmed" mt="xs">
                    {article.source}
                    {article.published_at && ` • ${new Date(article.published_at).toLocaleDateString()}`}
                  </Text>
                </div>
              ))
            ) : (
              <Text c="dimmed">No articles available for this element.</Text>
            )}
          </Stack>
        )}
      </Modal>
    </div>
  );
};
