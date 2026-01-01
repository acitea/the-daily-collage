import { Button, Group, ScrollArea } from '@mantine/core';
import { IconMapPin } from '@tabler/icons-react';
import type { Location } from '../types/vibe';

interface LocationHeaderProps {
  locations: Location[];
  selectedLocation: string;
  onLocationChange: (locationId: string) => void;
}

export const LocationHeader = ({
  locations,
  selectedLocation,
  onLocationChange,
}: LocationHeaderProps) => {
  return (
    <div className="border-b border-gray-200 bg-white shadow-sm">
      <ScrollArea className="px-4 py-3">
        <Group gap="xs" wrap="nowrap">
          {locations.map((location) => (
            <Button
              key={location.id}
              variant={selectedLocation === location.id ? 'filled' : 'light'}
              leftSection={<IconMapPin size={16} />}
              onClick={() => onLocationChange(location.id)}
              className="flex-shrink-0"
            >
              {location.name}
            </Button>
          ))}
        </Group>
      </ScrollArea>
    </div>
  );
};
