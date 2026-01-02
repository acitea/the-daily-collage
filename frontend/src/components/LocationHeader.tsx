import { Button, Group, ScrollArea } from '@mantine/core';
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
    <ScrollArea>
      <Group gap="xs" wrap="nowrap">
        {locations.map((location) => (
          <Button
            key={location.id}
            variant={selectedLocation === location.id ? 'filled' : 'light'}
            onClick={() => onLocationChange(location.id)}
            className="flex-shrink-0"
            styles={{
              root: {
                backgroundColor: selectedLocation === location.id ? '#000000' : 'transparent',
                borderColor: selectedLocation === location.id ? '#000000' : '#d1d5db',
                color: selectedLocation === location.id ? '#ffffff' : '#1f2937',
              },
            }}
          >
            {location.name}
          </Button>
        ))}
      </Group>
    </ScrollArea>
  );
};
