import { describe, it, expect } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MantineProvider } from '@mantine/core';
import { VibeCanvas } from '../components/VibeCanvas';
import type { Hitbox } from '../types/vibe';

const mockHitboxes: Hitbox[] = [
  {
    x: 100,
    y: 100,
    w: 50,
    h: 50,
    category: 'emergencies',
    tag: 'fire',
    articles: [
      {
        title: 'Test Article',
        url: 'https://example.com',
        source: 'Test Source',
      },
    ],
  },
];

describe('VibeCanvas', () => {
  it('renders the image', () => {
    render(
      <MantineProvider>
        <VibeCanvas
          imageUrl="https://example.com/image.jpg"
          hitboxes={mockHitboxes}
          alt="Test Image"
        />
      </MantineProvider>
    );

    const image = screen.getByAltText('Test Image');
    expect(image).toBeInTheDocument();
    expect(image).toHaveAttribute('src', 'https://example.com/image.jpg');
  });

  it('renders hitbox overlays after image loads', async () => {
    const { container } = render(
      <MantineProvider>
        <VibeCanvas
          imageUrl="https://example.com/image.jpg"
          hitboxes={mockHitboxes}
        />
      </MantineProvider>
    );

    const image = container.querySelector('img');
    expect(image).toBeInTheDocument();

    if (image) {
      // Mock offsetWidth/offsetHeight before firing load event
      Object.defineProperty(image, 'offsetWidth', { 
        value: 800,
        configurable: true 
      });
      Object.defineProperty(image, 'offsetHeight', { 
        value: 600,
        configurable: true 
      });
      
      // Trigger load event
      image.dispatchEvent(new Event('load'));
    }

    // Wait for hitboxes to render
    await waitFor(() => {
      const hitboxElements = container.querySelectorAll('[title]');
      expect(hitboxElements.length).toBeGreaterThan(0);
    });
  });
});
