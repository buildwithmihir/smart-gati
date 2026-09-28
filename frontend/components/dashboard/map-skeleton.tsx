"use client";

/**
 * The map's loading placeholder.
 *
 * Not grey bars, and not a spinner over an empty canvas: a placeholder should
 * look like the thing that is arriving, and what is arriving here is a street
 * network. A stylised one drawn from the same muted ink as the real road layer
 * reads as "the map is coming" at a glance, and it holds the frame steady so
 * the basemap does not snap into existence against a blank slab.
 *
 * The geometry is invented — it is a texture, not data — so it is `aria-hidden`
 * and carries no labels. Nothing here is meant to be read.
 *
 * The fade is split between an animation and a transition, which is not
 * redundancy: a transition cannot fire on an element that is appearing, because
 * there is no previous value to move from, so the entrance is a keyframe. The
 * exit removes that class in the same commit that flips the opacity, handing
 * the property back to the transition — a class that filled `opacity` forwards
 * would otherwise outrank it and the exit would never run.
 */

import { cn } from "cn";

export default function MapSkeleton({ visible }: { visible: boolean }) {
  return (
    <div
      data-map-skeleton
      data-visible={String(visible)}
      aria-hidden
      className={cn(
        // Above the basemap and its controls, below the hero — the skeleton is
        // shown *because* the hero has gone.
        "absolute inset-0 z-20 overflow-hidden bg-canvas-sunken transition-opacity duration-500 ease-out",
        visible ? "qgati-fade-in opacity-100" : "pointer-events-none opacity-0",
      )}
    >
      <svg
        // `slice` gives the SVG the same cover behaviour the basemap has, so
        // the grid fills any viewport instead of letterboxing inside it.
        className="absolute inset-0 h-full w-full animate-pulse text-road opacity-60"
        viewBox="0 0 400 300"
        preserveAspectRatio="xMidYMid slice"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
      >
        {/* Arterials. */}
        <g strokeWidth="2.5">
          <path d="M0 62 H400" />
          <path d="M0 158 H400" />
          <path d="M0 252 H400" />
          <path d="M84 0 V300" />
          <path d="M214 0 V300" />
          <path d="M330 0 V300" />
        </g>

        {/* Two diagonals, because a pure grid reads as a table rather than a city. */}
        <g strokeWidth="2.5">
          <path d="M-10 288 L410 46" />
          <path d="M-10 118 L268 300" />
        </g>

        {/* The residential infill between them. */}
        <g strokeWidth="1.2" opacity="0.65">
          <path d="M0 110 H400" />
          <path d="M0 205 H400" />
          <path d="M40 0 V300" />
          <path d="M150 0 V300" />
          <path d="M272 0 V300" />
          <path d="M372 0 V300" />
          <path d="M0 24 H400" />
          <path d="M0 282 H400" />
        </g>
      </svg>
    </div>
  );
}
