"use client";

/**
 * The ambient loop behind the empty state, and the scrim that keeps the copy on
 * top of it readable.
 *
 * Three things here are deliberate rather than incidental:
 *
 * 1. **The gradient underneath the video is not a placeholder.** It is what the
 *    screen is made of when the loop is missing, still loading, or suppressed
 *    by a reduced-motion preference. The empty state has to read as finished in
 *    all of those cases, so the video is enhancement over a surface that
 *    already works — never the surface itself.
 * 2. **Playback is driven from an effect, not the `autoPlay` attribute.** An
 *    attribute cannot be conditional, and the loop must not start at all for
 *    someone who has asked their OS for less motion. CSS could hide the
 *    element; only this stops it decoding.
 * 3. **The fade-out is why this stays mounted after `visible` goes false.** The
 *    caller unmounts it once a solution exists; until then it sits at zero
 *    opacity, `inert`, so the crossfade to the map has something to cross from.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import { cn } from "cn";

/**
 * Where the loop is expected. Both files are optional — see the note in
 * `frontend/README.md`. MP4/H.264 is the only format asked for: it is the one
 * every browser this will be demoed in can decode without a fallback ladder.
 */
const HERO_VIDEO_SRC = "/hero.mp4";
const HERO_POSTER_SRC = "/hero-poster.jpg";

type HeroBackdropProps = {
  /** False once loading starts, which fades the whole backdrop out. */
  visible: boolean;
  children: ReactNode;
};

export default function HeroBackdrop({ visible, children }: HeroBackdropProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoUnavailable, setVideoUnavailable] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => {
      // A pause here is also what stops a 1080p loop burning GPU behind a
      // skeleton that is already covering it.
      if (reducedMotion.matches || !visible) {
        video.pause();
      } else {
        void video.play().catch(() => {
          // Autoplay refused — the gradient underneath is the fallback and it
          // is already on screen, so there is nothing to recover from.
        });
      }
    };

    sync();
    reducedMotion.addEventListener("change", sync);
    return () => reducedMotion.removeEventListener("change", sync);
  }, [visible]);

  return (
    <div
      data-hero-visible={String(visible)}
      // `inert` rather than a bare `aria-hidden`: it takes the button out of
      // the tab order and out of pointer reach in one attribute. An `aria-hidden`
      // wrapper with a focusable button still inside it is a trap.
      inert={!visible}
      className={cn(
        // `z-30` clears the basemap's own controls, which MapLibre positions
        // absolutely inside the map container and which would otherwise float
        // over the hero — an attribution credit hovering on top of a graphic
        // that is not the map.
        "absolute inset-0 z-30 overflow-hidden transition-opacity duration-500 ease-out",
        visible ? "opacity-100" : "pointer-events-none opacity-0",
      )}
    >
      {/* The surface the video plays over, and the whole surface when it cannot. */}
      <div className="absolute inset-0 bg-linear-to-br from-ink via-[#16233d] to-[#2b4166]" />

      {videoUnavailable ? null : (
        <video
          ref={videoRef}
          className="qgati-hero-video"
          src={HERO_VIDEO_SRC}
          poster={HERO_POSTER_SRC}
          muted
          loop
          playsInline
          preload="metadata"
          aria-hidden
          onError={() => setVideoUnavailable(true)}
        />
      )}

      {/* The dark overlay the copy sits on. A flat scrim for an everywhere floor
          on contrast, then a vignette so the frame's brightest region — usually
          the middle, where the card lands — is dimmed hardest. */}
      <div className="absolute inset-0 bg-ink/55" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(11,18,32,0.75),rgba(11,18,32,0)_70%)]" />

      <div className="relative flex h-full items-center justify-center p-6">{children}</div>
    </div>
  );
}
