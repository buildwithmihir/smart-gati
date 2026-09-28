# Q-Gati frontend

The dashboard for Q-Gati: a scenario's stop list on the left, the solved routes
drawn over the real Delhi road network in the middle, and the solver's own
figures on the right.

## Running it

The dashboard is a client for the FastAPI backend and shows an error card
without it, so start the backend first:

```bash
cd ../backend
uv run uvicorn qgati.api.main:app --reload      # http://localhost:8000
```

Then:

```bash
npm run dev                                      # http://localhost:3000
```

`NEXT_PUBLIC_API_URL` overrides the backend address; see `.env.example`. It is
read at build time, so a change needs a restart.

## The hero video

The empty state — everything before the first solve — plays a muted ambient loop
behind the "Load Sample Scenario" card. **Drop the file at:**

```
frontend/public/hero.mp4
```

Optionally add a poster, shown until the first frame decodes:

```
frontend/public/hero-poster.jpg
```

Both files are optional. When `hero.mp4` is missing the request 404s, the
`<video>` is dropped, and a deep-navy gradient stands in its place — that
gradient is the surface the video plays *over*, not a placeholder for it, so the
empty state reads as finished either way.

### What to encode

| | |
|---|---|
| Container / codec | MP4, H.264 (AVC). Nothing else is required — this is the one codec every browser it will be demoed in can decode without a fallback ladder. |
| Pixel format | `yuv420p`. Anything else renders green or black in some browsers. |
| Audio | None needed — playback is muted regardless. Dropping the track is free file size. |
| Duration | 8–20 s. It loops, so it should loop cleanly: first and last frames matching hides the seam. |
| Size | Under ~8 MB, ideally ~4 MB. It is decorative and loads on first paint. |
| Faststart | Move the MP4 moov atom to the front (`-movflags +faststart`) so it starts before it has fully downloaded. |

```
ffmpeg -i source.mov -an -c:v libx264 -pix_fmt yuv420p -crf 23 -movflags +faststart public/hero.mp4
```

### Compose it for a near-square window

The loop is drawn with `object-fit: cover`, so it fills the frame and crops
rather than letterboxing. At the 1366×768 this is most likely demoed at, the map
area is **694 × 668** — nearly square, and much squarer than 16:9. A 1920×1080
source therefore loses roughly **40% of its width**, about a fifth off each
side.

Keep whatever matters near the centre and assume the outer thirds may not
survive. A route that enters and leaves at the edges of a 16:9 frame will be
cropped mid-street on a laptop.

[`prefers-reduced-motion`](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion)
is honoured: the video is never started and is hidden outright, leaving the
gradient. Playback is started from an effect rather than the `autoPlay`
attribute, because an attribute cannot be conditional.

Once a scenario is solved the video is torn down — the real MapLibre map takes
the space and the loop never returns.

## The MapLibre worker in `public/`

`public/maplibre/` holds two files copied verbatim out of `maplibre-gl/dist`:

```
maplibre-gl-worker.mjs
maplibre-gl-shared.mjs
```

MapLibre parses every source — basemap tiles and our GeoJSON alike — inside a
web worker, and works out that worker's URL from its own `import.meta.url` plus
a sibling filename. Under Turbopack that resolves to a path that is not an
emitted asset, so the request comes back as the HTML shell, the module worker is
rejected on its MIME type, and the worker never starts.

The failure is quiet and looks like a styling bug rather than a loading one: the
style's background layer still paints, the DOM markers still appear, and every
vector source silently stays empty. No roads, no routes, and none of the
basemap's own tiles. `route-map.tsx` calls `setWorkerUrl()` to point at these
copies, and that is what makes the map draw at all.

**These files must match the installed `maplibre-gl` version.** They are a copy
of a dependency rather than a dependency, so bumping maplibre-gl without
recopying them can desynchronise the main thread and the worker protocol in ways
that surface as strange tile behaviour rather than a clean error. After any
maplibre-gl upgrade:

```bash
cp node_modules/maplibre-gl/dist/maplibre-gl-worker.mjs \
   node_modules/maplibre-gl/dist/maplibre-gl-shared.mjs \
   public/maplibre/
```

## Layout

Four regions, all reachable without a scrollbar at 1366×768:

- `components/dashboard/route-map.tsx` — the MapLibre canvas. Redraws cross-fade
  rather than swapping; see the note at the top of the file, and `lib/motion.ts`
  for the timings the map and the dashboard agree on.
- `components/dashboard/stops-panel.tsx` — the stop list, grouped by vehicle.
- `components/dashboard/route-summary.tsx` — every figure read from the optimize
  response, including the solver name.
- `components/dashboard/hero-backdrop.tsx` — the loop and its scrim.

There is no dark theme. `globals.css` carries shadcn's `.dark` block, but nothing
ever applies the class.
