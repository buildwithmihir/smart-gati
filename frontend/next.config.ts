import type { NextConfig } from "next";

// Where the FastAPI backend lives. The browser never calls this directly: it
// calls /gati-api/* on the same origin as the page, and Next forwards the
// request here. Same-origin requests are not subject to CORS, so the backend
// needs no allow-list entry for the deployed site.
const BACKEND_URL = (
  process.env.BACKEND_PROXY_TARGET ??
  "https://smart-gati-production.up.railway.app"
).replace(/\/+$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/gati-api/:path*",
        destination: `${BACKEND_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
