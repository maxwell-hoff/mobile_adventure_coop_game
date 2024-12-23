let worldData = null;
let currentView = "world"; // "world", "region", or "section"
let currentRegion = null;
let currentSection = null;

const HEX_SIZE = 30; // radius of each hex
const SQRT3 = Math.sqrt(3);

// We'll assume the <svg> is 800x600
const SVG_WIDTH = 800;
const SVG_HEIGHT = 600;

/**
 * Axial -> pixel (pointy-top).
 * x = size * sqrt(3) * (q + r/2)
 * y = size * 3/2 * r
 */
function axialToPixel(q, r) {
  const x = HEX_SIZE * SQRT3 * (q + r/2);
  const y = HEX_SIZE * (3 / 2) * r;
  return { x, y };
}

/**
 * Return 6-corner points for a pointy-top hex centered at (cx, cy).
 */
function hexPolygonPoints(cx, cy) {
  let points = [];
  for (let i = 0; i < 6; i++) {
    let angle_deg = 60 * i + 30; // offset 30° for pointy-top
    let angle_rad = Math.PI / 180 * angle_deg;
    let px = cx + HEX_SIZE * Math.cos(angle_rad);
    let py = cy + HEX_SIZE * Math.sin(angle_rad);
    points.push(`${px},${py}`);
  }
  return points.join(" ");
}

// =========== NEW UTILITY FUNCTIONS FOR BOUNDING BOX ===========

function getHexBoundingBox(hexList, axialToPixelFn) {
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;

  hexList.forEach(({q, r}) => {
    const {x, y} = axialToPixelFn(q, r);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  });
  if (hexList.length === 0) {
    minX = 0; maxX = 0; minY = 0; maxY = 0;
  }
  return {minX, maxX, minY, maxY};
}

function centerHexGroup(hexList, group, axialToPixelFn, {
  svgWidth = 800,
  svgHeight = 600,
  scale = 1,
  rotation = 0
} = {}) {
  const { minX, maxX, minY, maxY } = getHexBoundingBox(hexList, axialToPixelFn);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const targetX = svgWidth / 2;
  const targetY = svgHeight / 2;

  let transformStr = `
    translate(${targetX}, ${targetY})
    scale(${scale})
    rotate(${rotation})
    translate(${-centerX}, ${-centerY})
  `.replace(/\s+/g, ' ');
  group.setAttribute("transform", transformStr);
}

// ============================================================

window.addEventListener("DOMContentLoaded", async () => {
  await loadWorldData();
  drawWorldView();
});

async function loadWorldData() {
  const resp = await fetch("/api/map_data");
  if (!resp.ok) {
    console.error("Failed to load world data");
    return;
  }
  worldData = await resp.json();
}

/**
 * WORLD VIEW
 */
function drawWorldView() {
  currentView = "world";
  currentRegion = null;
  currentSection = null;

  // Hide the toggle button at world level
  const toggleBtn = document.getElementById("toggleZoomBtn");
  toggleBtn.style.display = "none";

  const svg = document.getElementById("map-svg");
  svg.innerHTML = ""; // clear existing

  // Re-add hover label (for region name on hover)
  const hoverLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  hoverLabel.setAttribute("id", "hoverLabel");
  hoverLabel.setAttribute("x", "400");
  hoverLabel.setAttribute("y", "30");
  hoverLabel.setAttribute("text-anchor", "middle");
  hoverLabel.setAttribute("font-size", "16");
  hoverLabel.setAttribute("fill", "#222");
  svg.appendChild(hoverLabel);

  // Create a group for the entire world
  let gWorld = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gWorld.setAttribute("id", "world-group");
  svg.appendChild(gWorld);

  // We'll store the entire list of hex coords for the bounding box
  let worldHexList = [];

  // For each region, draw hexes
  worldData.regions.forEach(region => {
    let sumX = 0, sumY = 0, count = 0;
    let regionCorners = [];

    region.worldHexes.forEach(hex => {
      worldHexList.push(hex);

      const { x, y } = axialToPixel(hex.q, hex.r);
      sumX += x; 
      sumY += y; 
      count++;

      // Gather corners for hull
      let corners = getHexCorners(hex.q, hex.r);
      regionCorners.push(...corners);

      // Draw the actual hex polygon
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      poly.setAttribute("class", "hex-region");
      poly.setAttribute("points", hexPolygonPoints(x, y));
      poly.setAttribute("fill", regionColor(region.regionId));

      // Hover: show region name
      poly.addEventListener("mouseenter", () => {
        hoverLabel.textContent = region.name;
      });
      poly.addEventListener("mouseleave", () => {
        hoverLabel.textContent = "";
      });

      // Click to zoom into region
      poly.addEventListener("click", () => {
        currentRegion = region;
        drawRegionView(region);
      });

      gWorld.appendChild(poly);
    });

    // Place a label near the center of that region
    if (count > 0) {
      const centerX = sumX / count;
      const centerY = sumY / count;

      const regionLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      regionLabel.setAttribute("x", centerX);
      regionLabel.setAttribute("y", centerY);
      regionLabel.setAttribute("text-anchor", "middle");
      regionLabel.setAttribute("fill", "#333");
      regionLabel.setAttribute("font-size", "14");
      regionLabel.textContent = region.name;

      gWorld.appendChild(regionLabel);
    }

    // Now compute the hull outline for the region
    const hull = computeConvexHull(regionCorners);
    const hullPoints = hull.map(pt => `${pt.x},${pt.y}`).join(" ");

    let outlinePoly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    outlinePoly.setAttribute("points", hullPoints);
    outlinePoly.setAttribute("fill", "none");
    outlinePoly.setAttribute("stroke", "#000");  // or "red" or ...
    outlinePoly.setAttribute("stroke-width", "2");
    outlinePoly.setAttribute("class", "region-outline");

    // IMPORTANT: Append the outline polygon
    gWorld.appendChild(outlinePoly);
  });

  // Now center the entire world group
  // rotation=30 if you want to keep that orientation
  centerHexGroup(worldHexList, gWorld, axialToPixel, {
    svgWidth: SVG_WIDTH,
    svgHeight: SVG_HEIGHT,
    scale: 1,
    // rotation: 30
  });
}

/**
 * REGION VIEW
 */
function drawRegionView(region) {
  currentView = "region";
  currentSection = null;

  const toggleBtn = document.getElementById("toggleZoomBtn");
  toggleBtn.style.display = "inline-block";
  toggleBtn.textContent = "World View"; // region -> world

  const svg = document.getElementById("map-svg");
  svg.innerHTML = "";

  // Hover label
  const hoverLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  hoverLabel.setAttribute("id", "hoverLabel");
  hoverLabel.setAttribute("x", "400");
  hoverLabel.setAttribute("y", "30");
  hoverLabel.setAttribute("text-anchor", "middle");
  hoverLabel.setAttribute("font-size", "16");
  hoverLabel.setAttribute("fill", "#222");
  svg.appendChild(hoverLabel);

  let gRegion = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gRegion.setAttribute("id", "region-group");
  svg.appendChild(gRegion);

  // We'll store the region's hexes in an array for bounding box
  let regionHexList = [];

  region.worldHexes.forEach(hex => {
    regionHexList.push(hex);

    const { x, y } = axialToPixel(hex.q, hex.r);

    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("class", "hex-region");
    poly.setAttribute("points", hexPolygonPoints(x, y));
    poly.setAttribute("fill", regionColor(region.regionId));

    // Hover label
    poly.addEventListener("mouseenter", () => {
      hoverLabel.textContent = region.name;
    });
    poly.addEventListener("mouseleave", () => {
      hoverLabel.textContent = "";
    });

    // Click -> detail view
    poly.addEventListener("click", () => {
      drawHexDetailView(region, hex);
    });

    gRegion.appendChild(poly);
  });

  // Now center the region's group
  centerHexGroup(regionHexList, gRegion, axialToPixel, {
    svgWidth: SVG_WIDTH,
    svgHeight: SVG_HEIGHT,
    scale: 2,
    // rotation: 30
  });
}

/**
 * NEW SECTION (detail) VIEW
 */
function drawHexDetailView(region, clickedHex) {
  currentView = "section";

  const toggleBtn = document.getElementById("toggleZoomBtn");
  toggleBtn.style.display = "inline-block";
  toggleBtn.textContent = "Region View";

  const svg = document.getElementById("map-svg");
  svg.innerHTML = "";

  // Hover label
  const hoverLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  hoverLabel.setAttribute("id", "hoverLabel");
  hoverLabel.setAttribute("x", "400");
  hoverLabel.setAttribute("y", "30");
  hoverLabel.setAttribute("text-anchor", "middle");
  hoverLabel.setAttribute("font-size", "16");
  hoverLabel.setAttribute("fill", "#222");
  svg.appendChild(hoverLabel);

  let gDetail = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gDetail.setAttribute("id", "hex-detail-group");
  svg.appendChild(gDetail);

  // Sub-hex geometry
  const SUB_GRID_RADIUS = 5; // example
  const SUB_HEX_SIZE = 10;
  function subAxialToPixel(q, r) {
    const x = SUB_HEX_SIZE * SQRT3 * (q + r / 2);
    const y = SUB_HEX_SIZE * (3 / 2) * r;
    return { x, y };
  }

  function subHexPolygonPoints(cx, cy) {
    let pts = [];
    for (let i = 0; i < 6; i++) {
      let angle_deg = 60 * i + 30;
      let angle_rad = Math.PI / 180 * angle_deg;
      let px = cx + SUB_HEX_SIZE * Math.cos(angle_rad);
      let py = cy + SUB_HEX_SIZE * Math.sin(angle_rad);
      pts.push(`${px},${py}`);
    }
    return pts.join(" ");
  }

  // Build a hex-shaped array of sub-hex coords
  let subHexList = [];
  for (let q = -SUB_GRID_RADIUS; q <= SUB_GRID_RADIUS; q++) {
    for (let r = -SUB_GRID_RADIUS; r <= SUB_GRID_RADIUS; r++) {
      if (Math.abs(q + r) <= SUB_GRID_RADIUS) {
        subHexList.push({q, r});
      }
    }
  }

  // Render each small sub-hex
  subHexList.forEach(subHex => {
    const {x, y} = subAxialToPixel(subHex.q, subHex.r);
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("class", "hex-region");
    poly.setAttribute("points", subHexPolygonPoints(x, y));
    poly.setAttribute("fill", regionColor(region.regionId));

    poly.addEventListener("mouseenter", () => {
      hoverLabel.textContent = `Sub-Hex (q=${subHex.q}, r=${subHex.r}) of ${region.name}`;
    });
    poly.addEventListener("mouseleave", () => {
      hoverLabel.textContent = "";
    });

    gDetail.appendChild(poly);
  });

  // Now center the sub-hex grid
  centerHexGroup(subHexList, gDetail, subAxialToPixel, {
    svgWidth: SVG_WIDTH,
    svgHeight: SVG_HEIGHT,
    scale: 2,
    rotation: 30
  });
}

/**
 * Toggle Zoom:
 * - region -> world
 * - section -> region
 */
function handleToggleZoom() {
  if (currentView === "region") {
    drawWorldView();
  } else if (currentView === "section") {
    drawRegionView(currentRegion);
  }
}

/** Simple color palette */
function regionColor(regionId) {
  const palette = [
    "#cce5ff", "#ffe5cc", "#e5ffcc",
    "#f5ccff", "#fff5cc", "#ccf0ff",
    "#e0cce5", "#eed5cc"
  ];
  return palette[regionId % palette.length];
}

/**
 * A small utility to compute the convex hull (Monotone chain).
 * pointsArr should be an array of objects: [{x, y}, {x, y}, ...]
 * Returns hull as an array of {x, y} in CCW order.
 */
function computeConvexHull(pointsArr) {
  // Sort by x, then by y
  let sorted = [...pointsArr].sort((a, b) =>
    a.x === b.x ? a.y - b.y : a.x - b.x
  );

  // Build lower hull
  let lower = [];
  for (let pt of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], pt) <= 0) {
      lower.pop();
    }
    lower.push(pt);
  }

  // Build upper hull
  let upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    let pt = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], pt) <= 0) {
      upper.pop();
    }
    upper.push(pt);
  }

  // Remove duplicates at the seam
  upper.pop();
  lower.pop();
  return lower.concat(upper);

  function cross(o, a, b) {
    // cross product of OA x OB
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  }
}

/**
 * Return 6 corners in pixel coords for a single hex.
 */
function getHexCorners(q, r) {
  // First get the pixel center
  const { x: cx, y: cy } = axialToPixel(q, r);
  // Then compute each corner
  let corners = [];
  for (let i = 0; i < 6; i++) {
    let angle_deg = 60 * i + 30;
    let angle_rad = Math.PI / 180 * angle_deg;
    let px = cx + HEX_SIZE * Math.cos(angle_rad);
    let py = cy + HEX_SIZE * Math.sin(angle_rad);
    corners.push({ x: px, y: py });
  }
  return corners;
}
