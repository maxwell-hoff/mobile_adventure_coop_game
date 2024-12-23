let worldData = null;
let currentView = "world"; // "world", "region", or "section"
let currentRegion = null;
let currentSection = null;

const HEX_SIZE = 30; // radius of each hex
const SQRT3 = Math.sqrt(3);

// Offsets to prevent top/left clipping on the world map
// Adjust as needed if you have many negative q/r coordinates or large maps
const WORLD_OFFSET_X = 100;
const WORLD_OFFSET_Y = 100;

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
 * WORLD VIEW:
 * - Draw all region hexes in a single group, translated so they aren't clipped.
 * - Also place a region label (text) at the centroid of that region's hexes.
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

  // Group for the entire world
  let gWorld = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gWorld.setAttribute("id", "world-group");
  // Translate so top-left hex is not clipped
  gWorld.setAttribute("transform", `translate(${WORLD_OFFSET_X}, ${WORLD_OFFSET_Y})`);
  svg.appendChild(gWorld);

  // For each region, we store coordinates to compute a centroid for labeling
  worldData.regions.forEach(region => {
    let sumX = 0, sumY = 0, count = 0;

    region.worldHexes.forEach(hex => {
      const { x, y } = axialToPixel(hex.q, hex.r);
      sumX += x;
      sumY += y;
      count++;

      // Draw the hex polygon
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

    // Place a label in the average center of the region's hexes
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

      // Optional: region label doesn't handle click, just a label
      // If you want it clickable, you could do so

      gWorld.appendChild(regionLabel);
    }
  });
}

/**
 * REGION VIEW: Zoom in on one region's hexes.
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

  // We'll offset similarly so nothing is cut off. 
  // Then scale up 2x or 3x, as you prefer.
  gRegion.setAttribute("transform", `translate(100,100) scale(2)`);

  svg.appendChild(gRegion);

  region.worldHexes.forEach(hex => {
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

    // Check if hex belongs to a section
    poly.addEventListener("click", () => {
        drawHexDetailView(region, hex);
    });

    gRegion.appendChild(poly);
  });
}

/**
 * Draw detail for a single hex, subdivided into smaller hexes.
 * @param {Object} region - the region data
 * @param {Object} clickedHex - the axial coords {q, r} for the hex that was clicked
 */
function drawHexDetailView(region, clickedHex) {
    currentView = "section";
  
    // Show toggle button, letting user go back to region
    const toggleBtn = document.getElementById("toggleZoomBtn");
    toggleBtn.style.display = "inline-block";
    toggleBtn.textContent = "Region View"; // so we can step back up
  
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
  
    // Create a group for the subdivided hex grid
    let gDetail = document.createElementNS("http://www.w3.org/2000/svg", "g");
    gDetail.setAttribute("id", "hex-detail-group");
  
    // Place it in the center, scale, then rotate 30° to match the region
    gDetail.setAttribute("transform", "translate(200,200) scale(2) rotate(30)");
  
    svg.appendChild(gDetail);
  
    // We'll define a sub-hex "radius" so it forms a hex shape
    const SUB_GRID_RADIUS = 5; // yields 19 sub-hexes
    const SUB_HEX_SIZE = 10;
    function subAxialToPixel(q, r) {
      const x = SUB_HEX_SIZE * SQRT3 * (q + r / 2);
      const y = SUB_HEX_SIZE * (3 / 2) * r;
      return { x, y };
    }
    function subHexPolygonPoints(cx, cy) {
      let points = [];
      for (let i = 0; i < 6; i++) {
        let angle_deg = 60 * i + 30;
        let angle_rad = Math.PI / 180 * angle_deg;
        let px = cx + SUB_HEX_SIZE * Math.cos(angle_rad);
        let py = cy + SUB_HEX_SIZE * Math.sin(angle_rad);
        points.push(`${px},${py}`);
      }
      return points.join(" ");
    }
  
    // Build sub-hex coords in a proper hex shape
    let subHexList = [];
    for (let subQ = -SUB_GRID_RADIUS; subQ <= SUB_GRID_RADIUS; subQ++) {
      for (let subR = -SUB_GRID_RADIUS; subR <= SUB_GRID_RADIUS; subR++) {
        if (Math.abs(subQ + subR) <= SUB_GRID_RADIUS) {
          subHexList.push({ q: subQ, r: subR });
        }
      }
    }
  
    // Render each small sub-hex
    subHexList.forEach(subHex => {
      const { x, y } = subAxialToPixel(subHex.q, subHex.r);
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      poly.setAttribute("class", "hex-region");
      poly.setAttribute("points", subHexPolygonPoints(x, y));
      poly.setAttribute("fill", regionColor(region.regionId));
  
      // On hover, show sub-hex coords
      poly.addEventListener("mouseenter", () => {
        hoverLabel.textContent = `Sub-Hex (q=${subHex.q}, r=${subHex.r}) of ${region.name}`;
      });
      poly.addEventListener("mouseleave", () => {
        hoverLabel.textContent = "";
      });
  
      gDetail.appendChild(poly);
    });
  }
  
  
/**
 * Single toggle button:
 * - If region -> world
 * - If section -> region
 */
function handleToggleZoom() {
  if (currentView === "region") {
    drawWorldView();
  } else if (currentView === "section") {
    drawRegionView(currentRegion);
  }
}

/** Different fill colors per region ID. */
function regionColor(regionId) {
  const palette = [
    "#cce5ff", "#ffe5cc", "#e5ffcc",
    "#f5ccff", "#fff5cc", "#ccf0ff",
    "#e0cce5", "#eed5cc" // you can add more to vary
  ];
  return palette[regionId % palette.length];
}
