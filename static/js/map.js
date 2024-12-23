let worldData = null;
let currentView = "world"; // "world", "region", or "section"
let currentRegion = null;
let currentSection = null;

const HEX_SIZE = 30;       // size of each hex's radius
const SQRT3 = Math.sqrt(3);

/**
 * Axial to pixel (pointy-top hex layout).
 * Ref: https://www.redblobgames.com/grids/hexagons/#coordinates-axial
 * For a pointy-top orientation:
 *   x = size * sqrt(3) * (q + r/2)
 *   y = size * 3/2 * r
 */
function axialToPixel(q, r) {
  const x = HEX_SIZE * SQRT3 * (q + r/2);
  const y = HEX_SIZE * (3 / 2) * r;
  return { x, y };
}

/**
 * Build a polygon "points" string for a pointy-top hex centered at (cx, cy).
 * The corner angle for i-th corner: 60° * i + 30° offset = 2π*(i/6) + π/6
 */
function hexPolygonPoints(cx, cy) {
  let points = [];
  for (let i = 0; i < 6; i++) {
    let angle_deg = 60 * i + 30;
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
 * Draw the world map: all regions, each with their hexes, at a uniform scale.
 */
function drawWorldView() {
  currentView = "world";
  currentRegion = null;
  currentSection = null;

  // Hide or show toggle button
  const toggleBtn = document.getElementById("toggleZoomBtn");
  toggleBtn.style.display = "none"; // no button at world level

  const svg = document.getElementById("map-svg");
  svg.innerHTML = ""; // clear existing

  // Add hoverLabel text again (it was cleared)
  const hoverLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  hoverLabel.setAttribute("id", "hoverLabel");
  hoverLabel.setAttribute("x", "400");
  hoverLabel.setAttribute("y", "30");
  hoverLabel.setAttribute("text-anchor", "middle");
  hoverLabel.setAttribute("font-size", "16");
  hoverLabel.setAttribute("fill", "#222");
  svg.appendChild(hoverLabel);

  // Place all region hexes in one group
  let gWorld = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gWorld.setAttribute("id", "world-group");
  svg.appendChild(gWorld);

  worldData.regions.forEach(region => {
    region.worldHexes.forEach(hex => {
      const { x, y } = axialToPixel(hex.q, hex.r);
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      poly.setAttribute("class", "hex-region");
      poly.setAttribute("points", hexPolygonPoints(x, y));
      poly.setAttribute("fill", regionColor(region.regionId));

      // Hover: show region name in #hoverLabel text
      poly.addEventListener("mouseenter", () => {
        hoverLabel.textContent = region.name;
      });
      poly.addEventListener("mouseleave", () => {
        hoverLabel.textContent = "";
      });

      // Click: zoom into region
      poly.addEventListener("click", () => {
        currentRegion = region;
        drawRegionView(region);
      });

      gWorld.appendChild(poly);
    });
  });
}

/**
 * Draw a single region in detail (zoomed in).
 * We apply an SVG transform (scale) or just draw in the same coordinate space but bigger.
 */
function drawRegionView(region) {
  currentView = "region";
  currentSection = null;

  const toggleBtn = document.getElementById("toggleZoomBtn");
  toggleBtn.style.display = "inline-block";
  toggleBtn.textContent = "World View"; // because from region → world

  const svg = document.getElementById("map-svg");
  svg.innerHTML = "";

  // Re-add the hover label
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

  // For a bigger scale (e.g., 2x)
  gRegion.setAttribute("transform", "scale(2) translate(50,50)");

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

    // If the hex belongs to a section, clicking might zoom further
    poly.addEventListener("click", () => {
      let foundSection = region.sections.find(sec =>
        sec.sectionHexes.some(sHex => sHex.q === hex.q && sHex.r === hex.r)
      );
      if (foundSection) {
        currentSection = foundSection;
        drawSectionView(region, foundSection);
      }
    });

    gRegion.appendChild(poly);
  });
}

/**
 * Draw a single section (subset of region) in even more detail.
 */
function drawSectionView(region, section) {
  currentView = "section";

  const toggleBtn = document.getElementById("toggleZoomBtn");
  toggleBtn.style.display = "inline-block";
  toggleBtn.textContent = "Region View"; // because from section → region

  const svg = document.getElementById("map-svg");
  svg.innerHTML = "";

  // Re-add hover label
  const hoverLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  hoverLabel.setAttribute("id", "hoverLabel");
  hoverLabel.setAttribute("x", "400");
  hoverLabel.setAttribute("y", "30");
  hoverLabel.setAttribute("text-anchor", "middle");
  hoverLabel.setAttribute("font-size", "16");
  hoverLabel.setAttribute("fill", "#222");
  svg.appendChild(hoverLabel);

  let gSection = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gSection.setAttribute("id", "section-group");
  // Increase scale further, e.g. 3x
  gSection.setAttribute("transform", "scale(3) translate(80,80)");
  svg.appendChild(gSection);

  section.sectionHexes.forEach(hex => {
    const { x, y } = axialToPixel(hex.q, hex.r);
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("class", "hex-region");
    poly.setAttribute("points", hexPolygonPoints(x, y));
    poly.setAttribute("fill", regionColor(region.regionId));

    // Hover label
    poly.addEventListener("mouseenter", () => {
      hoverLabel.textContent = section.name + " (" + region.name + ")";
    });
    poly.addEventListener("mouseleave", () => {
      hoverLabel.textContent = "";
    });

    gSection.appendChild(poly);
  });
}

/**
 * Single button to handle toggling:
 * - If currentView == "region", clicking goes to drawWorldView()
 * - If currentView == "section", clicking goes to drawRegionView(region)
 */
function handleToggleZoom() {
  if (currentView === "region") {
    // Go back to world
    drawWorldView();
  } else if (currentView === "section") {
    // Go back to region
    drawRegionView(currentRegion);
  }
}

/**
 * Simple deterministic color palette for region IDs
 */
function regionColor(regionId) {
  const palette = ["#cce5ff","#ffe5cc","#e5ffcc","#f5ccff","#fff5cc","#ccf0ff"];
  return palette[regionId % palette.length];
}
