let worldData = null;
let currentRegionId = null;
let currentSectionName = null;

// Hex geometry parameters
const HEX_SIZE = 20; // base radius for hex
const SQRT3 = Math.sqrt(3);

// On load, fetch data and draw
window.addEventListener("DOMContentLoaded", async () => {
  await loadWorldData();
  drawWorldHexes();
});

async function loadWorldData() {
  const resp = await fetch("/api/map_data");
  if (!resp.ok) {
    console.error("Error fetching map data");
    return;
  }
  worldData = await resp.json();
}

// Convert axial coords (q, r) to pixel (x, y) for SVG
// Reference: https://www.redblobgames.com/grids/hexagons/#coordinates-axial
function axialToPixel(q, r) {
  let x = HEX_SIZE * (3/2 * q);
  let y = HEX_SIZE * (SQRT3 * (r + q/2));
  return {x, y};
}

// Build the polygon points for a single hex (centered at x,y).
function hexPolygonPoints(cx, cy) {
  // 6 corners, each 60 degrees apart
  let points = [];
  for (let i = 0; i < 6; i++) {
    let angle = 2 * Math.PI * (i + 0.5) / 6; 
    // i+0.5 ensures flat-top orientation
    let px = cx + HEX_SIZE * Math.cos(angle);
    let py = cy + HEX_SIZE * Math.sin(angle);
    points.push(`${px},${py}`);
  }
  return points.join(" ");
}

// Draw all region hexes on the world map
function drawWorldHexes() {
  const svg = document.getElementById("map-svg");
  // Clear existing
  svg.innerHTML = "";

  // Optionally, place everything in a <g> for easier transforms
  let worldGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  worldGroup.setAttribute("id", "world-group");
  svg.appendChild(worldGroup);

  // For each region in worldData
  worldData.regions.forEach(region => {
    region.worldHexes.forEach(hex => {
      // Convert axial to pixel
      const {x, y} = axialToPixel(hex.q, hex.r);
      // Build a polygon
      let poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      poly.setAttribute("class", "hex-region");
      poly.setAttribute("fill", regionColor(region.regionId));
      poly.setAttribute("points", hexPolygonPoints(x, y));

      // On hover, highlight
      poly.addEventListener("mouseenter", () => {
        poly.classList.add("hex-hover");
      });
      poly.addEventListener("mouseleave", () => {
        poly.classList.remove("hex-hover");
      });

      // On click, “zoom” into region
      poly.addEventListener("click", () => {
        currentRegionId = region.regionId;
        drawRegionView(region);
      });

      worldGroup.appendChild(poly);
    });
  });
}

// Example coloring by region id
function regionColor(regionId) {
  // Simple deterministic color for demonstration
  const colorPalette = ["#cce5ff","#ffe5cc","#e5ffcc","#f5ccff","#fff5cc","#ccf0ff"];
  return colorPalette[regionId % colorPalette.length];
}

// Show the entire region in detail
function drawRegionView(region) {
  const svg = document.getElementById("map-svg");
  svg.innerHTML = "";

  // Create a group for region hexes
  let regionGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  regionGroup.setAttribute("id", "region-group");
  svg.appendChild(regionGroup);

  // We can apply a transform scale to “zoom in”
  // For demonstration, let’s do an arbitrary scale
  regionGroup.setAttribute("transform", "scale(2.5) translate(0, 0)");

  // Draw each hex belonging to this region
  region.worldHexes.forEach(hex => {
    const {x, y} = axialToPixel(hex.q, hex.r);
    let poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("class", "hex-region");
    poly.setAttribute("fill", regionColor(region.regionId));
    poly.setAttribute("points", hexPolygonPoints(x, y));

    // Click a region hex → go to “section” if it exists
    poly.addEventListener("click", () => {
      // We can check which section this hex belongs to
      let foundSection = region.sections.find(sec =>
        sec.sectionHexes.some(sh => sh.q === hex.q && sh.r === hex.r)
      );
      if (foundSection) {
        currentSectionName = foundSection.name;
        drawSectionView(region, foundSection);
      }
    });

    regionGroup.appendChild(poly);
  });
}

// Show a single section within the region
function drawSectionView(region, section) {
  const svg = document.getElementById("map-svg");
  svg.innerHTML = "";

  // Let’s do an even bigger scale for the section
  let sectionGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  sectionGroup.setAttribute("id", "section-group");
  svg.appendChild(sectionGroup);

  sectionGroup.setAttribute("transform", "scale(4.0) translate(0,0)");

  // Draw only the hexes that belong to the chosen section
  section.sectionHexes.forEach(hex => {
    const {x, y} = axialToPixel(hex.q, hex.r);
    let poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("class", "hex-region");
    poly.setAttribute("fill", regionColor(region.regionId));
    poly.setAttribute("points", hexPolygonPoints(x, y));
    sectionGroup.appendChild(poly);
  });
}

// UI Buttons
function showWorldView() {
  drawWorldHexes();
}

function showRegionView() {
  if (!currentRegionId) return;
  const region = worldData.regions.find(r => r.regionId === currentRegionId);
  drawRegionView(region);
}

function showSectionView() {
  if (!currentRegionId || !currentSectionName) return;
  const region = worldData.regions.find(r => r.regionId === currentRegionId);
  const section = region.sections.find(sec => sec.name === currentSectionName);
  drawSectionView(region, section);
}
