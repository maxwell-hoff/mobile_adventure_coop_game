let worldData = null;  // We'll store the YAML->JSON data here
let currentRegion = null;

// On page load, fetch data and render
window.addEventListener("DOMContentLoaded", async () => {
  await loadWorldData();
  renderWorldMap();
});

async function loadWorldData() {
  const response = await fetch("/api/map_data");
  if (!response.ok) {
    console.error("Error loading world data");
    return;
  }
  worldData = await response.json();
  // worldData.regions is now an array of region objects
}

function renderWorldMap() {
  const worldView = document.getElementById("world-view");
  const regionView = document.getElementById("region-view");
  const sectionView = document.getElementById("section-view");

  // Show world view, hide others
  worldView.style.display = "block";
  regionView.style.display = "none";
  sectionView.style.display = "none";

  // Clear any existing content
  worldView.innerHTML = "";

  // For each region, create a "hex-like" cell
  // We'll do a simplistic layout: just in a row or grid
  // Real hex layout requires more advanced math/positioning.
  const regions = worldData.regions;

  regions.forEach(region => {
    const cell = document.createElement("div");
    cell.classList.add("hex-cell");
    cell.textContent = region.name;

    // We could interpret region.size/width/height to scale cell
    // For now, let's just store them as data
    cell.dataset.regionName = region.name;

    // On click, we "zoom" into that region
    cell.addEventListener("click", () => {
      zoomIntoRegion(region);
    });

    worldView.appendChild(cell);
  });
}

// Called when user clicks on a region from the world map
function zoomIntoRegion(region) {
  currentRegion = region;

  const worldView = document.getElementById("world-view");
  const regionView = document.getElementById("region-view");
  const sectionView = document.getElementById("section-view");
  const regionButton = document.getElementById("regionButton");

  // Hide world map, show region map
  worldView.style.display = "none";
  regionView.style.display = "block";
  sectionView.style.display = "none";
  regionButton.style.display = "inline-block"; // show the "Back to Region" button for sections

  // Clear regionView
  regionView.innerHTML = "";

  // Render region info
  const title = document.createElement("h2");
  title.textContent = `Region: ${region.name}`;
  regionView.appendChild(title);

  const desc = document.createElement("p");
  desc.textContent = region.description;
  regionView.appendChild(desc);

  // If you have "sections" for each region, you'd do something similar:
  // region.sections.forEach(...)

  // For demonstration, let's create a grid of region's "width" x "height"
  // to represent "sections" or sub-areas
  if (!region.width) region.width = 1;  // fallback
  if (!region.height) region.height = 1;

  const regionGrid = document.createElement("div");
  regionGrid.style.display = "inline-block";
  regionGrid.style.width = "auto";
  
  for (let r = 0; r < region.height; r++) {
    const rowDiv = document.createElement("div");
    rowDiv.style.whiteSpace = "nowrap";

    for (let c = 0; c < region.width; c++) {
      const subCell = document.createElement("div");
      subCell.classList.add("hex-cell");
      subCell.style.backgroundColor = "#ffe8a8";

      subCell.textContent = `(${r},${c})`;
      subCell.addEventListener("click", () => {
        zoomIntoSection(region, r, c);
      });
      rowDiv.appendChild(subCell);
    }
    regionGrid.appendChild(rowDiv);
  }

  regionView.appendChild(regionGrid);
}

// Called when you click on a specific subCell in the region
function zoomIntoSection(region, row, col) {
  const worldView = document.getElementById("world-view");
  const regionView = document.getElementById("region-view");
  const sectionView = document.getElementById("section-view");

  // Hide region map, show section map
  worldView.style.display = "none";
  regionView.style.display = "none";
  sectionView.style.display = "block";

  sectionView.innerHTML = "";

  // Just a placeholder: in reality you'd retrieve data about the section
  const title = document.createElement("h2");
  title.textContent = `Section of ${region.name} at row=${row}, col=${col}`;
  sectionView.appendChild(title);

  const desc = document.createElement("p");
  desc.textContent = `Further details about sub-section could go here.`;
  sectionView.appendChild(desc);
}

// UI to go back to region-level from a section
function zoomOutToRegion() {
  const worldView = document.getElementById("world-view");
  const regionView = document.getElementById("region-view");
  const sectionView = document.getElementById("section-view");

  worldView.style.display = "none";
  regionView.style.display = "block";
  sectionView.style.display = "none";
}

// UI to go back to the world map
function zoomOutToWorld() {
  const worldView = document.getElementById("world-view");
  const regionView = document.getElementById("region-view");
  const sectionView = document.getElementById("section-view");
  const regionButton = document.getElementById("regionButton");

  worldView.style.display = "block";
  regionView.style.display = "none";
  sectionView.style.display = "none";
  regionButton.style.display = "none";
}
