let worldData = null;
let piecesData = null;
let characterData = null;
let currentView = "world"; // "world", "region", or "section"
let currentRegion = null;
let currentSection = null;
let isSelectingHex = false;
let currentHexSelector = null;
let pieceSelections = new Map(); // Make this global
let puzzleScenario = null;
let battleLog = [];
let blockedHexes = new Set(); 
let delayedAttacks = []; 
let turnCounter = 0; 

const HEX_SIZE = 3; // radius of each hex
// For world-only zoom
let worldZoom = 0.25;       // default scale (max "zoom in")
const MIN_WORLD_ZOOM = 0.1; // how far out the user can zoom
const MAX_WORLD_ZOOM = 1;   // do not allow zoom in beyond scale 1

// For region-level zoom
let regionZoom = 1;
const MIN_REGION_ZOOM = 0.5;
const MAX_REGION_ZOOM = 5;

// Panning offsets
let worldPanX = 0, worldPanY = 0;
let regionPanX = 0, regionPanY = 0;

const SQRT3 = Math.sqrt(3);

// We'll assume the <svg> is 800x600
const SVG_WIDTH = 800;
const SVG_HEIGHT = 600;

// ============= Basic Hex Conversions =============

/** Axial -> pixel (pointy-top). */
function axialToPixel(q, r) {
  const x = HEX_SIZE * SQRT3 * (q + r / 2);
  const y = HEX_SIZE * (3 / 2) * r;
  return { x, y };
}

/** Return a polygon "points" string for a single hex. */
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

/** Return the 6 corners (x,y) for a single hex center, for path-building. */
function getHexCorners(cx, cy) {
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

/**
 * Build a single SVG path string that traces all the hexes in 'regionHexes'
 * as sub-paths (M...L...Z).
 */
function buildRegionPath(regionHexes) {
  let pathStr = "";
  for (let hex of regionHexes) {
    let { x, y } = axialToPixel(hex.q, hex.r);
    let corners = getHexCorners(x, y);
    pathStr += `M ${corners[0].x},${corners[0].y} `;
    for (let i = 1; i < 6; i++) {
      pathStr += `L ${corners[i].x},${corners[i].y} `;
    }
    pathStr += "Z ";
  }
  return pathStr;
}

// ============= Mouse Drag Panning =============
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;

function onMouseDown(evt) {
  // Only allow dragging in World or Region view
  if (currentView === "world" || currentView === "region") {
    isDragging = true;
    lastMouseX = evt.clientX;
    lastMouseY = evt.clientY;
  }
}

function onMouseMove(evt) {
  if (!isDragging) return;
  const dx = evt.clientX - lastMouseX;
  const dy = evt.clientY - lastMouseY;
  lastMouseX = evt.clientX;
  lastMouseY = evt.clientY;

  if (currentView === "world") {
    worldPanX += dx;
    worldPanY += dy;
    drawWorldView();
  } else if (currentView === "region") {
    regionPanX += dx;
    regionPanY += dy;
    drawRegionView(currentRegion);
  }
}

function onMouseUp(evt) {
  isDragging = false;
}

// ============= Edges & Neighbors (Region Outlines) =============

/** Return an array of edges for hex (q,r). Each edge => [ [x1,y1],[x2,y2] ] */
function getHexEdges(q, r) {
  const center = axialToPixel(q, r);
  let edges = [];
  let corners = [];
  for (let i = 0; i < 6; i++) {
    let angle_deg = 60 * i + 30;
    let rad = Math.PI / 180 * angle_deg;
    let px = center.x + HEX_SIZE * Math.cos(rad);
    let py = center.y + HEX_SIZE * Math.sin(rad);
    corners.push({ x: px, y: py });
  }
  for (let i = 0; i < 6; i++) {
    let c1 = corners[i];
    let c2 = corners[(i + 1) % 6];
    edges.push([[c1.x, c1.y], [c2.x, c2.y]]);
  }
  return edges;
}

/** Axial neighbors (q +/- 1, r +/- 1, etc.). */
function getHexNeighbors(q, r) {
  return [
    { q: q + 1, r: r },
    { q: q - 1, r: r },
    { q: q, r: r + 1 },
    { q: q, r: r - 1 },
    { q: q + 1, r: r - 1 },
    { q: q - 1, r: r + 1 }
  ];
}

/** Used to detect region boundary lines (whether an edge is shared). */
function isEdgeShared(edge, neighborEdges) {
  let [A, B] = edge;
  for (let nEdge of neighborEdges) {
    let [C, D] = nEdge;
    if (almostEqual(A[0], C[0]) && almostEqual(A[1], C[1]) &&
        almostEqual(B[0], D[0]) && almostEqual(B[1], D[1])) {
      return true;
    }
    if (almostEqual(A[0], D[0]) && almostEqual(A[1], D[1]) &&
        almostEqual(B[0], C[0]) && almostEqual(B[1], C[1])) {
      return true;
    }
  }
  return false;
}

function almostEqual(a, b, eps = 0.0001) {
  return Math.abs(a - b) < eps;
}

// ============= Centering Groups in the SVG =============

function getHexBoundingBox(hexList, axialToPixelFn) {
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  hexList.forEach(({ q, r }) => {
    const { x, y } = axialToPixelFn(q, r);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  });
  if (hexList.length === 0) {
    minX = 0; maxX = 0; minY = 0; maxY = 0;
  }
  return { minX, maxX, minY, maxY };
}

function centerHexGroup(hexList, group, axialToPixelFn, {
  svgWidth = 800,
  svgHeight = 600,
  scale = 1,
  rotation = 0,
  translateX = 0,
  translateY = 0
} = {}) {
  const { minX, maxX, minY, maxY } = getHexBoundingBox(hexList, axialToPixelFn);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const targetX = svgWidth / 2;
  const targetY = svgHeight / 2;

  let transformStr = `
    translate(${targetX + translateX}, ${targetY + translateY})
    scale(${scale})
    rotate(${rotation})
    translate(${-centerX}, ${-centerY})
  `.replace(/\s+/g, ' ');

  group.setAttribute("transform", transformStr);
}

// ============= Main Initialization =============

window.addEventListener("DOMContentLoaded", async () => {
  const svg = document.getElementById("map-svg");
  svg.addEventListener("mousedown", onMouseDown);
  svg.addEventListener("mousemove", onMouseMove);
  svg.addEventListener("mouseup", onMouseUp);

  await loadWorldData();
  drawWorldView();
});

async function loadWorldData() {
  const resp = await fetch("/api/map_data");
  if (!resp.ok) {
    console.error("Failed to load world data");
    return;
  }
  const data = await resp.json();
  worldData = data.world;
  piecesData = data.pieces;
  characterData = data.characters;  // Load characters from backend
}

// ============= WORLD VIEW =============

function drawWorldView() {
  currentView = "world";
  currentRegion = null;
  currentSection = null;

  const toggleBtn = document.getElementById("toggleZoomBtn");
  toggleBtn.style.display = "none"; // Not used at world level

  const svg = document.getElementById("map-svg");
  svg.innerHTML = "";
  // Attach wheel listener for world zoom
  svg.onwheel = null;
  svg.addEventListener("wheel", handleWorldWheelZoom, { passive: false });

  // Create a group for the entire world
  let gWorld = document.createElementNS("http://www.w3.org/2000/svg", "g");
  gWorld.setAttribute("id", "world-group");
  svg.appendChild(gWorld);

  // Hover label for region name
  let hoverLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  hoverLabel.setAttribute("id", "hoverLabel");
  hoverLabel.setAttribute("x", "400");
  hoverLabel.setAttribute("y", "30");
  hoverLabel.setAttribute("text-anchor", "middle");
  hoverLabel.setAttribute("font-size", "16");
  hoverLabel.setAttribute("fill", "#222");
  svg.appendChild(hoverLabel);

  // Collect all region hexes for bounding/centering
  let worldHexList = [];

  // Create one path per region
  worldData.regions.forEach(region => {
    const pathStr = buildRegionPath(region.worldHexes);

    let path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", pathStr);
    path.setAttribute("fill", "#cccccc");
    path.setAttribute("stroke", "none");
    path.setAttribute("class", "region-path");

    path.addEventListener("mouseenter", () => {
      hoverLabel.textContent = region.name;
      path.classList.add("hovered");
    });
    path.addEventListener("mouseleave", () => {
      hoverLabel.textContent = "";
      path.classList.remove("hovered");
    });
    path.addEventListener("click", () => {
      currentRegion = region;
      drawRegionView(region);
    });
    gWorld.appendChild(path);

    // Draw POI markers (already in your code)
    drawPOIMarkers(gWorld, region, 3.5);

    // --- NEW: Draw puzzle markers ---
    if (region.puzzleScenarios && region.puzzleScenarios.length > 0) {
      region.puzzleScenarios.forEach(puzzle => {
        if (puzzle.triggerHex) {
          const { q, r } = puzzle.triggerHex;
          const { x, y } = axialToPixel(q, r);
          const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          marker.setAttribute("cx", x);
          marker.setAttribute("cy", y);
          marker.setAttribute("r", 3.5); // adjust marker size as needed
          marker.setAttribute("class", "puzzle-marker");
          marker.setAttribute("title", puzzle.name);
          gWorld.appendChild(marker);
        }
      });
    }

    // Add region hexes for centering
    region.worldHexes.forEach(hex => {
      worldHexList.push(hex);
    });
  });

  // Center the entire world map
  centerHexGroup(worldHexList, gWorld, axialToPixel, {
    svgWidth: SVG_WIDTH,
    svgHeight: SVG_HEIGHT,
    scale: worldZoom,
    rotation: 0,
    translateX: worldPanX,
    translateY: worldPanY
  });
  drawCharacterRegionMarkers(gWorld);
}

function handleWorldWheelZoom(evt) {
  if (currentView !== "world") return;
  evt.preventDefault();
  let delta = -Math.sign(evt.deltaY) * 0.1;
  let newZoom = worldZoom + delta;
  if (newZoom < MIN_WORLD_ZOOM) newZoom = MIN_WORLD_ZOOM;
  if (newZoom > MAX_WORLD_ZOOM) newZoom = MAX_WORLD_ZOOM;
  worldZoom = newZoom;
  drawWorldView();
}

/** Small circle markers in world view */
function drawPOIMarkers(gGroup, region, marker_size) {
  if (region.pointsOfInterest && region.pointsOfInterest.length > 0) {
    const hoverLabel = document.getElementById("hoverLabel");
    region.pointsOfInterest.forEach(poi => {
      const { x, y } = axialToPixel(poi.q, poi.r);
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", x);
      circle.setAttribute("cy", y);
      circle.setAttribute("r", marker_size);
      circle.setAttribute("fill", "green");
      circle.setAttribute("stroke", "white");
      circle.setAttribute("stroke-width", "1");
      // circle.classList.add("poi-marker");

      circle.addEventListener("mouseenter", (evt) => {
        evt.stopPropagation();
        hoverLabel.textContent = `POI: (${poi.q}, ${poi.r})`;
      });
      circle.addEventListener("mouseleave", (evt) => {
        evt.stopPropagation();
        hoverLabel.textContent = region.name || "";
      });
      circle.addEventListener("click", (evt) => {
        evt.stopPropagation();
        alert(`Clicked a Region POI at (${poi.q}, ${poi.r})`);
      });

      gGroup.appendChild(circle);
    });
  }
}

// ============= REGION VIEW =============

function drawRegionView(region) {
  currentView = "region";
  currentSection = null;

  const toggleBtn = document.getElementById("toggleZoomBtn");
  toggleBtn.style.display = "inline-block";
  toggleBtn.textContent = "World View";
  toggleBtn.onclick = () => {
    drawWorldView();
  };

  const svg = document.getElementById("map-svg");
  svg.innerHTML = "";
  svg.addEventListener("wheel", handleRegionWheelZoom, { passive: false });

  let hoverLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
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

  // Build a Set of region POI coords
  let poiSet = new Set();
  if (region.pointsOfInterest) {
    region.pointsOfInterest.forEach(poi => {
      poiSet.add(`${poi.q},${poi.r}`);
    });
  }

  let regionHexList = [];
  let regionSet = new Set();
  region.worldHexes.forEach(h => regionSet.add(`${h.q},${h.r}`));

  // Draw each hex in this region
  region.worldHexes.forEach(hex => {
    regionHexList.push(hex);
    const { x, y } = axialToPixel(hex.q, hex.r);

    let poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("class", "hex-region");
    poly.setAttribute("points", hexPolygonPoints(x, y));

    // If POI, color differently
    const key = `${hex.q},${hex.r}`;
    if (poiSet.has(key)) {
      poly.setAttribute("style", "fill:green;"); // distinct color for POI
    } else {
      poly.setAttribute("fill", regionColor(region.regionId));
    }

    poly.addEventListener("mouseenter", () => {
      hoverLabel.textContent = region.name;
    });
    poly.addEventListener("mouseleave", () => {
      hoverLabel.textContent = "";
    });
    poly.addEventListener("click", () => {
      if (isSelectingHex && currentHexSelector) {
        const pieceLabel = currentHexSelector.getAttribute("data-piece-label");
        const selection = pieceSelections.get(pieceLabel);
        if (selection) {
          const pieceClass = piecesData.classes[selection.class];
          const actionData = pieceClass.actions[selection.action];
          
          if (actionData.action_type === 'move') {
            const isOccupied = puzzleScenario.pieces.some(p => p.q === sh.q && p.r === sh.r);
            if (!isOccupied && poly.classList.contains("in-range")) {
              selection.targetHex = { q: sh.q, r: sh.r };
              currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
              currentHexSelector.classList.remove("selecting");
              isSelectingHex = false;
              currentHexSelector = null;
              document.querySelectorAll(".hex-region.in-range").forEach(hex => {
                hex.classList.remove("in-range");
              });
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'trap') {
            const isOccupied = puzzleScenario.pieces.some(p => p.q === sh.q && p.r === sh.r && !p.dead);
            const isBlocked = puzzleScenario.blockedHexes.some(h => h.q === sh.q && h.r === sh.r && !h.isTrap);
            
            if (!isOccupied && !isBlocked && poly.classList.contains("in-range")) {
              selection.targetHex = { q: sh.q, r: sh.r };
              currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
              currentHexSelector.classList.remove("selecting");
              isSelectingHex = false;
              currentHexSelector = null;
              document.querySelectorAll(".hex-region.in-range, .hex-region.trap").forEach(hex => {
                hex.classList.remove("in-range");
                hex.classList.remove("trap");
              });
              
              // Add the trap hex to blockedHexes with isTrap flag
              const trapKey = `${sh.q},${sh.r}`;
              blockedHexes.add(trapKey);
              puzzleScenario.blockedHexes.push({
                q: sh.q,
                r: sh.r,
                isTrap: true
              });
              poly.classList.add("trap");
              
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'single_target_attack') {
            const targetPiece = puzzleScenario.pieces.find(p => 
              p.q === sh.q && 
              p.r === sh.r && 
              p.side !== 'player'
            );
            if (targetPiece && poly.classList.contains("in-range")) {
              selection.targetHex = { q: sh.q, r: sh.r };
              currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
              currentHexSelector.classList.remove("selecting");
              isSelectingHex = false;
              currentHexSelector = null;
              document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
                hex.classList.remove("in-range");
                hex.classList.remove("attack");
              });
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'multi_target_attack') {
            const targetPiece = puzzleScenario.pieces.find(p => 
              p.q === sh.q && 
              p.r === sh.r && 
              p.side !== 'player'
            );
            if (targetPiece && poly.classList.contains("in-range")) {
              // Initialize targetHexes array if it doesn't exist
              if (!selection.targetHexes) {
                selection.targetHexes = [];
              }
              
              // Check if this hex is already selected
              const isAlreadySelected = selection.targetHexes.some(h => h.q === sh.q && h.r === sh.r);
              
              if (isAlreadySelected) {
                // Remove this hex from selection
                selection.targetHexes = selection.targetHexes.filter(h => !(h.q === sh.q && h.r === sh.r));
                poly.classList.remove("selected");
              } else if (selection.targetHexes.length < actionData.max_num_targets) {
                // Add this hex to selection
                selection.targetHexes.push({ q: sh.q, r: sh.r });
                poly.classList.add("selected");
              }
              
              // Update UI text to show number of targets selected
              currentHexSelector.textContent = `Selected ${selection.targetHexes.length}/${actionData.max_num_targets} targets`;
              
              // If we've reached max targets, clear selection mode
              if (selection.targetHexes.length === actionData.max_num_targets) {
                currentHexSelector.classList.remove("selecting");
                isSelectingHex = false;
                currentHexSelector = null;
                document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
                  if (!hex.classList.contains("selected")) {
                    hex.classList.remove("in-range");
                    hex.classList.remove("attack");
                  }
                });
              }
              
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'push') {
            // First step: select the piece to push
            if (!selection.pushTarget) {
              const targetPiece = puzzleScenario.pieces.find(p => p.q === sh.q && p.r === sh.r);
              if (targetPiece && poly.classList.contains("in-range")) {
                selection.pushTarget = { q: sh.q, r: sh.r };
                currentHexSelector.textContent = "Select destination";
                
                // Clear existing highlights
                document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
                  hex.classList.remove("in-range");
                  hex.classList.remove("attack");
                });
                
                // Show possible destination hexes around the target piece
                const distance = actionData.distance;
                for (let q = -distance; q <= distance; q++) {
                  for (let r = -distance; r <= distance; r++) {
                    if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * distance) {
                      const destQ = targetPiece.q + q;
                      const destR = targetPiece.r + r;
                      const destKey = `${destQ},${destR}`;
                      
                      // Only show unoccupied hexes that are further from the pushing piece
                      const isUnoccupied = !puzzleScenario.pieces.some(p => p.q === destQ && p.r === destR);
                      const isFurther = Math.abs(destQ - piece.q) + Math.abs(destR - piece.r) > 
                                      Math.abs(targetPiece.q - piece.q) + Math.abs(targetPiece.r - piece.r);
                      
                      if (isUnoccupied && isFurther && !blockedHexes.has(destKey)) {
                        const destHex = document.querySelector(`polygon[data-q="${destQ}"][data-r="${destR}"]`);
                        if (destHex) {
                          destHex.classList.add("in-range");
                          destHex.classList.add("destination");
                        }
                      }
                    }
                  }
                }
                updateActionDescriptions();
              }
            } else {
              // Second step: select the destination
              const isOccupied = puzzleScenario.pieces.some(p => p.q === sh.q && p.r === sh.r);
              if (!isOccupied && poly.classList.contains("in-range") && poly.classList.contains("destination")) {
                selection.targetHex = { q: sh.q, r: sh.r };
                currentHexSelector.textContent = `Push to (${sh.q}, ${sh.r})`;
                currentHexSelector.classList.remove("selecting");
                isSelectingHex = false;
                currentHexSelector = null;
                document.querySelectorAll(".hex-region.in-range, .hex-region.destination").forEach(hex => {
                  hex.classList.remove("in-range");
                  hex.classList.remove("destination");
                });
                updateActionDescriptions();
                validateTurnCompletion();
              }
            }
          } else if (actionData.action_type === 'pull') {
            // First step: select the piece to pull
            if (!selection.pullTarget) {
              const targetPiece = puzzleScenario.pieces.find(p => p.q === sh.q && p.r === sh.r);
              if (targetPiece && poly.classList.contains("in-range")) {
                selection.pullTarget = { q: sh.q, r: sh.r };
                currentHexSelector.textContent = "Select destination";
                
                // Clear existing highlights
                document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
                  hex.classList.remove("in-range");
                  hex.classList.remove("attack");
                });
                
                // Show possible destination hexes around the target piece
                const distance = actionData.distance;
                for (let q = -distance; q <= distance; q++) {
                  for (let r = -distance; r <= distance; r++) {
                    if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * distance) {
                      const destQ = targetPiece.q + q;
                      const destR = targetPiece.r + r;
                      const destKey = `${destQ},${destR}`;
                      
                      // Only show unoccupied hexes that are closer to the pulling piece
                      const isUnoccupied = !puzzleScenario.pieces.some(p => p.q === destQ && p.r === destR);
                      const isCloser = Math.abs(destQ - piece.q) + Math.abs(destR - piece.r) < 
                                     Math.abs(targetPiece.q - piece.q) + Math.abs(targetPiece.r - piece.r);
                      
                      if (isUnoccupied && isCloser && !blockedHexes.has(destKey)) {
                        const destHex = document.querySelector(`polygon[data-q="${destQ}"][data-r="${destR}"]`);
                        if (destHex) {
                          destHex.classList.add("in-range");
                          destHex.classList.add("destination"); // Add a new class for destination hexes
                        }
                      }
                    }
                  }
                }
                updateActionDescriptions();
              }
            } else {
              // Second step: select the destination
              const isOccupied = puzzleScenario.pieces.some(p => p.q === sh.q && p.r === sh.r);
              if (!isOccupied && poly.classList.contains("in-range") && poly.classList.contains("destination")) {
                selection.targetHex = { q: sh.q, r: sh.r };
                currentHexSelector.textContent = `Pull to (${sh.q}, ${sh.r})`;
                currentHexSelector.classList.remove("selecting");
                isSelectingHex = false;
                currentHexSelector = null;
                document.querySelectorAll(".hex-region.in-range, .hex-region.destination").forEach(hex => {
                  hex.classList.remove("in-range");
                  hex.classList.remove("destination");
                });
                updateActionDescriptions();
                validateTurnCompletion();
              }
            }
          }
        }
      } else {
        // This is the original click handler for region hexes
        drawHexDetailView(region, hex);
      }
    });
    gRegion.appendChild(poly);
  });

  // Outline perimeter edges
  region.worldHexes.forEach(hex => {
    let edges = getHexEdges(hex.q, hex.r);
    let neighbors = getHexNeighbors(hex.q, hex.r);
    edges.forEach(edge => {
      let shared = false;
      for (let n of neighbors) {
        if (regionSet.has(`${n.q},${n.r}`)) {
          let nEdges = getHexEdges(n.q, n.r);
          if (isEdgeShared(edge, nEdges)) {
            shared = true;
            break;
          }
        }
      }
      // If not shared, it's a perimeter edge
      if (!shared) {
        let line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", edge[0][0]);
        line.setAttribute("y1", edge[0][1]);
        line.setAttribute("x2", edge[1][0]);
        line.setAttribute("y2", edge[1][1]);
        line.setAttribute("stroke", "black");
        line.setAttribute("stroke-width", "2");
        line.setAttribute("class", "region-outline");
        gRegion.appendChild(line);
      }
    });
  });

  // Now center region
  centerHexGroup(regionHexList, gRegion, axialToPixel, {
    svgWidth: SVG_WIDTH,
    svgHeight: SVG_HEIGHT,
    scale: regionZoom,
    rotation: 0,
    translateX: regionPanX,
    translateY: regionPanY
  });
  drawCharacterRegionMarkers(gRegion);

  // After drawing all the hexes in the region (i.e. after the loop over region.worldHexes),
  // add code to overlay a marker on each puzzle trigger hex.
  if (region.puzzleScenarios && region.puzzleScenarios.length > 0) {
    region.puzzleScenarios.forEach(puzzle => {
      // puzzle.triggerHex should be of the form: { q: value, r: value }
      if (puzzle.triggerHex) {
        const { q, r } = puzzle.triggerHex;
        // Convert axial to pixel coordinates (using your existing function)
        const { x, y } = axialToPixel(q, r);
        // Create an SVG circle as the puzzle marker
        const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        marker.setAttribute("cx", x);
        marker.setAttribute("cy", y);
        marker.setAttribute("r", 2); // marker radius; adjust as needed
        marker.setAttribute("class", "puzzle-marker");
        // Optionally, add a tooltip or data attribute if needed:
        marker.setAttribute("title", puzzle.name);
        // Append the marker to your region group (gRegion)
        gRegion.appendChild(marker);
      }
    });
  }
}

function handleRegionWheelZoom(evt) {
  if (currentView !== "region") return;
  evt.preventDefault();
  let delta = -Math.sign(evt.deltaY) * 0.1;
  let newZoom = regionZoom + delta;
  if (newZoom < MIN_REGION_ZOOM) newZoom = MIN_REGION_ZOOM;
  if (newZoom > MAX_REGION_ZOOM) newZoom = MAX_REGION_ZOOM;
  regionZoom = newZoom;
  drawRegionView(currentRegion);
}

// Draw markers on the world/region view.
function drawCharacterRegionMarkers(g) {
  if (!characterData || !Array.isArray(characterData)) return;
  characterData.forEach(character => {
    // Parse the regionLocation string (expected to be "q,r")
    let parts = character.regionLocation.split(",");
    let q = parseFloat(parts[0]);
    let r = parseFloat(parts[1]);
    let { x, y } = axialToPixel(q, r);
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    marker.setAttribute("cx", x);
    marker.setAttribute("cy", y);
    marker.setAttribute("r", 4); // Adjust as needed
    marker.setAttribute("fill", character.color || "blue");
    marker.setAttribute("stroke", "black");
    marker.setAttribute("stroke-width", "1");
    marker.classList.add("character-region-marker");
    g.appendChild(marker);
  });
}

// Draw markers on the detail (sub-grid) view.
function drawCharacterDetailMarkers(g, subHexList, subAxialToPixel) {
  if (!characterData || !Array.isArray(characterData)) return;
  characterData.forEach(character => {
    // Parse the detailLocation string (expected to be "q,r")
    let parts = character.detailLocation.split(",");
    let q = parseFloat(parts[0]);
    let r = parseFloat(parts[1]);
    // Only draw if the character's detail location is valid (i.e. exists in the sub-hex list)
    if (subHexList.some(hex => hex.q === q && hex.r === r)) {
      let key = `${q},${r}`;
      if (!blockedHexes.has(key)) {
        let { x, y } = subAxialToPixel(q, r);
        // Use a distinct marker (a square, for example)
        const size = 3;
        const marker = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        marker.setAttribute("x", x - size / 2);
        marker.setAttribute("y", y - size / 2);
        marker.setAttribute("width", size);
        marker.setAttribute("height", size);
        marker.setAttribute("fill", character.color || "yellow");
        marker.setAttribute("stroke", "orange");
        marker.setAttribute("stroke-width", "1");
        marker.classList.add("character-detail-marker");
        g.appendChild(marker);
      }
    }
  });
}

/**
 * NEW SECTION VIEW
 */
function drawHexDetailView(region, clickedHex) {
  currentView = "section";
  currentSection = clickedHex;

  // --- Determine if a POI story should trigger ---
  let triggeredPOI = null;
  if (region.pointsOfInterest) {
    for (let poi of region.pointsOfInterest) {
      if (poi.detailHex) {
        // For POIs with a detailHex, only trigger if the clicked hex exactly matches the detailHex.
        if (Number(poi.detailHex.q) === Number(clickedHex.q) &&
            Number(poi.detailHex.r) === Number(clickedHex.r)) {
          triggeredPOI = poi;
          break;
        }
      } else {
        // For POIs without a detailHex, trigger immediately when the clicked hex matches the POI's own coordinates.
        if (Number(poi.q) === Number(clickedHex.q) &&
            Number(poi.r) === Number(clickedHex.r)) {
          triggeredPOI = poi;
          break;
        }
      }
    }
  }

  if (
    triggeredPOI &&
    triggeredPOI.story &&
    triggeredPOI.story.lines &&
    triggeredPOI.story.lines.length > 0 &&
    !triggeredPOI.story.shown
  ) {
    showStoryOverlay(triggeredPOI.story.lines, function() {
      // Mark the story as shown so that subsequent clicks won't re-trigger it.
      triggeredPOI.story.shown = true;
      // Redraw the detail view after closing the story overlay.
      drawHexDetailView(region, clickedHex);
    });
    return; // Exit early until the story overlay is closed.
  }

  // --- Continue drawing the normal detail view ---
  const toggleBtn = document.getElementById("toggleZoomBtn");
  toggleBtn.style.display = "inline-block";
  toggleBtn.textContent = "Region View";
  toggleBtn.onclick = () => {
    drawRegionView(region);
  };

  const svg = document.getElementById("map-svg");
  svg.innerHTML = "";

  // Create hover label
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

  const SUB_GRID_RADIUS = 5;
  const SUB_HEX_SIZE = 10;

  function subAxialToPixel(q, r) {
    let x = SUB_HEX_SIZE * SQRT3 * (q + r / 2);
    let y = SUB_HEX_SIZE * (3 / 2) * r;
    return { x, y };
  }
  function subHexPolygonPoints(cx, cy) {
    let pts = [];
    for (let i = 0; i < 6; i++) {
      let deg = 60 * i + 30;
      let rad = Math.PI / 180 * deg;
      let px = cx + SUB_HEX_SIZE * Math.cos(rad);
      let py = cy + SUB_HEX_SIZE * Math.sin(rad);
      pts.push(`${px},${py}`);
    }
    return pts.join(" ");
  }

  // --- Load puzzle scenario if present ---
  puzzleScenario = null;
  if (region.puzzleScenarios) {
    puzzleScenario = region.puzzleScenarios.find(ps => ps.triggerHex.q === clickedHex.q && ps.triggerHex.r === clickedHex.r);
  }

  const playerControls = document.getElementById("player-controls");
  const enemyControls = document.getElementById("enemy-controls");
  if (puzzleScenario) {
    if (window.selectedCharacter && window.selectedCharacter.location === "regionId=1|q=0|r=0") {
      const exists = puzzleScenario.pieces.some(p => p.label === window.selectedCharacter.name);
      if (!exists) {
        puzzleScenario.pieces.push({
          class: window.selectedCharacter.char_class,
          label: window.selectedCharacter.name,
          color: "#00FF00",
          side: "player",
          q: 0,
          r: 0
        });
      }
    }
    playerControls.style.display = "block";
    enemyControls.style.display = "block";
    setupPlayerControls(puzzleScenario);
    setupEnemyPiecesDisplay(puzzleScenario);
  } else {
    playerControls.style.display = "none";
    enemyControls.style.display = "none";
  }

  // --- Build the sub-grid for the detail view ---
  const gridRadius = puzzleScenario ? puzzleScenario.subGridRadius : SUB_GRID_RADIUS;
  let subHexList = [];
  for (let q = -gridRadius; q <= gridRadius; q++) {
    for (let r = -gridRadius; r <= gridRadius; r++) {
      if (Math.abs(q + r) <= gridRadius) {
        subHexList.push({ q, r });
      }
    }
  }

  // --- Set up blocked hexes if defined ---
  blockedHexes.clear();
  if (puzzleScenario && puzzleScenario.blockedHexes) {
    puzzleScenario.blockedHexes.forEach(h => {
      const key = `${h.q},${h.r}`;
      blockedHexes.add(key);
    });
  }

  // --- Draw the sub-hexes ---
  subHexList.forEach(sh => {
    let { x, y } = subAxialToPixel(sh.q, sh.r);
    let poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("class", "hex-region");
    poly.setAttribute("points", subHexPolygonPoints(x, y));
    poly.setAttribute("data-q", sh.q);
    poly.setAttribute("data-r", sh.r);
    const hexKey = `${sh.q},${sh.r}`;
    if (blockedHexes.has(hexKey)) {
      const blockedHex = puzzleScenario.blockedHexes.find(h => h.q === sh.q && h.r === sh.r);
      if (blockedHex && blockedHex.isTrap) {
        poly.setAttribute("class", "hex-region trap");
      } else {
        poly.setAttribute("fill", "#000000");
        poly.setAttribute("class", "hex-region blocked");
      }
    } else {
      poly.setAttribute("fill", regionColor(region.regionId));
    }
    poly.addEventListener("mouseenter", () => {
      hoverLabel.textContent = `(q=${sh.q},r=${sh.r}) of ${region.name}`;
    });
    poly.addEventListener("mouseleave", () => {
      hoverLabel.textContent = "";
    });
    poly.addEventListener("click", () => {
      if (isSelectingHex && currentHexSelector) {
        const pieceLabel = currentHexSelector.getAttribute("data-piece-label");
        const selection = pieceSelections.get(pieceLabel);
        if (selection) {
          const pieceClass = piecesData.classes[selection.class];
          const actionData = pieceClass.actions[selection.action];
          
          if (actionData.action_type === 'move') {
            const isOccupied = puzzleScenario.pieces.some(p => p.q === sh.q && p.r === sh.r);
            if (!isOccupied && poly.classList.contains("in-range")) {
              selection.targetHex = { q: sh.q, r: sh.r };
              currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
              currentHexSelector.classList.remove("selecting");
              isSelectingHex = false;
              currentHexSelector = null;
              document.querySelectorAll(".hex-region.in-range").forEach(hex => {
                hex.classList.remove("in-range");
              });
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'trap') {
            const isOccupied = puzzleScenario.pieces.some(p => p.q === sh.q && p.r === sh.r && !p.dead);
            const isBlocked = puzzleScenario.blockedHexes.some(h => h.q === sh.q && h.r === sh.r && !h.isTrap);
            
            if (!isOccupied && !isBlocked && poly.classList.contains("in-range")) {
              selection.targetHex = { q: sh.q, r: sh.r };
              currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
              currentHexSelector.classList.remove("selecting");
              isSelectingHex = false;
              currentHexSelector = null;
              document.querySelectorAll(".hex-region.in-range, .hex-region.trap").forEach(hex => {
                hex.classList.remove("in-range");
                hex.classList.remove("trap");
              });
              
              // Add the trap hex to blockedHexes with isTrap flag
              const trapKey = `${sh.q},${sh.r}`;
              blockedHexes.add(trapKey);
              puzzleScenario.blockedHexes.push({
                q: sh.q,
                r: sh.r,
                isTrap: true
              });
              poly.classList.add("trap");
              
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'single_target_attack') {
            const targetPiece = puzzleScenario.pieces.find(p => 
              p.q === sh.q && 
              p.r === sh.r && 
              p.side !== 'player'
            );
            if (targetPiece && poly.classList.contains("in-range")) {
              selection.targetHex = { q: sh.q, r: sh.r };
              currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
              currentHexSelector.classList.remove("selecting");
              isSelectingHex = false;
              currentHexSelector = null;
              document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
                hex.classList.remove("in-range");
                hex.classList.remove("attack");
              });
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'multi_target_attack') {
            const targetPiece = puzzleScenario.pieces.find(p => 
              p.q === sh.q && 
              p.r === sh.r && 
              p.side !== 'player'
            );
            if (targetPiece && poly.classList.contains("in-range")) {
              // Initialize targetHexes array if it doesn't exist
              if (!selection.targetHexes) {
                selection.targetHexes = [];
              }
              
              // Check if this hex is already selected
              const isAlreadySelected = selection.targetHexes.some(h => h.q === sh.q && h.r === sh.r);
              
              if (isAlreadySelected) {
                // Remove this hex from selection
                selection.targetHexes = selection.targetHexes.filter(h => !(h.q === sh.q && h.r === sh.r));
                poly.classList.remove("selected");
              } else if (selection.targetHexes.length < actionData.max_num_targets) {
                // Add this hex to selection
                selection.targetHexes.push({ q: sh.q, r: sh.r });
                poly.classList.add("selected");
              }
              
              // Update UI text to show number of targets selected
              currentHexSelector.textContent = `Selected ${selection.targetHexes.length}/${actionData.max_num_targets} targets`;
              
              // If we've reached max targets, clear selection mode
              if (selection.targetHexes.length === actionData.max_num_targets) {
                currentHexSelector.classList.remove("selecting");
                isSelectingHex = false;
                currentHexSelector = null;
                document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
                  if (!hex.classList.contains("selected")) {
                    hex.classList.remove("in-range");
                    hex.classList.remove("attack");
                  }
                });
              }
              
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'push') {
            // First step: select the piece to push
            if (!selection.pushTarget) {
              const targetPiece = puzzleScenario.pieces.find(p => p.q === sh.q && p.r === sh.r);
              if (targetPiece && poly.classList.contains("in-range")) {
                selection.pushTarget = { q: sh.q, r: sh.r };
                currentHexSelector.textContent = "Select destination";
                
                // Clear existing highlights
                document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
                  hex.classList.remove("in-range");
                  hex.classList.remove("attack");
                });
                
                // Show possible destination hexes around the target piece
                const distance = actionData.distance;
                for (let q = -distance; q <= distance; q++) {
                  for (let r = -distance; r <= distance; r++) {
                    if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * distance) {
                      const destQ = targetPiece.q + q;
                      const destR = targetPiece.r + r;
                      const destKey = `${destQ},${destR}`;
                      
                      // Only show unoccupied hexes that are further from the pushing piece
                      const isUnoccupied = !puzzleScenario.pieces.some(p => p.q === destQ && p.r === destR);
                      const isFurther = Math.abs(destQ - piece.q) + Math.abs(destR - piece.r) > 
                                      Math.abs(targetPiece.q - piece.q) + Math.abs(targetPiece.r - piece.r);
                      
                      if (isUnoccupied && isFurther && !blockedHexes.has(destKey)) {
                        const destHex = document.querySelector(`polygon[data-q="${destQ}"][data-r="${destR}"]`);
                        if (destHex) {
                          destHex.classList.add("in-range");
                          destHex.classList.add("destination");
                        }
                      }
                    }
                  }
                }
                updateActionDescriptions();
              }
            } else {
              // Second step: select the destination
              const isOccupied = puzzleScenario.pieces.some(p => p.q === sh.q && p.r === sh.r);
              if (!isOccupied && poly.classList.contains("in-range") && poly.classList.contains("destination")) {
                selection.targetHex = { q: sh.q, r: sh.r };
                currentHexSelector.textContent = `Push to (${sh.q}, ${sh.r})`;
                currentHexSelector.classList.remove("selecting");
                isSelectingHex = false;
                currentHexSelector = null;
                document.querySelectorAll(".hex-region.in-range, .hex-region.destination").forEach(hex => {
                  hex.classList.remove("in-range");
                  hex.classList.remove("destination");
                });
                updateActionDescriptions();
                validateTurnCompletion();
              }
            }
          } else if (actionData.action_type === 'pull') {
            // First step: select the piece to pull
            if (!selection.pullTarget) {
              const targetPiece = puzzleScenario.pieces.find(p => p.q === sh.q && p.r === sh.r);
              if (targetPiece && poly.classList.contains("in-range")) {
                selection.pullTarget = { q: sh.q, r: sh.r };
                currentHexSelector.textContent = "Select destination";
                
                // Clear existing highlights
                document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
                  hex.classList.remove("in-range");
                  hex.classList.remove("attack");
                });
                
                // Show possible destination hexes around the target piece
                const distance = actionData.distance;
                for (let q = -distance; q <= distance; q++) {
                  for (let r = -distance; r <= distance; r++) {
                    if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * distance) {
                      const destQ = targetPiece.q + q;
                      const destR = targetPiece.r + r;
                      const destKey = `${destQ},${destR}`;
                      
                      // Only show unoccupied hexes that are closer to the pulling piece
                      const isUnoccupied = !puzzleScenario.pieces.some(p => p.q === destQ && p.r === destR);
                      const isCloser = Math.abs(destQ - piece.q) + Math.abs(destR - piece.r) < 
                                     Math.abs(targetPiece.q - piece.q) + Math.abs(targetPiece.r - piece.r);
                      
                      if (isUnoccupied && isCloser && !blockedHexes.has(destKey)) {
                        const destHex = document.querySelector(`polygon[data-q="${destQ}"][data-r="${destR}"]`);
                        if (destHex) {
                          destHex.classList.add("in-range");
                          destHex.classList.add("destination"); // Add a new class for destination hexes
                        }
                      }
                    }
                  }
                }
                updateActionDescriptions();
              }
            } else {
              // Second step: select the destination
              const isOccupied = puzzleScenario.pieces.some(p => p.q === sh.q && p.r === sh.r);
              if (!isOccupied && poly.classList.contains("in-range") && poly.classList.contains("destination")) {
                selection.targetHex = { q: sh.q, r: sh.r };
                currentHexSelector.textContent = `Pull to (${sh.q}, ${sh.r})`;
                currentHexSelector.classList.remove("selecting");
                isSelectingHex = false;
                currentHexSelector = null;
                document.querySelectorAll(".hex-region.in-range, .hex-region.destination").forEach(hex => {
                  hex.classList.remove("in-range");
                  hex.classList.remove("destination");
                });
                updateActionDescriptions();
                validateTurnCompletion();
              }
            }
          }
        }
      }
    });
    gDetail.appendChild(poly);
  });

  // --- Draw the exclamation marker ONLY for the triggered POI (if it defines a detailHex) ---
  if (triggeredPOI && triggeredPOI.detailHex && triggeredPOI.story && triggeredPOI.story.lines && triggeredPOI.story.lines.length > 0) {
    const { q: dq, r: dr } = triggeredPOI.detailHex;
    const { x, y } = subAxialToPixel(dq, dr);
    // Only draw if the designated hex exists in the sub‑grid.
    const found = subHexList.some(sh => sh.q === dq && sh.r === dr);
    if (found) {
      const textEl = document.createElementNS("http://www.w3.org/2000/svg", "text");
      textEl.setAttribute("x", x);
      textEl.setAttribute("y", y);
      textEl.setAttribute("text-anchor", "middle");
      textEl.setAttribute("dominant-baseline", "middle");
      textEl.setAttribute("fill", "orange");
      textEl.setAttribute("font-size", SUB_HEX_SIZE * 1.5);
      textEl.textContent = "!";
      gDetail.appendChild(textEl);
    }
  }

  // --- Draw pieces if available ---
  if (puzzleScenario && puzzleScenario.pieces) {
    puzzleScenario.pieces.forEach(piece => {
      const { x, y } = subAxialToPixel(piece.q, piece.r);
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", x);
      circle.setAttribute("cy", y);
      circle.setAttribute("r", SUB_HEX_SIZE * 0.6);
      circle.setAttribute("fill", piece.color || "#000");
      circle.setAttribute("pointer-events", "none");
      circle.addEventListener("mouseenter", () => {
        const pieceClass = piecesData.classes[piece.class];
        if (pieceClass && pieceClass.actions.move) {
          const moveRange = pieceClass.actions.move.range;
          showMoveRange(piece.q, piece.r, moveRange, gDetail);
          hoverLabel.textContent = `${piece.class} (${piece.side}) - Move Range: ${moveRange}`;
        }
      });
      circle.addEventListener("mouseleave", () => {
        clearMoveRange();
        hoverLabel.textContent = "";
      });
      gDetail.appendChild(circle);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", x);
      text.setAttribute("y", y);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("dominant-baseline", "middle");
      text.setAttribute("fill", "#fff");
      text.setAttribute("font-size", SUB_HEX_SIZE);
      text.setAttribute("pointer-events", "none");
      text.setAttribute("transform", `rotate(-30, ${x}, ${y})`);
      text.textContent = piece.label;
      gDetail.appendChild(text);
    });
  }

  centerHexGroup(subHexList, gDetail, subAxialToPixel, {
    svgWidth: SVG_WIDTH,
    svgHeight: SVG_HEIGHT,
    scale: 2,
    rotation: 30
  });
  drawCharacterDetailMarkers(gDetail, subHexList, subAxialToPixel);
}

// Move this outside of setupPlayerControls to make it globally accessible
function updateActionDescriptions() {
    const actionDesc = document.getElementById("action-description");
    const descriptions = [];
    
    pieceSelections.forEach((selection, uniqueLabel) => {
        if (selection.action && selection.action !== "pass") {
            let desc = `${selection.class} (${selection.originalLabel}): ${selection.description}`;
            
            if (selection.targetHex) {
                if (selection.affectedHexes) {
                    // AOE attack
                    desc += ` centered at (${selection.targetHex.q}, ${selection.targetHex.r})`;
                    if (selection.affectedHexes.length > 0) {
                        desc += ` affecting enemies at ${selection.affectedHexes.map(h => `(${h.q},${h.r})`).join(', ')}`;
                    }
                } else {
                    // Single target attack or move
                    desc += ` to hex (${selection.targetHex.q}, ${selection.targetHex.r})`;
                }
            } else if (selection.targetHexes) {
                // Multi-target attack
                desc += ` targeting ${selection.targetHexes.map(h => `(${h.q},${h.r})`).join(', ')}`;
            }
            
            descriptions.push(desc);
        }
    });
    
    actionDesc.innerHTML = descriptions.length > 0 ? descriptions.join('<br><br>') : "";
}

// Move validateTurnCompletion outside of setupPlayerControls to make it globally accessible
function validateTurnCompletion() {
    let isValid = true;
    console.log("Validating turn completion...");
    
    pieceSelections.forEach((selection, uniqueLabel) => {
        console.log(`Checking ${uniqueLabel}:`, selection);
        
        if (selection.action === "pass") return;
        
        const [class_, q, r] = uniqueLabel.split('_');
        const piece = puzzleScenario.pieces.find(p => 
            p.class === class_ && 
            p.q === parseInt(q) && 
            p.r === parseInt(r)
        );
        if (!piece) return;
        
        const pieceClass = piecesData.classes[piece.class];
        const actionData = pieceClass.actions[selection.action];
        
        if (!actionData) return;
        
        // Check line of sight if required
        if (actionData.requires_los) {
          if (selection.targetHex) {
            const ignorePiece = actionData.action_type === 'push' ? selection.pushTarget : null;
            if (!hasLineOfSight(piece.q, piece.r, selection.targetHex.q, selection.targetHex.r, ignorePiece)) {
              console.log(`${uniqueLabel}'s ${selection.action} has no line of sight to target`);
              isValid = false;
              return;
            }
          }
          if (selection.targetHexes) {
            for (const target of selection.targetHexes) {
              const ignorePiece = actionData.action_type === 'push' ? selection.pushTarget : null;
              if (!hasLineOfSight(piece.q, piece.r, target.q, target.r, ignorePiece)) {
                console.log(`${uniqueLabel}'s ${selection.action} has no line of sight to one of its targets`);
                isValid = false;
                return;
              }
            }
          }
        }

        switch (actionData.action_type) {
            case 'move':
                if (!selection.targetHex) {
                    console.log(`${uniqueLabel} has no target hex`);
                    isValid = false;
                    break;
                }
                
                // Check if target hex is blocked
                const targetKey = `${selection.targetHex.q},${selection.targetHex.r}`;
                if (blockedHexes.has(targetKey)) {
                    console.log(`${uniqueLabel} target hex is blocked`);
                    isValid = false;
                    break;
                }
                
                // Check distance
                const dx = Math.abs(selection.targetHex.q - piece.q);
                const dy = Math.abs(selection.targetHex.r - piece.r);
                const dz = Math.abs(-selection.targetHex.q - selection.targetHex.r + piece.q + piece.r);
                const distance = Math.max(dx, dy, dz);
                
                if (distance > actionData.range) {
                    console.log(`${uniqueLabel} target is out of range`);
                    isValid = false;
                }
                
                // Check if occupied
                const isOccupied = puzzleScenario.pieces.some(p => 
                    p.q === selection.targetHex.q && p.r === selection.targetHex.r
                );
                if (isOccupied) {
                    console.log(`${uniqueLabel} target hex is occupied`);
                    isValid = false;
                }
                break;
                
            case 'swap_position':
                if (!selection.targetHex) {
                    console.log(`${uniqueLabel} has no target hex`);
                    isValid = false;
                    break;
                }

                // Check distance
                const swapDx = Math.abs(selection.targetHex.q - piece.q);
                const swapDy = Math.abs(selection.targetHex.r - piece.r);
                const swapDz = Math.abs(-selection.targetHex.q - selection.targetHex.r + piece.q + piece.r);
                const swapDistance = Math.max(swapDx, swapDy, swapDz);
                
                if (swapDistance > actionData.range) {
                    console.log(`${uniqueLabel} swap target is out of range`);
                    isValid = false;
                    break;
                }

                // Check if target hex has a piece to swap with
                const targetPiece = puzzleScenario.pieces.find(p => 
                    p.q === selection.targetHex.q && 
                    p.r === selection.targetHex.r && 
                    p !== piece
                );
                
                if (!targetPiece) {
                    console.log(`${uniqueLabel} target hex has no piece to swap with`);
                    isValid = false;
                    break;
                }

                // Check ally_only constraint
                if (actionData.ally_only && targetPiece.side !== piece.side) {
                    console.log(`${uniqueLabel} can only swap with allies`);
                    isValid = false;
                    break;
                }

                // Check if target hex is blocked
                const swapTargetKey = `${selection.targetHex.q},${selection.targetHex.r}`;
                if (blockedHexes.has(swapTargetKey)) {
                    console.log(`${uniqueLabel} swap target hex is blocked`);
                    isValid = false;
                }
                break;
                
            case 'single_target_attack':
                if (!selection.targetHex) {
                    console.log(`${uniqueLabel} has no target hex`);
                    isValid = false;
                    break;
                }
                
                // Check distance
                const attackDx = Math.abs(selection.targetHex.q - piece.q);
                const attackDy = Math.abs(selection.targetHex.r - piece.r);
                const attackDz = Math.abs(-selection.targetHex.q - selection.targetHex.r + piece.q + piece.r);
                const attackDistance = Math.max(attackDx, attackDy, attackDz);
                
                if (attackDistance > actionData.range) {
                    console.log(`${uniqueLabel} target is out of range`);
                    isValid = false;
                }
                
                // Check if target has enemy
                const hasEnemy = puzzleScenario.pieces.some(p => 
                    p.q === selection.targetHex.q && 
                    p.r === selection.targetHex.r && 
                    p.side !== 'player'
                );
                if (!hasEnemy) {
                    console.log(`${uniqueLabel} target hex has no enemy`);
                    isValid = false;
                }
                break;
                
            case 'multi_target_attack':
                if (!selection.targetHexes || selection.targetHexes.length === 0) {
                    console.log(`${uniqueLabel} has no targets`);
                    isValid = false;
                    break;
                }
                
                // Check each target
                selection.targetHexes.forEach(target => {
                    // Check distance
                    const mtaDx = Math.abs(target.q - piece.q);
                    const mtaDy = Math.abs(target.r - piece.r);
                    const mtaDz = Math.abs(-target.q - target.r + piece.q + piece.r);
                    const mtaDistance = Math.max(mtaDx, mtaDy, mtaDz);
                    
                    if (mtaDistance > actionData.range) {
                        console.log(`${uniqueLabel} target (${target.q},${target.r}) is out of range`);
                        isValid = false;
                    }
                    
                    // Check if target has enemy
                    const hasEnemy = puzzleScenario.pieces.some(p => 
                        p.q === target.q && 
                        p.r === target.r && 
                        p.side !== 'player'
                    );
                    if (!hasEnemy) {
                        console.log(`${uniqueLabel} target hex (${target.q},${target.r}) has no enemy`);
                        isValid = false;
                    }
                });
                break;
                
            case 'aoe':
                if (!selection.targetHex) {
                    console.log(`${uniqueLabel} has no target hex`);
                    isValid = false;
                    break;
                }
                
                // Check if center point is in range
                const aoeDx = Math.abs(selection.targetHex.q - piece.q);
                const aoeDy = Math.abs(selection.targetHex.r - piece.r);
                const aoeDz = Math.abs(-selection.targetHex.q - selection.targetHex.r + piece.q + piece.r);
                const aoeDistance = Math.max(aoeDx, aoeDy, aoeDz);
                
                if (aoeDistance > actionData.range) {
                    console.log(`${uniqueLabel} target center is out of range`);
                    isValid = false;
                }
                break;

            case 'pull':
                if (!selection.pullTarget || !selection.targetHex) {
                    console.log(`${uniqueLabel} has no pull target or destination`);
                    isValid = false;
                    break;
                }

                // Check if pull target is in range
                const pullDx = Math.abs(selection.pullTarget.q - piece.q);
                const pullDy = Math.abs(selection.pullTarget.r - piece.r);
                const pullDz = Math.abs(-selection.pullTarget.q - selection.pullTarget.r + piece.q + piece.r);
                const pullDistance = Math.max(pullDx, pullDy, pullDz);
                
                if (pullDistance > actionData.range) {
                    console.log(`${uniqueLabel} pull target is out of range`);
                    isValid = false;
                    break;
                }

                // Check if destination is within pull distance
                const destDx = Math.abs(selection.targetHex.q - piece.q);
                const destDy = Math.abs(selection.targetHex.r - piece.r);
                const destDz = Math.abs(-selection.targetHex.q - selection.targetHex.r + piece.q + piece.r);
                const destDistance = Math.max(destDx, destDy, destDz);
                
                if (destDistance > actionData.distance) {
                    console.log(`${uniqueLabel} destination is out of pull range`);
                    isValid = false;
                    break;
                }

                // Check if destination is occupied
                const isDestOccupied = puzzleScenario.pieces.some(p => 
                    p.q === selection.targetHex.q && p.r === selection.targetHex.r
                );
                if (isDestOccupied) {
                    console.log(`${uniqueLabel} destination is occupied`);
                    isValid = false;
                }
                break;

            case 'push':
                if (!selection.pushTarget || !selection.targetHex) {
                    console.log(`${uniqueLabel} has no push target or destination`);
                    isValid = false;
                    break;
                }

                // Check if push target is in range
                const pushDx = Math.abs(selection.pushTarget.q - piece.q);
                const pushDy = Math.abs(selection.pushTarget.r - piece.r);
                const pushDz = Math.abs(-selection.pushTarget.q - selection.pushTarget.r + piece.q + piece.r);
                const pushDistance = Math.max(pushDx, pushDy, pushDz);
                
                if (pushDistance > actionData.range) {
                    console.log(`${uniqueLabel} push target is out of range`);
                    isValid = false;
                    break;
                }

                // Check if destination is within push distance from the target piece
                const pushDestDx = Math.abs(selection.targetHex.q - selection.pushTarget.q);
                const pushDestDy = Math.abs(selection.targetHex.r - selection.pushTarget.r);
                const pushDestDz = Math.abs(-selection.targetHex.q - selection.targetHex.r + selection.pushTarget.q + selection.pushTarget.r);
                const pushDestDistance = Math.max(pushDestDx, pushDestDy, pushDestDz);
                
                if (pushDestDistance > actionData.distance) {
                    console.log(`${uniqueLabel} push destination is out of push range`);
                    isValid = false;
                    break;
                }

                // Check if destination is occupied
                const isPushDestOccupied = puzzleScenario.pieces.some(p => 
                    p.q === selection.targetHex.q && p.r === selection.targetHex.r
                );
                if (isPushDestOccupied) {
                    console.log(`${uniqueLabel} push destination is occupied`);
                    isValid = false;
                }
                break;

            case 'trap':
                if (!selection.targetHex) {
                    console.log(`${uniqueLabel} has no target hex`);
                    isValid = false;
                    break;
                }

                // Check if target hex is blocked by a non-trap obstacle
                const trapTargetKey = `${selection.targetHex.q},${selection.targetHex.r}`;
                const isBlockedByNonTrap = puzzleScenario.blockedHexes.some(h => 
                    h.q === selection.targetHex.q && 
                    h.r === selection.targetHex.r &&
                    !h.isTrap
                );
                
                if (isBlockedByNonTrap) {
                    console.log(`${uniqueLabel} trap target hex is blocked by non-trap obstacle`);
                    isValid = false;
                    break;
                }
                break;
        }
    });
    
    console.log("Turn validation result:", isValid);
    
    const completeTurnBtn = document.getElementById("complete-turn");
    if (completeTurnBtn) {
        completeTurnBtn.disabled = !isValid;
    }
}

// Update createPieceInfoSection to add hover handlers for each action
function createPieceInfoSection(piece, pieceClass) {
  const infoSection = document.createElement("div");
  infoSection.className = "piece-info-section";
  
  // Add class name and description
  const title = document.createElement("h4");
  title.textContent = piece.class;
  infoSection.appendChild(title);
  
  if (pieceClass.description) {
    const desc = document.createElement("div");
    desc.className = "description";
    desc.textContent = pieceClass.description;
    infoSection.appendChild(desc);
  }
  
  // Add list of actions
  const actionList = document.createElement("ul");
  actionList.className = "action-list";
  
  Object.entries(pieceClass.actions).forEach(([actionName, actionData]) => {
    const actionItem = document.createElement("li");
    actionItem.className = "action-item";
    actionItem.setAttribute("data-action", actionName);
    
    const actionTitle = document.createElement("div");
    actionTitle.className = "action-name";
    actionTitle.textContent = actionName.charAt(0).toUpperCase() + actionName.slice(1);
    actionItem.appendChild(actionTitle);
    
    const actionAttributes = document.createElement("div");
    actionAttributes.className = "action-attributes";
    
    // Add all relevant attributes
    const attributes = [];
    if (actionData.action_type) attributes.push(`Type: ${actionData.action_type}`);
    if (actionData.range) attributes.push(`Range: ${actionData.range}`);
    if (actionData.description) attributes.push(actionData.description);
    if (actionData.max_num_targets) attributes.push(`Max Targets: ${actionData.max_num_targets}`);
    if (actionData.radius) attributes.push(`Radius: ${actionData.radius}`);
    
    actionAttributes.textContent = attributes.join(" • ");
    actionItem.appendChild(actionAttributes);

    // Add hover handlers for the action item
    actionItem.addEventListener("mouseenter", () => {
      showPieceActionRange(piece, pieceClass, actionName);
    });

    actionItem.addEventListener("mouseleave", () => {
      // Clear all range indicators
      document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
        hex.classList.remove("in-range");
        hex.classList.remove("attack");
      });
    });
    
    actionList.appendChild(actionItem);
  });
  
  infoSection.appendChild(actionList);
  return infoSection;
}

// Update setupEnemyPiecesDisplay to remove piece-level hover
function setupEnemyPiecesDisplay(scenario) {
  const enemyPiecesList = document.getElementById("enemy-pieces");
  enemyPiecesList.innerHTML = ""; // Clear existing
  
  // Filter for enemy pieces
  const enemyPieces = scenario.pieces.filter(p => p.side === "enemy");
  
  // Create display for each enemy piece
  enemyPieces.forEach(piece => {
    const li = document.createElement("li");
    li.className = "enemy-piece-item";
    
    // Create piece label with color circle
    const labelDiv = document.createElement("div");
    labelDiv.className = "piece-label";
    
    const colorSpan = document.createElement("span");
    colorSpan.className = "piece-color";
    colorSpan.style.backgroundColor = piece.color;
    
    const labelSpan = document.createElement("span");
    labelSpan.textContent = `${piece.class} (${piece.label})`;
    
    // Add expand button
    const expandButton = document.createElement("button");
    expandButton.className = "piece-info-button";
    expandButton.textContent = "ℹ️";
    expandButton.title = "Show piece information";
    
    labelDiv.appendChild(colorSpan);
    labelDiv.appendChild(labelSpan);
    labelDiv.appendChild(expandButton);
    
    // Get piece class data and create info section
    const pieceClass = piecesData.classes[piece.class];
    const infoSection = createPieceInfoSection(piece, pieceClass);
    
    // Add expand button click handler
    expandButton.addEventListener("click", (e) => {
      e.stopPropagation();
      infoSection.classList.toggle("visible");
      expandButton.textContent = infoSection.classList.contains("visible") ? "🔼" : "ℹ️";
    });
    
    li.appendChild(labelDiv);
    li.appendChild(infoSection);
    enemyPiecesList.appendChild(li);
  });
}

// Helper function to show movement range
function showMoveRange(centerQ, centerR, range, parentGroup) {
  // Only clear move-range class, not in-range class
  const rangeHexes = document.getElementsByClassName("move-range");
  while (rangeHexes.length > 0) {
    rangeHexes[0].remove();
  }
  
  // Create a set of occupied positions
  const occupiedPositions = new Set(
    puzzleScenario.pieces.map(p => `${p.q},${p.r}`)
  );
  
  // For each hex within range
  for (let q = -range; q <= range; q++) {
    for (let r = -range; r <= range; r++) {
      // Check if hex is within range (using axial distance)
      if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * range) {
        const targetQ = centerQ + q;
        const targetR = centerR + r;
        
        // Don't highlight if:
        // 1. It's the piece's own hex
        // 2. The hex is occupied by any piece
        if (q === 0 && r === 0 || occupiedPositions.has(`${targetQ},${targetR}`)) continue;
        
        const {x,y} = subAxialToPixel(targetQ, targetR);
        
        // Create range indicator
        const rangeHex = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
        rangeHex.setAttribute("points", subHexPolygonPoints(x,y));
        rangeHex.setAttribute("class", "move-range");
        rangeHex.setAttribute("fill", "rgba(255,255,0,0.2)");
        rangeHex.setAttribute("stroke", "rgba(255,255,0,0.5)");
        rangeHex.setAttribute("stroke-width", "1");
        parentGroup.insertBefore(rangeHex, parentGroup.firstChild); // Add behind pieces
      }
    }
  }
}

// Helper function to clear movement range
function clearMoveRange() {
  // Only clear move-range class, not in-range class
  const rangeHexes = document.getElementsByClassName("move-range");
  while (rangeHexes.length > 0) {
    rangeHexes[0].remove();
  }
}

// Add this new function for showing move range during selection
function showMoveRangeForSelection(centerQ, centerR, range, parentGroup) {
  // Remove any existing range indicators
  clearMoveRange();
  
  // For each hex within range
  for (let q = -range; q <= range; q++) {
    for (let r = -range; r <= range; r++) {
      // Check if hex is within range (using axial distance)
      if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * range) {
        const targetQ = centerQ + q;
        const targetR = centerR + r;
        
        // Don't highlight the piece's own hex
        if (q === 0 && r === 0) continue;
        
        // Find the hex at these coordinates
        const hex = document.querySelector(`polygon[data-q="${targetQ}"][data-r="${targetR}"]`);
        if (hex) {
          hex.classList.add("in-range");
        }
      }
    }
  }
}

// /** handle zoom */
// function handleToggleZoom(){
//   if(currentView==="region"){
//     drawWorldView();
//   } else if(currentView==="section"){
//     drawRegionView(currentRegion);
//   }
// }

// ============= Additional Helpers =============
function regionColor(id) {
  let pal = ["#cce5ff","#ffe5cc","#e5ffcc","#f5ccff","#fff5cc","#ccf0ff","#e0cce5","#eed5cc"];
  return pal[id % pal.length];
}

/**
 * Check if a given edge [ [x1,y1],[x2,y2] ] 
 * appears in neighborEdges (within tolerance). 
 * We'll consider an edge "the same" if it has the same 2 endpoints 
 * ignoring order (since reversed is the same edge).
 */
function isEdgeShared(edge, neighborEdges){
  let [A,B] = edge;
  for(let nEdge of neighborEdges){
    let [C,D] = nEdge;
    // We can do a small epsilon if needed, or direct compare
    // Check if A==C & B==D or A==D & B==C
    if(almostEqual(A[0],C[0]) && almostEqual(A[1],C[1]) &&
       almostEqual(B[0],D[0]) && almostEqual(B[1],D[1])){
      return true;
    }
    if(almostEqual(A[0],D[0]) && almostEqual(A[1],D[1]) &&
       almostEqual(B[0],C[0]) && almostEqual(B[1],C[1])){
      return true;
    }
  }
  return false;
}

// optional small float compare 
function almostEqual(a,b, eps=0.0001){
  return Math.abs(a-b)<eps;
}

// Update the CSS for the disabled button state
function updateButtonStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .complete-turn-btn:disabled {
      background: #cccccc;
      cursor: not-allowed;
    }
  `;
  document.head.appendChild(style);
}

// Call this when the page loads
window.addEventListener("DOMContentLoaded", () => {
  updateButtonStyles();
  // ... existing DOMContentLoaded code ...
});

// Add this function to handle logging
function addBattleLog(message) {
    battleLog.push(message);
    const logEntries = document.getElementById("log-entries");
    const entry = document.createElement("div");
    entry.textContent = message;
    entry.style.marginBottom = "5px";
    logEntries.appendChild(entry);
    logEntries.scrollTop = logEntries.scrollHeight;
}

// Add this new function to show piece range and targets
function showPieceActionRange(piece, pieceClass, actionName) {
  // Clear existing highlights
  document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
    hex.classList.remove("in-range");
    hex.classList.remove("attack");
  });

  const actionData = pieceClass.actions[actionName];
  if (!actionData || !actionData.range) return;

  const range = actionData.range;
  const requiresLOS = actionData.requires_los; // Change to check action's requires_los instead of piece's

  // For each hex within range
  for (let q = -range; q <= range; q++) {
    for (let r = -range; r <= range; r++) {
      if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * range) {
        const targetQ = piece.q + q;
        const targetR = piece.r + r;
        
        // Skip if target hex is blocked
        const targetKey = `${targetQ},${targetR}`;
        if (blockedHexes.has(targetKey)) continue;

        // Check line of sight if required
        if (requiresLOS && !hasLineOfSight(piece.q, piece.r, targetQ, targetR)) {
          continue; // Skip this hex if no line of sight
        }

        // Rest of the action type logic remains the same...
        if (actionData.action_type === 'move') {
          const isOccupied = puzzleScenario.pieces.some(p => 
            p.q === targetQ && p.r === targetR
          );
          
          if (!isOccupied && !(q === 0 && r === 0)) {
            const hex = document.querySelector(`polygon[data-q="${targetQ}"][data-r="${targetR}"]`);
            if (hex) {
              hex.classList.add("in-range");
            }
          }
        } else if (actionData.action_type === 'swap_position') {
          // Only highlight hexes with valid swap targets
          const targetPiece = puzzleScenario.pieces.find(p => 
            p.q === targetQ && p.r === targetR && p !== piece
          );
          
          if (targetPiece) {
            // Check if we can swap with this piece based on ally_only flag
            const canSwap = !actionData.ally_only || targetPiece.side === piece.side;
            
            if (canSwap) {
              const hex = document.querySelector(`polygon[data-q="${targetQ}"][data-r="${targetR}"]`);
              if (hex) {
                hex.classList.add("in-range");
                // Add attack class for enemy swaps to make them visually distinct
                if (targetPiece.side !== piece.side) {
                  hex.classList.add("attack");
                }
              }
            }
          }
        } else if (actionData.action_type === 'pull') {
          // For pull, highlight hexes that have pieces (allies or enemies) within range
          const targetPiece = puzzleScenario.pieces.find(p => 
            p.q === targetQ && p.r === targetR && p !== piece
          );
          
          if (targetPiece) {
            const hex = document.querySelector(`polygon[data-q="${targetQ}"][data-r="${targetR}"]`);
            if (hex) {
              hex.classList.add("in-range");
              // Add attack class for enemy pulls to make them visually distinct
              if (targetPiece.side !== piece.side) {
                hex.classList.add("attack");
              }
            }
          }
        } else if (actionData.action_type === 'single_target_attack' || 
                   actionData.action_type === 'multi_target_attack' || 
                   actionData.action_type === 'dark_bolt') {
          const hasValidTarget = puzzleScenario.pieces.some(p => 
            p.q === targetQ && p.r === targetR && p.side !== 'player'
          );
          
          if (hasValidTarget) {
            const hex = document.querySelector(`polygon[data-q="${targetQ}"][data-r="${targetR}"]`);
            if (hex) {
              hex.classList.add("in-range");
              hex.classList.add("attack");
            }
          }
        } else if (actionData.action_type === 'aoe') {
          const hex = document.querySelector(`polygon[data-q="${targetQ}"][data-r="${targetR}"]`);
          if (hex) {
            hex.classList.add("in-range");
            const hasValidTarget = puzzleScenario.pieces.some(p => 
              p.q === targetQ && p.r === targetR && p.side !== 'player'
            );
            if (hasValidTarget) {
              hex.classList.add("attack");
            }
          }
        } else if (actionData.action_type === 'push') {
          // For push, highlight hexes that have pieces (allies or enemies) within range
          const targetPiece = puzzleScenario.pieces.find(p => 
            p.q === targetQ && p.r === targetR && p !== piece
          );
          
          if (targetPiece) {
            const hex = document.querySelector(`polygon[data-q="${targetQ}"][data-r="${targetR}"]`);
            if (hex) {
              hex.classList.add("in-range");
              // Add attack class for enemy pushes to make them visually distinct
              if (targetPiece.side !== piece.side) {
                hex.classList.add("attack");
              }
            }
          }
        } else if (actionData.action_type === 'trap') {
          // For trap, highlight any unoccupied hex within range
          const isOccupied = puzzleScenario.pieces.some(p => 
            p.q === targetQ && p.r === targetR && !p.dead
          );
          const isBlocked = puzzleScenario.blockedHexes.some(h => 
            h.q === targetQ && h.r === targetR && !h.isTrap
          );
          
          if (!isOccupied && !isBlocked) {
            const hex = document.querySelector(`polygon[data-q="${targetQ}"][data-r="${targetR}"]`);
            if (hex) {
              hex.classList.add("in-range");
              hex.classList.add("trap");
            }
          }
        }
      }
    }
  }
}

function hasLineOfSight(startQ, startR, endQ, endR, ignorePiece = null) {
  if (startQ === endQ && startR === endR) return true;

  // Build the supercover line:
  const hexLine = getHexesInLineSupercover(startQ, startR, endQ, endR);

  // For each interior hex (skip the very first if you don't want the caster's own tile to block)
  // but definitely check everything else. 
  // If any blocked hex or piece occupies those hexes, LOS is blocked.
  for (let i = 1; i < hexLine.length - 1; i++) {
    const { q, r } = hexLine[i];
    const key = `${q},${r}`;
    // blocked?
    if (blockedHexes.has(key)) {
      return false;
    }
    // piece in the way?
    const hasPiece = puzzleScenario.pieces.some(p => 
      p.q === q && p.r === r && 
      (!ignorePiece || (p.q !== ignorePiece.q || p.r !== ignorePiece.r))
    );
    if (hasPiece) {
      return false;
    }
  }

  return true;
}


function axialToPixel(q, r) {
  const x = HEX_SIZE * SQRT3 * (q + r / 2);
  const y = HEX_SIZE * (3 / 2) * r;
  return { x, y };
}

function pixelToAxial(x, y) {
  // We invert the same transform:
  //    x = HEX_SIZE * sqrt3 * (q + r/2)
  //    y = HEX_SIZE * (3/2) * r
  // Solve for q,r.
  const q = (SQRT3/3 * x - 1/3 * y) / (HEX_SIZE * 1);  // approximate
  const r = (2/3 * y) / (HEX_SIZE * 1);               // approximate
  return { q, r };
}

// Then to get integer hex coords, you typically do a "cube round":
function pixelToAxialRound(px, py) {
  const { q, r } = pixelToAxial(px, py);
  // Convert q,r => cube => round => back to axial
  const cubeQ = q;
  const cubeR = r;
  const cubeS = -cubeQ - cubeR;

  let rq = Math.round(cubeQ);
  let rr = Math.round(cubeR);
  let rs = Math.round(cubeS);

  // fix rounding drift so that rq + rr + rs = 0
  const qDiff = Math.abs(rq - cubeQ);
  const rDiff = Math.abs(rr - cubeR);
  const sDiff = Math.abs(rs - cubeS);

  if (qDiff > rDiff && qDiff > sDiff) {
    rq = -rr - rs;
  } else if (rDiff > sDiff) {
    rr = -rq - rs;
  } else {
    rs = -rq - rr;
  }

  return { q: rq, r: rr };
}

/**
 * getHexesInLineSupercover
 * Returns *all* hexes the line (in pixel space) from (startQ,startR) 
 * to (endQ,endR) touches, even if it only crosses a corner. 
 */
function getHexesInLineSupercover(startQ, startR, endQ, endR) {
  // Convert axial -> pixel
  const startPixel = axialToPixel(startQ, startR);
  const endPixel   = axialToPixel(endQ, endR);

  // Decide how many steps to take. 
  // A common choice is something like 2 or 3 steps per pixel of distance.
  // Or you can base it on the max hex distance times some factor.
  const dx = endPixel.x - startPixel.x;
  const dy = endPixel.y - startPixel.y;
  const distPix = Math.sqrt(dx*dx + dy*dy);
  // We'll do something like 2 samples per pixel:
  const steps = Math.ceil(distPix * 2);

  const touched = [];
  const visited = new Set();

  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const px = startPixel.x + dx * t;
    const py = startPixel.y + dy * t;
    
    const { q, r } = pixelToAxialRound(px, py);
    const key = `${q},${r}`;
    if (!visited.has(key)) {
      visited.add(key);
      touched.push({ q, r });
    }
  }
  
  return touched;
}

// Add this function to get hexes in a line using cube coordinates
function getHexesInLine(startQ, startR, endQ, endR) {
  const N = Math.max(
    Math.abs(endQ - startQ),
    Math.abs(endR - startR),
    Math.abs((startQ + startR) - (endQ + endR))
  );

  const hexes = [];
  
  // If N is 0, return just the start point
  if (N === 0) {
    return [{q: startQ, r: startR}];
  }
  
  // Use cube coordinates for linear interpolation
  const startS = -startQ - startR;
  const endS = -endQ - endR;
  
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    // Interpolate in cube coordinates
    const q = Math.round(startQ + (endQ - startQ) * t);
    const r = Math.round(startR + (endR - startR) * t);
    const s = Math.round(startS + (endS - startS) * t);
    
    // Ensure we maintain q + r + s = 0
    const sum = q + r + s;
    if (sum !== 0) {
      // Fix any rounding errors by adjusting the coordinate with the largest change
      const dq = Math.abs(q - startQ);
      const dr = Math.abs(r - startR);
      const ds = Math.abs(s - startS);
      
      if (dq > dr && dq > ds) {
        hexes.push({q: -r-s, r: r});
      } else if (dr > ds) {
        hexes.push({q: q, r: -q-s});
      } else {
        hexes.push({q: q, r: r});
      }
    } else {
      hexes.push({q: q, r: r});
    }
  }
  
  return hexes;
}


function showStoryOverlay(storyLines, onCloseCallback) {
  // Create (or reuse) the full-screen overlay
  let overlay = document.getElementById("story-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "story-overlay";
    overlay.style.position = "fixed";
    overlay.style.top = "0";
    overlay.style.left = "0";
    overlay.style.width = "100%";
    overlay.style.height = "100%";
    overlay.style.backgroundColor = "rgba(0, 0, 0, 0.85)";
    overlay.style.color = "#fff";
    overlay.style.fontFamily = "Arial, sans-serif";
    overlay.style.fontSize = "18px";
    overlay.style.padding = "40px";
    overlay.style.overflowY = "auto";
    overlay.style.zIndex = "1000";
    document.body.appendChild(overlay);
  } else {
    overlay.innerHTML = "";
    overlay.style.display = "block";
  }
  
  // Create a container for the story text.
  const textContainer = document.createElement("div");
  textContainer.id = "story-text";
  overlay.appendChild(textContainer);
  
  let lineIndex = 0;
  let storyActive = true; // Flag to control printing
  
  function showNextLine() {
    if (!storyActive) return;
    if (lineIndex < storyLines.length) {
      const lineP = document.createElement("p");
      lineP.textContent = storyLines[lineIndex];
      textContainer.appendChild(lineP);
      lineIndex++;
      overlay.scrollTop = overlay.scrollHeight;
      setTimeout(showNextLine, 2000);
    } else {
      // All lines printed; create a Close button.
      const closeBtn = document.createElement("button");
      closeBtn.textContent = "Close";
      closeBtn.style.marginTop = "20px";
      closeBtn.style.padding = "10px 20px";
      closeBtn.style.fontSize = "16px";
      closeBtn.addEventListener("click", () => {
        storyActive = false; // Stop further printing
        overlay.style.display = "none";
        if (typeof onCloseCallback === "function") {
          onCloseCallback();
        }
      });
      textContainer.appendChild(closeBtn);
    }
  }
  
  showNextLine();
}

// Add back the setupPlayerControls function
function setupPlayerControls(scenario) {
  const playerPiecesList = document.getElementById("player-pieces");
  playerPiecesList.innerHTML = ""; // Clear existing
  pieceSelections.clear(); // Clear the global map

  // Filter for player pieces
  const playerPieces = scenario.pieces.filter(p => p.side === "player");

  // Create action selection list
  playerPieces.forEach(piece => {
    const li = document.createElement("li");
    li.className = "piece-item";
    
    // Create piece label with color circle
    const labelDiv = document.createElement("div");
    labelDiv.className = "piece-label";
    
    const colorSpan = document.createElement("span");
    colorSpan.className = "piece-color";
    colorSpan.style.backgroundColor = piece.color;
    
    // Create a unique identifier for the piece by combining class and position
    const uniqueLabel = `${piece.class}_${piece.q}_${piece.r}`;
    const labelSpan = document.createElement("span");
    labelSpan.textContent = `${piece.class} (${piece.label})`;
    
    // Add expand button
    const expandButton = document.createElement("button");
    expandButton.className = "piece-info-button";
    expandButton.textContent = "ℹ️";
    expandButton.title = "Show piece information";
    
    labelDiv.appendChild(colorSpan);
    labelDiv.appendChild(labelSpan);
    labelDiv.appendChild(expandButton);
    
    // Create action select dropdown
    const select = document.createElement("select");
    select.className = "action-select";
    
    // Add pass option first (default)
    const passOption = document.createElement("option");
    passOption.value = "pass";
    passOption.textContent = "Pass";
    select.appendChild(passOption);
    
    // Get piece class data and add its actions
    const pieceClass = piecesData.classes[piece.class];
    if (pieceClass && pieceClass.actions) {
      Object.entries(pieceClass.actions).forEach(([actionName, actionData]) => {
        const option = document.createElement("option");
        option.value = actionName;
        option.textContent = actionName.charAt(0).toUpperCase() + actionName.slice(1);
        select.appendChild(option);
      });
    }

    // Create hex selection text box
    const hexSelect = document.createElement("div");
    hexSelect.className = "hex-select";
    hexSelect.textContent = "Click to select hex";
    hexSelect.setAttribute("data-piece-label", uniqueLabel);
    
    // Create and add info section
    const infoSection = createPieceInfoSection(piece, pieceClass);
    
    // Add expand button click handler
    expandButton.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent piece item click
      infoSection.classList.toggle("visible");
      expandButton.textContent = infoSection.classList.contains("visible") ? "🔼" : "ℹ️";
    });

    // Initialize piece selection tracking with unique label
    pieceSelections.set(uniqueLabel, {
      class: piece.class,
      action: "pass",
      description: "",
      targetHex: null,
      originalLabel: piece.label // Store the original label for display
    });

    // Handle action selection
    select.addEventListener("change", (e) => {
      const actionName = e.target.value;
      const selection = pieceSelections.get(uniqueLabel);
      
      if (actionName === "pass") {
        selection.action = "pass";
        selection.description = "";
        selection.targetHex = null;
        hexSelect.textContent = "Click to select hex";
        hexSelect.style.display = "none";
      } else if (actionName && pieceClass.actions[actionName]) {
        const actionData = pieceClass.actions[actionName];
        selection.action = actionName;
        selection.description = actionData.description;
        selection.targetHex = null; // Reset target when action changes
        selection.targetHexes = null; // Reset multi-target list
        selection.affectedHexes = null; // Reset AOE affected hexes

        // Show hex selector for any action that needs target selection
        const needsTargetSelection = ['move', 'swap_position', 'single_target_attack', 'multi_target_attack', 'aoe', 'pull', 'push', 'trap'].includes(actionData.action_type);
        hexSelect.style.display = needsTargetSelection ? "block" : "none";
        hexSelect.textContent = "Click to select hex";
        
        // Reset pull-specific state
        if (actionData.action_type === 'pull') {
          selection.pullTarget = null;
        }
        // Reset push-specific state
        if (actionData.action_type === 'push') {
          selection.pushTarget = null;
        }
      }
      
      // Clear any existing range indicators
      document.querySelectorAll(".hex-region.in-range").forEach(hex => {
        hex.classList.remove("in-range");
        hex.classList.remove("attack");
      });
      
      updateActionDescriptions();
      validateTurnCompletion();
    });

    // Add hex selection click handler
    hexSelect.addEventListener("click", () => {
      console.log("Hex selection clicked"); // Debug log
      
      // Clear any existing hex selection mode and ranges
      if (currentHexSelector) {
        console.log("Clearing previous selection"); // Debug log
        currentHexSelector.classList.remove("selecting");
        currentHexSelector.textContent = "Click to select hex";
        document.querySelectorAll(".hex-region.in-range, .hex-region.attack, .hex-region.destination").forEach(hex => {
          hex.classList.remove("in-range");
          hex.classList.remove("attack");
          hex.classList.remove("destination");
        });
      }
      
      // Enter hex selection mode
      isSelectingHex = true;
      currentHexSelector = hexSelect;
      hexSelect.classList.add("selecting");
      hexSelect.textContent = "Selecting...";

      // Find the piece's current position and show range
      const pieceLabel = hexSelect.getAttribute("data-piece-label");
      console.log("Selected piece:", pieceLabel); // Debug log
      
      const piece = scenario.pieces.find(p => `${p.class}_${p.q}_${p.r}` === pieceLabel);
      const pieceClass = piecesData.classes[piece.class];
      const selection = pieceSelections.get(pieceLabel);
      const actionData = pieceClass.actions[selection.action];
      
      console.log("Piece data:", { piece, action: selection.action, actionData }); // Debug log
      
      if (piece && pieceClass && actionData) {
        if (actionData.action_type === 'pull' && selection.pullTarget) {
          // If we're selecting a destination for a pull action, show destination hexes
          const targetPiece = scenario.pieces.find(p => 
            p.q === selection.pullTarget.q && 
            p.r === selection.pullTarget.r
          );
          
          if (targetPiece) {
            const distance = actionData.distance;
            for (let q = -distance; q <= distance; q++) {
              for (let r = -distance; r <= distance; r++) {
                if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * distance) {
                  const destQ = targetPiece.q + q;
                  const destR = targetPiece.r + r;
                  const destKey = `${destQ},${destR}`;
                  
                  // Only show unoccupied hexes that are closer to the pulling piece
                  const isUnoccupied = !scenario.pieces.some(p => p.q === destQ && p.r === destR);
                  const isCloser = Math.abs(destQ - piece.q) + Math.abs(destR - piece.r) < 
                                 Math.abs(targetPiece.q - piece.q) + Math.abs(targetPiece.r - piece.r);
                  
                  if (isUnoccupied && isCloser && !blockedHexes.has(destKey)) {
                    const destHex = document.querySelector(`polygon[data-q="${destQ}"][data-r="${destR}"]`);
                    if (destHex) {
                      destHex.classList.add("in-range");
                      destHex.classList.add("destination");
                    }
                  }
                }
              }
            }
          }
        } else if (actionData.action_type === 'push' && selection.pushTarget) {
          // If we're selecting a destination for a push action, show destination hexes
          const targetPiece = scenario.pieces.find(p => 
            p.q === selection.pushTarget.q && 
            p.r === selection.pushTarget.r
          );
          
          if (targetPiece) {
            const distance = actionData.distance;
            for (let q = -distance; q <= distance; q++) {
              for (let r = -distance; r <= distance; r++) {
                if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * distance) {
                  const destQ = targetPiece.q + q;
                  const destR = targetPiece.r + r;
                  const destKey = `${destQ},${destR}`;
                  
                  // Only show unoccupied hexes that are further from the pushing piece
                  const isUnoccupied = !scenario.pieces.some(p => p.q === destQ && p.r === destR);
                  const isFurther = Math.abs(destQ - piece.q) + Math.abs(destR - piece.r) > 
                               Math.abs(targetPiece.q - piece.q) + Math.abs(targetPiece.r - piece.r);
                  
                  if (isUnoccupied && isFurther && !blockedHexes.has(destKey)) {
                    const destHex = document.querySelector(`polygon[data-q="${destQ}"][data-r="${destR}"]`);
                    if (destHex) {
                      destHex.classList.add("in-range");
                      destHex.classList.add("destination");
                      // Add click handler for destination selection
                      destHex.addEventListener("click", () => {
                        selection.targetHex = { q: destQ, r: destR };
                        currentHexSelector.textContent = "Destination selected";
                        currentHexSelector.classList.remove("selecting");
                        isSelectingHex = false;
                        validateTurnCompletion();
                      });
                    }
                  }
                }
              }
            }
          }
        } else {
          // For all other cases, use the normal range highlighting
          showPieceActionRange(piece, pieceClass, selection.action);
        }
      }
      validateTurnCompletion();
    });

    // Initially hide hex select
    hexSelect.style.display = "none";
    
    li.appendChild(labelDiv);
    li.appendChild(select);
    li.appendChild(hexSelect);
    li.appendChild(infoSection);
    playerPiecesList.appendChild(li);
  });

  // Initialize Sortable for drag-and-drop
  new Sortable(playerPiecesList, {
    animation: 150,
    ghostClass: 'sortable-ghost'
  });

  // Set up complete turn button handler with piece movement
  const completeTurnBtn = document.getElementById("complete-turn");
  completeTurnBtn.addEventListener("click", () => {
    // Only proceed if all moves are valid
    if (completeTurnBtn.disabled) {
      return;
    }

    turnCounter++; // Increment turn counter

    // Process delayed attacks first
    const remainingAttacks = [];
    for (const attack of delayedAttacks) {
        if (turnCounter >= attack.executionTurn) {
            // Check if attacker has moved
            const attacker = scenario.pieces.find(p => p.label === attack.attackerLabel);
            if (attacker.q !== attack.attackerQ || attacker.r !== attack.attackerR) {
                addBattleLog(`${attacker.class} (${attack.attackerLabel})'s cast is canceled due to movement`);
                continue;
            }

            // Handle different attack types
            if (attack.type === 'single_target_attack' || attack.type === 'multi_target_attack') {
                // For single/multi target attacks, check if target(s) moved
                const targets = Array.isArray(attack.targets) ? attack.targets : [attack.targets];
                let targetMoved = false;

                for (const target of targets) {
                    const targetPiece = scenario.pieces.find(p => 
                        p.q === target.originalQ && 
                        p.r === target.originalR && 
                        p.side !== 'player'
                    );

                    if (!targetPiece || targetPiece.q !== target.originalQ || targetPiece.r !== target.originalR) {
                        addBattleLog(`${attacker.class} (${attack.attackerLabel})'s attack missed - target moved`);
                        targetMoved = true;
                        break;
                    }
                }

                if (!targetMoved) {
                    // Execute the attack
                    for (const target of targets) {
                        const targetIndex = scenario.pieces.findIndex(p => 
                            p.q === target.originalQ && 
                            p.r === target.originalR && 
                            p.side !== 'player'
                        );
                        
                        if (targetIndex !== -1) {
                            const removedPiece = scenario.pieces[targetIndex];
                            scenario.pieces.splice(targetIndex, 1);
                            addBattleLog(`${attacker.class} (${attack.attackerLabel})'s delayed ${attack.actionName} eliminated ${removedPiece.class} (${removedPiece.label})`);
                        }
                    }
                }
            } else if (attack.type === 'aoe') {
                // For AOE, check which pieces are still in the affected area
                const affectedPieces = [];
                const radius = attack.radius;
                
                for (let q = -radius; q <= radius; q++) {
                    for (let r = -radius; r <= radius; r++) {
                        if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * radius) {
                            const targetQ = attack.centerQ + q;
                            const targetR = attack.centerR + r;
                            
                            const targetPiece = scenario.pieces.find(p => 
                                p.q === targetQ && 
                                p.r === targetR && 
                                p.side !== 'player'
                            );
                            
                            if (targetPiece) {
                                affectedPieces.push(targetPiece);
                            }
                        }
                    }
                }

                // Remove affected pieces
                for (const piece of affectedPieces) {
                    const index = scenario.pieces.indexOf(piece);
                    if (index !== -1) {
                        scenario.pieces.splice(index, 1);
                        addBattleLog(`${attacker.class} (${attack.attackerLabel})'s delayed ${attack.actionName} eliminated ${piece.class} (${piece.label})`);
                    }
                }
            }
        } else {
            remainingAttacks.push(attack);
        }
    }
    delayedAttacks = remainingAttacks;

    // Process current turn moves and attacks
    pieceSelections.forEach((selection, uniqueLabel) => {
        const [class_, q, r] = uniqueLabel.split('_');
        const piece = puzzleScenario.pieces.find(p => 
            p.class === class_ && 
            p.q === parseInt(q) && 
            p.r === parseInt(r)
        );
        if (!piece) return;
        
        const pieceClass = piecesData.classes[piece.class];
        const actionData = pieceClass.actions[selection.action];
        
        if (!actionData) return;

        switch (actionData.action_type) {
            case 'move':
                if (selection.targetHex) {
                    piece.q = selection.targetHex.q;
                    piece.r = selection.targetHex.r;
                    addBattleLog(`${piece.class} (${uniqueLabel}) moved to (${selection.targetHex.q}, ${selection.targetHex.r})`);
                }
                break;

            case 'swap_position':
                if (selection.targetHex) {
                    const targetPiece = scenario.pieces.find(p => 
                        p.q === selection.targetHex.q && 
                        p.r === selection.targetHex.r && 
                        p !== piece
                    );
                    
                    if (targetPiece) {
                        const originalQ = piece.q;
                        const originalR = piece.r;
                        piece.q = targetPiece.q;
                        piece.r = targetPiece.r;
                        targetPiece.q = originalQ;
                        targetPiece.r = originalR;
                        addBattleLog(`${piece.class} (${uniqueLabel}) swapped positions with ${targetPiece.class} (${targetPiece.label})`);
                    }
                }
                break;

            case 'single_target_attack':
            case 'multi_target_attack':
            case 'aoe':
                // Check if this action has a cast_speed
                if (actionData.cast_speed > 0) {
                    const executionTurn = turnCounter + actionData.cast_speed;
                    const attackInfo = {
                        type: actionData.action_type,
                        actionName: selection.action,
                        attackerLabel: uniqueLabel,
                        attackerQ: piece.q,
                        attackerR: piece.r,
                        executionTurn: executionTurn
                    };

                    if (actionData.action_type === 'single_target_attack') {
                        const target = scenario.pieces.find(p => 
                            p.q === selection.targetHex.q && 
                            p.r === selection.targetHex.r && 
                            p.side !== 'player'
                        );
                        if (target) {
                            attackInfo.targets = {
                                originalQ: selection.targetHex.q,
                                originalR: selection.targetHex.r,
                                label: target.label
                            };
                            addBattleLog(`${piece.class} (${uniqueLabel}) begins casting ${selection.action} on ${target.class} (${target.label})`);
                            delayedAttacks.push(attackInfo);
                        }
                    } else if (actionData.action_type === 'multi_target_attack') {
                        if (selection.targetHexes && selection.targetHexes.length > 0) {
                            attackInfo.targets = selection.targetHexes.map(hex => {
                                const target = scenario.pieces.find(p => 
                                    p.q === hex.q && 
                                    p.r === hex.r && 
                                    p.side !== 'player'
                                );
                                return target ? {
                                    originalQ: hex.q,
                                    originalR: hex.r,
                                    label: target.label
                                } : null;
                            }).filter(t => t !== null);
                            
                            if (attackInfo.targets.length > 0) {
                                const targetLabels = attackInfo.targets.map(t => t.label).join(', ');
                                addBattleLog(`${piece.class} (${uniqueLabel}) begins casting ${selection.action} on targets: ${targetLabels}`);
                                delayedAttacks.push(attackInfo);
                            }
                        }
                    } else if (actionData.action_type === 'aoe') {
                        attackInfo.centerQ = selection.targetHex.q;
                        attackInfo.centerR = selection.targetHex.r;
                        attackInfo.radius = actionData.radius;
                        addBattleLog(`${piece.class} (${uniqueLabel}) begins casting ${selection.action} centered at (${selection.targetHex.q}, ${selection.targetHex.r})`);
                        delayedAttacks.push(attackInfo);
                    }
                } else {
                    // Handle immediate attacks
                    if (actionData.action_type === 'single_target_attack' && selection.targetHex) {
                        const targetIndex = scenario.pieces.findIndex(p => 
                            p.q === selection.targetHex.q && 
                            p.r === selection.targetHex.r && 
                            p.side !== 'player'
                        );
                        
                        if (targetIndex !== -1) {
                            const removedPiece = scenario.pieces[targetIndex];
                            scenario.pieces.splice(targetIndex, 1);
                            addBattleLog(`${piece.class} (${uniqueLabel}) eliminated ${removedPiece.class} (${removedPiece.label}) with ${selection.action}`);
                        }
                    } else if (actionData.action_type === 'multi_target_attack' && selection.targetHexes) {
                        selection.targetHexes.forEach(targetHex => {
                            const targetIndex = scenario.pieces.findIndex(p => 
                                p.q === targetHex.q && 
                                p.r === targetHex.r && 
                                p.side !== 'player'
                            );
                            
                            if (targetIndex !== -1) {
                                const removedPiece = scenario.pieces[targetIndex];
                                scenario.pieces.splice(targetIndex, 1);
                                addBattleLog(`${piece.class} (${uniqueLabel}) eliminated ${removedPiece.class} (${removedPiece.label}) with ${selection.action}`);
                            }
                        });
                    } else if (actionData.action_type === 'aoe' && selection.affectedHexes) {
                        selection.affectedHexes.forEach(targetHex => {
                            const targetIndex = scenario.pieces.findIndex(p => 
                                p.q === targetHex.q && 
                                p.r === targetHex.r && 
                                p.side !== 'player'
                            );
                            
                            if (targetIndex !== -1) {
                                const removedPiece = scenario.pieces[targetIndex];
                                scenario.pieces.splice(targetIndex, 1);
                                addBattleLog(`${piece.class} (${uniqueLabel}) eliminated ${removedPiece.class} (${removedPiece.label}) with ${selection.action}`);
                            }
                        });
                    }
                }
                break;

            case 'push':
                if (selection.pushTarget && selection.targetHex) {
                    const targetPiece = scenario.pieces.find(p => 
                        p.q === selection.pushTarget.q && 
                        p.r === selection.pushTarget.r
                    );
                    
                    if (targetPiece) {
                        targetPiece.q = selection.targetHex.q;
                        targetPiece.r = selection.targetHex.r;
                        addBattleLog(`${piece.class} (${uniqueLabel}) pushed ${targetPiece.class} (${targetPiece.label}) to (${selection.targetHex.q}, ${selection.targetHex.r})`);
                    }
                }
                break;

            case 'pull':
                if (selection.pullTarget && selection.targetHex) {
                    const targetPiece = scenario.pieces.find(p => 
                        p.q === selection.pullTarget.q && 
                        p.r === selection.pullTarget.r
                    );
                    
                    if (targetPiece) {
                        const originalQ = targetPiece.q;
                        const originalR = targetPiece.r;
                        targetPiece.q = selection.targetHex.q;
                        targetPiece.r = selection.targetHex.r;
                        addBattleLog(`${piece.class} (${uniqueLabel}) pulled ${targetPiece.class} (${targetPiece.label}) from (${originalQ},${originalR}) to (${targetPiece.q},${targetPiece.r})`);
                    }
                }
                break;

            case 'push':
                if (!selection.pushTarget || !selection.targetHex) {
                    console.log(`${uniqueLabel} has no push target or destination`);
                    isValid = false;
                    break;
                }

                // Check if push target is in range
                const pushDx = Math.abs(selection.pushTarget.q - piece.q);
                const pushDy = Math.abs(selection.pushTarget.r - piece.r);
                const pushDz = Math.abs(-selection.pushTarget.q - selection.pushTarget.r + piece.q + piece.r);
                const pushDistance = Math.max(pushDx, pushDy, pushDz);
                
                if (pushDistance > actionData.range) {
                    console.log(`${uniqueLabel} push target is out of range`);
                    isValid = false;
                    break;
                }

                // Check if destination is within push distance from the target piece
                const pushDestDx = Math.abs(selection.targetHex.q - selection.pushTarget.q);
                const pushDestDy = Math.abs(selection.targetHex.r - selection.pushTarget.r);
                const pushDestDz = Math.abs(-selection.targetHex.q - selection.targetHex.r + selection.pushTarget.q + selection.pushTarget.r);
                const pushDestDistance = Math.max(pushDestDx, pushDestDy, pushDestDz);
                
                if (pushDestDistance > actionData.distance) {
                    console.log(`${uniqueLabel} push destination is out of push range`);
                    isValid = false;
                    break;
                }

                // Check if destination is occupied
                const isPushDestOccupied = puzzleScenario.pieces.some(p => 
                    p.q === selection.targetHex.q && p.r === selection.targetHex.r
                );
                if (isPushDestOccupied) {
                    console.log(`${uniqueLabel} push destination is occupied`);
                    isValid = false;
                }
                break;

            case 'trap':
                if (!selection.targetHex) {
                    console.log(`${uniqueLabel} has no target hex`);
                    isValid = false;
                    break;
                }

                // Check if target hex is blocked by a non-trap obstacle
                const trapTargetKey = `${selection.targetHex.q},${selection.targetHex.r}`;
                const isBlockedByNonTrap = puzzleScenario.blockedHexes.some(h => 
                    h.q === selection.targetHex.q && 
                    h.r === selection.targetHex.r &&
                    !h.isTrap
                );
                
                if (isBlockedByNonTrap) {
                    console.log(`${uniqueLabel} trap target hex is blocked by non-trap obstacle`);
                    isValid = false;
                    break;
                }
                break;
        }
    });

    pieceSelections.clear();
    drawHexDetailView(currentRegion, currentSection);
    enemyTakePPOAction();
    // enemyTurnFull();

  });

  // Initial validation
  validateTurnCompletion();

  // Initial update of descriptions
  updateActionDescriptions();
  
  return pieceSelections; // Return this so we can use it in hex click handlers
}

async function enemyTakePPOAction() {
  if (!puzzleScenario) {
    console.warn("No puzzle scenario to act on.");
    return;
  }

  console.log("=== Enemy attempting 1 PPO action ===");

  // Prepare request
  const bodyData = {
    scenario: puzzleScenario,
    approach: "ppo"
  };

  let response;
  try {
    response = await fetch("/api/enemy_action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bodyData)
    });
  } catch (err) {
    console.error("Network or server error calling /api/enemy_action:", err);
    return;
  }

  if (!response.ok) {
    console.error("Server responded with non-OK status:", response.status);
    return;
  }

  const result = await response.json();
  if (result.error) {
    // e.g. "No valid actions for enemy side."
    console.log("Enemy cannot act:", result.error);
    return;
  }

  // Apply that single action
  applyEnemyActionToScenario(result);

  // If the puzzle scenario flips to player or is done, we won't repeat
  if (puzzleScenario.turn_side === "player") {
    console.log("Enemy turn is done (player's turn now)!");
  }

  // Re-draw board
  drawHexDetailView(currentRegion, currentSection);

  console.log("=== One PPO action complete ===");
}

/**
 * Takes the action returned by the server and updates puzzleScenario in place,
 * so that the enemy piece is moved or does the correct attack, etc.
 */
function applyEnemyActionToScenario(actionResult) {
  const subAction = actionResult.sub_action;
  const pieceLabel = actionResult.piece_label;
  
  const piece = puzzleScenario.pieces.find(p => p.label === pieceLabel);
  if (!piece) {
    console.warn("Enemy piece not found:", pieceLabel);
    return;
  }

  // Mark that piece as having moved:
  piece.moved_this_turn = true;

  switch (subAction.type) {
    case "move":
      {
        const tq = subAction.dest[0];
        const tr = subAction.dest[1];
        
        // Check if there's a trap at the destination
        if (puzzleScenario.traps) {
          const trapIndex = puzzleScenario.traps.findIndex(t => t.q === tq && t.r === tr);
          if (trapIndex !== -1) {
            const trap = puzzleScenario.traps[trapIndex];
            
            // Apply trap effect
            if (trap.effect === "immobilize") {
              piece.immobilized = true;
              piece.immobilizedTurns = trap.duration;
              addBattleLog(`${piece.class} (${piece.label}) stepped on a trap and is immobilized for ${trap.duration} turns`);
            }
            
            // Remove the trap after it's triggered
            puzzleScenario.traps.splice(trapIndex, 1);
          }
        }
        
        piece.q = tq;
        piece.r = tr;
        addBattleLog(`Enemy ${piece.class} (${piece.label}) moved to (${tq},${tr})`);
      }
      break;

    case "pass":
      addBattleLog(`Enemy ${piece.class} (${piece.label}) passed`);
      break;

    case "swap_position":
      {
        const tgt = subAction.target_piece;
        if (tgt) {
          // find the target in puzzleScenario
          const swapPiece = puzzleScenario.pieces.find(pp => pp.label === tgt.label);
          if (swapPiece) {
            // swap coords
            const oldQ = piece.q, oldR = piece.r;
            piece.q = swapPiece.q; piece.r = swapPiece.r;
            swapPiece.q = oldQ; swapPiece.r = oldR;
            addBattleLog(`Enemy ${piece.class} (${piece.label}) swapped with ${swapPiece.class} (${swapPiece.label})`);
          }
        }
      }
      break;

    case "single_target_attack":
      {
        const tgt = subAction.target_piece;
        if (tgt) {
          // remove that piece from scenario, or mark dead
          const idx = puzzleScenario.pieces.findIndex(pp => pp.label === tgt.label);
          if (idx >= 0) {
            const removed = puzzleScenario.pieces[idx];
            // Just remove it from array for simplicity
            puzzleScenario.pieces.splice(idx, 1);
            addBattleLog(`Enemy ${piece.class} (${piece.label}) killed ${removed.class} (${removed.label})`);
          }
        }
      }
      break;

    case "multi_target_attack":
      {
        const targets = subAction.targets || [];
        targets.forEach(tgt => {
          // remove each piece
          const idx = puzzleScenario.pieces.findIndex(pp => pp.label === tgt.label);
          if (idx >= 0) {
            const removed = puzzleScenario.pieces[idx];
            puzzleScenario.pieces.splice(idx, 1);
            addBattleLog(`Enemy ${piece.class} (${piece.label}) multi-attack killed ${removed.class} (${removed.label})`);
          }
        });
      }
      break;

    case "aoe":
      {
        // If subAction.targets is an array of targets, kill them
        const arr = subAction.targets || [];
        arr.forEach(tgt => {
          const idx = puzzleScenario.pieces.findIndex(pp => pp.label === tgt.label);
          if (idx >= 0) {
            const removed = puzzleScenario.pieces[idx];
            puzzleScenario.pieces.splice(idx, 1);
            addBattleLog(`Enemy ${piece.class} (${piece.label}) AOE killed ${removed.class} (${removed.label})`);
          }
        });
      }
      break;

    case "push":
      {
        const tgt = subAction.target_piece;
        if (tgt) {
          // find the target in puzzleScenario
          const pushPiece = puzzleScenario.pieces.find(pp => pp.label === tgt.label);
          if (pushPiece) {
            // Move the target piece to the destination
            const oldQ = pushPiece.q, oldR = pushPiece.r;
            pushPiece.q = subAction.dest[0];
            pushPiece.r = subAction.dest[1];
            addBattleLog(`Enemy ${piece.class} (${piece.label}) pushed ${pushPiece.class} (${pushPiece.label}) from (${oldQ},${oldR}) to (${pushPiece.q},${pushPiece.r})`);
          }
        }
      }
      break;

    case "trap":
      {
        // Add trap to the scenario
        if (!puzzleScenario.traps) {
          puzzleScenario.traps = [];
        }
        
        const trap = {
          q: subAction.q,
          r: subAction.r,
          effect: subAction.effect,
          duration: subAction.duration,
          caster: piece.class,
          caster_side: piece.side
        };
        
        puzzleScenario.traps.push(trap);
        addBattleLog(`Enemy ${piece.class} (${piece.label}) set a ${trap.effect} trap at (${trap.q},${trap.r})`);
      }
      break;

    default:
      console.log("Unknown sub_action.type:", subAction.type);
      break;
  }

  // Now check if all enemy pieces moved
  const enemies = puzzleScenario.pieces.filter(pp => pp.side === "enemy" && !pp.dead);
  const allMoved = enemies.every(pp => pp.moved_this_turn);
  if (allMoved) {
    // End the enemy turn, switch to player
    puzzleScenario.turn_side = "player";
    // Reset flags so next round of enemy moves will be allowed
    enemies.forEach(pp => {
      pp.moved_this_turn = false;
    });
  }

  // If your puzzle scenario needs any other updates (like if a piece is "dead" but not removed),
  // do it here. Then your scenario is consistent for the next re-draw.
}

async function pollActions() {
  try {
    const resp = await fetch('/api/get_actions');
    if (!resp.ok) {
      console.error('Failed to fetch actions:', resp.status, resp.statusText);
      return;
    }
    const data = await resp.json();
    
    if (data.error) {
      console.error('Error from server:', data.error);
      return;
    }

    // Process each action
    data.forEach(actionObj => {
      if (actionObj.type === 'enemy_action') {
        const { piece_label, sub_action } = actionObj.data;
        let actionMsg = `${piece_label} performs `;
        
        if (sub_action.type === 'move') {
          actionMsg += `move to (${sub_action.dest[0]}, ${sub_action.dest[1]})`;
        } else if (sub_action.type === 'attack') {
          actionMsg += `${sub_action.action_name} on ${sub_action.target_piece.label}`;
        } else if (sub_action.type === 'push') {
          actionMsg += `push on ${sub_action.target_piece.label}`;
        } else if (sub_action.type === 'pass') {
          actionMsg += 'pass';
        } else {
          actionMsg += JSON.stringify(sub_action);
        }
        
        addBattleLog(actionMsg);
      }
    });
  } catch (err) {
    console.error('Error polling actions:', err);
  }
}

// Poll every 3 seconds, but add exponential backoff on errors
let pollInterval = 3000;
let errorCount = 0;
const MAX_ERRORS = 5;

function startPolling() {
  pollActions();
  setTimeout(() => {
    if (errorCount >= MAX_ERRORS) {
      console.error('Too many polling errors, stopping');
      return;
    }
    startPolling();
  }, pollInterval);
}

// Start polling when the page loads
startPolling();