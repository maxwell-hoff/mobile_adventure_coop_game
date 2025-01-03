let worldData = null;
let piecesData = null;
let currentView = "world"; // "world", "region", or "section"
let currentRegion = null;
let currentSection = null;
let isSelectingHex = false;
let currentHexSelector = null;
let pieceSelections = new Map(); // Make this global

const HEX_SIZE = 30; // radius of each hex
const SQRT3 = Math.sqrt(3);

// We'll assume the <svg> is 800x600
const SVG_WIDTH = 800;
const SVG_HEIGHT = 600;

/** Axial -> pixel (pointy-top). */
function axialToPixel(q, r) {
  const x = HEX_SIZE * SQRT3 * (q + r/2);
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

/** 
 * For a single hex (q,r), return an array of edges, 
 * each edge is [ [x1, y1], [x2, y2] ] in pixel coords.
 */
function getHexEdges(q, r) {
  const center = axialToPixel(q, r);
  let edges = [];
  let corners = [];
  // compute 6 corners
  for (let i = 0; i < 6; i++) {
    let angle_deg = 60*i + 30;
    let rad = Math.PI/180 * angle_deg;
    let px = center.x + HEX_SIZE * Math.cos(rad);
    let py = center.y + HEX_SIZE * Math.sin(rad);
    corners.push({ x:px, y:py });
  }
  // Each edge is corner[i] -> corner[i+1], wrapping at i=5
  for (let i = 0; i < 6; i++) {
    let c1 = corners[i];
    let c2 = corners[(i+1) % 6];
    edges.push([[c1.x, c1.y],[c2.x, c2.y]]);
  }
  return edges;
}

/** 
 * Return axial neighbors for pointy-top in axial coords 
 * (q+/-1, r?), etc. 
 */
function getHexNeighbors(q, r) {
  // Standard pointy axial neighbors
  return [
    {q: q+1, r: r},
    {q: q-1, r: r},
    {q: q,   r: r+1},
    {q: q,   r: r-1},
    {q: q+1, r: r-1},
    {q: q-1, r: r+1}
  ];
}

// =========== BOUNDING BOX + CENTERING ===========

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
  const data = await resp.json();
  worldData = data.world;
  piecesData = data.pieces;
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

  // Re-add hover label
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
  svg.appendChild(gWorld);

  // For bounding box
  let worldHexList = [];

  // For each region, draw hexes + build perimeter
  worldData.regions.forEach(region => {
    let sumX=0, sumY=0, count=0;
    // We'll store region's hex coords in a set for quick adjacency check
    let regionSet = new Set();
    region.worldHexes.forEach(h => regionSet.add(`${h.q},${h.r}`));

    // 1) Draw each hex
    region.worldHexes.forEach(hex => {
      worldHexList.push(hex);
      const { x, y } = axialToPixel(hex.q, hex.r);
      sumX += x; sumY += y; count++;

      // Draw polygon
      let p = document.createElementNS("http://www.w3.org/2000/svg","polygon");
      p.setAttribute("class","hex-region");
      p.setAttribute("points", hexPolygonPoints(x,y));
      p.setAttribute("fill", regionColor(region.regionId));

      p.addEventListener("mouseenter", () => { hoverLabel.textContent = region.name; });
      p.addEventListener("mouseleave", () => { hoverLabel.textContent = ""; });
      p.addEventListener("click", () => {
        currentRegion = region;
        drawRegionView(region);
      });
      gWorld.appendChild(p);
    });

    // // 2) Put region label near center
    // if(count>0){
    //   let centerX = sumX/count, centerY = sumY/count;
    //   let lbl = document.createElementNS("http://www.w3.org/2000/svg","text");
    //   lbl.setAttribute("x",centerX);
    //   lbl.setAttribute("y",centerY);
    //   lbl.setAttribute("text-anchor","middle");
    //   lbl.setAttribute("fill","#333");
    //   lbl.setAttribute("font-size","10");
    //   lbl.textContent = region.name;
    //   gWorld.appendChild(lbl);
    // }

    // 3) Outline the perimeter edges only
    // For each hex in region, get its 6 edges in pixel coords
    // Check if that edge is shared with a neighbor => skip if shared
    region.worldHexes.forEach(hex => {
      let edges = getHexEdges(hex.q, hex.r);
      // For adjacency check
      let neighbors = getHexNeighbors(hex.q, hex.r);

      edges.forEach(edge => {
        // edge is [ [x1,y1], [x2,y2] ]
        // We find the neighbor that would share this edge
        // We do so by checking if the neighbor is in regionSet
        // and if the neighbor is the *correct* one for that edge, 
        // but simpler approach: if *any* neighbor is in regionSet with same edge, skip
        // We can do a small trick: if that neighbor's coords match the direction.

        let shared = false;
        for(let n of neighbors){
          if(regionSet.has(`${n.q},${n.r}`)){
            // => that neighbor is in region
            // Now: does that neighbor actually share this edge in pixel space?
            // We can compute neighbor's corners and see if it has the same edge
            let nEdges = getHexEdges(n.q, n.r);
            // If any nEdge matches our current edge (allowing reversed coords),
            // then it's shared.
            if(isEdgeShared(edge, nEdges)){
              shared = true;
              break;
            }
          }
        }
        if(!shared){
          // draw a line for this edge
          let line = document.createElementNS("http://www.w3.org/2000/svg","line");
          line.setAttribute("x1", edge[0][0]);
          line.setAttribute("y1", edge[0][1]);
          line.setAttribute("x2", edge[1][0]);
          line.setAttribute("y2", edge[1][1]);
          line.setAttribute("stroke","black");
          line.setAttribute("stroke-width","2");
          line.setAttribute("class","region-outline");
          gWorld.appendChild(line);
        }
      });
    });
  });

  // center the whole world group
  centerHexGroup(worldHexList, gWorld, axialToPixel, {
    svgWidth: SVG_WIDTH,
    svgHeight: SVG_HEIGHT,
    scale: 1,
    rotation: 0
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

  let regionHexList = [];
  let regionSet = new Set();
  region.worldHexes.forEach(h => regionSet.add(`${h.q},${h.r}`));

  // Draw hexes
  region.worldHexes.forEach(hex => {
    regionHexList.push(hex);
    let {x,y} = axialToPixel(hex.q,hex.r);

    let poly = document.createElementNS("http://www.w3.org/2000/svg","polygon");
    poly.setAttribute("class","hex-region");
    poly.setAttribute("points",hexPolygonPoints(x,y));
    poly.setAttribute("fill",regionColor(region.regionId));
    poly.addEventListener("mouseenter",()=>{ hoverLabel.textContent=region.name; });
    poly.addEventListener("mouseleave",()=>{ hoverLabel.textContent=""; });
    poly.addEventListener("click",()=>{ drawHexDetailView(region,hex); });
    gRegion.appendChild(poly);
  });

  // Outline perimeter edges in region view
  region.worldHexes.forEach(hex => {
    let edges = getHexEdges(hex.q, hex.r);
    let neighbors = getHexNeighbors(hex.q, hex.r);
    edges.forEach(edge=>{
      let shared=false;
      for(let n of neighbors){
        if(regionSet.has(`${n.q},${n.r}`)){
          // neighbor is in region => check if it shares edge
          let nEdges = getHexEdges(n.q,n.r);
          if(isEdgeShared(edge, nEdges)){
            shared=true; 
            break;
          }
        }
      }
      if(!shared){
        let line = document.createElementNS("http://www.w3.org/2000/svg","line");
        line.setAttribute("x1",edge[0][0]);
        line.setAttribute("y1",edge[0][1]);
        line.setAttribute("x2",edge[1][0]);
        line.setAttribute("y2",edge[1][1]);
        line.setAttribute("stroke","black");
        line.setAttribute("stroke-width","2");
        line.setAttribute("class","region-outline");
        gRegion.appendChild(line);
      }
    });
  });

  // center region
  centerHexGroup(regionHexList, gRegion, axialToPixel, {
    svgWidth: SVG_WIDTH,
    svgHeight: SVG_HEIGHT,
    scale: 2,
    rotation: 0
  });
}

/**
 * NEW SECTION VIEW
 */
function drawHexDetailView(region, clickedHex) {
  currentView = "section";
  currentSection = clickedHex;

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

  const SUB_GRID_RADIUS = 5;
  const SUB_HEX_SIZE = 10;

  function subAxialToPixel(q, r){
    let x = SUB_HEX_SIZE * SQRT3 * (q + r/2);
    let y = SUB_HEX_SIZE * (3/2)*r;
    return {x,y};
  }
  function subHexPolygonPoints(cx,cy){
    let pts=[];
    for(let i=0;i<6;i++){
      let deg=60*i+30;
      let rad=Math.PI/180*deg;
      let px=cx+SUB_HEX_SIZE*Math.cos(rad);
      let py=cy+SUB_HEX_SIZE*Math.sin(rad);
      pts.push(`${px},${py}`);
    }
    return pts.join(" ");
  }

  // Find matching puzzle scenario for this hex
  let puzzleScenario = null;
  if (region.puzzleScenarios) {
    puzzleScenario = region.puzzleScenarios.find(ps => 
      ps.triggerHex.q === clickedHex.q && ps.triggerHex.r === clickedHex.r
    );
  }

  // Show/hide player controls based on whether this is a puzzle scenario
  const playerControls = document.getElementById("player-controls");
  if (puzzleScenario) {
    playerControls.style.display = "block";
    setupPlayerControls(puzzleScenario);
  } else {
    playerControls.style.display = "none";
  }

  // If we found a puzzle scenario, use its radius and blocked hexes
  const gridRadius = puzzleScenario ? puzzleScenario.subGridRadius : SUB_GRID_RADIUS;

  // build sub-hex coords
  let subHexList=[];
  for(let q=-gridRadius; q<=gridRadius;q++){
    for(let r=-gridRadius; r<=gridRadius; r++){
      if(Math.abs(q+r)<=gridRadius){
        subHexList.push({q,r});
      }
    }
  }

  // Create set of blocked hexes for quick lookup
  const blockedHexes = new Set();
  if (puzzleScenario && puzzleScenario.blockedHexes) {
    puzzleScenario.blockedHexes.forEach(h => {
      blockedHexes.add(`${h.q},${h.r}`);
    });
  }

  // Draw the hexes
  subHexList.forEach(sh => {
    let {x,y} = subAxialToPixel(sh.q,sh.r);
    let poly = document.createElementNS("http://www.w3.org/2000/svg","polygon");
    poly.setAttribute("class","hex-region");
    poly.setAttribute("points",subHexPolygonPoints(x,y));
    poly.setAttribute("data-q", sh.q);
    poly.setAttribute("data-r", sh.r);
    
    // If hex is blocked in puzzle scenario, make it darker
    if (blockedHexes.has(`${sh.q},${sh.r}`)) {
      poly.setAttribute("fill", "#666");
    } else {
      poly.setAttribute("fill", regionColor(region.regionId));
    }
    
    poly.addEventListener("mouseenter",()=>{
      hoverLabel.textContent=`(q=${sh.q},r=${sh.r}) of ${region.name}`;
    });
    poly.addEventListener("mouseleave",()=>{hoverLabel.textContent="";});

    // Add click handler for hex selection
    poly.addEventListener("click", () => {
      if (isSelectingHex && currentHexSelector) {
        const pieceLabel = currentHexSelector.getAttribute("data-piece-label");
        const selection = pieceSelections.get(pieceLabel);
        
        // Check if the target hex is occupied
        const isOccupied = puzzleScenario.pieces.some(p => p.q === sh.q && p.r === sh.r);
        
        // Only allow selection if the hex is not occupied
        if (selection && !isOccupied) {
          selection.targetHex = { q: sh.q, r: sh.r };
          currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
          currentHexSelector.classList.remove("selecting");
          isSelectingHex = false;
          currentHexSelector = null;
          
          // Clear all move range highlights
          document.querySelectorAll(".hex-region.in-range").forEach(hex => {
            hex.classList.remove("in-range");
          });
          
          updateActionDescriptions();
        }
      }
    });

    gDetail.appendChild(poly);
  });

  // Draw pieces if we have a puzzle scenario
  if (puzzleScenario && puzzleScenario.pieces) {
    puzzleScenario.pieces.forEach(piece => {
      const {x,y} = subAxialToPixel(piece.q, piece.r);
      
      // Add piece circle
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", x);
      circle.setAttribute("cy", y);
      circle.setAttribute("r", SUB_HEX_SIZE * 0.6);
      circle.setAttribute("fill", piece.color || "#000");
      
      // Add hover behavior to show movement range
      circle.addEventListener("mouseenter", () => {
        // Get piece class data
        const pieceClass = piecesData.classes[piece.class];
        if (pieceClass && pieceClass.actions.move) {
          const moveRange = pieceClass.actions.move.range;
          
          // Show movement range
          showMoveRange(piece.q, piece.r, moveRange, gDetail);
          
          // Update hover label
          hoverLabel.textContent = `${piece.class} (${piece.side}) - Move Range: ${moveRange}`;
        }
      });
      
      circle.addEventListener("mouseleave", () => {
        // Clear movement range indicators
        clearMoveRange();
        hoverLabel.textContent = "";
      });
      
      gDetail.appendChild(circle);

      // Add piece label
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", x);
      text.setAttribute("y", y);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("dominant-baseline", "middle");
      text.setAttribute("fill", "#fff");
      text.setAttribute("font-size", SUB_HEX_SIZE);
      text.setAttribute("pointer-events", "none");
      text.textContent = piece.label;
      gDetail.appendChild(text);
    });
  }

  centerHexGroup(subHexList, gDetail, subAxialToPixel,{
    svgWidth:SVG_WIDTH,
    svgHeight:SVG_HEIGHT,
    scale:2,
    rotation:0
  });
}

// New function to set up player controls
function setupPlayerControls(scenario) {
  const playerPiecesList = document.getElementById("player-pieces");
  playerPiecesList.innerHTML = ""; // Clear existing
  pieceSelections.clear(); // Clear the global map

  // Filter for player pieces
  const playerPieces = scenario.pieces.filter(p => p.side === "player");

  function updateActionDescriptions() {
    const actionDesc = document.getElementById("action-description");
    const descriptions = [];
    
    pieceSelections.forEach((selection, pieceLabel) => {
      if (selection.action && selection.action !== "pass") {
        let desc = `${selection.class} (${pieceLabel}): ${selection.description}`;
        if (selection.targetHex) {
          desc += ` to hex (${selection.targetHex.q}, ${selection.targetHex.r})`;
        }
        descriptions.push(desc);
      }
    });
    
    actionDesc.innerHTML = descriptions.length > 0 ? descriptions.join('<br><br>') : "";
  }

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
    
    const labelSpan = document.createElement("span");
    labelSpan.textContent = `${piece.class} (${piece.label})`;
    
    labelDiv.appendChild(colorSpan);
    labelDiv.appendChild(labelSpan);
    
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
    hexSelect.setAttribute("data-piece-label", piece.label);
    
    // Add hover handlers for hex highlighting
    hexSelect.addEventListener("mouseenter", () => {
      const selection = pieceSelections.get(piece.label);
      if (selection && selection.targetHex) {
        const hex = document.querySelector(`polygon[data-q="${selection.targetHex.q}"][data-r="${selection.targetHex.r}"]`);
        if (hex) {
          hex.classList.add("highlighted");
        }
      }
    });

    hexSelect.addEventListener("mouseleave", () => {
      // Remove highlight from all hexes
      document.querySelectorAll(".hex-region.highlighted").forEach(hex => {
        hex.classList.remove("highlighted");
      });
    });

    // Initialize piece selection tracking
    pieceSelections.set(piece.label, {
      class: piece.class,
      action: "pass",
      description: "",
      targetHex: null
    });

    // Handle action selection
    select.addEventListener("change", (e) => {
      const actionName = e.target.value;
      const selection = pieceSelections.get(piece.label);
      
      if (actionName === "pass") {
        selection.action = "pass";
        selection.description = "";
        selection.targetHex = null;
        hexSelect.textContent = "Click to select hex";
        hexSelect.style.display = "none";
      } else if (actionName && pieceClass.actions[actionName]) {
        selection.action = actionName;
        selection.description = pieceClass.actions[actionName].description;
        hexSelect.style.display = actionName === "move" ? "block" : "none";
      }
      
      updateActionDescriptions();
    });

    // Handle hex selection
    hexSelect.addEventListener("click", () => {
      // Clear any existing hex selection mode and ranges
      if (currentHexSelector) {
        currentHexSelector.classList.remove("selecting");
        // Clear any existing range indicators
        document.querySelectorAll(".hex-region.in-range").forEach(hex => {
          hex.classList.remove("in-range");
        });
      }
      
      // Enter hex selection mode
      isSelectingHex = true;
      currentHexSelector = hexSelect;
      hexSelect.classList.add("selecting");
      hexSelect.textContent = "Selecting...";

      // Find the piece's current position and show its move range
      const pieceLabel = hexSelect.getAttribute("data-piece-label");
      const piece = scenario.pieces.find(p => p.label === pieceLabel);
      const pieceClass = piecesData.classes[piece.class];
      
      if (piece && pieceClass && pieceClass.actions.move) {
        const range = pieceClass.actions.move.range;
        
        // Create a set of occupied positions
        const occupiedPositions = new Set(
          scenario.pieces.map(p => `${p.q},${p.r}`)
        );
        
        // For each hex within range
        for (let q = -range; q <= range; q++) {
          for (let r = -range; r <= range; r++) {
            // Check if hex is within range (using axial distance)
            if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * range) {
              const targetQ = piece.q + q;
              const targetR = piece.r + r;
              
              // Don't highlight if:
              // 1. It's the piece's own hex
              // 2. The hex is occupied by any piece
              if (q === 0 && r === 0 || occupiedPositions.has(`${targetQ},${targetR}`)) continue;
              
              // Find and highlight the hex
              const hex = document.querySelector(`polygon[data-q="${targetQ}"][data-r="${targetR}"]`);
              if (hex) {
                hex.classList.add("in-range");
              }
            }
          }
        }
      }
    });

    // Initially hide hex select
    hexSelect.style.display = "none";
    
    li.appendChild(labelDiv);
    li.appendChild(select);
    li.appendChild(hexSelect);
    playerPiecesList.appendChild(li);
  });

  // Initialize Sortable for drag-and-drop
  new Sortable(playerPiecesList, {
    animation: 150,
    ghostClass: 'sortable-ghost'
  });

  // Set up complete turn button handler
  const completeTurnBtn = document.getElementById("complete-turn");
  completeTurnBtn.addEventListener("click", () => {
    // This will be implemented later
    console.log("Turn completed!");
  });

  // Initial update of descriptions
  updateActionDescriptions();
  
  return pieceSelections; // Return this so we can use it in hex click handlers
}

// Helper function to show movement range
function showMoveRange(centerQ, centerR, range, parentGroup) {
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
  const rangeHexes = document.getElementsByClassName("move-range");
  while (rangeHexes.length > 0) {
    rangeHexes[0].remove();
  }
  
  // Also clear in-range highlights
  document.querySelectorAll(".hex-region.in-range").forEach(hex => {
    hex.classList.remove("in-range");
  });
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

/** handle zoom */
function handleToggleZoom(){
  if(currentView==="region"){
    drawWorldView();
  } else if(currentView==="section"){
    drawRegionView(currentRegion);
  }
}

/** color palette */
function regionColor(id){
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
