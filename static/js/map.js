let worldData = null;
let piecesData = null;
let currentView = "world"; // "world", "region", or "section"
let currentRegion = null;
let currentSection = null;
let isSelectingHex = false;
let currentHexSelector = null;
let pieceSelections = new Map(); // Make this global
let puzzleScenario = null;
let battleLog = [];
let blockedHexes = new Set(); // Add this global variable
let delayedAttacks = []; // Array to store attacks that are delayed due to cast_speed
let turnCounter = 0; // Track number of turns completed

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
      p.setAttribute("data-region-id", region.regionId);

      p.addEventListener("mouseenter", () => { 
        hoverLabel.textContent = region.name;
        // Highlight all hexes in this region
        document.querySelectorAll(`polygon[data-region-id="${region.regionId}"]`).forEach(hex => {
          hex.classList.add("highlighted");
        });
      });
      
      p.addEventListener("mouseleave", () => { 
        hoverLabel.textContent = "";
        // Remove highlight from all hexes in this region
        document.querySelectorAll(`polygon[data-region-id="${region.regionId}"]`).forEach(hex => {
          hex.classList.remove("highlighted");
        });
      });
      
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
  puzzleScenario = null;  // Reset the global variable
  if (region.puzzleScenarios) {
    puzzleScenario = region.puzzleScenarios.find(ps => 
      ps.triggerHex.q === clickedHex.q && ps.triggerHex.r === clickedHex.r
    );
  }

  // Show/hide player controls based on whether this is a puzzle scenario
  const playerControls = document.getElementById("player-controls");
  const enemyControls = document.getElementById("enemy-controls");
  if (puzzleScenario) {
    playerControls.style.display = "block";
    enemyControls.style.display = "block";
    setupPlayerControls(puzzleScenario);
    setupEnemyPiecesDisplay(puzzleScenario);
  } else {
    playerControls.style.display = "none";
    enemyControls.style.display = "none";
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

  // Reset and populate blocked hexes
  blockedHexes.clear(); // Clear previous blocked hexes
  if (puzzleScenario && puzzleScenario.blockedHexes) {
    puzzleScenario.blockedHexes.forEach(h => {
      const key = `${h.q},${h.r}`;
      blockedHexes.add(key);
      console.log("Added blocked hex:", key); // Debug log
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
    
    const hexKey = `${sh.q},${sh.r}`;
    // If hex is blocked in puzzle scenario, make it black
    if (blockedHexes.has(hexKey)) {
      console.log("Coloring hex black:", hexKey); // Debug log
      poly.setAttribute("fill", "#000000");
      poly.setAttribute("class", "hex-region blocked"); // Add blocked class
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
          } else if (actionData.action_type === 'swap_position') {
            const targetPiece = puzzleScenario.pieces.find(p => 
              p.q === sh.q && p.r === sh.r && 
              (!actionData.ally_only || p.side === selection.side)
            );
            
            if (targetPiece && poly.classList.contains("in-range")) {
              selection.targetHex = { q: sh.q, r: sh.r };
              currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
              currentHexSelector.classList.remove("selecting");
              isSelectingHex = false;
              currentHexSelector = null;
              
              document.querySelectorAll(".hex-region.in-range").forEach(hex => {
                hex.classList.remove("in-range");
                hex.classList.remove("attack");
              });
              
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'single_target_attack') {
            const targetPiece = puzzleScenario.pieces.find(p => 
              p.q === sh.q && p.r === sh.r && p.side !== 'player'
            );
            
            if (targetPiece && poly.classList.contains("in-range")) {
              selection.targetHex = { q: sh.q, r: sh.r };
              currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
              currentHexSelector.classList.remove("selecting");
              isSelectingHex = false;
              currentHexSelector = null;
              
              document.querySelectorAll(".hex-region.in-range").forEach(hex => {
                hex.classList.remove("in-range");
                hex.classList.remove("attack");
              });
              
              updateActionDescriptions();
              validateTurnCompletion();
            }
          } else if (actionData.action_type === 'multi_target_attack') {
            const targetPiece = puzzleScenario.pieces.find(p => 
              p.q === sh.q && p.r === sh.r && p.side !== 'player'
            );
            
            if (targetPiece && poly.classList.contains("in-range")) {
              // Initialize or add to target list
              if (!selection.targetHexes) {
                selection.targetHexes = [];
              }
              
              // Check if we haven't reached max targets
              if (selection.targetHexes.length < actionData.max_num_targets) {
                selection.targetHexes.push({ q: sh.q, r: sh.r });
                currentHexSelector.textContent = selection.targetHexes
                  .map(h => `(${h.q},${h.r})`)
                  .join(', ');
                
                // If we've reached max targets, end selection
                if (selection.targetHexes.length === actionData.max_num_targets) {
                  currentHexSelector.classList.remove("selecting");
                  isSelectingHex = false;
                  currentHexSelector = null;
                  
                  document.querySelectorAll(".hex-region.in-range").forEach(hex => {
                    hex.classList.remove("in-range");
                    hex.classList.remove("attack");
                  });
                }
                
                updateActionDescriptions();
                validateTurnCompletion();
              }
            }
          } else if (actionData.action_type === 'aoe') {
            if (poly.classList.contains("in-range")) {
              selection.targetHex = { q: sh.q, r: sh.r };
              // Also store affected hexes for visualization
              selection.affectedHexes = [];
              
              // Calculate all hexes within radius
              const radius = actionData.radius;
              for (let q = -radius; q <= radius; q++) {
                for (let r = -radius; r <= radius; r++) {
                  if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * radius) {
                    const affectedQ = sh.q + q;
                    const affectedR = sh.r + r;
                    
                    // Check if there's an enemy in this hex
                    const hasEnemy = puzzleScenario.pieces.some(p => 
                      p.q === affectedQ && p.r === affectedR && p.side !== 'player'
                    );
                    
                    if (hasEnemy) {
                      selection.affectedHexes.push({ q: affectedQ, r: affectedR });
                    }
                  }
                }
              }
              
              currentHexSelector.textContent = `(${sh.q}, ${sh.r})`;
              currentHexSelector.classList.remove("selecting");
              isSelectingHex = false;
              currentHexSelector = null;
              
              document.querySelectorAll(".hex-region.in-range").forEach(hex => {
                hex.classList.remove("in-range");
                hex.classList.remove("attack");
              });
              
              updateActionDescriptions();
              validateTurnCompletion();
            }
          }
        }
      }
    });

    // Add hover behavior for AOE preview during hex selection
    poly.addEventListener("mouseenter", () => {
      if (isSelectingHex && currentHexSelector) {
        const pieceLabel = currentHexSelector.getAttribute("data-piece-label");
        const selection = pieceSelections.get(pieceLabel);
        
        if (selection) {
          const pieceClass = piecesData.classes[selection.class];
          const actionData = pieceClass.actions[selection.action];
          
          if (actionData.action_type === 'aoe' && poly.classList.contains("in-range")) {
            const q = parseInt(poly.getAttribute("data-q"));
            const r = parseInt(poly.getAttribute("data-r"));
            
            // Calculate and highlight affected hexes
            const radius = actionData.radius;
            for (let dq = -radius; dq <= radius; dq++) {
              for (let dr = -radius; dr <= radius; dr++) {
                if (Math.abs(dq) + Math.abs(dr) + Math.abs(-dq-dr) <= 2 * radius) {
                  const affectedQ = q + dq;
                  const affectedR = r + dr;
                  
                  const affectedHex = document.querySelector(`polygon[data-q="${affectedQ}"][data-r="${affectedR}"]`);
                  if (affectedHex) {
                    affectedHex.classList.add("aoe-preview");
                    
                    // If hex contains an enemy, also mark it as an attack hex
                    const hasEnemy = puzzleScenario.pieces.some(p => 
                      p.q === affectedQ && p.r === affectedR && p.side !== 'player'
                    );
                    if (hasEnemy) {
                      affectedHex.classList.add("attack");
                    }
                  }
                }
              }
            }
          }
        }
      }
      
      // Original hover behavior for hex coordinates
      const q = poly.getAttribute("data-q");
      const r = poly.getAttribute("data-r");
      hoverLabel.textContent = `(q=${q},r=${r}) of ${currentRegion.name}`;
    });

    poly.addEventListener("mouseleave", () => {
      // Clear AOE preview
      document.querySelectorAll(".hex-region.aoe-preview").forEach(hex => {
        hex.classList.remove("aoe-preview");
        hex.classList.remove("attack");
      });
      
      // Original behavior to clear hover label
      hoverLabel.textContent = "";
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
      circle.setAttribute("pointer-events", "none"); // Make circle ignore pointer events
      
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
      text.setAttribute("pointer-events", "none"); // Make text ignore pointer events
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

// Move this outside of setupPlayerControls to make it globally accessible
function updateActionDescriptions() {
    const actionDesc = document.getElementById("action-description");
    const descriptions = [];
    
    pieceSelections.forEach((selection, pieceLabel) => {
        if (selection.action && selection.action !== "pass") {
            let desc = `${selection.class} (${pieceLabel}): ${selection.description}`;
            
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
    
    pieceSelections.forEach((selection, pieceLabel) => {
        console.log(`Checking ${pieceLabel}:`, selection);
        
        if (selection.action === "pass") return;
        
        const piece = puzzleScenario.pieces.find(p => p.label === pieceLabel);
        if (!piece) return;
        
        const pieceClass = piecesData.classes[piece.class];
        const actionData = pieceClass.actions[selection.action];
        
        if (!actionData) return;
        
        switch (actionData.action_type) {
            case 'move':
                if (!selection.targetHex) {
                    console.log(`${pieceLabel} has no target hex`);
                    isValid = false;
                    break;
                }
                
                // Check if target hex is blocked
                const targetKey = `${selection.targetHex.q},${selection.targetHex.r}`;
                if (blockedHexes.has(targetKey)) {
                    console.log(`${pieceLabel} target hex is blocked`);
                    isValid = false;
                    break;
                }
                
                // Check distance
                const dx = Math.abs(selection.targetHex.q - piece.q);
                const dy = Math.abs(selection.targetHex.r - piece.r);
                const dz = Math.abs(-selection.targetHex.q - selection.targetHex.r + piece.q + piece.r);
                const distance = Math.max(dx, dy, dz);
                
                if (distance > actionData.range) {
                    console.log(`${pieceLabel} target is out of range`);
                    isValid = false;
                }
                
                // Check if occupied
                const isOccupied = puzzleScenario.pieces.some(p => 
                    p.q === selection.targetHex.q && p.r === selection.targetHex.r
                );
                if (isOccupied) {
                    console.log(`${pieceLabel} target hex is occupied`);
                    isValid = false;
                }
                break;
                
            case 'swap_position':
                if (!selection.targetHex) {
                    console.log(`${pieceLabel} has no target hex`);
                    isValid = false;
                    break;
                }

                // Check distance
                const swapDx = Math.abs(selection.targetHex.q - piece.q);
                const swapDy = Math.abs(selection.targetHex.r - piece.r);
                const swapDz = Math.abs(-selection.targetHex.q - selection.targetHex.r + piece.q + piece.r);
                const swapDistance = Math.max(swapDx, swapDy, swapDz);
                
                if (swapDistance > actionData.range) {
                    console.log(`${pieceLabel} swap target is out of range`);
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
                    console.log(`${pieceLabel} target hex has no piece to swap with`);
                    isValid = false;
                    break;
                }

                // Check ally_only constraint
                if (actionData.ally_only && targetPiece.side !== piece.side) {
                    console.log(`${pieceLabel} can only swap with allies`);
                    isValid = false;
                    break;
                }

                // Check if target hex is blocked
                const swapTargetKey = `${selection.targetHex.q},${selection.targetHex.r}`;
                if (blockedHexes.has(swapTargetKey)) {
                    console.log(`${pieceLabel} swap target hex is blocked`);
                    isValid = false;
                }
                break;
                
            case 'single_target_attack':
                if (!selection.targetHex) {
                    console.log(`${pieceLabel} has no target hex`);
                    isValid = false;
                    break;
                }
                
                // Check distance
                const attackDx = Math.abs(selection.targetHex.q - piece.q);
                const attackDy = Math.abs(selection.targetHex.r - piece.r);
                const attackDz = Math.abs(-selection.targetHex.q - selection.targetHex.r + piece.q + piece.r);
                const attackDistance = Math.max(attackDx, attackDy, attackDz);
                
                if (attackDistance > actionData.range) {
                    console.log(`${pieceLabel} target is out of range`);
                    isValid = false;
                }
                
                // Check if target has enemy
                const hasEnemy = puzzleScenario.pieces.some(p => 
                    p.q === selection.targetHex.q && 
                    p.r === selection.targetHex.r && 
                    p.side !== 'player'
                );
                if (!hasEnemy) {
                    console.log(`${pieceLabel} target hex has no enemy`);
                    isValid = false;
                }
                break;
                
            case 'multi_target_attack':
                if (!selection.targetHexes || selection.targetHexes.length === 0) {
                    console.log(`${pieceLabel} has no targets`);
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
                        console.log(`${pieceLabel} target (${target.q},${target.r}) is out of range`);
                        isValid = false;
                    }
                    
                    // Check if target has enemy
                    const hasEnemy = puzzleScenario.pieces.some(p => 
                        p.q === target.q && 
                        p.r === target.r && 
                        p.side !== 'player'
                    );
                    if (!hasEnemy) {
                        console.log(`${pieceLabel} target hex (${target.q},${target.r}) has no enemy`);
                        isValid = false;
                    }
                });
                break;
                
            case 'aoe':
                if (!selection.targetHex) {
                    console.log(`${pieceLabel} has no target hex`);
                    isValid = false;
                    break;
                }
                
                // Check if center point is in range
                const aoeDx = Math.abs(selection.targetHex.q - piece.q);
                const aoeDy = Math.abs(selection.targetHex.r - piece.r);
                const aoeDz = Math.abs(-selection.targetHex.q - selection.targetHex.r + piece.q + piece.r);
                const aoeDistance = Math.max(aoeDx, aoeDy, aoeDz);
                
                if (aoeDistance > actionData.range) {
                    console.log(`${pieceLabel} target center is out of range`);
                    isValid = false;
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
  // Clear any existing highlights first
  document.querySelectorAll(".hex-region.in-range, .hex-region.attack").forEach(hex => {
    hex.classList.remove("in-range");
    hex.classList.remove("attack");
  });

  const actionData = pieceClass.actions[actionName];
  if (!actionData || !actionData.range) return;

  const range = actionData.range;
  console.log("Showing range for", piece.class, "action:", actionName, "range:", range);

  // For each hex within range
  for (let q = -range; q <= range; q++) {
    for (let r = -range; r <= range; r++) {
      if (Math.abs(q) + Math.abs(r) + Math.abs(-q-r) <= 2 * range) {
        const targetQ = piece.q + q;
        const targetR = piece.r + r;
        
        // Skip if target hex is blocked
        const targetKey = `${targetQ},${targetR}`;
        if (blockedHexes.has(targetKey)) continue;

        if (actionData.action_type === 'move') {
          // Don't highlight if occupied
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
        } else if (actionData.action_type === 'single_target_attack' || 
                   actionData.action_type === 'multi_target_attack' || 
                   actionData.action_type === 'dark_bolt') {
          // For player pieces, highlight enemy pieces as targets
          // For enemy pieces, highlight player pieces as targets
          const hasValidTarget = puzzleScenario.pieces.some(p => 
            p.q === targetQ && p.r === targetR && p.side !== piece.side
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
            // For player pieces, highlight hexes with enemy pieces
            // For enemy pieces, highlight hexes with player pieces
            const hasValidTarget = puzzleScenario.pieces.some(p => 
              p.q === targetQ && p.r === targetR && p.side !== piece.side
            );
            if (hasValidTarget) {
              hex.classList.add("attack");
            }
          }
        }
      }
    }
  }
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
    hexSelect.setAttribute("data-piece-label", piece.label);
    
    // Create and add info section
    const infoSection = createPieceInfoSection(piece, pieceClass);
    
    // Add expand button click handler
    expandButton.addEventListener("click", (e) => {
      e.stopPropagation(); // Prevent piece item click
      infoSection.classList.toggle("visible");
      expandButton.textContent = infoSection.classList.contains("visible") ? "🔼" : "ℹ️";
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
        const actionData = pieceClass.actions[actionName];
        selection.action = actionName;
        selection.description = actionData.description;
        selection.targetHex = null; // Reset target when action changes
        selection.targetHexes = null; // Reset multi-target list
        selection.affectedHexes = null; // Reset AOE affected hexes

        // Show hex selector for any action that needs target selection
        const needsTargetSelection = ['move', 'swap_position', 'single_target_attack', 'multi_target_attack', 'aoe'].includes(actionData.action_type);
        hexSelect.style.display = needsTargetSelection ? "block" : "none";
        hexSelect.textContent = "Click to select hex";
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
        document.querySelectorAll(".hex-region.in-range").forEach(hex => {
          hex.classList.remove("in-range");
          hex.classList.remove("attack");
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
      
      const piece = scenario.pieces.find(p => p.label === pieceLabel);
      const pieceClass = piecesData.classes[piece.class];
      const selection = pieceSelections.get(pieceLabel);
      const actionData = pieceClass.actions[selection.action];
      
      console.log("Piece data:", { piece, action: selection.action, actionData }); // Debug log
      
      if (piece && pieceClass && actionData) {
        showPieceActionRange(piece, pieceClass, selection.action);
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
    pieceSelections.forEach((selection, pieceLabel) => {
        const piece = scenario.pieces.find(p => p.label === pieceLabel);
        const pieceClass = piecesData.classes[piece.class];
        const actionData = pieceClass.actions[selection.action];
        
        if (!actionData) return;

        switch (actionData.action_type) {
            case 'move':
                if (selection.targetHex) {
                    piece.q = selection.targetHex.q;
                    piece.r = selection.targetHex.r;
                    addBattleLog(`${piece.class} (${pieceLabel}) moved to (${selection.targetHex.q}, ${selection.targetHex.r})`);
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
                        addBattleLog(`${piece.class} (${pieceLabel}) swapped positions with ${targetPiece.class} (${targetPiece.label})`);
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
                        attackerLabel: pieceLabel,
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
                        attackInfo.targets = {
                            originalQ: selection.targetHex.q,
                            originalR: selection.targetHex.r,
                            label: target.label
                        };
                        addBattleLog(`${piece.class} (${pieceLabel}) begins casting ${selection.action} on ${target.class} (${target.label})`);
                    } else if (actionData.action_type === 'multi_target_attack') {
                        attackInfo.targets = selection.targetHexes.map(hex => {
                            const target = scenario.pieces.find(p => 
                                p.q === hex.q && 
                                p.r === hex.r && 
                                p.side !== 'player'
                            );
                            return {
                                originalQ: hex.q,
                                originalR: hex.r,
                                label: target.label
                            };
                        });
                        const targetLabels = attackInfo.targets.map(t => t.label).join(', ');
                        addBattleLog(`${piece.class} (${pieceLabel}) begins casting ${selection.action} on targets: ${targetLabels}`);
                    } else if (actionData.action_type === 'aoe') {
                        attackInfo.centerQ = selection.targetHex.q;
                        attackInfo.centerR = selection.targetHex.r;
                        attackInfo.radius = actionData.radius;
                        addBattleLog(`${piece.class} (${pieceLabel}) begins casting ${selection.action} centered at (${selection.targetHex.q}, ${selection.targetHex.r})`);
                    }

                    delayedAttacks.push(attackInfo);
                } else {
                    // Handle immediate attacks as before
                    if (actionData.action_type === 'single_target_attack') {
                        if (selection.targetHex) {
                            const targetIndex = scenario.pieces.findIndex(p => 
                                p.q === selection.targetHex.q && 
                                p.r === selection.targetHex.r && 
                                p.side !== 'player'
                            );
                            
                            if (targetIndex !== -1) {
                                const removedPiece = scenario.pieces[targetIndex];
                                scenario.pieces.splice(targetIndex, 1);
                                addBattleLog(`${piece.class} (${pieceLabel}) eliminated ${removedPiece.class} (${removedPiece.label}) with ${selection.action}`);
                            }
                        }
                    } else if (actionData.action_type === 'multi_target_attack') {
                        if (selection.targetHexes) {
                            selection.targetHexes.forEach(targetHex => {
                                const targetIndex = scenario.pieces.findIndex(p => 
                                    p.q === targetHex.q && 
                                    p.r === targetHex.r && 
                                    p.side !== 'player'
                                );
                                
                                if (targetIndex !== -1) {
                                    const removedPiece = scenario.pieces[targetIndex];
                                    scenario.pieces.splice(targetIndex, 1);
                                    addBattleLog(`${piece.class} (${pieceLabel}) eliminated ${removedPiece.class} (${removedPiece.label}) with ${selection.action}`);
                                }
                            });
                        }
                    } else if (actionData.action_type === 'aoe') {
                        if (selection.affectedHexes) {
                            selection.affectedHexes.forEach(targetHex => {
                                const targetIndex = scenario.pieces.findIndex(p => 
                                    p.q === targetHex.q && 
                                    p.r === targetHex.r && 
                                    p.side !== 'player'
                                );
                                
                                if (targetIndex !== -1) {
                                    const removedPiece = scenario.pieces[targetIndex];
                                    scenario.pieces.splice(targetIndex, 1);
                                    addBattleLog(`${piece.class} (${pieceLabel}) eliminated ${removedPiece.class} (${removedPiece.label}) with ${selection.action}`);
                                }
                            });
                        }
                    }
                }
                break;
        }
    });

    pieceSelections.clear();
    drawHexDetailView(currentRegion, currentSection);
  });

  // Initial validation
  validateTurnCompletion();

  // Initial update of descriptions
  updateActionDescriptions();
  
  return pieceSelections; // Return this so we can use it in hex click handlers
}
