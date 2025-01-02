let worldData = null;
let currentView = "world"; // "world", "region", or "section"
let currentRegion = null;
let currentSection = null;

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

  // In drawRegionView, after we build each hex polygon:
  poly.addEventListener("click", () => {
    currentRegion = region;
    // check if there's a puzzle scenario for this hex
    let puzzle = findPuzzleScenario(region, { q: hex.q, r: hex.r });
    drawHexDetailView(region, hex, puzzle);
  });

  // We'll define a helper:
  function findPuzzleScenario(region, clickedHex) {
    if (!region.puzzleScenarios) return null;
    for (let sc of region.puzzleScenarios) {
      if (sc.triggerHex.q === clickedHex.q && sc.triggerHex.r === clickedHex.r) {
        return sc; // found the matching scenario
      }
    }
    return null;
  }

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

  // build sub-hex coords
  let subHexList=[];
  for(let q=-SUB_GRID_RADIUS; q<=SUB_GRID_RADIUS;q++){
    for(let r=-SUB_GRID_RADIUS; r<=SUB_GRID_RADIUS; r++){
      if(Math.abs(q+r)<=SUB_GRID_RADIUS){
        subHexList.push({q,r});
      }
    }
  }

  subHexList.forEach(sh=>{
    let {x,y}= subAxialToPixel(sh.q,sh.r);
    let poly=document.createElementNS("http://www.w3.org/2000/svg","polygon");
    poly.setAttribute("class","hex-region");
    poly.setAttribute("points",subHexPolygonPoints(x,y));
    poly.setAttribute("fill", regionColor(region.regionId));
    poly.addEventListener("mouseenter",()=>{
      hoverLabel.textContent=`(q=${sh.q},r=${sh.r}) of ${region.name}`;
    });
    poly.addEventListener("mouseleave",()=>{hoverLabel.textContent="";});
    gDetail.appendChild(poly);
  });

  // If puzzleScenario is non-null, do puzzle logic:
  if (puzzleScenario) {
    drawPuzzleScenario(gDetail, puzzleScenario);
  } else {
    // do your default sub-hex or "drawDefaultSubGrid"
    drawDefaultSubGrid(gDetail, region, clickedHex, hoverLabel);
  }

  centerHexGroup(subHexList, gDetail, subAxialToPixel,{
    svgWidth:SVG_WIDTH,
    svgHeight:SVG_HEIGHT,
    scale:2,
    rotation:0
  });
}

function drawPuzzleScenario(gDetail, scenario) {
  const SUB_GRID_RADIUS = scenario.subGridRadius || 3;  // fallback if missing
  const SUB_HEX_SIZE = 30;

  const blockedSet = new Set(
    scenario.blockedHexes.map(b => `${b.q},${b.r}`)
  );

  // Build a map from (q,r) -> piece info
  let pieceMap = new Map();
  (scenario.pieces || []).forEach(p => {
    pieceMap.set(`${p.q},${p.r}`, p);
  });

  // Build sub-hex coords
  let subHexList = [];
  for (let q = -SUB_GRID_RADIUS; q <= SUB_GRID_RADIUS; q++) {
    for (let r = -SUB_GRID_RADIUS; r <= SUB_GRID_RADIUS; r++) {
      if (Math.abs(q + r) <= SUB_GRID_RADIUS) {
        subHexList.push({q, r});
      }
    }
  }

  // We'll define local axialToPixel for puzzle
  function puzzleAxialToPixel(q, r){
    let x = SUB_HEX_SIZE * SQRT3 * (q + r/2);
    let y = SUB_HEX_SIZE * (3/2)*r;
    return {x,y};
  }
  function puzzleHexPoints(cx, cy){
    let pts=[];
    for(let i=0;i<6;i++){
      let deg=60*i+30;
      let rad=Math.PI/180*deg;
      let px=cx + SUB_HEX_SIZE*Math.cos(rad);
      let py=cx + SUB_HEX_SIZE*Math.sin(rad);
      pts.push(`${px},${py}`);
    }
    return pts.join(" ");
  }

  // Draw each sub-hex
  subHexList.forEach(sh => {
    let {x,y} = puzzleAxialToPixel(sh.q, sh.r);

    let poly = document.createElementNS("http://www.w3.org/2000/svg","polygon");
    poly.setAttribute("stroke","#666");
    poly.setAttribute("stroke-width","1");
    poly.setAttribute("points", puzzleHexPoints(x,y));

    const key = `${sh.q},${sh.r}`;
    if (blockedSet.has(key)) {
      poly.setAttribute("fill","lightgray");
    } else {
      if (pieceMap.has(key)) {
        let piece = pieceMap.get(key);
        poly.setAttribute("fill", piece.color || "#f0f0f0");

        // add text label for piece
        let t = document.createElementNS("http://www.w3.org/2000/svg","text");
        t.setAttribute("x", x);
        t.setAttribute("y", y+5);
        t.setAttribute("text-anchor","middle");
        t.setAttribute("font-size","14");
        t.setAttribute("fill","#fff");
        t.textContent = piece.label || piece.name.substr(0,1);
        gDetail.appendChild(t);
      } else {
        poly.setAttribute("fill","#fafafa");
      }
    }

    gDetail.appendChild(poly);
  });

  // center them
  centerHexGroup(subHexList, gDetail, puzzleAxialToPixel, {
    svgWidth: SVG_WIDTH,
    svgHeight: SVG_HEIGHT,
    scale: 1.5,
    rotation: 0
  });

  // optional puzzle legend
  addPuzzleLegend(gDetail, scenario);
}

function addPuzzleLegend(gDetail, scenario){
  const legendX=50, legendY=70;
  // We'll gather unique piece types from scenario
  let uniqueTypes = new Map();
  (scenario.pieces||[]).forEach(p=>{
    if(!uniqueTypes.has(p.name)){
      uniqueTypes.set(p.name, { color:p.color, label:p.label });
    }
  });

  // Also add "Blocked Hex" to the legend
  uniqueTypes.set("Blocked", { color:"lightgray", label:"Blocked"});

  let i=0;
  uniqueTypes.forEach((val, key)=>{
    let boxY = legendY + i*20;
    i++;
    let rect = document.createElementNS("http://www.w3.org/2000/svg","rect");
    rect.setAttribute("x", legendX);
    rect.setAttribute("y", boxY);
    rect.setAttribute("width","15");
    rect.setAttribute("height","15");
    rect.setAttribute("fill", val.color||"#ccc");
    gDetail.appendChild(rect);

    let txt = document.createElementNS("http://www.w3.org/2000/svg","text");
    txt.setAttribute("x", legendX+20);
    txt.setAttribute("y", boxY+12);
    txt.setAttribute("font-size","14");
    txt.setAttribute("fill","#222");
    txt.textContent = `${key} (${val.label})`;
    gDetail.appendChild(txt);
  });
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

// ============== HELPER FOR PERIMETER OUTLINE ==================

/**
 * Return an array [ [x1,y1],[x2,y2] ] for each of 6 edges in pixel coords.
 */
function getHexEdges(q, r){
  let c = axialToPixel(q,r);
  let corners=[];
  for(let i=0;i<6;i++){
    let deg=60*i+30;
    let rad=Math.PI/180*deg;
    let px=c.x+ HEX_SIZE*Math.cos(rad);
    let py=c.y+ HEX_SIZE*Math.sin(rad);
    corners.push({x:px,y:py});
  }
  let edges=[];
  for(let i=0;i<6;i++){
    let c1=corners[i];
    let c2=corners[(i+1)%6];
    edges.push([[c1.x,c1.y],[c2.x,c2.y]]);
  }
  return edges;
}

/**
 * Return axial neighbors for a hex
 */
function getHexNeighbors(q,r){
  return [
    {q:q+1,r:r},
    {q:q-1,r:r},
    {q:q,r:r+1},
    {q:q,r:r-1},
    {q:q+1,r:r-1},
    {q:q-1,r:r+1}
  ];
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
