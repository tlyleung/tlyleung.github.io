const FRAME_ASPECT_RATIO = 5 / 6;

let canvas;
let pg;
let seed;

function setup() {
  const renderer = RENDERER === "P2D" ? P2D : WEBGL;
  canvas = createCanvas(WIDTH, HEIGHT, renderer).parent("sketch");
  pg = createGraphics(WIDTH, HEIGHT);

  // Initialize seed from query or generate a new one
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.has("seed")) {
    seed = urlParams.get("seed");
  }
  noiseSeed(seed);
  randomSeed(seed);

  // Draw initial content and scale the canvas
  sketch();
  scaleCanvas();

  // Event listeners for resizing and button controls
  window.addEventListener("resize", scaleCanvas);
  setupControls();
}

function generateSeed() {
  return Math.floor(Math.random() * 1_000_000);
}

function scaleCanvas() {
  const frameContainer = document.getElementById("frame-container");
  const frameContainerWidth = frameContainer ? frameContainer.offsetWidth : 0;
  const frameContainerHeight = frameContainer ? frameContainer.offsetHeight : 0;

  // Calculate frame dimensions to fit aspect ratio within frame container
  let frameWidth, frameHeight;
  if (frameContainerWidth / FRAME_ASPECT_RATIO > frameContainerHeight) {
    frameWidth = frameContainerHeight * FRAME_ASPECT_RATIO;
    frameHeight = frameContainerHeight;
  } else {
    frameWidth = frameContainerWidth;
    frameHeight = frameContainerWidth * FRAME_ASPECT_RATIO;
  }

  const maxDimension = Math.max(WIDTH, HEIGHT);
  const minDimension = Math.min(WIDTH, HEIGHT);

  const scale = Math.min(frameWidth / maxDimension, frameHeight / minDimension) * SCALE;

  // Apply scaled dimensions to `sketch` container and transform to canvas
  document.getElementById("sketch").style.width = `${WIDTH * scale}px`;
  document.getElementById("sketch").style.height = `${HEIGHT * scale}px`;
  canvas.style("transform", `scale(${scale})`).style("transform-origin", "top left");
}

function setupControls() {
  document.getElementById("generate").addEventListener("click", () => {
    const url = new URL(window.location.href);
    url.searchParams.delete("seed");
    window.history.replaceState({}, document.title, url);
    seed = generateSeed();
    noiseSeed(seed);
    randomSeed(seed);
    sketch();
  });

  document.getElementById("share").addEventListener("click", () => {
    const url = new URL(window.location.href);
    url.searchParams.set("seed", seed);
    navigator.clipboard.writeText(url.href).then(() => showTemporaryMessage("share", "Link Copied"));
  });
}

function showTemporaryMessage(elementId, message) {
  const element = document.getElementById(elementId);
  const span = element.querySelector("span");
  const originalText = span.innerText;
  span.innerText = message;
  setTimeout(() => (span.innerText = originalText), 2000);
}