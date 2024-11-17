---
layout: artwork
artist: Tim Sheridan
title: Emergency
dimensions: 19 ⅝ × 15 ¾ inches (50 × 40 cm)
medium: Acrylic lacquer on canvas
p5: true
---

<script>
const WIDTH = 1000;
const HEIGHT = 800;
const SCALE = 0.67;
const RENDERER = 'P2D';
const SEED = 0;

const COLORS = [
  '#7f888e',  // grey
  '#00704c',  // green
  '#ffffff',  // white
  '#ff5000',  // orange
  '#ff0000',  // light orange
  '#000000',  // black
  '#cfab00',  // yellow
  '#eeee00',  // bright yellow
  '#a70017',  // red
  '#0050a8',  // blue
];

function preload() {
  seed = SEED;
}

function sketch() {
  canvas.elt.setAttribute("title", `Seed: ${seed}`);

  pg.clear();
  
  // Randomize colors and set background
  const colors = shuffle(COLORS);
  pg.background(colors[0]);
  pg.fill(colors[colors.length - 1]);
  pg.noStroke();

  // Draw shapes with the specified vertex pattern
  for (let c = -800; c < 800; c += 200) {
    pg.beginShape();
    pg.vertex(500, c);
    pg.vertex(1000, 500 + c);
    pg.vertex(1000, 600 + c);
    pg.vertex(500, 100 + c);
    pg.vertex(0, 600 + c);
    pg.vertex(0, 500 + c);
    pg.endShape(CLOSE);
  }

  image(pg, 0, 0, WIDTH, HEIGHT);
}
</script>