---
layout: artwork
artist: Charles Lorent
title: Divisions
dimensions: 19 ⅝ × 15 ¾ inches (50 × 40 cm)
medium: Acrylic glass, paint and roach
p5: true
---

<script>
const WIDTH = 1000;
const HEIGHT = 800;
const SCALE = 0.67;
const RENDERER = 'P2D';
const SEED = 62178;

let fish;

function preload() {
  seed = SEED;
  fish = loadImage('/assets/images/artworks/divisions/fish.png');
}

function sketch() {
  const divisions = Math.floor(random(0, 16));
  canvas.elt.setAttribute("title", `Seed: ${seed}\nDivisions: ${divisions}`);

  pg.clear();
  pg.background('#ffffff');
  pg.noStroke();
  pg.rectMode(CORNERS);

  let xs = [0];
  let ys = [HEIGHT];

  for (let i = 0; i < divisions; i++) {
    xs.push(xs[xs.length - 1] + WIDTH / Math.pow(2, i + 1));
    ys.push(ys[ys.length - 1] - HEIGHT / Math.pow(2, i + 1));
  }

  for (let i = 0; i < divisions + 1; i++) {
    pg.fill(229, 77, 30, 256 / Math.pow(1.15, i));
    let j = Math.floor(i / 2);

    if (i % 2 === 0) {
      pg.rect(xs[j], 0, i < divisions ? xs[j + 1] : WIDTH, ys[j]);
    } else {
      pg.rect(WIDTH, ys[j], xs[j + 1], i < divisions ? ys[j + 1] : 0);
    }
  }

  pg.image(fish, (3000 / 6) - (1896 / 6), (2400 / 6) - (570 / 6), 3000 / 6, 2000 / 6);

  image(pg, 0, 0, WIDTH, HEIGHT);
}
</script>