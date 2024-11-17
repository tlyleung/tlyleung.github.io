---
layout: artwork
artist: Liam Osterfeld
title: Black Acid
dimensions: 19 ⅝ × 15 ¾ inches (50 × 40 cm)
medium: Acrylic on wood
p5: true
---

<script>
const WIDTH = 1000;
const HEIGHT = 800;
const SCALE = 0.67;
const RENDERER = 'P2D';
const SEED = 0;

const COLORS = [
  '#eccfd4',
  '#dfd8c6',
  '#374c2b',
  '#68376f',
  '#326d97',
  '#02306b',
  '#bd5d10',
  '#e6cab5',
  '#a8421a',
  '#8a944b',
  '#bb6eb2',
  '#7088de',
  '#8a0c00',
  '#e8af3b',
  '#ebcfa8',
  '#4c7347',
  '#639d6b',
  '#ddabb4',
  '#f2eeeb',
];

function preload() {
  seed = SEED;
}

function sketch() {
  canvas.elt.setAttribute("title", `Seed: ${seed}`);

  pg.clear();
  pg.background('#0a1616');
  
  for (let x = 100; x <= 900; x += 25) {
    for (let y = 100; y <= 700; y += 25) {
      const colors = shuffle(COLORS);
      new Fleck(x, y, 8, 11, colors[0]).draw();
      new Fleck(x, y, 5, 8, colors[1]).draw();
      new Fleck(x, y, 3, 5, colors[2]).draw();
    }
  }

  image(pg, 0, 0, WIDTH, HEIGHT);
}

class Fleck {
  constructor(x, y, rangeMin, rangeMax, color) {
    this.pos = createVector(x, y);
    this.angles = [];
    this.values = [];
    this.color = color;

    // Generate random angles until completing one rotation
    let total = 0;
    while (true) {
      total += random(0.5, 0.9);
      if (total < TWO_PI) {
        this.angles.push(total);
        this.values.push(loopingNoise(total, rangeMin, rangeMax));
      } else {
        break;
      }
    }
  }

  draw() {
    pg.push();
    pg.translate(this.pos.x, this.pos.y);
    pg.noStroke();
    pg.fill(this.color);
    pg.beginShape();

    for (let i = 0; i < this.angles.length; i++) {
      const v = circlePosition(this.angles[i], this.values[i]);
      pg.curveVertex(v.x, v.y);
    }

    // Redo first 3 values for smooth shape closure
    for (let i = 0; i < 3; i++) {
      const v = circlePosition(this.angles[i], this.values[i]);
      pg.curveVertex(v.x, v.y);
    }

    pg.endShape(CLOSE);
    pg.pop();
  }
}

function circlePosition(angle, value) {
  const x = cos(angle) * value;
  const y = sin(angle) * value;
  return createVector(x, y);
}

function loopingNoise(angle, rangeMin, rangeMax) {
  const x = cos(angle) * 50;
  const y = sin(angle) * 50;
  return rangeMin + noise(x, y) * (rangeMax - rangeMin);
}
</script>